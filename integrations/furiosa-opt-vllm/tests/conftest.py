"""Tests run on the inventory's checkpoints, served with the flags their model pages document.

A test that takes `row` runs once per `.github/benchmarks.json` row whose task is its module's
`TASK`, so a new row is tested without editing a test. The chips a module's in-process models hold
are free before the next module's tests start.
"""

from __future__ import annotations

import gc
import os
import socket
import subprocess
import time
import urllib.request
from contextlib import contextmanager, suppress
from pathlib import Path
from signal import SIGKILL, SIGTERM
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, NotRequired, Required, TypeVar

import pytest
from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError, with_config
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Pydantic parses only this module's TypedDict before Python 3.12.
from typing_extensions import TypedDict
from vllm.config.cache import CacheDType
from vllm.config.model import ConvertOption, ModelDType, RunnerOption

if TYPE_CHECKING:
    from collections.abc import Generator
    from http.client import HTTPResponse

    import torch
    from pydantic import BaseModel
    from transformers import PreTrainedModel


class Reference(NamedTuple):
    model: PreTrainedModel
    weight: torch.Tensor
    bias: torch.Tensor | None = None

    def rows(
        self, sequences: list[list[int]], side: Literal["right", "left"] = "right"
    ) -> list[torch.Tensor]:
        import torch

        width = max(map(len, sequences))
        ids = torch.full((len(sequences), width), self.model.config.pad_token_id)
        mask = torch.zeros_like(ids)
        spans = [
            slice(0, len(tokens)) if side == "right" else slice(width - len(tokens), width)
            for tokens in sequences
        ]
        for row, (tokens, span) in enumerate(zip(sequences, spans, strict=True)):
            ids[row, span] = torch.tensor(tokens)
            mask[row, span] = 1
        with torch.inference_mode():
            hidden = self.model.base_model(input_ids=ids, attention_mask=mask).last_hidden_state
        return [hidden[row, span] for row, span in enumerate(spans)]

    def check(self, actual: torch.Tensor, rows: torch.Tensor) -> None:
        # Each BF16 hidden value can move a score by its weighted relative precision.
        import torch

        expected = torch.nn.functional.linear(rows, self.weight, self.bias)
        bound = torch.nn.functional.linear(rows.abs(), self.weight.abs()) * 2**-8
        assert actual.shape == expected.shape
        assert ((actual - expected).abs() <= bound).all(), (actual, expected, bound)


class Task(TypedDict, total=False):
    """The flags every row of a task documents."""

    serve: list[str]


class Row(TypedDict):
    """A checkpoint of the inventory and the task its tests serve."""

    id: str
    task: str
    serve: NotRequired[list[str]]
    revision: NotRequired[str]
    download: NotRequired[list[str]]
    reference: NotRequired[str]


class Inventory(TypedDict):
    tasks: dict[str, Task]
    models: list[Row]


# Unknown flags fail the row's tests rather than reach `LLM` unchecked.
@with_config(ConfigDict(extra="forbid"))
class Options(TypedDict, total=False):
    """`LLM` keyword arguments of the flags the inventory documents."""

    model: Required[str]
    runner: RunnerOption
    convert: ConvertOption
    dtype: ModelDType
    generation_config: str
    hf_overrides: dict[str, JsonValue]
    chat_template: str
    kv_cache_dtype: CacheDType
    language_model_only: bool
    max_model_len: int
    tensor_parallel_size: int


INVENTORY: Final = TypeAdapter(Inventory).validate_json(
    (Path(__file__).resolve().parents[3] / ".github/benchmarks.json").read_bytes()
)
# A flag's value, where it is JSON.
JSON: Final = TypeAdapter[JsonValue](JsonValue)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "row" in metafunc.fixturenames:
        rows = [row for row in INVENTORY["models"] if row["task"] == metafunc.module.TASK]
        metafunc.parametrize("row", rows, ids=[row["id"] for row in rows], scope="module")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: pytest.Item | None
) -> Generator[None, object, object]:
    finished = yield
    module = item.getparent(pytest.Module)
    # vLLM-Omni's VAEs and PyTorch's parametrized modules form reference cycles, which keep a model
    # they reach, and its chips, until a full collection: in a long session, minutes later.
    if nextitem is not None and nextitem.getparent(pytest.Module) is not module:
        _ = gc.collect()
    return finished


@pytest.fixture(scope="module")
def flags(row: Row) -> list[str]:
    """The row's documented `vllm serve` flags: its task's, then its own."""
    return [*INVENTORY["tasks"][row["task"]].get("serve", []), *row.get("serve", [])]


@pytest.fixture(scope="module")
def options(row: Row, flags: list[str]) -> Options:
    """`LLM` keyword arguments equal to the row's documented `vllm serve` command.

    Values parse as JSON where they are JSON, as vLLM parses numbers and `--hf-overrides`; a flag
    without a value is `True`; `{checkpoint}/<file>` is that file of the downloaded checkpoint.
    """
    from huggingface_hub import hf_hub_download

    arguments: dict[str, JsonValue] = {"model": row["id"]}
    name = ""
    for flag in flags:
        if flag.startswith("--"):
            name = flag.removeprefix("--").replace("-", "_")
            arguments[name] = True
            continue
        if flag.startswith("{checkpoint}/"):
            flag = hf_hub_download(row["id"], flag.removeprefix("{checkpoint}/"))
        try:
            arguments[name] = JSON.validate_json(flag)
        except ValidationError:
            arguments[name] = flag
    return TypeAdapter(Options).validate_python(arguments)


Reply = TypeVar("Reply", bound="BaseModel")


@pytest.fixture(scope="module")
def serving() -> list[str]:
    """The `vllm serve` flags a module adds to its rows' documented ones."""
    return []


@pytest.fixture(scope="module")
def server(
    row: Row, flags: list[str], serving: list[str], tmp_path_factory: pytest.TempPathFactory
) -> Generator[str]:
    """`vllm serve` with the row's documented flags and the module's `serving` ones, on a chip of
    its own."""
    with serve([row["id"], *flags, *serving], tmp_path_factory) as url:
        yield url


@contextmanager
def serve(arguments: list[str], tmp_path_factory: pytest.TempPathFactory) -> Generator[str]:
    """The URL of `vllm serve` with `arguments`, until it and every process it started exited, so
    that a server started next finds the chips free."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
    log = tmp_path_factory.mktemp("serve") / "serve.log"
    with log.open("wb") as output:
        # A group of its own reaches the engine core even when it outlives the server.
        process = subprocess.Popen(
            ["vllm", "serve", *arguments, "--port", str(port)],
            stdout=output,
            stderr=subprocess.STDOUT,
            process_group=0,
        )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 300

    def ready() -> bool:
        try:
            response: HTTPResponse = urllib.request.urlopen(f"{url}/health", timeout=5)
        except OSError:
            return False
        with response:
            return response.status == 200

    try:
        while not ready():
            assert process.poll() is None and time.monotonic() < deadline, log.read_text()[-4000:]
            time.sleep(2)
        yield url
    finally:
        stop(process, 60)


def stop(process: subprocess.Popen[bytes], timeout: float) -> None:
    """End `process` and the processes of its group: terminate them, kill those still running
    `timeout` seconds later, and wait as long again for them."""
    import psutil

    def running() -> bool:
        for member in psutil.process_iter():
            with suppress(psutil.NoSuchProcess, ProcessLookupError):
                if (
                    os.getpgid(member.pid) == process.pid
                    and member.status() != psutil.STATUS_ZOMBIE
                ):
                    return True
        return False

    for signal in (SIGTERM, SIGKILL):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal)
        deadline = time.monotonic() + timeout
        while process.poll() is None or running():
            if time.monotonic() > deadline:
                break
            time.sleep(0.2)
        else:
            return
    raise TimeoutError(f"processes of group {process.pid} outlived SIGKILL by {timeout} s")


def post(url: str, request: BaseModel, reply: type[Reply]) -> Reply:
    """The server's answer to `request` posted to `url`, parsed as vLLM's `reply`."""
    body = request.model_dump_json(exclude_unset=True).encode()
    response: HTTPResponse = urllib.request.urlopen(
        urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=300
    )
    with response:
        return reply.model_validate_json(response.read())


@pytest.fixture(scope="module")
def reference(row: Row) -> Path:
    """The row's reference outputs, where the checkpoints action downloads them."""
    if "reference" not in row:
        raise LookupError(f"{row['id']} documents no reference outputs")
    return Path(os.environ["FURIOSA_OPT_REFERENCES"], row["reference"])


@pytest.fixture
def served(tmp_path: Path) -> Path:
    """The configuration of a checkpoint the engine serves, without its weights, in `tmp_path`."""
    from transformers import AutoConfig

    AutoConfig.from_pretrained("Qwen/Qwen3-0.6B").save_pretrained(tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    return tokenizer


@pytest.fixture
def platform_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The process environment restored after the test, since the platform's configuration check
    sets the worker start method and numba's thread count for the whole process."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Registering the thread count restores its value or its absence.
    current = os.environ.get("NUMBA_NUM_THREADS")
    monkeypatch.setenv("NUMBA_NUM_THREADS", current or "1")
    if current is None:
        monkeypatch.delenv("NUMBA_NUM_THREADS")
