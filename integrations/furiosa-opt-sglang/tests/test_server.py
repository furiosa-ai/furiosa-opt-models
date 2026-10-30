"""SGLang's HTTP server returns the checkpoint's answer with its scheduling and caching intact.

Tests run on the checkpoints of `.github/benchmarks.json`, so a new row is tested without editing
a test.
"""

import json
import os
import subprocess
import sys
import textwrap
import time
import uuid
from collections.abc import Callable, Generator
from contextlib import ExitStack, contextmanager
from http.client import HTTPResponse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, NamedTuple, NotRequired, final
from urllib.request import Request, urlopen

import pytest
from pydantic import ConfigDict, TypeAdapter, with_config

# Pydantic parses only this module's TypedDict before Python 3.12.
from typing_extensions import TypedDict

if TYPE_CHECKING:
    import torch

INVENTORY: dict[str, Any] = json.loads(
    (Path(__file__).resolve().parents[3] / ".github/benchmarks.json").read_text()
)
TEXT = [row for row in INVENTORY["models"] if row["task"] == "text"]
EMBEDDING = [row for row in INVENTORY["models"] if row["task"] == "embedding"]
RERANKING = [row for row in INVENTORY["models"] if row["task"] == "reranking"]
CLASSIFICATION = [row for row in INVENTORY["models"] if row["task"] == "classification"]
DEFAULT = next(row for row in TEXT if row["id"] == "Qwen/Qwen3-1.7B")
WATERMARK_KEY = "0x5eed"
WATERMARKED = DEFAULT | {
    "flags": [
        "--enable-watermark",
        "--watermark-config",
        json.dumps({"key": WATERMARK_KEY}),
        "--context-length",
        "1024",
    ]
}
PROCESSING = DEFAULT | {"flags": ["--enable-custom-logit-processor"]}
# Parsers split a chat answer into tool calls and reasoning; a chat with tools exceeds 128 tokens.
PARSING = DEFAULT | {
    "flags": [
        "--tool-call-parser",
        "qwen25",
        "--reasoning-parser",
        "qwen3",
        "--context-length",
        "1024",
    ]
}
PROMPTS = ["The capital of France is Paris. " * 5, "What is a compiler?"]
CACHE_DTYPE = "--kv-cache-dtype"
"""The one row flag SGLang reads as vLLM does; the others select vLLM's own loading."""
TOP = 5
# What serving each inventory checkpoint without a listed decoder waits for.
PENDING = {
    "mistralai/Mistral-7B-v0.3": "a Mistral line",
    "mistralai/Mistral-7B-v0.1": "a Mistral line",
    "RedHatAI/Mistral-Nemo-Instruct-2407-FP8": "a Mistral line of FP8 weights and an FP8 cache",
    "unsloth/gemma-4-12b-it-NVFP4": "a Gemma 4 line of NVFP4 weights",
}
# Contracts of the engine's scores, which a decoder call does not return yet.
SCORES = pytest.mark.skip(reason="pending: reads scores, which a decoder call does not return yet")


def served(rows: list[dict[str, Any]]) -> list[Any]:
    """`rows` as server parameters, those without a listed decoder pending with what they wait
    for."""
    return [
        pytest.param(
            row,
            id=row["id"],
            marks=[pytest.mark.skip(reason=f"pending: {PENDING[row['id']]}")]
            if row["id"] in PENDING
            else [],
        )
        for row in rows
    ]


def shared(flags: list[str]) -> list[str]:
    """The cache dtype flag of `flags`, in either spelling, with its value."""
    for index, flag in enumerate(flags):
        if flag == CACHE_DTYPE:
            return flags[index : index + 2]
        if flag.startswith(f"{CACHE_DTYPE}="):
            return [flag]
    return []


@pytest.fixture(scope="session", autouse=True)
def plugin(tmp_path_factory: pytest.TempPathFactory) -> None:
    """SGLang's plugin loader registers the furiosa platform, worker and hooks in a new interpreter,
    as each server process loads them; SGLang only logs a plugin or hook that fails to apply."""
    script = textwrap.dedent(
        """
        from sglang.srt.plugins import load_plugins
        from sglang.srt.plugins.hook_registry import HookRegistry
        load_plugins()
        from sglang.srt.managers import tp_worker
        from sglang.srt.platforms import current_platform
        from furiosa_opt_sglang.platform import Platform
        from furiosa_opt_sglang.worker import Worker
        assert isinstance(current_platform, Platform), current_platform
        assert tp_worker.TpModelWorker is Worker, tp_worker.TpModelWorker
        assert HookRegistry._hooks.keys() <= HookRegistry._patched, HookRegistry._patched
        """
    )
    loader = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        # The servers also start in an empty directory, so both import the same plugin.
        cwd=tmp_path_factory.mktemp("loader"),
    )
    assert loader.returncode == 0, loader.stderr


class Live(NamedTuple):
    """A started server and the function that sends it requests."""

    process: subprocess.Popen[bytes]
    send: Callable[..., HTTPResponse]


@final
class Chips:
    """The chips the tests serve on, held by one server at a time: starting a server stops the one
    before it, so each finds the chips its row needs."""

    def __init__(self) -> None:
        self._live: Live | None = None

    @contextmanager
    def serve(
        self, tmp_path_factory: pytest.TempPathFactory, row: dict[str, Any]
    ) -> Generator[Callable[..., HTTPResponse], None, None]:
        """Serve an inventory row on the chips for this scope."""
        log = tmp_path_factory.mktemp("sglang") / "server.log"
        process, url = self.start(row, log)

        def send(
            path: str, body: dict[str, object] | None = None, timeout: float = 60
        ) -> HTTPResponse:
            return urlopen(
                Request(
                    url + path,
                    data=json.dumps(body).encode() if body is not None else None,
                    headers={"Content-Type": "application/json"},
                ),
                timeout=timeout,
            )

        self._live = Live(process, send)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail(f"SGLang did not start; see {log}")
                try:
                    with urlopen(url + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(0.1)
            yield send
        finally:
            if self.holds(send):
                self.stop()

    def refusal(self, tmp_path_factory: pytest.TempPathFactory, row: dict[str, Any]) -> str:
        """The log of a server for `row` that exits before it serves."""
        log = tmp_path_factory.mktemp("sglang") / "server.log"
        process, _ = self.start(row, log)
        try:
            _ = process.wait(timeout=300)
        finally:
            self.end(process)
        return log.read_text()

    def holds(self, send: Callable[..., HTTPResponse]) -> bool:
        return self._live is not None and self._live.send is send

    def start(self, row: dict[str, Any], log: Path) -> tuple[subprocess.Popen[bytes], str]:
        """Stop the server holding the chips, then start one for `row` that logs to `log`, with
        the row's `env` over this process's environment."""
        from sglang.srt.utils.network import get_free_port

        self.stop()
        embedding = row["task"] == "embedding"
        pooling = embedding or row["task"] == "classification"
        port = get_free_port()
        with log.open("w") as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    row["id"],
                    *(["--is-embedding"] if pooling else []),
                    # SGLang serves `dimensions` only for Matryoshka embeddings, and its
                    # `--language-model-only` refuses most multimodal architectures, so a checkpoint
                    # serves its text model through the config, as language_model_only=True does.
                    "--json-model-override-args",
                    json.dumps(
                        {"is_matryoshka": True} if embedding else {"language_model_only": True}
                    ),
                    *shared(row.get("serve", [])),
                    *(["--disable-radix-cache"] if row.get("disable_radix_cache") else []),
                    # Small budgets prefill every prompt in several chunks and batch two requests.
                    "--context-length",
                    "128",
                    "--chunked-prefill-size",
                    "32",
                    "--max-running-requests",
                    "2",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    *row.get("flags", []),
                ],
                stdout=output,
                stderr=subprocess.STDOUT,
                cwd=log.parent,
                env=os.environ | row.get("env", {}),
            )
        return process, f"http://127.0.0.1:{port}"

    def stop(self) -> None:
        if self._live is not None:
            self.end(self._live.process)
            self._live = None

    @staticmethod
    def end(process: subprocess.Popen[bytes]) -> None:
        """Stop `process` and the scheduler processes that hold its chips."""
        import psutil
        from sglang.srt.utils import kill_process_tree

        if process.poll() is None:
            children = psutil.Process(process.pid).children(recursive=True)
            kill_process_tree(process.pid)
            _ = psutil.wait_procs(children, timeout=30)
        _ = process.wait(timeout=10)


@pytest.fixture(scope="session")
def chips() -> Generator[Chips, None, None]:
    held = Chips()
    yield held
    held.stop()


@pytest.fixture(scope="module")
def server(
    tmp_path_factory: pytest.TempPathFactory, chips: Chips, request: pytest.FixtureRequest
) -> Generator[Callable[..., HTTPResponse], None, None]:
    """Serve one inventory row and yield a function that sends a JSON request to a path.

    `request.param` is the row, optionally with `disable_radix_cache` and SGLang `flags` that
    override the defaults; tests that pass none share a Qwen3-1.7B server. The row's flags that
    mean the same in SGLang are passed on; the task flags are vLLM's. A test that starts its own
    server stops this one, which starts again for the next request.
    """
    row = getattr(request, "param", DEFAULT)
    with ExitStack() as stack:
        live = stack.enter_context(chips.serve(tmp_path_factory, row))

        def send(path: str, body: dict[str, object] | None = None) -> HTTPResponse:
            nonlocal live
            if not chips.holds(live):
                live = stack.enter_context(chips.serve(tmp_path_factory, row))
            return live(path, body)

        yield send


@pytest.mark.parametrize(
    ("flags", "kept"),
    [
        (
            ["--kv-cache-dtype", "fp8_e4m3", "--tokenizer-mode", "mistral"],
            ["--kv-cache-dtype", "fp8_e4m3"],
        ),
        (["--config-format", "hf", "--kv-cache-dtype=fp8_e4m3"], ["--kv-cache-dtype=fp8_e4m3"]),
        (["--language-model-only", "--config-format", "hf"], []),
    ],
    ids=["split", "joined", "absent"],
)
def test_passes_on_only_the_cache_dtype(flags: list[str], kept: list[str]) -> None:
    assert shared(flags) == kept


def assert_close(
    server: Callable[..., HTTPResponse],
    prompts: list[list[int]],
    answers: list[list[int]],
    expected: list[list[int]],
) -> None:
    """Each answer equals its expected one, or from the first token they differ at, each one's
    tokens are greedy on its own prefix by Transformers' scores of the served checkpoint: within 5%
    of the top probability, as BF16 rounding can tie the two highest scores."""
    import torch
    from transformers import AutoModelForCausalLM

    differing = []
    for prompt, answer, reference in zip(prompts, answers, expected, strict=True):
        assert len(answer) == len(reference)
        pairs = enumerate(zip(answer, reference, strict=True))
        if (index := next((i for i, (a, b) in pairs if a != b), None)) is not None:
            differing.append((prompt, (answer, reference), index))
    if not differing:
        return
    with server("/model_info") as response:
        checkpoint = json.load(response)["model_path"]
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float32).eval()
    for prompt, pair, index in differing:
        for tokens in pair:
            with torch.inference_mode():
                logits = model(torch.tensor([prompt + tokens])).logits[0, len(prompt) - 1 : -1]
            for token, scores in list(zip(tokens, logits, strict=True))[index:]:
                assert (scores[token] - scores.max()).exp() >= 0.95


@pytest.mark.parametrize("server", served(TEXT), indirect=True)
@pytest.mark.feature("Text generation")
@pytest.mark.feature("FP8 KV cache")
@pytest.mark.feature("Quantized checkpoints (FP8, NVFP4)")
def test_batch_parity(server: Callable[..., HTTPResponse]) -> None:
    """Prompts batched together answer as each alone does, up to Transformers' greedy tolerance:
    attention rounds a row by the other rows of its call."""
    texts = [
        "The capital of France is",
        "You are a helpful assistant. Answer briefly. The largest planet in our solar system is",
    ]
    with server("/v1/tokenize", {"prompt": texts}) as response:
        prompts: list[list[int]] = json.load(response)["tokens"]
    params = {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True}
    expected = []
    for prompt in prompts:
        with server("/generate", {"input_ids": prompt, "sampling_params": params}) as response:
            expected.append(json.load(response)["output_ids"])
    with server("/generate", {"input_ids": prompts, "sampling_params": params}) as response:
        answers = [item["output_ids"] for item in json.load(response)]
    assert_close(server, prompts, answers, expected)


@pytest.mark.parametrize(
    "server",
    [DEFAULT | {"flags": ["--dp-size", "2", "--load-balance-method", "round_robin"]}],
    indirect=True,
    ids=["dp2"],
)
@pytest.mark.feature("Data parallelism")
def test_replicas_answer_alike(server: Callable[..., HTTPResponse]) -> None:
    """Two data-parallel replicas, each on its own chip and taking requests in turn, answer one
    prompt with the same greedy tokens."""
    params = {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True}
    answers = []
    for _ in range(2):
        body = {"text": "The capital of France is", "sampling_params": params}
        with server("/generate", body) as response:
            answers.append(json.load(response)["output_ids"])
    assert answers[0] == answers[1]


NGRAM = ("--speculative-algorithm", "NGRAM", "--speculative-ngram-max-bfs-breadth", "1")
STANDALONE = ("--speculative-algorithm", "STANDALONE", "--speculative-draft-model-path")


@pytest.mark.parametrize(
    "flags",
    [(*NGRAM, "--speculative-num-draft-tokens", "4"), (*STANDALONE, "Qwen/Qwen3-0.6B")],
    ids=["ngram", "standalone"],
)
@pytest.mark.feature("Speculative decoding (ngram, suffix, draft model)")
def test_speculative_parity(
    tmp_path_factory: pytest.TempPathFactory, chips: Chips, flags: tuple[str, ...]
) -> None:
    """Speculation accepts drafts and returns the greedy tokens of ordinary generation, for a batch
    of requests of different lengths and up to the context's last token."""
    repeated = "Repeat: red blue red blue red blue"
    long = "Explain briefly. " * 24 + "The capital of France is"
    with chips.serve(tmp_path_factory, DEFAULT) as send:
        with send("/v1/tokenize", {"prompt": long}) as response:
            length = len(json.load(response)["tokens"])

        def generate(texts: list[str], tokens: int) -> list[dict[str, Any]]:
            body = {"text": texts, "sampling_params": {"temperature": 0, "max_new_tokens": tokens}}
            with send("/generate", body) as response:
                return json.load(response)

        # The context holds 128 tokens; the last request's final chains run out of room.
        requests = [([repeated, "The capital of France is"], 16), ([long], 128 - length)]
        expected = [[item["output_ids"] for item in generate(*request)] for request in requests]
    with chips.serve(tmp_path_factory, DEFAULT | {"flags": list(flags)}) as send:
        answers = [generate(*request) for request in requests]
    assert [[item["output_ids"] for item in answer] for answer in answers] == expected
    assert max(item["meta_info"]["spec_accept_length"] for answer in answers for item in answer) > 1


@pytest.mark.parametrize(
    ("row", "requirement"),
    [
        (
            DEFAULT | {"flags": [*STANDALONE, "mistralai/Mistral-7B-v0.1"]},
            "speculative_draft_model_path requires the target's token-to-id mapping",
        ),
        (
            next(row for row in EMBEDDING if row["id"] == "Qwen/Qwen3-Embedding-0.6B")
            | {"flags": list(NGRAM)},
            "speculative_algorithm requires a causal attention language model",
        ),
    ],
    ids=["draft-tokenizer", "embedding-target"],
)
@pytest.mark.feature("Speculative decoding (ngram, suffix, draft model)")
def test_refuses_checkpoints(
    tmp_path_factory: pytest.TempPathFactory,
    chips: Chips,
    row: dict[str, Any],
    requirement: str,
) -> None:
    """A draft or target checkpoint speculation cannot serve stops the server with the reason."""
    assert f"furiosa-opt-sglang {requirement}" in chips.refusal(tmp_path_factory, row)


GREEDY = "serves only greedy decoding, without logprobs or logit changes"


@pytest.mark.parametrize(
    ("body", "requirement"),
    [
        (
            {"sampling_params": {"temperature": 0.8}},
            f"requires temperature 0 or top_k 1: {GREEDY}",
        ),
        ({"sampling_params": {"temperature": 0, "frequency_penalty": 0.2}}, GREEDY),
        # SGLang biases the scores of a request given any logit bias, even an empty one.
        ({"sampling_params": {"temperature": 0, "logit_bias": {}}}, GREEDY),
        ({"sampling_params": {"temperature": 0, "regex": " (Paris|Rome)"}}, GREEDY),
        ({"sampling_params": {"temperature": 0}, "return_logprob": True}, GREEDY),
    ],
    ids=["sampling", "penalty", "empty-bias", "grammar", "logprobs"],
)
@pytest.mark.feature("Text generation")
@pytest.mark.feature("Structured outputs")
def test_refuses_scored_requests(
    server: Callable[..., HTTPResponse], body: dict[str, object], requirement: str
) -> None:
    """A request that reads scores, which the engine's calls do not return, fails alone with the
    reason, and the server goes on answering greedy requests."""
    from urllib.error import HTTPError

    prompt = {"text": "The capital of France is"}
    with pytest.raises(HTTPError) as error:
        server("/generate", prompt | body)
    assert error.value.code == 400
    assert json.load(error.value)["error"]["message"] == f"furiosa-opt-sglang {requirement}"
    greedy = {"sampling_params": {"temperature": 0, "max_new_tokens": 1}}
    with server("/generate", prompt | greedy) as response:
        assert "Paris" in json.load(response)["text"]


@pytest.mark.parametrize(
    ("row", "requirement"),
    [
        pytest.param(
            DEFAULT | {"flags": ["--enable-watermark"]},
            "serves no watermarking, which samples from scores the engine does not return",
            marks=pytest.mark.feature("Watermarking"),
            id="watermark",
        ),
        pytest.param(
            DEFAULT | {"serve": [CACHE_DTYPE, "fp8_e4m3"]},
            "kv_cache: unsupported, every implementation caches keys and values in BF16",
            marks=pytest.mark.feature("FP8 KV cache"),
            id="fp8-cache",
        ),
        pytest.param(
            CLASSIFICATION[0],
            "ForSequenceClassification is no served decoder",
            marks=pytest.mark.feature("Classification and rewards from a score head"),
            id="score-head",
        ),
        pytest.param(
            DEFAULT | {"flags": ["--tp-size", "2"]},
            "chips: unsupported, every implementation runs on one chip",
            marks=pytest.mark.feature("Tensor parallelism over a compiled chip count"),
            id="tensor-parallel",
        ),
        # A checkpoint of a family the engine lists no model of, refused with the engine's reason.
        pytest.param(
            {"id": "hf-internal-testing/tiny-random-MambaForCausalLM", "task": "text"},
            "MambaForCausalLM: unsupported, no model type is implemented",
            id="recurrent",
        ),
    ],
)
def test_refuses_at_startup(
    tmp_path_factory: pytest.TempPathFactory,
    chips: Chips,
    row: dict[str, object],
    requirement: str,
) -> None:
    """An option or checkpoint the plugin cannot serve yet stops the server with the reason."""
    assert f"furiosa-opt-sglang {requirement}" in chips.refusal(tmp_path_factory, row)


@pytest.mark.feature("Long contexts, up to the compiled window")
def test_bounds_context_by_the_window(
    tmp_path_factory: pytest.TempPathFactory, chips: Chips, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A context one position past the decoder's window stops the server, naming the window, even
    where SGLang allows a context past the checkpoint's own; the window itself serves."""
    monkeypatch.setenv("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
    # Qwen3-1.7B's decoder is compiled for 131072 positions.
    past = DEFAULT | {"flags": ["--context-length", "131073"]}
    log = chips.refusal(tmp_path_factory, past)
    assert "furiosa-opt-sglang requires context_length at most 131072" in log
    body = {
        "text": "The capital of France is",
        "sampling_params": {"temperature": 0, "max_new_tokens": 1},
    }
    whole = DEFAULT | {"flags": ["--context-length", "131072"]}
    with chips.serve(tmp_path_factory, whole) as send, send("/generate", body) as response:
        assert "Paris" in json.load(response)["text"]


@pytest.mark.parametrize(
    "server",
    [
        *served(TEXT),
        # Deterministic inference keeps the radix cache, whose prefixes a batch-invariant load reads
        # as it wrote them.
        pytest.param(
            DEFAULT | {"flags": ["--enable-deterministic-inference"]},
            id="deterministic",
            marks=pytest.mark.feature("Batch invariance"),
        ),
    ],
    indirect=True,
)
@pytest.mark.feature("Automatic prefix caching")
def test_prefix_cache(server: Callable[..., HTTPResponse]) -> None:
    prompt = "Explain briefly. " * 20 + "The capital of France is"
    body = {"text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": 4}}
    with server("/generate", body) as response:
        first = json.load(response)
    with server("/generate", body) as response:
        second = json.load(response)
    assert second["output_ids"] == first["output_ids"]
    assert second["meta_info"]["cached_tokens"] > 0


# A device pool that one request and its answer mostly fill.
SMALL: Final = ("--max-total-tokens", "128")
# A host tier twice the device pool, by SGLang's own hierarchical cache options.
HIERARCHICAL: Final = ("--enable-hierarchical-cache", "--hicache-ratio", "2")
PAGE: Final = 16
# Whose home and hobby each prompt asks about; the first prompt ends inside a page, so a hit on
# either tier leaves the same tokens to compute.
NOTES: Final = (
    (("Dmitri", "Kyiv", "painting"), ("Esra", "Izmir", "cycling"), ("Femi", "Lagos", "drumming")),
    (("Alice", "Paris", "the violin"), ("Bruno", "Lima", "chess"), ("Chen", "Oslo", "rowing")),
    (("Gita", "Pune", "hiking"), ("Hugo", "Lyon", "baking"), ("Ines", "Porto", "surfing")),
)


@with_config(ConfigDict(extra="forbid"))
class Tiers(TypedDict):
    """The cached prompt tokens SGLang found on each tier of its hierarchical cache."""

    device: int
    host: int


class Meta(TypedDict):
    cached_tokens: int
    # None for a prompt none of whose tokens SGLang found cached.
    cached_tokens_details: NotRequired[Tiers | None]


class Generated(TypedDict):
    output_ids: list[int]
    meta_info: Meta


@with_config(ConfigDict(extra="forbid"))
class Allocation(TypedDict):
    """A host-pool allocation, as the watcher logs it: its pages that held a copy, and those whose
    copy still ran."""

    reused: int
    running: int


GENERATED: Final = TypeAdapter(Generated)
ALLOCATION: Final = TypeAdapter(Allocation)


def ask(checkpoint: str, people: tuple[tuple[str, str, str], ...]) -> list[int]:
    """A chat question whose answer only the notes at its start hold."""
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    facts = " ".join(
        f"{name} lives in {city} and spends every weekend on {hobby}."
        for name, city, hobby in people
    )
    question = (
        f"Read these notes. {facts} "
        f"Where does {people[1][0]} live, and what does {people[2][0]} do on weekends?"
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert isinstance(text, str)
    return tokenizer.encode(text, add_special_tokens=False)


def generate(
    send: Callable[[str, dict[str, object]], HTTPResponse], prompt: list[int], tokens: int = 32
) -> Generated:
    """The server's greedy answer to `prompt`, `tokens` long."""
    params = {"temperature": 0, "max_new_tokens": tokens, "ignore_eos": True}
    with send("/generate", {"input_ids": prompt, "sampling_params": params}) as response:
        return GENERATED.validate_json(response.read())


# A checkpoint whose cache splits over two chips, each holding its part of every host page.
SPLIT: Final = "RedHatAI/Mistral-Nemo-Instruct-2407-FP8"


@pytest.mark.parametrize(
    ("model", "parallel"),
    [
        pytest.param(DEFAULT["id"], (), id="one-chip"),
        pytest.param(
            SPLIT,
            ("--tp-size", "2"),
            marks=pytest.mark.skip(reason=f"pending: {PENDING[SPLIT]}"),
            id="two-chips-fp8",
        ),
    ],
)
@pytest.mark.feature("KV cache offloading")
def test_reloads_evicted_prefix(
    model: str, parallel: tuple[str, ...], chips: Chips, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A prompt whose pages other prompts evicted from the device reloads them from host memory:
    the hierarchical cache answers it as a server without one answers from its device cache."""
    row = next(row for row in TEXT if row["id"] == model)
    first, *others = (ask(model, people) for people in NOTES)
    cached = len(first) // PAGE * PAGE
    flags = [*parallel, *SMALL]
    with chips.serve(tmp_path_factory, row | {"flags": flags}) as server:
        _ = generate(server, first)
        resident = generate(server, first)
    assert resident["meta_info"]["cached_tokens"] == cached
    with chips.serve(tmp_path_factory, row | {"flags": [*flags, *HIERARCHICAL]}) as server:
        _ = generate(server, first)
        for prompt in others:
            _ = generate(server, prompt)
        reloaded = generate(server, first)
    assert reloaded["meta_info"].get("cached_tokens_details") == {"device": 0, "host": cached}
    assert reloaded["output_ids"] == resident["output_ids"]


# A host tier smaller than the device pool, which SGLang fills only with the prompts it evicts.
WRITE_BACK: Final = (
    "--enable-hierarchical-cache",
    "--hicache-ratio",
    "0.5",
    "--hicache-write-policy",
    "write_back",
)


@pytest.mark.feature("KV cache offloading")
def test_reloads_host_pages_reallocated_mid_copy(
    chips: Chips, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Host pages SGLang reallocates while a copy into them still runs hold the prompt copied there
    next: that prompt reloads as a server without a hierarchical cache answers it from its device
    cache.

    Under write-back, a prompt that evicts two others backs them up one after the other, and the
    second backup evicts the first from the host tier to take its pages.
    """
    model: str = DEFAULT["id"]
    reloaded_prompt = ask(model, NOTES[0])
    assert 4 * PAGE < len(reloaded_prompt) < 5 * PAGE
    prefix = reloaded_prompt[: 4 * PAGE]
    # Each prompt prefills in one call, and an answer of one token leaves only the prompt's pages.
    flags = [*SMALL, "--chunked-prefill-size", str(8 * PAGE)]
    with chips.serve(tmp_path_factory, DEFAULT | {"flags": flags}) as server:
        _ = generate(server, prefix, tokens=1)
        resident = generate(server, reloaded_prompt)
    assert resident["meta_info"]["cached_tokens"] == 4 * PAGE
    # SGLang loads the watcher in each server process as the plugin of a distribution here.
    plugin = tmp_path_factory.mktemp("watcher")
    (plugin / "watcher.py").symlink_to(Path(__file__).with_name("watcher.py"))
    (plugin / "watcher-0.dist-info").mkdir()
    _ = (plugin / "watcher-0.dist-info/METADATA").write_text(
        "Metadata-Version: 2.1\nName: watcher\nVersion: 0\n"
    )
    _ = (plugin / "watcher-0.dist-info/entry_points.txt").write_text(
        "[sglang.srt.plugins]\nwatcher = watcher:register\n"
    )
    path = os.pathsep.join(filter(None, [str(plugin), os.environ.get("PYTHONPATH")]))
    hierarchical = DEFAULT | {"flags": [*flags, *WRITE_BACK], "env": {"PYTHONPATH": path}}
    with chips.serve(tmp_path_factory, hierarchical) as server:
        # The last prompt evicts both others: SGLang frees a page beyond a prefill's own pages.
        for prompt in (ask(model, NOTES[1])[: 2 * PAGE], prefix, ask(model, NOTES[2])[: 4 * PAGE]):
            _ = generate(server, prompt, tokens=1)
        reloaded = generate(server, reloaded_prompt)
    log = (plugin / "allocations.jsonl").read_text().splitlines()
    allocations = [ALLOCATION.validate_json(line) for line in log]
    assert any(allocation["running"] for allocation in allocations), allocations
    assert reloaded["meta_info"].get("cached_tokens_details") == {"device": 0, "host": 4 * PAGE}
    assert reloaded["output_ids"] == resident["output_ids"]


# SGLang accepts input embeddings only with the radix cache off.
@pytest.mark.parametrize(
    "server",
    [DEFAULT | {"disable_radix_cache": True}],
    indirect=True,
    ids=lambda row: row["id"],
)
@pytest.mark.feature("Prompt embeddings")
@pytest.mark.skip(reason="pending: a decoder call takes token IDs, not input embeddings")
def test_embedding_inputs(server: Callable[..., HTTPResponse]) -> None:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    params = {"temperature": 0, "max_new_tokens": 16}
    with server("/model_info") as response:
        checkpoint = json.load(response)["model_path"]
    directory = (
        Path(checkpoint) if Path(checkpoint).is_dir() else Path(snapshot_download(checkpoint))
    )
    index = directory / "model.safetensors.index.json"
    shard = (
        json.loads(index.read_text())["weight_map"]["model.embed_tokens.weight"]
        if index.exists()
        else "model.safetensors"
    )
    tokens = [642, 643, 644] * 12
    with safe_open(directory / shard, framework="pt") as tensors:
        weight = tensors.get_slice("model.embed_tokens.weight")
        embeddings = [weight[token].float().tolist() for token in tokens]
    with server("/generate", {"input_ids": tokens, "sampling_params": params}) as response:
        expected = json.load(response)["output_ids"]
    with server("/generate", {"input_embeds": embeddings, "sampling_params": params}) as response:
        assert json.load(response)["output_ids"] == expected


def test_chat_completion(server: Callable[..., HTTPResponse]) -> None:
    with server(
        "/v1/chat/completions",
        {
            "model": "Qwen/Qwen3-1.7B",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "temperature": 0,
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    ) as response:
        result = json.load(response)
    assert "4" in result["choices"][0]["message"]["content"]
    assert result["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "server",
    [DEFAULT, DEFAULT | {"flags": ["--disable-overlap-schedule"]}],
    indirect=True,
    ids=["overlap", "serial"],
)
@pytest.mark.feature("Structured outputs")
@SCORES
def test_follows_grammar(server: Callable[..., HTTPResponse]) -> None:
    """Constrained decoding keeps the answer inside a regex and a JSON schema."""
    params = {"temperature": 0, "max_new_tokens": 16}
    body = {
        "text": "The capital of France is",
        "sampling_params": params | {"regex": " (Paris|Rome)"},
    }
    with server("/generate", body) as response:
        assert json.load(response)["text"] in {" Paris", " Rome"}
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string", "maxLength": 12}},
        "required": ["city"],
        "additionalProperties": False,
    }
    constrained = params | {"max_new_tokens": 32, "json_schema": json.dumps(schema)}
    body = {"text": "The capital of France, in JSON:", "sampling_params": constrained}
    with server("/generate", body) as response:
        assert set(json.loads(json.load(response)["text"])) == {"city"}


@pytest.mark.feature("Per-request metrics")
def test_reports_latency(server: Callable[..., HTTPResponse]) -> None:
    """Each finished request reports its own end-to-end latency and token count."""
    params = {"temperature": 0, "max_new_tokens": 4, "ignore_eos": True}
    with server(
        "/generate", {"text": "The capital of France is", "sampling_params": params}
    ) as response:
        meta = json.load(response)["meta_info"]
    assert meta["e2e_latency"] > 0
    assert meta["completion_tokens"] == 4


@pytest.mark.parametrize("server", [PARSING], indirect=True, ids=lambda row: row["id"])
@pytest.mark.feature("Tool calling and reasoning outputs")
def test_calls_tools(server: Callable[..., HTTPResponse]) -> None:
    """A request offering a tool returns the call the model writes, parsed into the tool's name and
    the arguments its schema requires."""
    tool = {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "The current weather of a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    body = {
        "model": DEFAULT["id"],
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [tool],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with server("/v1/chat/completions", body) as response:
        (call,) = json.load(response)["choices"][0]["message"]["tool_calls"]
    assert call["function"]["name"] == "weather"
    assert "city" in json.loads(call["function"]["arguments"])


@pytest.mark.parametrize("server", [PARSING], indirect=True, ids=lambda row: row["id"])
@pytest.mark.feature("Tool calling and reasoning outputs")
def test_separates_reasoning(server: Callable[..., HTTPResponse]) -> None:
    """The reasoning parser returns the thinking apart from the answer."""
    body = {
        "model": DEFAULT["id"],
        "messages": [{"role": "user", "content": "What is 2+2? Answer with the number."}],
        "temperature": 0,
        "max_tokens": 768,
        "separate_reasoning": True,
    }
    with server("/v1/chat/completions", body) as response:
        message = json.load(response)["choices"][0]["message"]
    assert message["reasoning_content"]
    assert "<think>" not in (message["content"] or "")


@pytest.mark.parametrize("server", [WATERMARKED], indirect=True, ids=lambda row: row["id"])
@pytest.mark.feature("Watermarking")
@SCORES
def test_detects_watermark(server: Callable[..., HTTPResponse]) -> None:
    """SGLang's detector finds the server key in a watermarked answer and not in a plain one."""
    from sglang.srt.sampling.watermarking.detector import WatermarkDetector

    with server("/v1/tokenize", {"prompt": "Write a story about a lighthouse keeper."}) as response:
        prompt: list[int] = json.load(response)["tokens"]
    detector = WatermarkDetector(WATERMARK_KEY)
    params = {"temperature": 1.0, "max_new_tokens": 200, "ignore_eos": True, "sampling_seed": 7}
    p_values = {}
    for enabled in (True, False):
        body = {
            "input_ids": prompt,
            "sampling_params": params | {"watermark": {"enabled": enabled}},
        }
        with server("/generate", body) as response:
            answer = json.load(response)["output_ids"]
        detection = detector.detect_tokens(answer, prompt_token_ids=prompt)
        p_values[enabled] = detection.combined.p_value
    # One decision threshold: a plain answer falls below it with probability 1e-3.
    assert p_values[True] < 1e-3 <= p_values[False]


@pytest.mark.parametrize("server", CLASSIFICATION, indirect=True, ids=lambda row: row["id"])
@pytest.mark.feature("Classification and rewards from a score head")
@pytest.mark.skip(reason="pending: an SGLang route that reads a score head's calls")
def test_classifies(server: Callable[..., HTTPResponse]) -> None:
    """Each prompt's label scores equal the checkpoint's score head on its last hidden row, up to
    each BF16 hidden value's weighted relative precision."""
    import torch
    from transformers import AutoModelForSequenceClassification

    with server("/model_info") as response:
        checkpoint = json.load(response)["model_path"]
    with server("/v1/tokenize", {"prompt": PROMPTS}) as response:
        prompts: list[list[int]] = json.load(response)["tokens"]
    with server("/encode", {"input_ids": prompts}) as response:
        actual = torch.tensor([item["embedding"] for item in json.load(response)])
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).float().eval()
    weight = model.score.weight
    with torch.inference_mode():
        rows = torch.stack(
            [
                model.base_model(input_ids=torch.tensor([prompt])).last_hidden_state[0, -1]
                for prompt in prompts
            ]
        )
        expected = rows @ weight.T
        bound = rows.abs() @ weight.abs().T * 2**-8
    assert actual.shape == expected.shape
    assert ((actual - expected).abs() <= bound).all(), (actual, expected, bound)


@pytest.mark.parametrize("server", [PROCESSING], indirect=True, ids=lambda row: row["id"])
@pytest.mark.feature("Custom logits processors")
@SCORES
def test_applies_logit_processor(server: Callable[..., HTTPResponse]) -> None:
    """A request's custom logit processor reaches the step's scores: a disallowed greedy token is
    never chosen, and a request without the processor keeps it."""
    from sglang.srt.sampling.custom_logit_processor import DisallowedTokensLogitsProcessor

    params = {"temperature": 0, "max_new_tokens": 1}
    body = {"text": "The capital of France is", "sampling_params": params}
    with server("/generate", body) as response:
        (greedy,) = json.load(response)["output_ids"]
    processed = body | {
        "sampling_params": params | {"custom_params": {"token_ids": [greedy]}},
        "custom_logit_processor": DisallowedTokensLogitsProcessor.to_str(),
    }
    with server("/generate", processed) as response:
        assert json.load(response)["output_ids"] != [greedy]
    with server("/generate", body) as response:
        assert json.load(response)["output_ids"] == [greedy]


def test_abort_releases(server: Callable[..., HTTPResponse]) -> None:
    rid = uuid.uuid4().hex
    with server(
        "/generate",
        {
            "rid": rid,
            "text": "Count upwards: 1, 2, 3,",
            "stream": True,
            # Stays below the 128-token context, so only the abort ends it.
            "sampling_params": {"temperature": 0, "max_new_tokens": 110, "ignore_eos": True},
        },
    ) as response:
        assert response.readline().startswith(b"data: ")
        with server("/abort_request", {"rid": rid}) as aborted:
            assert aborted.status == 200
        events = [json.loads(line[6:]) for line in response if line.startswith(b"data: {")]
    assert events[-1]["meta_info"]["finish_reason"]["type"] == "abort"
    # SGLang flushes its cache only once no request holds the scheduler.
    with server("/flush_cache?timeout=10") as response:
        assert response.status == 200
    with server(
        "/generate",
        {
            "text": "The capital of France is",
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
        },
    ) as response:
        assert "Paris" in json.load(response)["text"]


@SCORES
def test_sampling_and_logprobs(server: Callable[..., HTTPResponse]) -> None:
    with server(
        "/generate",
        {
            "text": "The capital of France is",
            "return_logprob": True,
            "logprob_start_len": 0,
            "sampling_params": {"temperature": 0.8, "max_new_tokens": 4, "frequency_penalty": 0.2},
        },
    ) as response:
        result = json.load(response)
    meta = result["meta_info"]
    # Each output token carries its logprob, and each prompt token from `logprob_start_len`.
    assert [token for _, token, _ in meta["output_token_logprobs"]] == result["output_ids"]
    assert len(meta["input_token_logprobs"]) == meta["prompt_tokens"]


@pytest.fixture(scope="module")
def reference(server: Callable[..., HTTPResponse]) -> "torch.Tensor":
    """Transformers' last-token hidden state per prompt for the served checkpoint."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    with server("/model_info") as response:
        checkpoint = json.load(response)["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    inputs = tokenizer(PROMPTS, padding=True, return_tensors="pt")
    model = AutoModel.from_pretrained(checkpoint, dtype=torch.bfloat16).eval()
    with torch.inference_mode():
        return model(**inputs).last_hidden_state[:, -1].float()


def embed(server: Callable[..., HTTPResponse], **options: object) -> "torch.Tensor":
    """`/v1/embeddings` for `PROMPTS`, one row per prompt."""
    import torch

    with server("/model_info") as response:
        checkpoint = json.load(response)["model_path"]
    body = {"model": checkpoint, "input": PROMPTS} | options
    with server("/v1/embeddings", body) as response:
        return torch.tensor([item["embedding"] for item in json.load(response)["data"]])


@pytest.mark.parametrize("server", EMBEDDING, indirect=True, ids=lambda row: row["id"])
@pytest.mark.feature("Embeddings")
def test_embeddings_match_transformers(
    server: Callable[..., HTTPResponse], reference: "torch.Tensor"
) -> None:
    import torch

    actual = embed(server)
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones(len(PROMPTS)))
    similarity = torch.nn.functional.cosine_similarity(actual, reference)
    assert torch.all(similarity > 0.999), similarity


@pytest.mark.parametrize("server", EMBEDDING, indirect=True, ids=lambda row: row["id"])
def test_embeddings_repeat_exactly(server: Callable[..., HTTPResponse]) -> None:
    import torch

    torch.testing.assert_close(embed(server), embed(server), atol=0, rtol=0)


@pytest.mark.parametrize("server", EMBEDDING, indirect=True, ids=lambda row: row["id"])
def test_pooled_hidden_states_normalize_to_embeddings(server: Callable[..., HTTPResponse]) -> None:
    import torch

    with server("/encode", {"text": PROMPTS, "return_pooled_hidden_states": True}) as response:
        outputs = json.load(response)
    raw = torch.tensor([item["pooled_hidden_state"] for item in outputs])
    embeddings = torch.tensor([item["embedding"] for item in outputs])
    torch.testing.assert_close(
        torch.nn.functional.normalize(raw, dim=-1), embeddings, atol=0, rtol=0
    )


@pytest.mark.parametrize("server", EMBEDDING, indirect=True, ids=lambda row: row["id"])
def test_dimensions_truncate_and_normalize(
    server: Callable[..., HTTPResponse], reference: "torch.Tensor"
) -> None:
    import torch

    reduced = embed(server, dimensions=128)
    assert reduced.shape == (len(PROMPTS), 128)
    torch.testing.assert_close(reduced.norm(dim=-1), torch.ones(len(PROMPTS)))
    similarity = torch.nn.functional.cosine_similarity(reduced, reference[:, :128])
    assert torch.all(similarity > 0.999), similarity


@pytest.mark.parametrize("server", RERANKING, indirect=True, ids=lambda row: row["id"])
@pytest.mark.feature("Reranking")
@SCORES
def test_reranking_ranks_relevant_document_first(server: Callable[..., HTTPResponse]) -> None:
    """The relevant document ranks first, and a repeated request, whose prompts the cache holds,
    ranks and scores the documents as the first does, up to rounding."""
    import torch

    request = {
        "query": "What is the capital of France?",
        "documents": ["Tokyo is the capital of Japan.", "Paris is the capital of France."],
    }
    with server("/v1/rerank", request) as response:
        first = json.load(response)
    with server("/v1/rerank", request) as response:
        repeated = json.load(response)
    assert [item["index"] for item in first] == [1, 0], first
    assert [item["index"] for item in repeated] == [1, 0], repeated
    # A cached prompt runs in other calls than the first, whose attention rounds differently; the
    # vLLM suite holds a reranking score to this tolerance against transformers.
    torch.testing.assert_close(
        torch.tensor([item["score"] for item in repeated]),
        torch.tensor([item["score"] for item in first]),
        atol=0.01,
        rtol=0.02,
    )
    assert 0 <= first[1]["score"] < first[0]["score"] <= 1
