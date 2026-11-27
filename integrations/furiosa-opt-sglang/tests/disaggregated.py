"""Servers of prefill-decode disaggregation for the RNGD tests: a server without disaggregation
as the oracle, a prefill server and decode servers on chips of their own, every one loading the
model batch invariant, and requests sent as SGLang's router sends them.

The servers need three chips, a fourth for a second decode replica, and Mooncake's
`mooncake-transfer-engine-non-cuda` package."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, NotRequired
from urllib.request import Request, urlopen

import pytest
from pydantic import TypeAdapter
from test_server import Tiers
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping, Sequence
    from contextlib import AbstractContextManager
    from http.client import HTTPResponse
    from pathlib import Path

MODEL: Final = "Qwen/Qwen3-0.6B"
TRANSFER: Final = ("--disaggregation-transfer-backend", "mooncake_tcp")
TOKENS: Final = 16
# Enough decode steps to retract a request before it ends.
LONG: Final = 96
# Enough that a request still decodes once the decode server receives another a second later.
PAUSED: Final = 512
# The platform's page of tokens, and a prefill chunk of several pages, shorter than the prompts.
PAGE: Final = 16
CHUNK: Final = 8 * PAGE
# A prefill server's device pool that one short prompt and its answer mostly fill, beside a host
# tier twice its size.
RELOADING: Final = (
    *("--max-total-tokens", "128"),
    *("--enable-hierarchical-cache", "--hicache-ratio", "2"),
)
SAVING: Final = ("--enable-memory-saver",)
# A draft model, as both servers of a pair run it.
STANDALONE: Final = (
    *("--speculative-algorithm", "STANDALONE", "--speculative-draft-model-path", MODEL),
)
SWITCHING: Final = ("--enable-pd-role-switch",)
# The prefill servers tests ask for by name, beside the default one.
PREFILLS: Final[Mapping[str, tuple[str, ...]]] = {
    "chunked": ("--chunked-prefill-size", str(CHUNK)),
    "hierarchical": RELOADING,
    "memory-saver": SAVING,
    "standalone": STANDALONE,
    "role-switch": SWITCHING,
}


def prompt(topic: str, items: int = 24) -> str:
    """A prompt about `topic` of several cache pages, the last one partial. A run's own words keep
    a chip's memory from holding its cache from an earlier run, which a missed transfer would then
    read."""
    listed = " ".join(f"Item {i} of the list is a word for {topic}." for i in range(items))
    return f"Run {secrets.token_hex(8)}. {listed}"


class Sampling(TypedDict):
    """The sampling parameters of the tests' requests."""

    temperature: float
    max_new_tokens: int
    ignore_eos: bool


def greedy(tokens: int) -> Sampling:
    """Greedy sampling of `tokens` tokens, a payload of its own for each request."""
    return {"temperature": 0, "max_new_tokens": tokens, "ignore_eos": True}


class Room(TypedDict):
    """A bootstrap room, which a request joins on both servers."""

    bootstrap_host: str
    bootstrap_port: int
    bootstrap_room: int


class Finish(TypedDict):
    type: str


class Meta(TypedDict):
    cached_tokens: int
    # None for a prompt none of whose tokens SGLang found cached.
    cached_tokens_details: NotRequired[Tiers | None]
    num_retractions: int
    # Tokens a verification of a speculative server commits on average, once it verified any.
    spec_accept_length: NotRequired[float]
    # A stream's chunks before the last one carry none.
    finish_reason: Finish | None


class Answer(TypedDict):
    """The part of a `/generate` answer the tests read."""

    output_ids: list[int]
    meta_info: Meta


class Tokenized(TypedDict):
    tokens: list[int]


ANSWER: Final = TypeAdapter(Answer)
TOKENIZED: Final = TypeAdapter(Tokenized)


def post(url: str, body: Mapping[str, object]) -> HTTPResponse:
    """The response of `url` to the JSON `body`."""
    headers = {"Content-Type": "application/json"}
    # A request ends in seconds; a server that died fails the test instead of stalling it.
    response: HTTPResponse = urlopen(Request(url, json.dumps(body).encode(), headers), timeout=120)
    return response


def read(url: str, body: Mapping[str, object]) -> bytes:
    with post(url, body) as response:
        return response.read()


def tokens(url: str, text: str) -> list[int]:
    return TOKENIZED.validate_json(read(f"{url}/v1/tokenize", {"prompt": text}))["tokens"]


def generate(url: str, ids: list[int], params: Sampling, room: Room | None = None) -> Answer:
    """The answer of the server at `url` to prompt `ids`, joining bootstrap `room`."""
    body = {"input_ids": ids, "sampling_params": params, **(room or {})}
    return ANSWER.validate_json(read(f"{url}/generate", body))


def stream(
    url: str, ids: list[int], params: Sampling, room: Room, seen: Callable[[int], None]
) -> Answer:
    """`generate` as a stream, reporting the tokens generated so far to `seen`."""
    body = {"input_ids": ids, "sampling_params": params, "stream": True, **room}
    answer = None
    with post(f"{url}/generate", body) as response:
        for line in response:
            if line.startswith(b"data: {"):
                answer = ANSWER.validate_json(line.removeprefix(b"data: "))
                seen(len(answer["output_ids"]))
    assert answer is not None
    return answer


@contextmanager
def serve(
    chip: str, flags: Sequence[str], log: Path, env: Mapping[str, str] | None = None
) -> Generator[str, None, None]:
    """`sglang.launch_server` of `MODEL` on `chip` with `flags` and environment `env`."""
    from sglang.srt.utils import kill_process_tree
    from sglang.srt.utils.network import get_free_port

    port = get_free_port()
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                *(sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL),
                *("--context-length", "1024", "--host", "127.0.0.1", "--port", str(port)),
                # A request's tokens then keep their bits whichever calls wrote its cache.
                "--enable-deterministic-inference",
                *flags,
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            cwd=log.parent,
            env=os.environ | dict(env or {}) | {"FURIOSA_VISIBLE_CHIPS": chip},
        )
    url = f"http://127.0.0.1:{port}"

    def healthy() -> bool:
        try:
            response: HTTPResponse = urlopen(f"{url}/health", timeout=1)
        except OSError:
            return False
        with response:
            return response.status == 200

    try:
        # SGLang reports readiness only over HTTP, so this polls it until the deadline; the server's
        # exit ends the wait at once.
        deadline = time.monotonic() + 600
        while not healthy():
            try:
                code = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                assert time.monotonic() < deadline, log.read_text()[-4000:]
            else:
                raise AssertionError(f"exit {code}: {log.read_text()[-4000:]}")
        yield url
    finally:
        # SGLang waits until every process it kills is gone, and its schedulers end with it.
        if process.poll() is None:
            kill_process_tree(process.pid)
        _ = process.wait(timeout=30)


@dataclass(frozen=True, slots=True)
class Prefill:
    """A prefill server and the port decode servers bootstrap its transfers on."""

    url: str
    bootstrap: int

    def room(self) -> Room:
        """A bootstrap room of its own."""
        return {
            "bootstrap_host": "127.0.0.1",
            "bootstrap_port": self.bootstrap,
            "bootstrap_room": secrets.randbits(62),
        }

    def decode(self, decode: str, ids: list[int], tokens: int = TOKENS) -> tuple[Answer, Answer]:
        """The prefill and decode servers' greedy answers of `tokens` tokens to `ids`, sent to
        both at once as SGLang's router does."""
        room = self.room()
        with ThreadPoolExecutor(2) as pool:
            sent = pool.submit(generate, self.url, ids, greedy(tokens), room)
            decoded = pool.submit(generate, decode, ids, greedy(tokens), room)
            return sent.result(), decoded.result()


@pytest.fixture(scope="module")
def chips() -> list[str]:
    return os.environ["FURIOSA_VISIBLE_CHIPS"].split(",")


@pytest.fixture(scope="module")
def oracle(
    chips: list[str], tmp_path_factory: pytest.TempPathFactory
) -> Generator[str, None, None]:
    """A server without disaggregation."""
    with serve(chips[0], [], tmp_path_factory.mktemp("oracle") / "server.log") as url:
        yield url


@pytest.fixture(scope="module")
def prefill(
    chips: list[str], tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> Generator[Prefill, None, None]:
    """A prefill server with the flags of `PREFILLS` a test names as the fixture's parameter, if it
    does. pytest keeps one server while consecutive tests ask for one parameter, and stops it
    before it starts one for another."""
    from sglang.srt.utils.network import get_free_port

    bootstrap = get_free_port()
    flags = ["--disaggregation-mode", "prefill", *TRANSFER]
    flags += ["--disaggregation-bootstrap-port", str(bootstrap)]
    # pytest sets the parameter of an indirectly parametrized fixture only on its request.
    name: object = getattr(request, "param", None)
    match name:
        case None:
            pass
        case str() if name in PREFILLS:
            flags += PREFILLS[name]
        case _:
            raise TypeError(f"the prefill servers are {sorted(PREFILLS)}, not {name!r}")
    with serve(chips[1], flags, tmp_path_factory.mktemp("prefill") / "server.log") as url:
        yield Prefill(url, bootstrap)


def decoding(
    chips: list[str],
    tmp_path: Path,
    flags: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> AbstractContextManager[str]:
    """A decode server with `flags` and environment `env`, on the chips after the prefill
    server's: one, or one a data-parallel replica."""
    replicas = int(flags[flags.index("--dp-size") + 1]) if "--dp-size" in flags else 1
    assert len(chips) >= 2 + replicas, f"{replicas} decode replicas need {2 + replicas} chips"
    arguments = ["--disaggregation-mode", "decode", *TRANSFER, *flags]
    return serve(",".join(chips[2 : 2 + replicas]), arguments, tmp_path / "decode.log", env)


def ended(answer: Answer) -> bool:
    """Whether `answer` reached its token limit, as SGLang ends a request it did not abort."""
    finish = answer["meta_info"]["finish_reason"]
    return finish is not None and finish["type"] == "length"
