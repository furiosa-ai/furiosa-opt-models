"""Long prompts, tensor parallelism, deterministic inference and the memory saver preserve the
served generation contract."""

import json
import os
import shutil
from array import array
from collections.abc import Callable, Generator
from contextlib import contextmanager
from http.client import HTTPResponse
from pathlib import Path
from typing import Final, NotRequired, TypedDict

import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load, save
from test_server import DEFAULT, EMBEDDING, TEXT, TOP, Chips, assert_close
from test_server import chips as chips
from test_server import plugin as plugin
from test_server import server as serving

server = serving


class Row(TypedDict):
    id: str
    task: str
    serve: NotRequired[list[str]]
    flags: NotRequired[list[str]]


@pytest.mark.feature("Long contexts, up to the compiled window")
@pytest.mark.parametrize(
    "server",
    [DEFAULT | {"flags": ["--context-length", "2048"]}],
    indirect=True,
    ids=["long-context"],
)
def test_preserves_long_prompt(server: Callable[..., HTTPResponse]) -> None:
    """A prompt spanning many prefill chunks answers top-five-close alone and in a batch."""
    texts = ["Explain briefly. " * 180 + "The capital of France is", "What is a compiler?"]
    with server("/v1/tokenize", {"prompt": texts}) as response:
        prompts: list[list[int]] = json.load(response)["tokens"]
    assert 128 < len(prompts[0]) < 2048 - 8
    params = {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True}
    expected = []
    for prompt in prompts:
        with server("/generate", {"input_ids": prompt, "sampling_params": params}) as response:
            expected.append(json.load(response)["output_ids"])
    with server("/generate", {"input_ids": prompts, "sampling_params": params}) as response:
        answers = [item["output_ids"] for item in json.load(response)]
    assert_close(server, prompts, answers, expected)


@pytest.mark.feature("Tensor parallelism over a compiled chip count")
@pytest.mark.skip(reason="pending: a line split over two chips, and the scores of its tokens")
@pytest.mark.parametrize(
    "row",
    [row for row in TEXT if row["id"] == "RedHatAI/Mistral-Nemo-Instruct-2407-FP8"],
    ids=lambda row: row["id"],
)
def test_preserves_tokens(tmp_path_factory: pytest.TempPathFactory, chips: Chips, row: Row) -> None:
    """TP2 and TP1 greedy tokens are mutually top-five-close from their first differing token."""
    results = []
    for tensor in (1, 2):
        configured = row | {"flags": ["--tp-size", str(tensor), "--context-length", "1024"]}
        with chips.serve(tmp_path_factory, configured) as send:
            body = {
                "text": "The capital of France is",
                "return_logprob": True,
                "top_logprobs_num": TOP,
                "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
            }
            with send("/generate", body) as response:
                results.append(json.load(response))
    first, second = results
    assert len(first["output_ids"]) == len(second["output_ids"]) == 8
    differing = False
    for a, b, left, right in zip(
        first["output_ids"],
        second["output_ids"],
        first["meta_info"]["output_top_logprobs"],
        second["meta_info"]["output_top_logprobs"],
        strict=True,
    ):
        differing |= a != b
        if differing:
            assert a in {token for _, token, _ in right}
            assert b in {token for _, token, _ in left}


# Up to 16 requests share a step, which prefills 128 prompt tokens at most: behind another
# request's 76-token prompt, this 100-token prompt prefills over two steps, alone in one.
DETERMINISTIC: Final = [
    "--enable-deterministic-inference",
    "--chunked-prefill-size",
    "128",
    "--max-running-requests",
    "16",
]
GENERATING: Final = next(row for row in TEXT if row["id"] == "Qwen/Qwen3-0.6B") | {
    "flags": DETERMINISTIC
}
# The context holds the 1,100-token prompt beside whose last rows the cached prefix prefills.
POOLING: Final = next(row for row in EMBEDDING if row["id"] == "Qwen/Qwen3-Embedding-0.6B") | {
    "flags": [*DETERMINISTIC, "--context-length", "2048"]
}
REQUEST: Final = [100, 101, 102, 103] * 25
LONGEST: Final = [200, 201, 202, 203] * 275
GREEDY: Final = {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True}
# Prompts beside it: long ones that move its prompt's chunks to other steps, short ones that share
# its steps.
OTHERS: Final = {
    "long": [[200, 201, 202, 203] * 19],
    "short": [[300 + row, 7, 9] for row in range(5)],
}
COMPOSITIONS: Final = pytest.mark.parametrize(
    ("before", "after"),
    [("long", "short"), ("short", "long"), ("long", "long"), ("short", "short")],
)


def batch(before: str, after: str) -> tuple[list[list[int]], int]:
    """The request between three copies of the `before` prompts and the `after` ones, and its
    index."""
    ahead = OTHERS[before] * 3
    return [*ahead, REQUEST, *OTHERS[after]], len(ahead)


def flush(server: Callable[..., HTTPResponse]) -> None:
    """Empty SGLang's radix cache, so the prompts sent next prefill every row."""
    server("/flush_cache").close()


def bits(values: list[float]) -> bytes:
    """The bytes of `values` as float64, which tell -0.0 from 0.0 as `==` does not."""
    return array("d", values).tobytes()


def pooled(
    server: Callable[..., HTTPResponse], prompts: list[list[int]]
) -> list[tuple[bytes, bytes]]:
    """Each prompt's last hidden row and its embedding, as bits."""
    body = {"input_ids": prompts, "return_pooled_hidden_states": True}
    with server("/encode", body) as response:
        items: list[dict[str, list[float]]] = json.load(response)
    return [(bits(item["pooled_hidden_state"]), bits(item["embedding"])) for item in items]


@pytest.mark.feature("Batch invariance")
@pytest.mark.parametrize("server", [GENERATING], indirect=True, ids=["deterministic"])
@COMPOSITIONS
def test_tokens_keep_bits(server: Callable[..., HTTPResponse], before: str, after: str) -> None:
    """The request alone, its prompt prefilled in one step, decodes the tokens it decodes between
    other requests: its prompt then prefills over several steps, its rows sit at other offsets of
    its calls, and the calls hold other requests' prefill and decode rows."""
    flush(server)
    with server("/generate", {"input_ids": REQUEST, "sampling_params": GREEDY}) as response:
        alone = json.load(response)["output_ids"]
    prompts, at = batch(before, after)
    flush(server)
    with server("/generate", {"input_ids": prompts, "sampling_params": GREEDY}) as response:
        assert json.load(response)[at]["output_ids"] == alone


@pytest.mark.feature("Batch invariance")
@pytest.mark.parametrize("server", [POOLING], indirect=True, ids=["deterministic"])
@COMPOSITIONS
def test_embedding_keeps_bits(server: Callable[..., HTTPResponse], before: str, after: str) -> None:
    """The request's last hidden row and its embedding have the same bits alone as between other
    requests whose prompts share its steps."""
    flush(server)
    alone = pooled(server, [REQUEST])[0]
    prompts, at = batch(before, after)
    flush(server)
    assert pooled(server, prompts)[at] == alone


@pytest.mark.feature("Batch invariance")
@pytest.mark.parametrize("server", [POOLING], indirect=True, ids=["deterministic"])
def test_cached_prefix_keeps_bits(server: Callable[..., HTTPResponse]) -> None:
    """A request whose prefix the radix cache holds from its prefill beside a long prompt's last
    rows pools the bits it pools from an empty cache."""
    flush(server)
    alone = pooled(server, [REQUEST])[0]
    flush(server)
    _ = pooled(server, [LONGEST, REQUEST])
    assert pooled(server, [REQUEST])[0] == alone


SLEEPING: Final = "Qwen/Qwen3-0.6B"
# Tokens each request of the memory saver test generates.
GENERATED: Final = 16


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A copy of the checkpoint, whose files the tests rewrite."""
    copy = tmp_path_factory.mktemp("checkpoint") / "Qwen3-0.6B"
    _ = shutil.copytree(snapshot_download(SLEEPING), copy)
    return copy


@contextmanager
def swapped(files: Path, tokens: tuple[int, int]) -> Generator[None]:
    """The checkpoint's weight file with the token table rows of `tokens` swapped, until the block
    ends: weights of the same names and shapes that read each token as the other."""
    [original] = Path(snapshot_download(SLEEPING)).glob("*.safetensors")
    tensors = load(original.read_bytes())
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        rows = tensors[name]
        rows[list(tokens)] = rows[list(reversed(tokens))]
    try:
        _ = (files / original.name).write_bytes(save(tensors))
        yield
    finally:
        _ = shutil.copyfile(original, files / original.name)


def greedy(directory: Path, prompts: list[list[int]], count: int) -> list[list[int]]:
    """Each prompt's `count` greedy tokens by Transformers' scores of the checkpoint in
    `directory`."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()
    answers: list[list[int]] = []
    with torch.inference_mode():
        for prompt in prompts:
            output, tokens = model(torch.tensor([prompt])), []
            while len(tokens) < count:
                tokens.append(int(output.logits[0, -1].argmax()))
                cache = output.past_key_values
                output = model(torch.tensor([tokens[-1:]]), past_key_values=cache)
            answers.append(tokens)
    return answers


@pytest.mark.feature("Sleep mode")
@pytest.mark.parametrize("backup", [False, True], ids=["files", "host-copy"])
def test_resumes_the_served_weights(
    tmp_path_factory: pytest.TempPathFactory, chips: Chips, files: Path, backup: bool
) -> None:
    """Released memory resumes to the answers of before in any tag order; once the checkpoint's
    files change under the served weights, released weights resume to the loaded ones with their
    backup, else to the files' as Transformers answers with them, also for a prompt cached under the
    old ones."""
    flags = ["--enable-memory-saver", *(["--enable-weights-cpu-backup"] if backup else [])]
    with chips.serve(tmp_path_factory, {"id": str(files), "task": "text", "flags": flags}) as send:
        # The first prompt names France within its first cache page, which a later request reuses.
        texts = [
            "The capital of France is a city that many people visit every year, and its name is",
            "What is a compiler?",
            "The capital of France is",
            "The capital of Japan is",
        ]
        with send("/v1/tokenize", {"prompt": texts}) as response:
            *prompts, france, japan = json.load(response)["tokens"]
        params = {"temperature": 0, "max_new_tokens": GENERATED, "ignore_eos": True}

        def answers() -> list[list[int]]:
            with send("/generate", {"input_ids": prompts, "sampling_params": params}) as response:
                return [item["output_ids"] for item in json.load(response)]

        def move(path: str, tags: list[str] | None) -> None:
            with send(path, {"tags": tags}) as response:
                assert response.status == 200

        expected = answers()
        for order in ([None], [["weights"], ["kv_cache"]], [["kv_cache"], ["weights"]]):
            for tags in order:
                move("/release_memory_occupation", tags)
            for tags in order:
                move("/resume_memory_occupation", tags)
            assert answers() == expected
        # A checkpoint that reads France as Japan answers the first prompt otherwise.
        [tokens] = [pair for pair in zip(france, japan, strict=True) if pair[0] != pair[1]]
        # With the backup every tag leaves, so no cache rows of the loaded weights answer for the
        # ones that come back; without it only the weights leave, so a kept cache would answer.
        tags = None if backup else ["weights"]
        with swapped(files, tokens):
            move("/release_memory_occupation", tags)
            move("/resume_memory_occupation", tags)
            changed = answers()
            if backup:
                assert changed == expected
            else:
                other = greedy(files, prompts, GENERATED)
                assert other[0] != expected[0]
                assert_close(send, prompts, changed, other)


# A test-only SGLang plugin, which its distribution's entry point beside it loads into every SGLang
# process whose path holds this directory: `/server_info` reports the served decoder's free device
# memory.
PROBE: Final = Path(__file__).resolve().parent / "probe"


@pytest.mark.feature("Sleep mode")
def test_releases_the_memory_of_each_tag(
    tmp_path_factory: pytest.TempPathFactory, chips: Chips, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Releasing the weights frees at least their bytes and releasing the KV cache at least its
    token capacity's; the cache resumes into its own bytes while the weights stay released, and the
    weights then load into theirs, taking no more than they freed."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(SLEEPING)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    # The distinct weights in BF16, a head tied to the token table once.
    weights = sum(parameter.nbytes for parameter in model.parameters())
    # A token's keys and values in every layer, in BF16.
    token = 2 * config.num_hidden_layers * config.num_key_value_heads * config.head_dim * 2
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(PROBE), os.environ.get("PYTHONPATH", "")])
    )
    row = {"id": SLEEPING, "task": "text", "flags": ["--enable-memory-saver"]}
    with chips.serve(tmp_path_factory, row) as send:

        def available() -> int:
            with send("/server_info") as response:
                return json.load(response)["internal_states"][0]["device_available"]

        def move(path: str, tags: list[str]) -> int:
            with send(path, {"tags": tags}) as response:
                assert response.status == 200
            return available()

        with send("/server_info") as response:
            state = json.load(response)["internal_states"][0]
        awake, cache = state["device_available"], state["memory_usage"]["token_capacity"] * token
        released = move("/release_memory_occupation", ["weights"])
        assert released - awake >= weights
        empty = move("/release_memory_occupation", ["kv_cache"])
        assert empty - released >= cache
        allocated = move("/resume_memory_occupation", ["kv_cache"])
        assert empty - allocated >= cache
        assert allocated - awake >= weights
        loaded = move("/resume_memory_occupation", ["weights"])
        assert allocated - loaded >= weights
        assert loaded >= awake
