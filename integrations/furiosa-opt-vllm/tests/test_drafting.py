"""A draft model's proposals, checked on the host against a causal model held in memory.

- A draft model drafts each position after the first from the draft before it.
- A parallel draft model reads its mask token at every draft position after the first, in the
  calls that append the known tokens, as one forward of the known tokens and the masks would.
- No drafter call runs an adapter: the target alone runs a request's adapter.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Final, cast

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from test_vllm_platform import refused
from transformers import BambaConfig, MambaConfig, PretrainedConfig
from vllm import SamplingParams
from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

import furiosa_opt_engine as engine
from furiosa_opt_vllm import admission, drafting
from furiosa_opt_vllm.additional import Section
from furiosa_opt_vllm.admission import Draft, Refused, speculative
from furiosa_opt_vllm.drafting import Drafter, Drafting
from furiosa_opt_vllm.runner import Runner

TARGET: Final = "Qwen/Qwen3-0.6B"
PARALLEL: Final = "amd/PARD-Qwen3-0.6B"
VOCABULARY: Final = 151936
BLOCK: Final = 16
# Rows one call of the model in memory holds: fewer than a long prompt's or many masks' rows.
ROWS: Final = 8
PROMPTS: Final = {"short": list(range(100, 105)), "long": list(range(200, 200 + ROWS + 3))}
TABLES: Final = {"short": list(range(128)), "long": list(range(128, 256))}


def greedy(prefix: Sequence[int]) -> int:
    """The token the model in memory selects after `prefix`, which every token of it decides."""
    return zlib.crc32(np.asarray(prefix, dtype=np.int64).tobytes()) % VOCABULARY


@dataclass(frozen=True, slots=True)
class Tensor:
    """A tensor of the engine in memory, whose device is the host."""

    values: tuple[int, ...]

    @staticmethod
    def from_dlpack(value: NDArray[np.int32]) -> Tensor:
        return Tensor(tuple(int(item) for item in value))

    def to_hbm(self, _device: object) -> Tensor:
        return self

    def to_host(self) -> Tensor:
        return self

    def numpy(self) -> NDArray[np.int32]:
        return np.asarray(self.values, dtype=np.int32)

    def __getitem__(self, rows: slice) -> Tensor:
        return Tensor(self.values[rows])


@dataclass(frozen=True, slots=True)
class Request:
    """A request of the engine in memory, with the arguments the drafter passes."""

    rows: Sequence[Sequence[int]]
    cached: int
    tokens: Tensor
    keep: engine.decoder.Keep | range
    adapter: int | None = field(default=None, kw_only=True)


@dataclass(slots=True)
class Cache:
    """Each written cache slot's token."""

    spec: object
    blocks: int
    slots: dict[int, int] = field(default_factory=dict)


@dataclass(slots=True)
class Model:
    """A causal model: a call writes each appended token at its slot and returns, for each kept
    row, `greedy` of the tokens its cache holds up to that row's position."""

    cache: SimpleNamespace = field(
        default_factory=lambda: SimpleNamespace(
            pools=[SimpleNamespace(block=BLOCK)], groups=[SimpleNamespace(pool=0)]
        )
    )
    rows: int = ROWS
    window: int = 2048
    # Every call's requests, in call order.
    calls: list[list[Request]] = field(default_factory=list)

    def forward(self, requests: Sequence[Request], cache: Cache) -> Tensor:
        assert sum(len(request.tokens.values) for request in requests) <= self.rows
        self.calls.append(list(requests))
        kept: list[int] = []
        for request in requests:
            table = request.rows[0]

            def slot(position: int, table: Sequence[int] = table) -> int:
                return table[position // BLOCK] * BLOCK + position % BLOCK

            appended = len(request.tokens.values)
            for offset, token in enumerate(request.tokens.values):
                cache.slots[slot(request.cached + offset)] = token
            match request.keep:
                case range():
                    rows = request.keep
                case engine.decoder.Keep.ALL:
                    rows = range(appended)
                case engine.decoder.Keep.LAST:
                    rows = range(appended - 1, appended)
                case engine.decoder.Keep.NONE:
                    rows = range(0)
            kept.extend(
                greedy([cache.slots[slot(p)] for p in range(request.cached + row + 1)])
                for row in rows
            )
        return Tensor(tuple(kept))


@dataclass(frozen=True, slots=True)
class Session:
    """A drafting of the two `PROMPTS` by the model in memory, and the runner whose batch and
    requests it reads."""

    drafting: Drafting
    model: Model
    cache: Cache
    runner: Runner

    def step(self, proposed: dict[str, list[int]], rejected: Sequence[str], count: int) -> None:
        """Sample a step's tokens, which reject a `rejected` request's first draft and accept every
        other's drafts and one more, then propose up to `count` drafts for each into `proposed`."""
        sampled = {
            req_id: [(proposed[req_id][0] + 1) % VOCABULARY]
            if req_id in rejected
            else [*proposed[req_id], 13]
            for req_id in PROMPTS
        }
        for req_id, tokens in sampled.items():
            self.runner.extend(self.runner.requests[req_id], tokens)
        drafts = self.drafting.propose(self.runner.input_batch, list(sampled.values()), count)
        proposed.update(zip(PROMPTS, drafts, strict=True))

    def known(self, req_id: str) -> list[int]:
        request = self.runner.requests[req_id]
        return [*PROMPTS[req_id], *request.output_token_ids]


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> Callable[[VllmConfig], Session]:
    """A session of a configuration's draft model, run by the engine in memory."""
    fake = SimpleNamespace(
        HostTensor=Tensor,
        decoder=SimpleNamespace(Request=Request, Cache=Cache, Keep=engine.decoder.Keep),
    )
    monkeypatch.setattr(drafting, "engine", fake)

    def start(config: VllmConfig) -> Session:
        draft = speculative(config, Section.parse(config))
        assert isinstance(draft, Draft)
        model = Model()
        runner = Runner(config)
        runner.input_batch = InputBatch(
            max_num_reqs=2,
            max_model_len=2048,
            max_num_batched_tokens=32,
            device=torch.device("cpu"),
            vocab_size=VOCABULARY,
            block_sizes=[BLOCK],
            kernel_block_sizes=[BLOCK],
            max_num_blocks_per_req=[128],
        )
        for req_id, prompt in PROMPTS.items():
            runner.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=prompt,
                mm_features=[],
                sampling_params=SamplingParams(temperature=0),
                generator=None,
                block_ids=(TABLES[req_id],),
                num_computed_tokens=0,
                output_token_ids=[],
            )
            _ = runner.input_batch.add_request(runner.requests[req_id])
        # The engine in memory stands in for the engine's decoder, device and cache.
        decoder = cast("engine.Decoder", cast("object", model))
        drafter = Drafter(decoder, cast("engine.Device", cast("object", None)), draft)
        proposer = Drafting(drafter, 256, runner.requests)
        return Session(proposer, model, cast("Cache", cast("object", proposer.cache)), runner)

    return start


def configuration(draft: str, *, parallel: bool) -> VllmConfig:
    return EngineArgs(
        model=TARGET,
        dtype="bfloat16",
        max_model_len=2048,
        skip_tokenizer_init=True,
        speculative_config={
            "method": "draft_model",
            "model": draft,
            "num_speculative_tokens": 3,
            "parallel_drafting": parallel,
        },
    ).create_engine_config()


# A step's tokens accept every draft, or reject the first draft of one request.
STEPS: Final = [(), ("short",), ("long",)]
# Three drafts fit one call; ten put their nine masks over two of the model's calls.
COUNTS: Final = [3, 10]


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize("count", COUNTS)
def test_drafts_from_previous_draft(session: Callable[[VllmConfig], Session], count: int) -> None:
    """Each request's drafts continue its known tokens greedily, one draft after another, also
    after rejected and accepted drafts and for known tokens that take more than one call."""
    run = session(configuration(TARGET, parallel=False))
    proposed: dict[str, list[int]] = {req_id: [] for req_id in PROMPTS}
    for rejected in STEPS:
        run.step(proposed, rejected, count)
        for req_id in PROMPTS:
            expected = run.known(req_id)
            for _ in range(count):
                expected.append(greedy(expected))
            assert proposed[req_id] == expected[-count:], (rejected, req_id)


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize("count", COUNTS)
def test_drafts_in_parallel(session: Callable[[VllmConfig], Session], count: int) -> None:
    """A parallel draft model's drafts are its greedy tokens at the last known position and at
    `count - 1` mask tokens after it, which its calls write there: proposed together, after
    rejected and accepted drafts, and for known tokens and masks that take more than one call."""
    mask = json.loads(Path(hf_hub_download(PARALLEL, "config.json")).read_text())["pard_token"]
    run = session(configuration(PARALLEL, parallel=True))
    proposed: dict[str, list[int]] = {req_id: [] for req_id in PROMPTS}
    for rejected in STEPS:
        run.step(proposed, rejected, count)
        for req_id in PROMPTS:
            known = run.known(req_id)
            expected = [greedy(known + [mask] * offset) for offset in range(count)]
            assert proposed[req_id] == expected, (rejected, req_id)
            table = TABLES[req_id]
            written = [
                run.cache.slots[table[p // BLOCK] * BLOCK + p % BLOCK]
                for p in range(len(known), len(known) + count - 1)
            ]
            assert written == [mask] * (count - 1), (rejected, req_id)


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(("draft", "parallel"), [(TARGET, False), (PARALLEL, True)])
def test_drafts_without_adapters(
    session: Callable[[VllmConfig], Session], draft: str, *, parallel: bool
) -> None:
    """A drafter's calls run the base model: no request names an adapter slot."""
    run = session(configuration(draft, parallel=parallel))
    proposed: dict[str, list[int]] = {req_id: [] for req_id in PROMPTS}
    for rejected in STEPS:
        run.step(proposed, rejected, 3)
    assert run.model.calls
    assert all(request.adapter is None for call in run.model.calls for request in call)


def parallel_draft(directory: Path, settings: dict[str, object]) -> EngineArgs:
    """The target drafted in parallel by the parallel draft model's config with `settings` over
    its mask token, saved in `directory`."""
    config = json.loads(Path(hf_hub_download(PARALLEL, "config.json")).read_text())
    del config["pard_token"]
    (directory / "config.json").write_text(json.dumps(config | settings))
    return EngineArgs(
        model=TARGET,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        speculative_config={
            "method": "draft_model",
            "model": str(directory),
            "num_speculative_tokens": 3,
            "parallel_drafting": True,
        },
    )


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(
    "settings",
    [
        {"pard_token": VOCABULARY},
        {"pard_token": -1},
        {"ptd_token_id": VOCABULARY + 7},
        {"pard_token": True},
        {},
    ],
    ids=["past-vocabulary", "negative", "ptd-past-vocabulary", "bool", "none"],
)
def test_refuses_mask_outside_vocabulary(tmp_path: Path, settings: dict[str, object]) -> None:
    """The engine reads no token ID, so startup refuses parallel drafting unless the draft config
    names a mask token in the draft vocabulary, before any weight loads."""
    error: Refused = refused(parallel_draft(tmp_path, settings))
    assert error.option == "speculative_config.parallel_drafting"


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(
    "recurrent",
    [
        BambaConfig(architectures=["BambaForCausalLM"], vocab_size=VOCABULARY),
        MambaConfig(architectures=["MambaForCausalLM"], vocab_size=VOCABULARY),
    ],
    ids=["hybrid", "recurrent"],
)
def test_refuses_recurrent_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recurrent: PretrainedConfig
) -> None:
    """A rejected draft rolls a draft cache back by rewriting attention blocks, which a recurrent
    state cannot undo, so startup refuses a recurrent draft model before the engine reads its
    checkpoint, whatever architectures the engine serves."""
    recurrent.save_pretrained(tmp_path)
    read: list[str] = []

    # An engine that serves every architecture reads each checkpoint it is asked about.
    def serves(
        _model: ModelConfig, _load: LoadConfig, option: str, _options: engine.LoadOptions
    ) -> None:
        read.append(option)

    monkeypatch.setattr(admission, "verify_checkpoint", serves)
    error = refused(
        EngineArgs(
            model=TARGET,
            dtype="bfloat16",
            skip_tokenizer_init=True,
            speculative_config={
                "method": "draft_model",
                "model": str(tmp_path),
                "num_speculative_tokens": 3,
            },
        )
    )
    assert (error.option, error.requirement) == (
        "speculative_config.model",
        "cannot roll back recurrent state for speculative decoding",
    )
    assert "speculative_config.model" not in read


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(
    "settings",
    [
        {"pard_token": 151670},
        {"ptd_token_id": 151671},
        {"mask_token_id": 151672},
        {"mask_token_id": None, "pard_token": 151670},
        {"mask_token_id": 151672, "pard_token": 151670},
        {"pard_token": 151670, "ptd_token_id": 151671},
        {"dflash_config": {"mask_token_id": 151673}, "mask_token_id": 151672},
        {"dflash_config": {"block_size": 16}, "pard_token": 151670},
        {"dspark_noise_token_id": 151674, "pard_token": 151670},
        {"mask_token_id": 151672, "dspark_noise_token_id": 151674},
    ],
    ids=[
        "pard",
        "ptd",
        "mask",
        "mask-none",
        "mask-first",
        "pard-first",
        "dflash-first",
        "dflash-without-mask",
        "dspark-first",
        "mask-before-dspark",
    ],
)
def test_reads_upstream_mask_token(tmp_path: Path, settings: dict[str, object]) -> None:
    """Parallel drafting reads the mask token vLLM's own proposer reads of the draft config."""
    config = parallel_draft(tmp_path, settings).create_engine_config()
    draft = speculative(config, Section.parse(config))
    assert isinstance(draft, Draft)
    assert config.speculative_config is not None
    upstream = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
    upstream.draft_model_config = config.speculative_config.draft_model_config
    upstream.pass_hidden_states_to_model = False
    _ = getattr(upstream, "_init_parallel_drafting_params")()  # noqa: B009
    assert draft.mask == upstream.parallel_drafting_token_id
