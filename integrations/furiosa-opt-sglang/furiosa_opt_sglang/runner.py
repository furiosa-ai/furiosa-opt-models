from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NoReturn, TypeAlias, assert_never, final

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    resolve_attention_backend_strs,
)
from sglang.srt.runtime_context import get_server_args, get_spec
from typing_extensions import override

import furiosa_opt_engine as engine
from furiosa_opt_sglang import load
from furiosa_opt_sglang.cache import Attention
from furiosa_opt_sglang.decoder import Decoder
from furiosa_opt_sglang.generation.model import Model
from furiosa_opt_sglang.platform import Ngram, Refused, Speculations, Standalone
from furiosa_opt_sglang.speculative import Speculative
from furiosa_opt_sglang.speculative.ngram import Corpus
from furiosa_opt_sglang.speculative.standalone import Draft

if TYPE_CHECKING:
    from sglang.srt.kv_canary.runner.canary_manager import CanaryManager
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

    from furiosa_opt_sglang.generation import Sampled, Token

    # The memory SGLang's memory saver releases, by the tag SGLang names it with.
    Tag: TypeAlias = Literal["weights", "kv_cache"]


@dataclass(frozen=True, slots=True)
class Weights:
    """Weights the memory saver released: their copy in host memory, or None when resuming reads
    the checkpoint's files."""

    copy: engine.Checkpoint | None


@dataclass(frozen=True, slots=True)
class Storage:
    """A KV cache's storage the memory saver released: the spec and block count resuming
    allocates."""

    spec: engine.decoder.CacheSpec
    blocks: int


@final
class Module(torch.nn.Module):
    """SGLang's model object; the worker runs scheduled batches on the loaded model, of which
    SGLang's memory saver releases and restores the weights and cache storage."""

    def __init__(self, loaded: Model) -> None:
        super().__init__()
        self.loaded = loaded
        # The released memory, by its tag, and what resuming restores it from.
        self.asleep: dict[Tag, Weights | Storage] = {}

    @override
    def forward(self) -> NoReturn:
        # SGLang only inspects this signature; the worker replaces its forward calls.
        raise RuntimeError("the plugin worker executes scheduled batches")


@final
class Runner(ModelRunner):
    model: Module
    req_to_token_pool: ReqToTokenPool | None
    # Each request row's next token while a greedy step holds it on the device, until SGLang
    # releases the row; SGLang's own relay holds a placeholder for these rows.
    selected: dict[int, Token]
    # The steps issued since the last one SGLang has certainly processed, oldest first.
    issued: deque[Sampled]
    canary_manager: CanaryManager | None = None
    prefill_aware_swa = False
    speculative: Speculative | None = None

    @property
    @override
    def preloaded_weights_bytes(self) -> int:
        return 0

    @property
    @override
    def logical_max_total_num_tokens(self) -> int:
        # The base asks SGLang's KV configurator, which only the replaced pool allocation builds.
        pool = self.req_to_token_pool
        if pool is None:
            raise RuntimeError("cache initialization must precede the token capacity")
        return pool.schedulable_token_capacity(self.max_total_num_tokens)

    @override
    def initialize(self) -> None:
        from furiosa_opt_sglang.paging import initialize

        self.selected, self.issued = {}, deque()
        loaded = load.bind(self.model_config, self.gpu_id)
        options = Speculations.parse(get_server_args())
        if options is not None:
            # Verification selects each draft position's token on the device, which only a causal
            # language model's head does.
            match loaded:
                case Model(
                    core=Decoder(
                        model=engine.Decoder(output=engine.decoder.Output.TOKENS)
                    ) as target,
                ):
                    pass
                case _:
                    raise Refused(
                        "speculative_algorithm",
                        "speculative_algorithm requires a causal attention language model",
                    )
            width = get_spec().speculative_num_draft_tokens
            match options:
                case Ngram():
                    proposer = Corpus(options, width)
                case Standalone():
                    proposer = Draft(options, target, self.model_config.context_len)
                case _:
                    assert_never(options)
            self.speculative = Speculative(target, proposer, width)
        self.model = Module(initialize(self, loaded))
        from sglang.srt.layers.sampler import create_sampler

        self.sampler = create_sampler()
        self.maybe_init_watermark_state()
        self.init_kv_index_translator()
        self.init_attention_backends()
        self.init_cuda_graphs()
        self.init_ngram_embedding_manager()

    @override
    def alloc_memory_pool(self, memory_pool_config: MemoryPoolConfig | None = None) -> None:
        pass

    @override
    def init_attention_backends(self) -> None:
        resolved = resolve_attention_backend_strs(model_runner=self)
        self.prefill_attention_backend_str = resolved.prefill
        self.decode_attention_backend_str = resolved.decode
        self.attn_backend: AttentionBackend = Attention()

    @override
    def init_cuda_graphs(self, capture_decode_cuda_graph: bool = True) -> None:
        self.decode_cuda_graph_runner: BaseRunner | None = None
        self.prefill_cuda_graph_runner: BaseRunner | None = None
