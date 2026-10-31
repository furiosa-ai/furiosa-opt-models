"""Install the plugin into SGLang: its model configurations, the attention backends deterministic
inference takes, its response encoder, its host-pool strategy, and each SGLang function it replaces
or precedes, beside the reason."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import (
        DecodePreallocQueue,
        DecodeRequest,
        DecodeTransferQueue,
    )
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


def register() -> None:
    """Register extensions before SGLang starts its workers."""
    from collections.abc import Callable
    from typing import cast

    import torch
    from fastapi.encoders import ENCODERS_BY_TYPE
    from sglang.srt.configs.linear_attn_model_registry import (
        LinearAttnModelSpec,
        register_linear_attn_model,
    )
    from sglang.srt.managers.io_struct import (
        ReleaseMemoryOccupationReqInput,
        ReleaseMemoryOccupationReqOutput,
        ResumeMemoryOccupationReqInput,
        ResumeMemoryOccupationReqOutput,
    )
    from sglang.srt.managers.scheduler_components.weight_updater import (
        SchedulerWeightUpdaterManager,
    )
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import register_stack_strategy
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.server_args import (
        ServerArgs,
        add_deterministic_attention_backend_choices,
        add_radix_supported_deterministic_attention_backend_choices,
    )
    from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
    from transformers import AutoConfig

    from furiosa_opt_sglang import speculative
    from furiosa_opt_sglang.cache import Cache
    from furiosa_opt_sglang.generation import Sampled
    from furiosa_opt_sglang.offload import Strategy
    from furiosa_opt_sglang.paging import assign
    from furiosa_opt_sglang.recurrent import BambaConfig, Config, Mamba2Config
    from furiosa_opt_sglang.request import render, validate_request
    from furiosa_opt_sglang.residency import release, resume
    from furiosa_opt_sglang.runner import Module
    from furiosa_opt_sglang.worker import Worker

    def unbound(*, device: str) -> None:
        _ = device

    def watermark(check: Callable[[object], None], arguments: ServerArgs) -> None:
        # SGLang requires CUDA for its Triton selector; the plugin samples with its torch one.
        # The check reads the last declaration of each option first.
        class Cuda:
            def __getattr__(self, name: str) -> object:
                if name == "_resolved_overrides":
                    declared = getattr(arguments, name, None) or ()
                    return [*declared, ("furiosa", {"device": "cuda"})]
                return getattr(arguments, name)

        check(Cuda())

    def speculation(_algorithm: object, _server_args: ServerArgs) -> type[speculative.Worker]:
        return speculative.Worker

    def collect(_scheduler: object, _batch: object, result: object) -> None:
        if isinstance(result, Sampled):
            result.collect()

    def saver(
        create: Callable[[bool], TorchMemorySaverAdapter], enable: bool
    ) -> TorchMemorySaverAdapter:
        _ = enable
        return create(False)

    def module(updater: SchedulerWeightUpdaterManager) -> Module:
        # SGLang leaves the updater's worker untyped; the plugin replaces SGLang's worker.
        return cast("Worker", updater.tp_worker).runner.model

    def released(
        _result: ReleaseMemoryOccupationReqOutput,
        updater: SchedulerWeightUpdaterManager,
        request: ReleaseMemoryOccupationReqInput,
    ) -> None:
        release(module(updater), request.tags, updater.flush_cache)

    def resumed(
        _result: ResumeMemoryOccupationReqOutput,
        updater: SchedulerWeightUpdaterManager,
        request: ResumeMemoryOccupationReqInput,
    ) -> None:
        resume(module(updater), request.tags)

    def send(
        scheduler: Scheduler, req: Req, last_chunk: bool = False, end_idx: int | None = None
    ) -> None:
        _ = last_chunk
        # SGLang leaves the scheduler's pools untyped.
        allocator = cast("BaseTokenToKVPoolAllocator", scheduler.token_to_kv_pool_allocator)
        requests = cast("ReqToTokenPool", scheduler.req_to_token_pool)
        cache = allocator.get_kvcache()
        assert isinstance(cache, Cache), "the platform disaggregates the engine's token pool only"
        # SGLang sends these tokens' pages, a non-last send all but a partial last page.
        end = end_idx
        if end is None:
            extended = req.extend_end
            assert extended is not None, "SGLang sends a request's pages once a step extended it"
            end = min(extended, len(req.origin_input_ids))
        cache.image.store(requests.req_to_token[req.kv.req_pool_idx, req.start_send_idx : end])

    def receive(
        queue: DecodePreallocQueue,
        decode_req: DecodeRequest,
        kv_indices: torch.Tensor,
        *_metadata: object,
        **options: object,
    ) -> None:
        cache = queue.token_to_kv_pool
        assert isinstance(cache, Cache), "the platform disaggregates the engine's token pool only"
        # SGLang's transfer destinations are string enums; the platform refuses host receive.
        assert options.get("destination") != "host", "a transfer writes into the host image only"
        row = decode_req.req.kv.req_pool_idx
        assert row is not None, "SGLang allocates a request's row before it receives the row's KV"
        cache.image.receive(row, kv_indices)

    def transferred(
        result: list[Req], queue: DecodeTransferQueue, *_args: object, **_options: object
    ) -> None:
        # SGLang leaves the scheduler's pools untyped.
        allocator = cast("BaseTokenToKVPoolAllocator", queue.scheduler.token_to_kv_pool_allocator)
        cache = allocator.get_kvcache()
        assert isinstance(cache, Cache), "the platform disaggregates the engine's token pool only"
        # Every transferred request holds the row it received into.
        cache.image.load(row for req in result if (row := req.kv.req_pool_idx) is not None)

    # SGLang stacks pooled tensors before HTTP serialization and exposes no response encoder hook.
    ENCODERS_BY_TYPE[torch.Tensor] = cast(
        "Callable[[torch.Tensor], list[float]]", torch.Tensor.tolist
    )
    # Transformers and SGLang leave their registries' entry parameters untyped.
    configure = cast("Callable[..., None]", AutoConfig.register)
    install = cast("Callable[..., None]", HookRegistry.register)

    # SGLang schedules a recurrent model's cache only for configurations it knows as linear
    # attention.
    for config, architecture in (
        (Config, "MambaForCausalLM"),
        (Mamba2Config, "Mamba2ForCausalLM"),
        (BambaConfig, "BambaForCausalLM"),
    ):
        configure(config.model_type, config, exist_ok=True)
        register_linear_attn_model(
            LinearAttnModelSpec(
                config_class=config,
                backend_class_name="furiosa_opt_sglang.runner.Attention",
                arch_names=[architecture],
                support_mamba_cache_extra_buffer=False,
            )
        )

    # Deterministic inference takes only the attention backends SGLang lists; the engine replaces
    # the host one SGLang names, and its batch-invariant load reads a cached prefix as it wrote it.
    for add in (
        add_deterministic_attention_backend_choices,
        add_radix_supported_deterministic_attention_backend_choices,
    ):
        add(["intel_amx", "torch_native"])

    # SGLang's hierarchical cache builds the host tier of the engine's token pool by this strategy.
    register_stack_strategy(Strategy())

    manager = f"{SchedulerWeightUpdaterManager.__module__}.{SchedulerWeightUpdaterManager.__name__}"
    for target, kind, function in (
        # The engine executes the scheduled batches.
        ("sglang.srt.managers.tp_worker.TpModelWorker", HookType.REPLACE, Worker),
        # SGLang's speculative workers run torch draft models; the engine drafts and verifies.
        (
            "sglang.srt.speculative.spec_info.SpeculativeAlgorithm.create_worker",
            HookType.REPLACE,
            speculation,
        ),
        # SGLang writes a step's cache slots to the request table with a Triton kernel unless its
        # own CPU kernels are installed; the table lives on the host.
        ("sglang.srt.mem_cache.allocation.assign_req_to_token_pool_func", HookType.REPLACE, assign),
        # A request with inputs the plugin does not serve is refused before tokenization.
        (
            "sglang.srt.managers.tokenizer_manager.TokenizerManager._validate_one_request",
            HookType.BEFORE,
            validate_request,
        ),
        # A rerank query and document render through the checkpoint's chat template.
        (
            "sglang.srt.entrypoints.openai.serving_rerank._render_jinja_chat_template",
            HookType.REPLACE,
            render,
        ),
        # SGLang binds the scheduler's threads to a NUMA node for a model that runs on the CPU;
        # the engine runs this one, so they stay unbound.
        ("sglang.srt.distributed.bootstrap._bind_threads_if_cpu", HookType.REPLACE, unbound),
        # An overlapped generation step's tokens are collected before SGLang reads its result.
        ("sglang.srt.managers.scheduler.Scheduler.process_batch_result", HookType.BEFORE, collect),
        # The engine, not torch's allocator, holds the device memory the memory saver releases.
        (
            "sglang.srt.utils.torch_memory_saver_adapter.TorchMemorySaverAdapter.create",
            HookType.AROUND,
            saver,
        ),
        # A transfer backend sends a prefill's pages from the token pool's host image, which holds
        # what the device's pages hold once the send's store returns.
        (
            "sglang.srt.disaggregation.prefill.SchedulerDisaggregationPrefillMixin.send_kv_chunk",
            HookType.BEFORE,
            send,
        ),
        # A decode server's transfer backend writes the token slots it names here into the host
        # image; they load as SGLang reports the transfer done, before it publishes, backs up or
        # runs the request.
        (
            "sglang.srt.disaggregation.decode.DecodePreallocQueue._send_kv_metadata",
            HookType.BEFORE,
            receive,
        ),
        (
            "sglang.srt.disaggregation.decode.DecodeTransferQueue.pop_transferred",
            HookType.AFTER,
            transferred,
        ),
        # Releasing and resuming memory by tag move the engine's weights and cache storage.
        (f"{manager}.release_memory_occupation", HookType.AFTER, released),
        (f"{manager}.resume_memory_occupation", HookType.AFTER, resumed),
        # Watermarking keeps every other check SGLang makes of its options.
        (
            "sglang.srt.arg_groups.validation_hook.check_watermark_server_args",
            HookType.AROUND,
            watermark,
        ),
    ):
        install(target, function, kind)
