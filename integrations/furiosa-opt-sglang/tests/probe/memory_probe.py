"""A test-only SGLang general plugin: the scheduler's internal state, which `/server_info` reports,
holds `device_available`, the free device memory per chip of the served decoder's chips."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import GetInternalStateReq, GetInternalStateReqOutput
    from sglang.srt.managers.scheduler import Scheduler


def register() -> None:
    """Report the device memory after every internal state SGLang's scheduler builds."""
    from collections.abc import Callable
    from typing import cast

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    from furiosa_opt_sglang.residency import served
    from furiosa_opt_sglang.worker import Worker

    def report(
        result: GetInternalStateReqOutput, scheduler: Scheduler, _request: GetInternalStateReq
    ) -> None:
        # SGLang leaves the scheduler's worker untyped; the plugin replaces SGLang's worker.
        _, decoder, _ = served(cast("Worker", scheduler.tp_worker).runner.model)
        result.internal_state["device_available"] = decoder.device.available

    # SGLang leaves its registry's entry parameters untyped.
    install = cast("Callable[..., None]", HookRegistry.register)
    install("sglang.srt.managers.scheduler.Scheduler.get_internal_state", report, HookType.AFTER)
