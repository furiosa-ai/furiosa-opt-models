"""The compatibility module re-exports the engine's shared batch API."""

import furiosa_opt_engine.batch as engine_batch
import furiosa_opt_vllm.batch as vllm_batch


def test_batch_exports_engine_symbols() -> None:
    assert vllm_batch.__all__ == engine_batch.__all__
    assert all(
        getattr(vllm_batch, name) is getattr(engine_batch, name) for name in engine_batch.__all__
    )
