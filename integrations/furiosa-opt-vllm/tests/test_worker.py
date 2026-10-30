from pathlib import Path

import pytest
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.core.sched.output import SchedulerOutput

from furiosa_opt_vllm.worker import FuriosaWorker


@pytest.mark.usefixtures("platform_env")
def test_propagates_failure(served: Path) -> None:
    """A step that vLLM issues before it collects a failed sample raises with that failure."""
    config = EngineArgs(
        model=str(served), dtype="bfloat16", skip_tokenizer_init=True
    ).create_engine_config()
    worker = FuriosaWorker(config, local_rank=0, rank=0, distributed_init_method="")
    worker.init_device()
    # Sampling before any execution fails, as a sampler error does.
    with pytest.raises(RuntimeError) as failed:
        _ = worker.sample_tokens(None)

    with pytest.raises(RuntimeError) as stopped:
        _ = worker.execute_model(SchedulerOutput.make_empty())

    assert stopped.value.__cause__ is failed.value
