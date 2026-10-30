"""A server started without sleep mode refuses to sleep and keeps serving."""

from __future__ import annotations

import pytest
from test_sleep import BLOCKS, CHECKPOINT, answers
from vllm import LLM


@pytest.mark.feature("Sleep mode")
def test_refuses_sleep_without_sleep_mode() -> None:
    """Only a load under `enable_sleep_mode` keeps the host copy that level 1 wakes from, so a
    server without it releases nothing and answers as before once its scheduling resumes."""
    llm = LLM(CHECKPOINT, dtype="bfloat16", max_model_len=512, num_gpu_blocks_override=BLOCKS)
    try:
        expected = answers(llm)
        with pytest.raises(Exception, match="enable_sleep_mode"):
            llm.sleep(level=1)
        # vLLM pauses the scheduler before the worker sleeps.
        llm.wake_up(tags=["scheduling"])
        assert answers(llm) == expected
    finally:
        llm.llm_engine.engine_core.shutdown()
