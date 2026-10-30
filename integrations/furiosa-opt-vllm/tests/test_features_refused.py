"""Unsupported serving features stop configuration or loading before device execution."""

from pathlib import Path

import pytest
from test_vllm_platform import refusal
from transformers import AutoConfig
from vllm.engine.arg_utils import EngineArgs

import furiosa_opt_engine as engine
from furiosa_opt_vllm.admission import Refused


@pytest.mark.usefixtures("platform_env")
@pytest.mark.feature("Context extension past the trained length")
def test_refuses_extension(served: Path) -> None:
    """Startup names the model for a rotary scaling override the engine does not compute."""
    arguments = EngineArgs(
        model=str(served),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        hf_overrides={
            "rope_parameters": {"rope_type": "dynamic", "factor": 2.0, "rope_theta": 1000000.0}
        },
    )
    assert refusal(arguments) == "model"


@pytest.mark.feature("Multimodal inputs (images, video, audio)")
def test_refuses_multimodal(tmp_path: Path) -> None:
    """A load of a vision-language checkpoint whole is refused before any weight is read, since the
    engine lists only its media encoder; a load of the media encoder alone is admitted."""
    AutoConfig.from_pretrained("unsloth/gemma-4-12b-it-NVFP4").save_pretrained(tmp_path)
    checkpoint = engine.Checkpoint(tmp_path)
    with pytest.raises(NotImplementedError, match="a media encoder loads alone"):
        engine.LoadOptions().parse(checkpoint)
    engine.LoadOptions().encoder().parse(checkpoint)


@pytest.mark.usefixtures("platform_env")
@pytest.mark.feature("Long contexts, up to the compiled window")
def test_bounds_context_by_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A context one position past the decoder's window stops startup, naming the option and the
    window, even where vLLM allows a context past the checkpoint's own; the window itself serves."""
    from vllm import LLM, SamplingParams

    def start(context: int) -> LLM:
        return LLM(model="Qwen/Qwen3-0.6B", dtype="bfloat16", max_model_len=context, max_num_seqs=1)

    # The checkpoint declares 40960 positions; its decoder is compiled for 131072.
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    llm = start(131072)
    try:
        greedy = SamplingParams(temperature=0, max_tokens=1)
        output = llm.generate("The capital of France is", greedy)[0].outputs[0]
    finally:
        llm.llm_engine.engine_core.shutdown()
    assert "Paris" in output.text
    # The engine core runs in this process, so the load's refusal reaches the test; it runs last,
    # as the error's frames keep the refused load's chip.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    with pytest.raises(Refused) as refused:
        _ = start(131073)
    assert (refused.value.option, refused.value.requirement) == (
        "max_model_len",
        "requires max_model_len at most 131072",
    )
