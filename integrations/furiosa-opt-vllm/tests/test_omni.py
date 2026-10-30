import json
from pathlib import Path

import pytest
from vllm_omni.diffusion.data import OmniDiffusionConfig

from furiosa_opt_vllm.omni.pipeline import build


def test_refuses_pipelines(tmp_path: Path) -> None:
    """vLLM-Omni's load of a diffusion pipeline is refused before a weight is read, with the
    engine's reason, which names the pipeline's architecture."""
    _ = (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "FluxPipeline"}))
    config = OmniDiffusionConfig(model=str(tmp_path), model_class_name="FluxPipeline")
    with pytest.raises(NotImplementedError) as refused:
        build(od_config=config)
    assert str(refused.value) == "FluxPipeline: unsupported, no model type is implemented"
