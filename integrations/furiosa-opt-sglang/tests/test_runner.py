import inspect

from furiosa_opt_sglang.generation.model import Model
from furiosa_opt_sglang.runner import Module


def test_model_forward_is_inspectable() -> None:
    """SGLang derives pipeline proxy support from the runner model's forward parameters."""
    model = Module(Model.__new__(Model))

    assert "pp_proxy_tensors" not in inspect.signature(model.forward).parameters
