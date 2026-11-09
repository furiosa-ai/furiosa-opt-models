"""vLLM serves LoRA adapters on the engine: each request runs with its own adapter, several
adapters share a step, and vLLM's LoRA calls load and unload them."""

import json
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import pytest
from pydantic import ValidationError
from test_vllm_platform import refusal

from furiosa_opt_vllm.admission import Refused

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.lora.request import LoRARequest

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "Explain briefly why the sky is blue."
TOKENS = 8
ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]


@pytest.fixture(scope="module")
def adapters(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """PEFT checkpoints of the model saved by PEFT: of rank 32, the adapter terms' rank, on every
    attention and MLP projection, of rank 8 under rsLoRA on the attention projections, and of rank
    4 on the query and output projections, whose larger scale keeps its update as large as the
    others' on fewer modules.

    Each factor is `0.02 * N(0, 1)`: its update changes the greedy tokens, while a much larger one
    flattens the scores until BF16 rounding picks the token.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    directory = tmp_path_factory.mktemp("adapters")
    forms = {
        # An update grows as alpha / r times sqrt(r): alpha 45 at rank 32 keeps rank 16's size.
        "full": LoraConfig(r=32, lora_alpha=45, target_modules=ATTENTION + MLP),
        "rslora": LoraConfig(r=8, lora_alpha=16, use_rslora=True, target_modules=ATTENTION),
        "partial": LoraConfig(r=4, lora_alpha=32, target_modules=["q_proj", "o_proj"]),
    }
    for seed, (name, config) in enumerate(forms.items()):
        model = get_peft_model(AutoModelForCausalLM.from_pretrained(MODEL), config)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for parameter, value in model.named_parameters():
                if ".lora_" in parameter:
                    _ = value.copy_(0.02 * torch.randn(value.shape, generator=generator))
        model.save_pretrained(str(directory / name))
    return {name: directory / name for name in forms}


@pytest.fixture(scope="module")
def llm() -> Generator["LLM"]:
    """Two adapter slots and three loaded adapters, as many as the checkpoints."""
    from vllm import LLM

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=2048,
        generation_config="vllm",
        enable_lora=True,
        max_loras=2,
        max_cpu_loras=3,
        max_lora_rank=32,
        max_num_seqs=4,
    )
    yield llm
    llm.llm_engine.engine_core.shutdown()


def generate(llm: "LLM", requests: "list[LoRARequest | None]") -> list[list[int]]:
    """The greedy tokens of `PROMPT` under each adapter, generated in one call."""
    from vllm import SamplingParams

    outputs = llm.generate(
        [PROMPT] * len(requests),
        SamplingParams(temperature=0, max_tokens=TOKENS, ignore_eos=True),
        # vLLM serves a `None` entry with the base model, which its annotation omits.
        lora_request=cast("list[LoRARequest]", requests),
    )
    return [list(output.outputs[0].token_ids) for output in outputs]


def assert_peft_greedy(tokens: list[int], adapter: Path | None) -> None:
    """Each of `tokens` is a greedy token of `PROMPT` and the tokens before it under Transformers
    with `adapter` merged by PEFT, in FP32: its probability is at least 95% of the highest one's."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reference = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    if adapter is not None:
        reference = PeftModel.from_pretrained(reference, str(adapter)).merge_and_unload()
    context = AutoTokenizer.from_pretrained(MODEL)(PROMPT, return_tensors="pt").input_ids
    with torch.inference_mode():
        for token in tokens:
            scores = reference(context).logits[0, -1]
            # BF16 rounding can tie the two highest scores; compare the chosen probability.
            assert (scores[token] - scores.max()).exp() >= 0.95, (adapter, tokens)
            context = torch.cat([context, torch.tensor([[token]])], dim=-1)


@pytest.mark.feature("LoRA")
def test_adapted_requests_match_peft(llm: "LLM", adapters: dict[str, Path]) -> None:
    """In one call of the base model and three adapters, more than the slots hold, each request
    generates the greedy tokens of Transformers with its adapter merged by PEFT."""
    from vllm.lora.request import LoRARequest

    named = [
        LoRARequest(name, lora_id, str(path))
        for lora_id, (name, path) in enumerate(adapters.items(), start=1)
    ]
    generated = generate(llm, [None, *named])
    for request, tokens in zip([None, *named], generated, strict=True):
        assert_peft_greedy(tokens, None if request is None else Path(request.lora_path))
    assert len({tuple(tokens) for tokens in generated}) == len(generated)


@pytest.mark.feature("LoRA")
def test_loads_and_unloads(llm: "LLM", adapters: dict[str, Path], tmp_path: Path) -> None:
    """vLLM's add, pin, list and remove calls keep an adapter's tokens: an adapter removed and
    named again, and a copy under another ID, generate the tokens it generated first."""
    from vllm.lora.request import LoRARequest

    full = LoRARequest("full", 1, str(adapters["full"]))
    (first,) = generate(llm, [full])
    engine = llm.llm_engine
    assert engine.add_lora(full)
    assert engine.pin_lora(full.lora_int_id)
    assert full.lora_int_id in engine.list_loras()
    assert engine.remove_lora(full.lora_int_id)
    assert full.lora_int_id not in engine.list_loras()
    (again,) = generate(llm, [full])
    copy = LoRARequest("copy", 9, str(shutil.copytree(adapters["full"], tmp_path / "copy")))
    assert engine.add_lora(copy)
    (copied,) = generate(llm, [copy])
    assert again == first
    assert copied == first


@pytest.mark.feature("LoRA")
def test_reloads_in_place(llm: "LLM", adapters: dict[str, Path], tmp_path: Path) -> None:
    """A request that loads its adapter in place replaces the adapter of its ID: it generates the
    greedy tokens of the new checkpoint merged by PEFT, as do the next requests of the ID, and a
    replacement the engine cannot load leaves that adapter as it was."""
    from vllm.lora.request import LoRARequest

    (before,) = generate(llm, [LoRARequest("swap", 7, str(adapters["full"]))])
    replaced = LoRARequest("swap", 7, str(adapters["rslora"]), load_inplace=True)
    (after,) = generate(llm, [replaced])
    assert after != before
    assert_peft_greedy(after, adapters["rslora"])
    # A configuration of a rank its factors do not have: vLLM accepts it, the engine does not.
    broken = Path(shutil.copytree(adapters["full"], tmp_path / "broken"))
    config = json.loads((broken / "adapter_config.json").read_text())
    _ = (broken / "adapter_config.json").write_text(json.dumps(config | {"r": 8}))
    engine = llm.llm_engine
    try:
        with pytest.raises(Exception, match="not the configured 8"):
            _ = engine.add_lora(LoRARequest("swap", 7, str(broken), load_inplace=True))
        (kept,) = generate(llm, [LoRARequest("swap", 7, str(adapters["rslora"]))])
        assert kept == after
    finally:
        _ = engine.remove_lora(7)


@pytest.mark.feature("LoRA")
def test_refuses_loads_past_pinned_slots(llm: "LLM", adapters: dict[str, Path]) -> None:
    """With both slots holding pinned adapters, as vLLM pins one in its host cache and its active
    slots, loading another is refused and changes nothing, though the host cache could unload an
    unpinned adapter for it: that adapter stays loaded, and the pinned ones keep their tokens.
    Pinning an adapter that is not loaded is refused. Once a pinned adapter is removed, the other
    loads and pins in its slot, and each pinned adapter generates its own tokens."""
    from vllm.lora.request import LoRARequest

    full, rslora, partial = (
        LoRARequest(name, lora_id, str(path))
        for lora_id, (name, path) in enumerate(adapters.items(), start=1)
    )
    spare = LoRARequest("spare", 4, str(adapters["partial"]))
    expected = generate(llm, [full, rslora, partial])
    engine = llm.llm_engine
    try:
        # vLLM's engine core returns the IDs as a list.
        for lora_id in list(engine.list_loras()):
            _ = engine.remove_lora(lora_id)
        assert engine.add_lora(spare)
        for request in [full, rslora]:
            assert engine.add_lora(request)
            assert engine.pin_lora(request.lora_int_id)
        assert set(engine.list_loras()) == {1, 2, 4}
        with pytest.raises(Exception, match="hold every slot"):
            _ = engine.add_lora(partial)
        assert set(engine.list_loras()) == {1, 2, 4}
        with pytest.raises(Exception, match="not loaded"):
            _ = engine.pin_lora(partial.lora_int_id)
        assert generate(llm, [full, rslora]) == expected[:2]
        assert engine.remove_lora(full.lora_int_id)
        assert engine.add_lora(partial)
        assert engine.pin_lora(partial.lora_int_id)
        assert set(engine.list_loras()) == {2, 3, 4}
        assert generate(llm, [rslora, partial]) == expected[1:]
    finally:
        for lora_id in [1, 2, 3, 4]:
            _ = engine.remove_lora(lora_id)


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("lora_target_modules", ["o_proj"]),
        ("lora_target_modules", ["qkv_proj", "o_proj"]),
        ("lora_dtype", "float16"),
        ("fully_sharded_loras", True),
        ("enable_tower_connector_lora", True),
        ("max_lora_cls_labels", 4),
        ("max_lora_rank", 64),
        ("max_loras", 3),
    ],
)
def test_refuses_lora_settings(served: Path, option: str, value: object) -> None:
    """Startup refuses each LoRA setting the engine's adapter terms cannot honor, naming it."""
    from vllm.engine.arg_utils import EngineArgs

    arguments = EngineArgs(
        model=str(served), dtype="bfloat16", skip_tokenizer_init=True, enable_lora=True
    )
    setattr(arguments, option, value)
    assert refusal(arguments) == option


# The engine's refusal of a load whose implementation takes no adapter, by its size.
NO_ADAPTER: Final = "adapters: unsupported, the Qwen3ForCausalLM {} implementation takes no adapter"


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(
    ("model", "quantization", "reason"),
    [
        ("furiosa-ai/Qwen3-8B-FP8", None, NO_ADAPTER.format("8B")),
        (MODEL, "fp8_per_tensor", NO_ADAPTER.format("0.6B")),
    ],
    ids=["block-scaled", "quantized"],
)
def test_refuses_lora_on_loads_without_adapters(
    tmp_path: Path, model: str, quantization: str | None, reason: str
) -> None:
    """Startup refuses LoRA with the engine's reason when the model or the quantization vLLM asks
    for loads a decoder whose projections take no adapter, from the checkpoint's configuration
    alone: no device opens and no weight is read."""
    from transformers import AutoConfig
    from vllm.engine.arg_utils import EngineArgs

    AutoConfig.from_pretrained(model).save_pretrained(tmp_path)
    arguments = EngineArgs(
        model=str(tmp_path),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        enable_lora=True,
        quantization=quantization,
    )
    with pytest.raises(ValidationError) as error:
        _ = arguments.create_engine_config()
    refused = error.value.errors()[0].get("ctx", {}).get("error")
    assert isinstance(refused, Refused)
    assert (refused.option, refused.requirement) == ("model", reason)
