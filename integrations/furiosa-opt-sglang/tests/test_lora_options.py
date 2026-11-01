"""Startup admits the LoRA options the engine's adapter terms serve, refuses the others by name,
and leaves SGLang's own LoRA rules to SGLang."""

import pytest
from sglang.srt.server_args import prepare_server_args
from test_platform import NGRAM, STANDALONE

from furiosa_opt_sglang.admission import Refused

# SGLang starts LoRA without adapters once it knows their rank and modules.
LORA = ("--enable-lora", "--max-lora-rank", "32", "--lora-target-modules", "all")

pytestmark = pytest.mark.feature("LoRA")


@pytest.mark.parametrize(
    ("flags", "option"),
    [
        (
            ["--enable-lora", "--max-lora-rank", "64", "--lora-target-modules", "all"],
            "max_lora_rank",
        ),
        (
            ["--enable-lora", "--max-lora-rank", "32", "--lora-target-modules", "lm_head"],
            "lora_target_modules",
        ),
        # Adapters named at startup enable LoRA, which SGLang resolves after the platform.
        (
            ["--lora-paths", "adapter=/nonexistent", "--lora-target-modules", "embed_tokens"],
            "lora_target_modules",
        ),
        ([*LORA, "--enable-lora-overlap-loading"], "enable_lora_overlap_loading"),
        ([*LORA, "--lora-use-virtual-experts"], "lora_use_virtual_experts"),
    ],
    ids=["rank", "head", "paths", "overlap", "experts"],
)
def test_refuses_unsupported_option(flags: list[str], option: str) -> None:
    """Startup names the refused LoRA option, also when only adapter paths enable LoRA."""
    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", *flags])
    with pytest.raises(Refused) as error:
        arguments.resolve_once()
    assert error.value.option == option


@pytest.mark.parametrize(
    "flags",
    [LORA, ("--lora-paths", "sky=/nonexistent")],
    ids=["enabled", "paths"],
)
def test_refuses_lora_on_loads_without_adapters(flags: tuple[str, ...]) -> None:
    """Startup refuses LoRA with the engine's reason when the model loads a decoder whose
    projections take no adapter, from the checkpoint's config alone, also when only adapter paths
    enable it."""
    arguments = prepare_server_args(["--model-path", "furiosa-ai/Qwen3-8B-FP8", *flags])
    with pytest.raises(Refused) as error:
        arguments.resolve_once()
    reason = "adapters: unsupported, the Qwen3ForCausalLM 8B implementation takes no adapter"
    assert (error.value.option, error.value.requirement) == ("model_path", reason)


@pytest.mark.parametrize(
    "flags",
    [
        ("--model-path", "Qwen/Qwen3-0.6B", *LORA, *NGRAM),
        ("--model-path", "Qwen/Qwen3-Embedding-0.6B", "--is-embedding", *LORA),
    ],
    ids=["ngram", "embedding"],
)
def test_serves_lora_with(flags: tuple[str, ...]) -> None:
    """LoRA starts with NGRAM speculation and on an embedding model, which SGLang serves."""
    arguments = prepare_server_args(list(flags))
    arguments.resolve_once()
    arguments.check_server_args()
    assert arguments.resolved_dict()["enable_lora"]


def test_inherits_the_refusal_of_lora_with_a_draft_model() -> None:
    """SGLang refuses LoRA with STANDALONE speculation itself, after the platform admits both,
    with its own reason."""
    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", *LORA, *STANDALONE])
    arguments.resolve_once()
    with pytest.raises(ValueError, match="LoRA is only compatible with NGRAM"):
        arguments.check_server_args()


@pytest.mark.parametrize(
    ("flags", "entries"),
    [([], 8), (["--max-loras-per-batch", "5"], 5), (["--max-loras-per-batch", "1"], 1)],
    ids=["default", "five", "fewer"],
)
def test_keeps_the_pool_capacity_of_sglang(flags: list[str], entries: int) -> None:
    """Startup keeps SGLang's pool capacity, its default included: the adapters one engine call
    takes bound a batch, not the pool."""
    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", *LORA, *flags])
    arguments.resolve_once()
    assert arguments.resolved_dict()["max_loras_per_batch"] == entries
