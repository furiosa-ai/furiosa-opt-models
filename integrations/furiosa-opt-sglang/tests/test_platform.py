"""The plugin rejects unsupported server options before a scheduler loads the model."""

import pickle
from typing import cast

import pytest
from sglang.srt.server_args import ServerArgs, prepare_server_args

from furiosa_opt_sglang.admission import (
    Disaggregation,
    Hierarchical,
    Recurrent,
    Refused,
    Requested,
    Resolved,
    Speculations,
    Standalone,
)
from furiosa_opt_sglang.platform import Platform


@pytest.mark.parametrize(
    ("flags", "option"),
    [
        pytest.param(["--enable-lora"], "enable_lora", marks=pytest.mark.feature("LoRA")),
        (["--lora-paths", "adapter=/nonexistent"], "lora_paths"),
        pytest.param(
            ["--speculative-algorithm", "EAGLE"],
            "speculative_algorithm",
            marks=pytest.mark.feature(
                "Speculative decoding (EAGLE, MTP, Medusa, MLP, parallel draft models)"
            ),
        ),
        pytest.param(
            ["--quantization", "awq"],
            "quantization",
            marks=pytest.mark.feature("Online quantization"),
        ),
        (["--device", "cuda"], "device"),
        (["--dtype", "float32"], "dtype"),
        (["--kv-cache-dtype", "fp8_e5m2"], "kv_cache_dtype"),
        (["--attention-backend", "flashinfer"], "attention_backend"),
        (["--prefill-attention-backend", "flashinfer"], "prefill_attention_backend"),
        (["--decode-attention-backend", "flashinfer"], "decode_attention_backend"),
        (["--startup-weight-load-mode", "overlap"], "startup_weight_load_mode"),
        (["--load-format", "dummy"], "load_format"),
        (["--enable-unified-cache-external-linker"], "enable_unified_cache_external_linker"),
        (["--ple-offload-embedding"], "ple_offload_embedding"),
        (["--return-hidden-states-mode", "last"], "enable_return_hidden_states"),
        pytest.param(
            ["--encoder-only"],
            "encoder_only",
            marks=pytest.mark.feature("Disaggregated encoder and encoder caches"),
        ),
        (["--language-only"], "language_only"),
        pytest.param(
            ["--enable-watermark"],
            "enable_watermark",
            marks=pytest.mark.feature("Watermarking"),
            id="watermark",
        ),
        pytest.param(
            ["--reasoning-parser", "qwen3", "--enable-strict-thinking"],
            "enable_strict_thinking",
            marks=pytest.mark.feature("Tool calling and reasoning outputs"),
            id="strict-thinking",
        ),
    ],
)
def test_refuses_unsupported_option(flags: list[str], option: str) -> None:
    """Startup names the refused option, also when SGLang derives it from another flag."""
    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", *flags])
    with pytest.raises(Refused) as error:
        arguments.resolve_once()
    assert error.value.option == option


@pytest.mark.feature("FP8 KV cache")
@pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3"])
def test_refuses_unserved_cache(dtype: str) -> None:
    """Startup refuses a cache dtype the engine does not cache the model in, with the engine's
    reason, before a scheduler starts."""
    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", "--kv-cache-dtype", dtype])
    with pytest.raises(Refused) as error:
        arguments.resolve_once()
    assert error.value.option == "model_path"


def test_reads_only_options_sglang_declares() -> None:
    """Every option the platform reads is an SGLang server option, so none reads as missing."""
    declared = set(ServerArgs.__annotations__)
    for served in (Requested, Resolved, Recurrent, Hierarchical, Disaggregation):
        assert set(served.model_fields) <= declared, served.__name__


NGRAM = ("--speculative-algorithm", "NGRAM", "--speculative-ngram-max-bfs-breadth", "1")
STANDALONE = (
    "--speculative-algorithm",
    "STANDALONE",
    "--speculative-draft-model-path",
    "Qwen/Qwen3-0.6B",
)
FP8 = ("--quantization", "fp8")


@pytest.mark.feature("Online quantization")
@pytest.mark.parametrize(
    "flags",
    [(), NGRAM, STANDALONE, (*STANDALONE, "--speculative-draft-model-quantization", "unquant")],
    ids=["alone", "ngram", "inherited-draft", "unquantized-draft"],
)
def test_accepts_online_quantization(flags: tuple[str, ...]) -> None:
    """`--quantization fp8` on a BF16 checkpoint starts, alone and with a speculative algorithm,
    whether a draft model takes the target's quantization or none."""
    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", *FP8, *flags])
    arguments.resolve_once()
    assert arguments.resolved_dict()["quantization"] == "fp8"


@pytest.mark.feature("Online quantization")
@pytest.mark.parametrize(
    ("checkpoint", "reason"),
    [
        pytest.param(
            "Qwen/Qwen3-8B",
            "a load quantizes the checkpoint, the line holds BF16 projections",
            id="8B",
        ),
        # The engine names its refusal of the stored form, not of a quantization at load.
        pytest.param(
            "Qwen/Qwen3-0.6B-FP8",
            "quantization_config: the size's lines here read BF16 checkpoints",
            id="quantized",
        ),
    ],
)
def test_refuses_unserved_quantization(checkpoint: str, reason: str) -> None:
    """Startup refuses `--quantization fp8` with the engine's reason, before a scheduler starts:
    on a size with no FP8 form, and on a quantized checkpoint whose stored form no line serves,
    which loads as stored, as SGLang loads it."""
    arguments = prepare_server_args(["--model-path", checkpoint, *FP8])
    with pytest.raises(Refused) as error:
        arguments.resolve_once()
    assert error.value.option == "model_path"
    assert reason in error.value.requirement


@pytest.mark.feature("Online quantization")
@pytest.mark.parametrize(
    ("flags", "quantization"),
    [((), "fp8"), (("--speculative-draft-model-quantization", "unquant"), None)],
    ids=["inherited", "unquant"],
)
def test_schedules_the_resolved_draft_quantization(
    flags: tuple[str, ...], quantization: str | None
) -> None:
    """A scheduler reads the draft quantization SGLang resolved in the server options it receives:
    the target's by default, none for `unquant`."""
    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", *FP8, *STANDALONE, *flags])
    arguments.resolve_once()
    received = cast("ServerArgs", pickle.loads(pickle.dumps(arguments)))
    options = Speculations.parse(received)
    assert isinstance(options, Standalone)
    assert options.speculative_draft_model_quantization == quantization


@pytest.mark.feature("Online quantization")
def test_quantizes_each_output_channel() -> None:
    """SGLang's `fp8` of a BF16 checkpoint scales each output channel of a projection by its own
    scale, which the engine names `fp8_per_channel`; any other method SGLang resolves, such as a
    ModelOpt checkpoint's own, quantizes nothing at load."""
    from furiosa_opt_sglang.load import quantization

    assert quantization("fp8", {}) == "fp8_per_channel"
    assert quantization("modelopt_fp8", {}) is None


def test_accepts_embedding_offload_options_without_the_offload() -> None:
    """The per-layer embedding offload's storage options take no effect while it is off."""
    flags = ["--ple-offload-backend", "file", "--ple-offload-dir", "/tmp/ple"]
    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", *flags])
    arguments.resolve_once()
    assert arguments.resolved_dict()["ple_offload_dir"] == "/tmp/ple"


def test_uses_host_device_name() -> None:
    assert Platform().get_device() == "cpu"


def test_reports_host_memory_and_process_backend() -> None:
    platform = Platform()
    assert platform.get_device_total_memory() > 0
    assert platform.get_torch_distributed_backend_str() == "gloo"


# The hierarchical cache with the host pool the plugin builds, sized as SGLang's default.
HIERARCHICAL = ("--enable-hierarchical-cache", "--hicache-ratio", "2")


@pytest.mark.parametrize(
    ("flags", "option", "requirement"),
    [
        (
            ("--hicache-storage-backend", "file"),
            "hicache_storage_backend",
            "keeps the hierarchical cache in host memory; omit --hicache-storage-backend",
        ),
        (
            ("--hicache-host-memory-mode", "buffer_only"),
            "hicache_host_memory_mode",
            "keeps host pages as a cache tier; omit --hicache-host-memory-mode",
        ),
        (
            ("--hicache-io-backend", "direct"),
            "hicache_io_backend",
            "moves host pages through the engine; omit --hicache-io-backend",
        ),
        (
            ("--hicache-mem-layout", "layer_first"),
            "hicache_mem_layout",
            "keeps host pages in the engine's layout; omit --hicache-mem-layout",
        ),
        (
            ("--model-path", "state-spaces/mamba-130m-hf"),
            "enable_hierarchical_cache",
            "keeps attention pages in host memory, not recurrent state",
        ),
        (
            ("--enable-memory-saver",),
            "enable_memory_saver",
            "releases no memory under a hierarchical cache yet; omit --enable-memory-saver",
        ),
    ],
    ids=["storage", "buffer-only", "io-backend", "layout", "recurrent", "memory-saver"],
)
@pytest.mark.feature("KV cache offloading")
def test_refuses_hierarchical_cache(flags: tuple[str, ...], option: str, requirement: str) -> None:
    """Hierarchical cache options the engine's host pool would ignore are refused by name and
    reason."""
    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", *HIERARCHICAL, *flags])
    with pytest.raises(Refused) as error:
        arguments.resolve_once()
    assert (error.value.option, error.value.requirement) == (option, requirement)


@pytest.mark.feature("KV cache offloading")
def test_refuses_sizing_hierarchical_cache_by_host_memory() -> None:
    """SGLang sizes a host pool by the host memory only for the device pools it builds itself, so
    the engine's needs a ratio or a size."""
    arguments = prepare_server_args(
        ["--model-path", "Qwen/Qwen3-1.7B", "--enable-hierarchical-cache"]
    )
    with pytest.raises(Refused) as error:
        arguments.resolve_once()
    assert (error.value.option, error.value.requirement) == (
        "hicache_host_memory_fraction",
        "sizes the hierarchical cache by --hicache-ratio or --hicache-size",
    )


@pytest.mark.parametrize(
    "size", [("--hicache-ratio", "2"), ("--hicache-size", "4")], ids=["ratio", "size"]
)
@pytest.mark.feature("KV cache offloading")
def test_accepts_hierarchical_cache_of_given_size(size: tuple[str, str]) -> None:
    """A hierarchical cache sized by a ratio or a size starts."""
    arguments = prepare_server_args(
        ["--model-path", "Qwen/Qwen3-1.7B", "--enable-hierarchical-cache", *size]
    )
    arguments.resolve_once()
    assert arguments.resolved_dict()["enable_hierarchical_cache"]


@pytest.mark.parametrize(
    ("flags", "option", "requirement"),
    [
        (
            ("--speculative-algorithm", "NGRAM", "--speculative-ngram-max-bfs-breadth", "2"),
            "speculative_ngram_max_bfs_breadth",
            "requires speculative_ngram_max_bfs_breadth=1",
        ),
        (
            (*NGRAM, "--speculative-ngram-min-bfs-breadth", "2"),
            "speculative_ngram_min_bfs_breadth",
            "requires speculative_ngram_min_bfs_breadth=1",
        ),
        (
            (*STANDALONE, "--speculative-eagle-topk", "2"),
            "speculative_eagle_topk",
            "requires speculative_eagle_topk=1",
        ),
        (
            (*NGRAM, "--speculative-adaptive"),
            "speculative_adaptive",
            "does not support speculative_adaptive",
        ),
        (
            (*NGRAM, "--speculative-ngram-external-corpus-path", "/nonexistent"),
            "speculative_ngram_external_corpus_path",
            "does not support speculative_ngram_external_corpus_path",
        ),
        (
            (*NGRAM, "--speculative-ngram-external-sam-budget", "1"),
            "speculative_ngram_external_sam_budget",
            "does not support speculative_ngram_external_sam_budget",
        ),
        (
            (*STANDALONE, "--enable-memory-saver"),
            "enable_memory_saver",
            "releases no memory under speculative decoding yet; omit --enable-memory-saver",
        ),
        pytest.param(
            (*STANDALONE, "--speculative-draft-model-quantization", "awq"),
            "speculative_draft_model_quantization",
            "quantizes a BF16 draft model at load only with fp8, one scale per output channel",
            marks=pytest.mark.feature("Online quantization"),
        ),
    ],
    ids=[
        "max-breadth",
        "min-breadth",
        "topk",
        "adaptive",
        "corpus",
        "sam-budget",
        "sleep",
        "draft-quantization",
    ],
)
@pytest.mark.feature("Speculative decoding (ngram, suffix, draft model)")
def test_refuses_speculation(flags: tuple[str, ...], option: str, requirement: str) -> None:
    """Speculative options outside one chain a request are refused by name and reason."""
    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", *flags])
    with pytest.raises(Refused) as error:
        arguments.resolve_once()
    assert (error.value.option, error.value.requirement) == (option, requirement)
