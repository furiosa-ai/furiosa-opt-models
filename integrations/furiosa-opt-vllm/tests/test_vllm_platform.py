from pathlib import Path
from typing import Final, Literal, TypedDict, get_args

import pytest
import torch
from pydantic import ValidationError
from transformers import (
    AutoConfig,
    BambaConfig,
    BertConfig,
    CLIPConfig,
    MambaConfig,
    PretrainedConfig,
    Qwen3MoeConfig,
    WhisperConfig,
    XLMRobertaConfig,
)
from vllm.config import (
    DeviceConfig,
    ECTransferConfig,
    KVTransferConfig,
    LoadConfig,
    VllmConfig,
    speculative,
)
from vllm.config.cache import CacheDType, KVOffloadingBackend
from vllm.config.speculative import DraftSampleMethod, RejectionSampleMethod, SpeculativeMethod
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import EngineInput, embeds_input, tokens_input
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.models import ModelRegistry
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

import furiosa_opt_engine as engine
from furiosa_opt_vllm import offload
from furiosa_opt_vllm.additional import Placement, Section
from furiosa_opt_vllm.admission import Draft, Method, Refused, quantization
from furiosa_opt_vllm.platform import FuriosaPlatform

# Every speculative method vLLM names.
UPSTREAM: Final = frozenset(map(str, get_args(SpeculativeMethod)))


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(
    "placement",
    [
        {"chips": [], "pes": [0, 8]},
        {"chips": [1, 1], "pes": [0, 8]},
        {"chips": [True], "pes": [0, 8]},
        {"chips": [1], "pes": [4, 2]},
    ],
)
def test_refuses_bad_placement(placement: dict[str, list[int]]) -> None:
    """Admission refuses a draft placement without distinct chips or with an empty PE range."""
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.additional_config = {"furiosa": {"draft_placement": placement}}
    with pytest.raises(Refused) as error:
        FuriosaPlatform.check_and_update_config(config)
    assert error.value.option == "additional_config.furiosa"


@pytest.mark.usefixtures("platform_env")
def test_keeps_given_placement() -> None:
    """The plugin's section keeps a draft placement's chips in rank order and its PE range."""
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.additional_config = {"furiosa": {"draft_placement": {"chips": [3, 1], "pes": [2, 6]}}}
    assert Section.parse(config).draft_placement == Placement(chips=(3, 1), pes=(2, 6))


def test_opens_visible_chips(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local rank names the chip at its rank in the runtime's mask of visible chips."""
    monkeypatch.setenv("FURIOSA_VISIBLE_CHIPS", "5,3")
    assert [FuriosaPlatform.device_id_to_physical_device_id(rank) for rank in range(2)] == [5, 3]


@pytest.mark.parametrize(
    ("inputs", "params", "option"),
    [
        (tokens_input([1, 2]), SamplingParams(temperature=1.0), "sampling_params"),
        (tokens_input([1, 2]), SamplingParams(temperature=0, logprobs=1), "sampling_params"),
        (embeds_input(torch.zeros(2, 8)), SamplingParams(temperature=0), "prompt_embeds"),
    ],
    ids=["sampling", "logprobs", "embeddings"],
)
def test_refuses_unserved_requests(
    inputs: EngineInput, params: SamplingParams, option: str
) -> None:
    """A request that needs vocabulary scores or takes embeddings is refused before it is queued."""
    with pytest.raises(Refused) as error:
        FuriosaPlatform.validate_request(inputs, params)
    assert error.value.option == option


@pytest.mark.parametrize("name", sorted(UPSTREAM - set(Method)))
@pytest.mark.feature("Speculative decoding (EAGLE, MTP, Medusa, MLP, parallel draft models)")
def test_refuses_unbuilt_methods(name: str) -> None:
    """Admission refuses every speculative method vLLM names but ngram, suffix and a draft model."""
    with pytest.raises(Refused) as error:
        _ = Method.parse(name)
    assert error.value.option == "speculative_config.method"


class Overrides(TypedDict, total=False):
    """The speculative settings a case sets beside its method and draft count."""

    rejection_sample_method: RejectionSampleMethod
    synthetic_acceptance_length: float
    num_speculative_tokens_per_batch_size: list[tuple[int, int, int]]
    draft_sample_method: DraftSampleMethod
    use_heterogeneous_vocab: bool
    kv_cache_dtype: CacheDType
    draft_load_config: LoadConfig


def refused(arguments: EngineArgs) -> Refused:
    """The platform's refusal while vLLM configures `arguments`."""
    # vLLM's configuration is a pydantic dataclass, which wraps the platform's refusal.
    with pytest.raises(ValidationError) as error:
        _ = arguments.create_engine_config()
    refusal = error.value.errors()[0].get("ctx", {}).get("error")
    assert isinstance(refusal, Refused)
    return refusal


def refusal(arguments: EngineArgs) -> str:
    """The option the platform refuses while vLLM configures `arguments`."""
    return refused(arguments).option


# Each case is a configuration vLLM serves, which the plugin refuses.
@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(
    ("method", "settings", "option"),
    [
        (
            "ngram",
            Overrides(rejection_sample_method="block"),
            "speculative_config.rejection_sample_method",
        ),
        (
            "ngram",
            Overrides(rejection_sample_method="synthetic", synthetic_acceptance_length=2.0),
            "speculative_config.rejection_sample_method",
        ),
        # A batch-size schedule that drafts 1 of the 3 tokens from the second request on.
        (
            "suffix",
            Overrides(num_speculative_tokens_per_batch_size=[(1, 1, 3), (2, 8, 1)]),
            "speculative_config.num_speculative_tokens_per_batch_size",
        ),
        ("ngram_gpu", Overrides(), "speculative_config.method"),
    ],
)
def test_refuses_unserved_speculation(
    method: SpeculativeMethod,
    settings: Overrides,
    option: str,
    served: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup refuses a speculative setting the plugin would not honor and names its option."""
    # vLLM checks for the suffix extra before the platform sees the configuration.
    monkeypatch.setattr(speculative, "has_arctic_inference", lambda: True)
    arguments = EngineArgs(
        model=str(served),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        speculative_config={"method": method, "num_speculative_tokens": 3, **settings},
    )
    assert refusal(arguments) == option


def drafting(checkpoint: Path, settings: Overrides) -> EngineArgs:
    """vLLM's arguments of the configuration saved in `checkpoint`, which drafts for itself with
    `settings`; the target's `/target` download directory is no draft's own."""
    return EngineArgs(
        model=str(checkpoint),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        download_dir="/target",
        speculative_config={
            "method": "draft_model",
            "model": str(checkpoint),
            "num_speculative_tokens": 3,
            **settings,
        },
    )


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(
    ("settings", "option"),
    [
        (
            Overrides(draft_sample_method="probabilistic"),
            "speculative_config.draft_sample_method",
        ),
        (Overrides(use_heterogeneous_vocab=True), "speculative_config.use_heterogeneous_vocab"),
        (Overrides(kv_cache_dtype="fp8_e5m2"), "speculative_config.kv_cache_dtype"),
        (
            Overrides(draft_load_config=LoadConfig(load_format="safetensors")),
            "speculative_config.draft_load_config",
        ),
    ],
)
def test_refuses_drafting(settings: Overrides, option: str, served: Path) -> None:
    """Startup refuses a draft model setting the drafter would not honor and names its option."""
    assert refusal(drafting(served, settings)) == option


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(
    ("settings", "dtype", "directory"),
    [
        (
            Overrides(
                kv_cache_dtype="bfloat16", draft_load_config=LoadConfig(download_dir="/draft")
            ),
            "bfloat16",
            "/draft",
        ),
        (Overrides(), None, "/target"),
    ],
    ids=["own", "target"],
)
def test_inherits_settings(
    settings: Overrides, dtype: str | None, directory: str, served: Path
) -> None:
    """A draft model takes its own cache dtype and download directory, else the target's."""
    config = drafting(served, settings).create_engine_config()
    assert config.speculative_config is not None
    draft = Draft.parse(config, config.speculative_config, Section.parse(config))
    assert (draft.cache_dtype, draft.download_dir) == (dtype, directory)


@pytest.mark.usefixtures("platform_env")
@pytest.mark.feature("FP8 KV cache")
@pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3"])
def test_refuses_unserved_cache(dtype: CacheDType, served: Path) -> None:
    """Startup refuses a cache dtype the engine does not cache the model in, with the engine's
    reason, before the engine core starts."""
    arguments = EngineArgs(
        model=str(served), dtype="bfloat16", skip_tokenizer_init=True, kv_cache_dtype=dtype
    )
    assert refusal(arguments) == "model"


@pytest.mark.parametrize(
    ("section", "option", "value"),
    [
        ("parallel", "data_parallel_backend", "ray"),
        ("parallel", "prefill_context_parallel_size", 2),
        ("parallel", "decode_context_parallel_size", 2),
        ("parallel", "nnodes", 2),
        ("parallel", "enable_expert_parallel", True),
        ("cache", "cache_dtype", "float32"),
        ("cache", "kv_cache_dtype_skip_layers", [0]),
    ],
)
@pytest.mark.usefixtures("platform_env")
def test_names_option(
    section: Literal["parallel", "cache"],
    option: str,
    value: str | int | bool | list[int],
) -> None:
    """Startup names the unsupported parallel or cache option it refuses."""
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    target = config.parallel_config if section == "parallel" else config.cache_config
    setattr(target, option, value)
    with pytest.raises(Refused) as error:
        FuriosaPlatform.check_and_update_config(config)
    assert error.value.option == option


@pytest.fixture
def hybrid(tmp_path: Path, platform_env: None) -> Path:
    """The configuration of a hybrid attention and Mamba-2 checkpoint, without its weights."""
    BambaConfig(architectures=["BambaForCausalLM"]).save_pretrained(tmp_path)
    return tmp_path


def test_refuses_hits(hybrid: Path) -> None:
    """A prefix match unit lets a hybrid prefix hit end inside a block, which needs a block copy."""
    arguments = EngineArgs(
        model=str(hybrid), dtype="bfloat16", enable_prefix_caching=True, prefix_match_unit=16
    )
    assert refusal(arguments) == "prefix_match_unit"


def test_refuses_recurrent_models(hybrid: Path) -> None:
    """Draft rejection cannot undo a recurrent state's updates."""
    arguments = EngineArgs(
        model=str(hybrid),
        dtype="bfloat16",
        speculative_config={"method": "ngram", "num_speculative_tokens": 3},
    )
    assert refusal(arguments) == "speculative_config"


@pytest.mark.feature("KV cache offloading")
def test_refuses_offloading_recurrent_state(hybrid: Path) -> None:
    """Offloading moves the attention cache the engine holds, not the state vLLM holds."""
    arguments = EngineArgs(model=str(hybrid), dtype="bfloat16", kv_offloading_size=1)
    refusal = refused(arguments)
    assert (refusal.option, refusal.requirement) == (
        "kv_offloading_size",
        "offloads the attention cache of a decoder, not recurrent state or encoder memory",
    )


@pytest.mark.usefixtures("platform_env")
@pytest.mark.feature("KV cache offloading")
def test_refuses_offloading_on_an_encoder_only_instance(tmp_path: Path) -> None:
    """An encoder-only instance runs a checkpoint's media encoder alone, which caches no
    attention."""
    AutoConfig.from_pretrained("unsloth/gemma-4-12b-it-NVFP4").save_pretrained(tmp_path)
    producer = ECTransferConfig(ec_connector="ECExampleConnector", ec_role="ec_producer")
    arguments = EngineArgs(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        ec_transfer_config=producer,
        kv_offloading_size=1,
    )
    refusal = refused(arguments)
    assert (refusal.option, refusal.requirement) == (
        "kv_offloading_size",
        "offloads the attention cache of a decoder, which an encoder-only instance does not run",
    )


@pytest.mark.usefixtures("platform_env")
@pytest.mark.feature("KV cache offloading")
def test_refuses_simple_offloading(served: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """vLLM's simple CPU offloading connector copies through CUDA tensors."""
    monkeypatch.setenv("VLLM_USE_SIMPLE_KV_OFFLOAD", "1")
    arguments = EngineArgs(
        model=str(served), dtype="bfloat16", skip_tokenizer_init=True, kv_offloading_size=1
    )
    refusal = refused(arguments)
    assert (refusal.option, refusal.requirement) == (
        "VLLM_USE_SIMPLE_KV_OFFLOAD",
        "offloads through RNGDOffloadingConnector, so it serves no VLLM_USE_SIMPLE_KV_OFFLOAD=1",
    )


@pytest.mark.usefixtures("platform_env")
def test_names_missing_package(served: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Suffix decoding without Arctic Inference stops startup with the package's name."""
    arguments = EngineArgs(
        model=str(served),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        speculative_config={"method": "suffix", "num_speculative_tokens": 3},
    )
    # vLLM refuses before the platform sees the configuration, so the plugin does not check.
    monkeypatch.setattr(speculative, "has_arctic_inference", lambda: False)
    with pytest.raises(ImportError, match="arctic-inference"):
        _ = arguments.create_engine_config()


class Ignored(TypedDict, total=False):
    quantization: QuantizationMethods
    cpu_offload_gb: float
    offload_group_size: int
    load_format: str
    kv_offloading_size: float
    kv_offloading_backend: KVOffloadingBackend
    kv_transfer_config: KVTransferConfig
    ec_transfer_config: ECTransferConfig
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    enable_sleep_mode: bool


def extra(**options: object) -> KVTransferConfig:
    """A transfer config that only adds offloading options to `--kv-offloading-size`."""
    return KVTransferConfig(kv_connector_extra_config=options)


def connector(name: str, module: str | None = None) -> KVTransferConfig:
    """A transfer config naming the connector `name`, from `module` if set, with a host tier of
    1 GiB."""
    return KVTransferConfig(
        kv_connector=name,
        kv_connector_module_path=module,
        kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use": 1 << 30},
    )


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(
    ("ignored", "options"),
    [
        (Ignored(kv_offloading_size=1), {}),
        (
            Ignored(kv_offloading_size=1, kv_transfer_config=extra(blocks_per_chunk=2)),
            {"blocks_per_chunk": 2},
        ),
        (Ignored(kv_transfer_config=connector("RNGDOffloadingConnector", offload.__name__)), {}),
    ],
    ids=["size", "size-and-options", "connector"],
)
@pytest.mark.feature("KV cache offloading")
def test_offloads_through_the_rngd_connector(
    ignored: Ignored, options: dict[str, object], served: Path
) -> None:
    """`--kv-offloading-size` alone, or with offloading options, makes vLLM's factory build the
    plugin's connector over a host tier of that many GiB, as naming the connector does."""
    arguments = EngineArgs(model=str(served), dtype="bfloat16", skip_tokenizer_init=True, **ignored)
    transfer = arguments.create_engine_config().kv_transfer_config
    assert transfer is not None
    assert KVConnectorFactory.get_connector_class(transfer) is offload.RNGDOffloadingConnector
    assert transfer.kv_connector_extra_config == {"cpu_bytes_to_use": 1 << 30, **options}


# The refusals that name a single worker's ranks.
RANKS = "{}: vLLM awaits each rank; one worker drives all chips"


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(
    ("ignored", "option", "requirement"),
    [
        (
            Ignored(kv_offloading_size=1, kv_offloading_backend="lmcache"),
            "kv_offloading_backend",
            "offloads the KV cache with kv_offloading_backend=native",
        ),
        (
            Ignored(kv_transfer_config=connector("OffloadingConnector")),
            "kv_transfer_config",
            "offloads through RNGDOffloadingConnector, which --kv-offloading-size selects: "
            + "vLLM's OffloadingConnector copies through CUDA streams",
        ),
        (
            Ignored(kv_transfer_config=connector("RNGDOffloadingConnector", "example.offloading")),
            "kv_transfer_config",
            "serves no KV connector but RNGDOffloadingConnector, which --kv-offloading-size "
            + "selects",
        ),
        (
            Ignored(
                kv_offloading_size=1, kv_transfer_config=extra(spec_name="TieringOffloadingSpec")
            ),
            "kv_connector_extra_config",
            "keeps offloaded blocks in host memory by vLLM's CPUOffloadingSpec, so it takes no "
            + "other spec_name or spec_module_path",
        ),
        (
            Ignored(kv_offloading_size=1, kv_transfer_config=extra(canonical_layout=True)),
            "kv_connector_extra_config",
            "keeps offloaded blocks in the device's order, so it takes no canonical_layout",
        ),
        (
            Ignored(
                kv_offloading_size=1,
                kv_transfer_config=KVTransferConfig(kv_load_failure_policy="recompute"),
            ),
            "kv_load_failure_policy",
            "stops the engine when an offloaded block fails to copy: kv_load_failure_policy=fail",
        ),
        (
            Ignored(kv_offloading_size=1, tensor_parallel_size=2),
            "tensor_parallel_size",
            RANKS.format("offloads at tensor_parallel_size=1"),
        ),
        pytest.param(
            Ignored(kv_offloading_size=1, pipeline_parallel_size=2),
            "pipeline_parallel_size",
            RANKS.format("offloads at pipeline_parallel_size=1"),
            marks=pytest.mark.skip(reason="pending: pipeline parallelism, refused for every model"),
        ),
        (
            Ignored(kv_offloading_size=1, data_parallel_size=2),
            "data_parallel_size",
            "offloads at data_parallel_size=1: vLLM ranks each replica's worker among all replicas",
        ),
        (
            Ignored(kv_offloading_size=1, enable_sleep_mode=True),
            "enable_sleep_mode",
            "offloads the KV cache without sleep mode, which frees the cache its host tier copies",
        ),
    ],
    ids=[
        "lmcache",
        "vllm-connector",
        "module",
        "tiering",
        "canonical",
        "recompute",
        "tp",
        "pp",
        "dp",
        "sleep",
    ],
)
@pytest.mark.feature("KV cache offloading")
def test_refuses_offloading(ignored: Ignored, option: str, requirement: str, served: Path) -> None:
    """Startup refuses the offloading settings the plugin's connector cannot serve on the device,
    naming the option and the reason."""
    arguments = EngineArgs(model=str(served), dtype="bfloat16", skip_tokenizer_init=True, **ignored)
    refusal = refused(arguments)
    assert (refusal.option, refusal.requirement) == (option, requirement)


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(
    ("ignored", "option"),
    [
        pytest.param(
            Ignored(quantization="fp8"),
            "quantization",
            marks=pytest.mark.feature("Online quantization"),
        ),
        (Ignored(cpu_offload_gb=1), "cpu_offload_gb"),
        (Ignored(offload_group_size=2), "offload_group_size"),
        (Ignored(load_format="safetensors"), "load_format"),
        (Ignored(load_format="pt"), "load_format"),
        pytest.param(
            Ignored(
                kv_transfer_config=KVTransferConfig(
                    kv_connector="ExampleConnector", kv_role="kv_both"
                )
            ),
            "kv_transfer_config",
            marks=pytest.mark.feature(
                "Disaggregated prefill and KV connectors (NIXL, Mooncake, MoRIIO, LMCache)"
            ),
        ),
        pytest.param(
            Ignored(
                ec_transfer_config=ECTransferConfig(
                    ec_connector="ECExampleConnector", ec_role="ec_both"
                )
            ),
            "ec_transfer_config",
            marks=pytest.mark.feature("Disaggregated encoder and encoder caches"),
        ),
        pytest.param(
            Ignored(
                ec_transfer_config=ECTransferConfig(
                    ec_connector="ECExampleConnector", ec_role="ec_consumer"
                )
            ),
            "ec_transfer_config",
            marks=pytest.mark.feature("Disaggregated encoder and encoder caches"),
        ),
    ],
)
def test_refuses_ignored(ignored: Ignored, option: str, served: Path) -> None:
    """Startup names each requested option the plugin cannot honor."""
    arguments = EngineArgs(model=str(served), dtype="bfloat16", skip_tokenizer_init=True, **ignored)
    assert refusal(arguments) == option


class Online(TypedDict, total=False):
    quantization: QuantizationMethods
    quantization_config: dict[str, object]


@pytest.mark.usefixtures("platform_env")
@pytest.mark.feature("Online quantization")
@pytest.mark.parametrize(
    "settings",
    [
        Online(quantization="fp8_per_tensor"),
        Online(quantization="online", quantization_config={"linear": "fp8_per_tensor"}),
    ],
    ids=["shorthand", "linear"],
)
def test_quantizes_online(settings: Online, served: Path) -> None:
    """vLLM's per-tensor online FP8 asks a load of the BF16 checkpoint for the engine's."""
    arguments = EngineArgs(
        model=str(served), dtype="bfloat16", skip_tokenizer_init=True, **settings
    )
    assert quantization(arguments.create_model_config()) == "fp8_per_tensor"


@pytest.mark.usefixtures("platform_env")
@pytest.mark.feature("Online quantization")
@pytest.mark.parametrize(
    ("settings", "option"),
    [
        (Online(quantization="fp8_per_block"), "quantization"),
        (Online(quantization="fp8_per_channel"), "quantization"),
        (Online(quantization="mxfp8"), "quantization"),
        (
            Online(
                quantization="online",
                quantization_config={"linear": "fp8_per_tensor", "ignore": ["lm_head"]},
            ),
            "quantization_config.ignore",
        ),
        (
            Online(
                quantization="online",
                quantization_config={"targets": {"re:.*_proj": "fp8_per_tensor"}},
            ),
            "quantization_config.targets",
        ),
        (
            Online(
                quantization="online",
                quantization_config={
                    "linear": {"weight": "fp8_per_tensor_static", "activation": "fp8_per_token"}
                },
            ),
            "quantization_config.linear.activation",
        ),
    ],
    ids=["block", "channel", "mxfp8", "ignore", "targets", "activation"],
)
def test_refuses_online_quantization(settings: Online, option: str, served: Path) -> None:
    """Startup names each online quantization option a load of a BF16 checkpoint cannot apply."""
    arguments = EngineArgs(
        model=str(served), dtype="bfloat16", skip_tokenizer_init=True, **settings
    )
    assert refusal(arguments) == option


@pytest.mark.usefixtures("platform_env")
@pytest.mark.feature("Online quantization")
def test_refuses_an_unquantizable_checkpoint_before_its_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Hub checkpoint whose size the engine does not quantize is refused from its config alone,
    before the plugin downloads any weight."""
    from furiosa_opt_vllm import checkpoint

    def download(model: str, *_: object) -> str:
        raise AssertionError(f"{model} downloaded for a refused quantization")

    monkeypatch.setattr(checkpoint, "checkout", download)
    arguments = EngineArgs(
        model="Qwen/Qwen3-8B",
        dtype="bfloat16",
        skip_tokenizer_init=True,
        quantization="fp8_per_tensor",
    )
    with pytest.raises(ValidationError) as error:
        _ = arguments.create_engine_config()
    refused = error.value.errors()[0].get("ctx", {}).get("error")
    assert isinstance(refused, Refused)
    assert refused.option == "model"
    assert "a load quantizes the checkpoint" in refused.requirement


@pytest.mark.usefixtures("platform_env")
def test_refuses_experts(tmp_path: Path) -> None:
    """Startup refuses routed-expert output rather than silently omitting it."""
    Qwen3MoeConfig(architectures=["Qwen3MoeForCausalLM"]).save_pretrained(tmp_path)
    assert (
        refusal(
            EngineArgs(
                model=str(tmp_path),
                dtype="bfloat16",
                skip_tokenizer_init=True,
                enable_return_routed_experts=True,
            )
        )
        == "enable_return_routed_experts"
    )


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize("named", [None, "fp8"], ids=["detected", "named"])
def test_keeps_quantization(named: QuantizationMethods | None, tmp_path: Path) -> None:
    """Checkpoint quantization remains accepted whether detected or explicitly named."""
    AutoConfig.from_pretrained("RedHatAI/Mistral-Nemo-Instruct-2407-FP8").save_pretrained(tmp_path)
    arguments = EngineArgs(
        model=str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True, quantization=named
    )
    assert arguments.create_engine_config().model_config.quantization == "fp8"


@pytest.mark.usefixtures("platform_env")
def test_refuses_kernels(served: Path) -> None:
    """Startup reports a model refusal when its architecture has no compiled kernels."""

    def unbuilt(architecture: str) -> bool:
        # Given only the name, the engine refuses an architecture without kernels as unimplemented
        # and asks one with a checkpoint loader for its fields.
        checkpoint = engine.Checkpoint(served, config={"architectures": [architecture]})
        try:
            engine.LoadOptions().parse(checkpoint)
        except NotImplementedError:
            return True
        except ValueError:
            return False
        return False

    config = AutoConfig.from_pretrained(served)
    config.architectures = [next(filter(unbuilt, sorted(ModelRegistry.get_supported_archs())))]
    config.save_pretrained(served)
    assert (
        refusal(EngineArgs(model=str(served), dtype="bfloat16", skip_tokenizer_init=True))
        == "model"
    )


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize(
    "config",
    [
        CLIPConfig(architectures=["CLIPModel"]),
        # BGE-M3's architecture; vLLM's model reads a field Transformers no longer defaults.
        XLMRobertaConfig.from_dict(
            {"architectures": ["XLMRobertaModel"], "position_embedding_type": "absolute"}
        ),
        BertConfig(architectures=["BertModel"]),
        WhisperConfig(architectures=["WhisperForConditionalGeneration"]),
        MambaConfig(architectures=["MambaForCausalLM"]),
        BambaConfig(architectures=["BambaForCausalLM"]),
    ],
    ids=["contrastive", "retrieval", "encoder", "encoder-decoder", "recurrent", "hybrid"],
)
def test_refuses_families(config: PretrainedConfig, tmp_path: Path) -> None:
    """Startup refuses a checkpoint of a family the engine lists no model of from its config, with
    the engine's reason, which names the architecture."""
    config.save_pretrained(tmp_path)
    arguments = EngineArgs(model=str(tmp_path), dtype="bfloat16", skip_tokenizer_init=True)
    with pytest.raises(ValidationError) as error:
        _ = arguments.create_engine_config()
    refused = error.value.errors()[0].get("ctx", {}).get("error")
    assert isinstance(refused, Refused)
    assert config.architectures is not None
    (architecture,) = config.architectures
    assert (refused.option, refused.requirement) == (
        "model",
        f"{architecture}: unsupported, no model type is implemented",
    )


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize("runner", ["0", "1"], ids=["v1", "v2"])
def test_admits_processors(served: Path, monkeypatch: pytest.MonkeyPatch, runner: str) -> None:
    """Custom processors are accepted under V1 and explicitly refused under V2."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", runner)
    arguments = EngineArgs(
        model=str(served),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        logits_processors=[MinPLogitsProcessor],
    )
    if runner == "1":
        assert refusal(arguments) == "logits_processors"
    else:
        config = arguments.create_engine_config()
        assert not config.use_v2_model_runner
        assert config.model_config.logits_processors == [MinPLogitsProcessor]


@pytest.mark.usefixtures("platform_env")
@pytest.mark.parametrize("method", list(Method))
@pytest.mark.parametrize("drafting", ["greedy", "probabilistic"])
def test_refuses_watermarked(
    method: Method, drafting: DraftSampleMethod, served: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Watermarked speculation preserves upstream refusal reasons for each draft policy."""
    monkeypatch.setattr(speculative, "has_arctic_inference", lambda: True)
    arguments = EngineArgs(
        model=str(served),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        watermark_config={"key": 7, "allow_target_only_watermarking": True},
        speculative_config={
            "method": method,
            "model": str(served),
            "num_speculative_tokens": 3,
            "draft_sample_method": drafting,
        },
    )
    with pytest.raises(ValidationError) as error:
        arguments.create_engine_config()
    refused = error.value.errors()[0].get("ctx", {}).get("error")
    assert isinstance(refused, ValueError) and not isinstance(refused, Refused)
    reason = (
        "Speculative decoding with watermarking requires draft_sample_method='probabilistic'."
        if drafting == "greedy"
        else "Watermarking supports only autoregressive model-based speculative decoding."
    )
    assert str(refused) == reason
    arguments.watermark_config = None
    if method == Method.MODEL and drafting == "probabilistic":
        assert refusal(arguments) == "speculative_config.draft_sample_method"
    else:
        assert arguments.create_engine_config().speculative_config is not None
