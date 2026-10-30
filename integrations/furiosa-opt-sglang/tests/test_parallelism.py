"""Startup refuses pipeline, expert, context and multi-node parallelism and names the refused
option, and serves a tensor degree from one scheduler that drives every chip, once per data-parallel
replica."""

import pytest


@pytest.mark.parametrize(
    "option",
    [
        "ep_size",
        "pp_size",
        "nnodes",
        "dcp_size",
        "attn_cp_size",
        "attn_dp_size",
    ],
)
@pytest.mark.feature("Pipeline, expert and context parallelism")
def test_refuses_parallelism_at_startup(option: str) -> None:
    from sglang.srt.server_args import prepare_server_args

    flag = "--" + option.replace("_", "-")
    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", flag, "2"])
    # The server resolves its arguments, including the platform's defaults, before it starts.
    with pytest.raises(ValueError, match=f"requires {option}=1"):
        arguments.resolve_once()


def test_refuses_parallelism_derived_from_another_option() -> None:
    """DWDP declares expert and attention data parallelism, which startup refuses."""
    from sglang.srt.server_args import prepare_server_args

    flags = ["--tp-size", "2", "--dwdp-size", "2"]
    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", *flags])
    with pytest.raises(ValueError, match="requires ep_size=1"):
        arguments.resolve_once()


@pytest.mark.feature("Tensor parallelism over a compiled chip count")
def test_drives_tensor_ranks_from_one_scheduler() -> None:
    """A tensor degree starts one scheduler, which reads that degree from the operator's options."""
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.server_args import prepare_server_args

    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", "--tp-size", "2"])
    arguments.resolve_once()
    assert resolving_view(arguments).tp_size == 1
    assert arguments.tp_size == 2


@pytest.mark.feature("Data parallelism")
def test_starts_one_scheduler_per_data_parallel_replica() -> None:
    """Data-parallel replicas start one scheduler each, every one over the tensor degree's chips."""
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.server_args import prepare_server_args

    flags = ["--tp-size", "2", "--dp-size", "3"]
    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", *flags])
    arguments.resolve_once()
    view = resolving_view(arguments)
    assert (view.tp_size, view.dp_size, arguments.tp_size) == (1, 3, 2)
    # SGLang's first device index of each replica lies `gpu_id_step` past the last one's.
    assert view.gpu_id_step == 2


@pytest.mark.feature("Data parallelism")
def test_refuses_replica_spacing() -> None:
    """An operator's spacing of the replicas' chips is refused: the platform spaces them by the
    tensor degree."""
    from sglang.srt.server_args import prepare_server_args

    flags = ["--dp-size", "2", "--gpu-id-step", "2"]
    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", *flags])
    with pytest.raises(ValueError, match="requires gpu_id_step=1"):
        arguments.resolve_once()


@pytest.mark.feature("Data parallelism")
def test_refuses_speculation() -> None:
    """Data-parallel replicas with speculative decoding are refused by name and reason."""
    from sglang.srt.server_args import prepare_server_args

    from furiosa_opt_sglang.platform import Refused

    flags = ["--dp-size", "2", "--speculative-algorithm", "NGRAM"]
    flags += ["--speculative-ngram-max-bfs-breadth", "1"]
    arguments = prepare_server_args(["--model-path", "Qwen/Qwen3-1.7B", *flags])
    with pytest.raises(Refused) as error:
        arguments.resolve_once()
    assert (error.value.option, error.value.requirement) == (
        "dp_size",
        "requires dp_size=1 with speculative decoding",
    )
