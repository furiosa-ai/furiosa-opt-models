"""Startup admits prefill-decode disaggregation over Mooncake's TCP transport beside the caches and
memory release SGLang keeps with it, and names each setting the token pool's host image does not
carry."""

from collections.abc import Sequence
from typing import Final

import pytest
from sglang.srt.server_args import ServerArgs, prepare_server_args
from test_platform import HIERARCHICAL, NGRAM, STANDALONE

from furiosa_opt_sglang.admission import Refused

TRANSFER: Final = ("--disaggregation-transfer-backend", "mooncake_tcp")


def disaggregated(mode: str, flags: Sequence[str], monkeypatch: pytest.MonkeyPatch) -> ServerArgs:
    """The options of a `mode` server of `flags`, in an environment without `MC_FORCE_TCP`, which
    SGLang's `mooncake_tcp` backend sets in the process."""
    monkeypatch.setenv("MC_FORCE_TCP", "")
    monkeypatch.delenv("MC_FORCE_TCP")
    return prepare_server_args(
        ["--model-path", "Qwen/Qwen3-1.7B", "--disaggregation-mode", mode, *flags]
    )


@pytest.mark.feature("Disaggregated prefill and KV connectors (NIXL, Mooncake, MoRIIO, LMCache)")
@pytest.mark.parametrize(
    ("mode", "flags"),
    [
        ("prefill", ()),
        ("decode", ()),
        ("prefill", ("--enable-memory-saver",)),
        ("decode", ("--enable-memory-saver",)),
        ("prefill", HIERARCHICAL),
        ("decode", ("--disaggregation-decode-enable-radix-cache",)),
        ("decode", ("--disaggregation-decode-enable-radix-cache", *HIERARCHICAL)),
        ("prefill", ("--dp-size", "2")),
        ("prefill", STANDALONE),
        ("decode", STANDALONE),
        ("prefill", ("--enable-pd-role-switch",)),
        ("decode", ("--enable-pd-role-switch",)),
    ],
    ids=[
        "prefill",
        "decode",
        "prefill-memory-saver",
        "decode-memory-saver",
        "prefill-hierarchical",
        "decode-radix",
        "decode-hierarchical",
        "data-parallel",
        "prefill-standalone",
        "decode-standalone",
        "prefill-role-switch",
        "decode-role-switch",
    ],
)
def test_accepts_disaggregation(
    mode: str, flags: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prefill or a decode server starts with Mooncake's TCP transport, beside the memory saver,
    the caches SGLang keeps with disaggregation, data-parallel replicas, a draft model and role
    switching."""
    arguments = disaggregated(mode, [*TRANSFER, *flags], monkeypatch)
    arguments.resolve_once()
    resolved = arguments.resolved_dict()
    assert resolved["disaggregation_mode"] == mode
    assert resolved["disaggregation_transfer_backend"] == "mooncake"


@pytest.mark.feature("Disaggregated prefill and KV connectors (NIXL, Mooncake, MoRIIO, LMCache)")
@pytest.mark.parametrize(
    ("mode", "flags"),
    [("prefill", ()), ("decode", ("--enable-pd-role-switch",))],
    ids=["prefill", "role-switch"],
)
def test_prefills_without_overlap(
    mode: str, flags: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server that prefills, or may switch to prefilling, schedules without overlap: SGLang's
    overlapped early send of a cached prefix records a CUDA event, and the scheduler fixes its
    overlap when it starts."""
    arguments = disaggregated(mode, [*TRANSFER, *flags], monkeypatch)
    arguments.resolve_once()
    assert arguments.resolved_dict()["disable_overlap_schedule"]


BACKEND: Final = "moves the cache only over --disaggregation-transfer-backend mooncake_tcp"


@pytest.mark.feature("Disaggregated prefill and KV connectors (NIXL, Mooncake, MoRIIO, LMCache)")
@pytest.mark.parametrize(
    ("mode", "flags", "option", "requirement"),
    [
        (
            "prefill",
            ("--disaggregation-transfer-backend", "nixl"),
            "disaggregation_transfer_backend",
            BACKEND,
        ),
        (
            "decode",
            ("--disaggregation-transfer-backend", "mori"),
            "disaggregation_transfer_backend",
            BACKEND,
        ),
        (
            "prefill",
            ("--disaggregation-transfer-backend", "mooncake"),
            "disaggregation_transfer_backend",
            BACKEND,
        ),
        (
            "decode",
            ("--disaggregation-transfer-backend", "fake"),
            "disaggregation_transfer_backend",
            "moves no KV over the fake transfer backend, a synthetic decode benchmark; use "
            + "--disaggregation-transfer-backend mooncake_tcp",
        ),
        (
            "prefill",
            (*TRANSFER, "--tp-size", "2"),
            "tp_size",
            "requires tp_size=1 with disaggregation: an image page holds each chip's part in turn, "
            + "which a server split over other chips reads differently",
        ),
        (
            "decode",
            (*TRANSFER, "--disaggregation-decode-host-receive-threshold", "0.5"),
            "disaggregation_decode_host_receive_threshold",
            "receives KV only into the token pool's host image: SGLang receives into a host pool "
            + "only beside an MHA token pool's separate key and value buffers",
        ),
        (
            "decode",
            (
                *TRANSFER,
                *("--disaggregation-decode-retraction-backup", "host_pool", "--hicache-ratio", "1"),
            ),
            "disaggregation_decode_retraction_backup",
            "backs a retracted request up only by the token pool's cpu_tensor copies: SGLang's "
            + "host-pool backup needs an MHA token pool",
        ),
        (
            "decode",
            (*TRANSFER, *NGRAM),
            "speculative_algorithm",
            "drafts no NGRAM candidates with disaggregation: SGLang's decode server hands a "
            + "received request's first token to an NGRAM schedule in a form it cannot read",
        ),
        (
            "prefill",
            (*TRANSFER, *NGRAM),
            "speculative_algorithm",
            "drafts no NGRAM candidates with disaggregation: SGLang's decode server hands a "
            + "received request's first token to an NGRAM schedule in a form it cannot read",
        ),
        (
            "decode",
            (*TRANSFER, "--disaggregation-decode-enable-offload-kvcache"),
            "disaggregation_decode_enable_offload_kvcache",
            "offloads no decode server's cache: the engine's host pool serves no storage backend",
        ),
        (
            "prefill",
            (*TRANSFER, "--disaggregation-enable-kv-checksum"),
            "disaggregation_enable_kv_checksum",
            "computes no KV checksum: SGLang's checksum runs a CUDA kernel over the transfer "
            + "regions",
        ),
    ],
    ids=[
        "nixl",
        "mori",
        "rdma",
        "fake",
        "split",
        "host-receive",
        "host-pool-backup",
        "decode-ngram",
        "prefill-ngram",
        "decode-offload",
        "checksum",
    ],
)
def test_refuses_disaggregation(
    mode: str,
    flags: tuple[str, ...],
    option: str,
    requirement: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup names each disaggregation setting the token pool's host image does not carry, with
    its reason."""
    arguments = disaggregated(mode, flags, monkeypatch)
    with pytest.raises(Refused) as error:
        arguments.resolve_once()
    assert (error.value.option, error.value.requirement) == (option, requirement)
