from __future__ import annotations

import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from furiosa_opt_engine import _mooncake
from furiosa_opt_vllm.mooncake_store import (
    FuriosaMooncakeStoreConnector,
    FuriosaMooncakeStoreWorker,
    _load_batches,
    _namespace_cache_prefix,
    _oversized_disk_groups,
    _staging_budget_bytes,
)
from furiosa_opt_vllm.offload import spec as offload_spec


@pytest.mark.parametrize(
    ("groups", "page_bytes", "disk_budget", "expected"),
    [
        ([], [4096], None, frozenset()),
        (
            [SimpleNamespace(layer_names=("layers.0", "layers.1"))],
            [4096, 4096],
            200_000,
            frozenset(),
        ),
        (
            [
                SimpleNamespace(layer_names=("layers.0",)),
                SimpleNamespace(layer_names=("layers.1",)),
            ],
            [65_536, 4096],
            65_536,
            frozenset({0}),
        ),
    ],
    ids=["no-mooncake-disk", "fits-raw-budget", "reject-single-oversized-key"],
)
def test_oversized_disk_groups_obey_upstream_key_budget(
    groups, page_bytes, disk_budget, expected
) -> None:
    assert _oversized_disk_groups(groups, page_bytes, disk_budget) == expected


def test_load_batches_use_upstream_disk_splitter_and_keep_slot_routes() -> None:
    keys = ["small", "large"]
    addresses = [[0x1000], [0x2000]]
    sizes = [[4096], [8192]]

    batches, oversized_key = _load_batches(keys, addresses, sizes, 20_000)

    assert oversized_key is None
    assert [(batch[0], batch[3]) for batch in batches] == [(["small"], [0]), (["large"], [1])]


def test_load_batches_reject_a_single_key_over_raw_disk_budget() -> None:
    batches, oversized_key = _load_batches(["oversized"], [[0x1000]], [[32_768]], 20_000)

    assert batches == []
    assert oversized_key == "oversized"


@pytest.mark.parametrize("value", [None, 0, -1, True, 1.5, "1024"])
def test_staging_budget_requires_positive_integer_megabytes(value: object) -> None:
    with pytest.raises(ValueError, match="staging_buffer_size_mb"):
        _staging_budget_bytes({"staging_buffer_size_mb": value})


def test_staging_budget_defaults_to_one_gibibyte() -> None:
    assert _staging_budget_bytes({}) == 1024**3


def test_store_prefix_namespaces_engine_bytes_and_is_idempotent() -> None:
    cache_config = SimpleNamespace(
        get_resolved_kv_cache_layout=lambda: SimpleNamespace(name="LBHNC"),
        cache_dtype="bfloat16",
    )
    transfer = SimpleNamespace(kv_connector_extra_config={"cache_prefix": "team"})
    config = SimpleNamespace(
        kv_transfer_config=transfer,
        cache_config=cache_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1),
    )
    kv_cache_config = SimpleNamespace(kv_cache_groups=[])

    _namespace_cache_prefix(config, kv_cache_config)
    prefix = transfer.kv_connector_extra_config["cache_prefix"]
    assert prefix.startswith("furiosa-engine-page-v1-")
    assert prefix.endswith(":team")

    _namespace_cache_prefix(config, kv_cache_config)
    assert transfer.kv_connector_extra_config["cache_prefix"] == prefix


def test_filesystem_secondary_tier_accepts_only_filesystem() -> None:
    assert offload_spec._tier_factory_config({"type": "fs"})["module_path"] == (
        "furiosa_opt_vllm.offload.storage"
    )
    with pytest.raises(ValueError, match="native Mooncake Store connector"):
        offload_spec._tier_factory_config({"type": "mooncake"})


def test_plugin_mooncake_extras_match_across_frameworks() -> None:
    root = Path(__file__).resolve().parents[3]
    manifests = [
        root / "integrations/furiosa-opt-vllm/pyproject.toml",
        root / "integrations/furiosa-opt-sglang/pyproject.toml",
    ]
    requirement_sets = [
        tomllib.loads(path.read_text())["project"]["optional-dependencies"]["mooncake"]
        for path in manifests
    ]
    assert requirement_sets[0] == requirement_sets[1]
    assert requirement_sets[0] == ["mooncake-transfer-engine-non-cuda>=0.3.12,<0.4.0"]


def test_save_progress_rewinds_when_reused_request_id_has_new_hashes() -> None:
    worker = object.__new__(FuriosaMooncakeStoreWorker)
    worker._save_locks = (threading.Lock(),)
    worker._state_lock = threading.Lock()
    worker._saved_offsets = {"request": 16}
    worker._saved_hashes = {"request": (b"old-prompt",)}
    observed_start: list[int] = []

    def operations(_request, *, load: bool, save_start: int):
        assert not load
        observed_start.append(save_start)
        return []

    worker._operations = operations
    request = SimpleNamespace(
        req_id="request",
        block_hashes=[b"new-prompt"],
        token_len_chunk=16,
    )

    assert worker._store_request(request)
    assert observed_start == [0]
    assert worker._saved_offsets["request"] == 16
    assert worker._saved_hashes["request"] == (b"new-prompt",)


def test_save_progress_keeps_offset_when_hashes_extend_same_prompt() -> None:
    worker = object.__new__(FuriosaMooncakeStoreWorker)
    worker._save_locks = (threading.Lock(),)
    worker._state_lock = threading.Lock()
    worker._saved_offsets = {"request": 16}
    worker._saved_hashes = {"request": (b"prompt",)}
    observed_start: list[int] = []
    worker._operations = lambda _request, *, load, save_start: (
        observed_start.append(save_start) or []
    )
    request = SimpleNamespace(
        req_id="request",
        block_hashes=[b"prompt", b"next"],
        token_len_chunk=24,
    )

    assert worker._store_request(request)
    assert observed_start == [16]


def test_native_connector_inherits_upstream_store_policy() -> None:
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import (
        MooncakeStoreConnector,
    )

    assert issubclass(FuriosaMooncakeStoreConnector, MooncakeStoreConnector)


@pytest.mark.parametrize("status", [-1, -600, -706, -900])
def test_unknown_or_unsafe_put_status_poisoned(status: int) -> None:
    with pytest.raises(_mooncake.UnconfirmedMooncakeIO):
        _mooncake.require_confirmed_put([status], 1)
