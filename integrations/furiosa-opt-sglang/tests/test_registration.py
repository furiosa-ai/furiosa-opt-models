from types import SimpleNamespace

import pytest


class _HostBuffer(bytearray):
    def __new__(cls, size: int):
        return super().__new__(cls, size)

    def __init__(self, size: int) -> None:
        super().__init__(size)

    def slice(self, offset: int, size: int) -> memoryview:
        return memoryview(self)[offset : offset + size]


class _DeviceBuffer:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size


def _pool(chips: int = 2):
    page_bytes, blocks = 24, 4
    layers = [_DeviceBuffer(page_bytes * blocks) for _ in range(2)]
    storage = SimpleNamespace(
        blocks=blocks,
        layers=layers,
        spec=SimpleNamespace(
            groups=[SimpleNamespace(pool=0, layers=[0, 1])],
            pools=[SimpleNamespace(bytes=page_bytes)],
        ),
    )
    return SimpleNamespace(
        pages=SimpleNamespace(chips=chips),
        storage=storage,
    ), page_bytes


def test_mooncake_registers_only_one_placeholder_page_per_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import furiosa_opt_engine as engine
    from furiosa_opt_sglang.cache.registration import MooncakeRegistration

    pool, page_bytes = _pool()
    monkeypatch.setattr(engine, "HostBuffer", _HostBuffer)
    registration = MooncakeRegistration(pool)

    infos = registration.infos()
    assert len(infos.addresses) == len(pool.storage.layers)
    assert infos.lengths == [page_bytes * pool.pages.chips] * len(pool.storage.layers)
    assert infos.page_sizes == [page_bytes * pool.pages.chips] * len(pool.storage.layers)
    assert len(registration._buffer) == sum(infos.lengths)
    assert [len(layer) for layer in registration._layers] == infos.lengths
    assert all(layer.obj is registration._buffer for layer in registration._layers)
    assert all(address > 0 for address in infos.addresses)
