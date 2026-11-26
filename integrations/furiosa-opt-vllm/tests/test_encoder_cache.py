"""Encoder cache reuse and eviction without creating an NPU device."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest


def test_step_reuses_and_evicts_encoder_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from furiosa_opt_vllm import encoder as encoder_module

    monkeypatch.setattr(encoder_module.engine, "HostBuffer", bytearray)

    class Transfer:
        def wait(self) -> None:
            pass

    class Buffer:
        def __init__(self, value: int, fail_transfer: bool = False) -> None:
            self.value = value
            self.fail_transfer = fail_transfer

        def __len__(self) -> int:
            return 4

        def to_host(self, host: bytearray) -> Transfer:
            if self.fail_transfer:
                raise RuntimeError("host transfer failed")
            raw = torch.tensor([[self.value, self.value + 1]], dtype=torch.bfloat16)
            memoryview(host)[:] = raw.view(torch.uint8).numpy().tobytes()
            return Transfer()

    class Rows:
        shape = (1, 2)

        def __init__(self, value: int) -> None:
            self.value = value
            self.buffer = Buffer(value)

    class Encoder:
        vision_input_width = 2
        audio_input_width = 2

        def __init__(self) -> None:
            self.vision_inputs: list[list[object]] = []
            self.audio_inputs: list[list[object]] = []

        def vision(self, items: list[object]) -> list[Rows]:
            self.vision_inputs.append(items)
            values = {"evicted": 1, "fresh": 2, "other": 3}
            return [Rows(values[item]) for item in items]

        def audio(self, items: list[object]) -> list[Rows]:
            self.audio_inputs.append(items)
            return [Rows(4 + index) for index, _ in enumerate(items)]

    class Adapter:
        def __init__(self) -> None:
            self.prepared: list[str] = []
            self.model: encoder_module.Encoder | None = None

        def prepare_vision(self, feature: object, _device: object) -> str:
            identifier = feature.identifier
            self.prepared.append(identifier)
            if identifier == "evicted":
                assert self.model is not None
                assert identifier not in self.model.transfers
            return identifier

        def prepare_audio(self, feature: object, _device: object) -> str:
            identifier = feature.identifier
            self.prepared.append(identifier)
            return identifier

    def feature(
        identifier: str, modality: str = "image", *, has_data: bool = True
    ) -> SimpleNamespace:
        return SimpleNamespace(
            identifier=identifier,
            modality=modality,
            data=object() if has_data else None,
            mm_position=SimpleNamespace(get_num_embeds=lambda: 1),
        )

    encoder = Encoder()
    adapter = Adapter()
    model = encoder_module.Encoder(encoder, object(), adapter)
    adapter.model = model
    model.transfers.update(
        kept=torch.tensor([[90, 91]], dtype=torch.bfloat16),
        evicted=torch.tensor([[91, 92]], dtype=torch.bfloat16),
    )
    scheduler = SimpleNamespace(
        free_encoder_mm_hashes=["evicted"],
        scheduled_encoder_inputs={"request": [2, 4, 5, 6, 7]},
    )
    requests = {
        "request": SimpleNamespace(
            mm_features=[
                feature("kept"),
                feature("evicted"),
                feature("fresh"),
                feature("fresh"),
                feature("other"),
                feature("audio-one", "audio"),
                feature("audio-two", "audio"),
                feature("data-less", has_data=False),
            ]
        )
    }
    connector_caches: list[dict[str, torch.Tensor]] = []
    saved: list[str] = []

    @contextmanager
    def connector_output(_scheduler: object, *, encoder_cache: dict[str, torch.Tensor]):
        connector_caches.append(dict(encoder_cache))
        yield

    def save_to_connector(cache: dict[str, torch.Tensor], identifier: str) -> None:
        assert identifier in cache
        saved.append(identifier)

    monkeypatch.setattr(
        encoder_module.ECConnectorModelRunnerMixin,
        "maybe_get_ec_connector_output",
        staticmethod(connector_output),
    )
    monkeypatch.setattr(
        encoder_module.ECConnectorModelRunnerMixin,
        "maybe_save_ec_to_connector",
        staticmethod(save_to_connector),
    )
    monkeypatch.setattr(encoder_module, "make_empty_encoder_model_runner_output", lambda _: "empty")

    output = encoder_module.step(model, scheduler, requests)

    assert output == "empty"
    assert "evicted" not in connector_caches[0]
    assert adapter.prepared == ["fresh", "other", "audio-one", "audio-two"]
    assert encoder.vision_inputs == [["fresh", "other"]]
    assert encoder.audio_inputs == [["audio-one", "audio-two"]]
    assert saved == ["fresh", "other", "audio-one", "audio-two"]
    assert set(model.transfers) == {
        "kept",
        "fresh",
        "other",
        "audio-one",
        "audio-two",
    }
    assert torch.equal(model.transfers["fresh"], torch.tensor([[2, 3]], dtype=torch.bfloat16))


def test_step_does_not_cache_rows_until_host_transfer_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from furiosa_opt_vllm import encoder as encoder_module

    monkeypatch.setattr(encoder_module.engine, "HostBuffer", bytearray)

    class Transfer:
        def wait(self) -> None:
            pass

    class Buffer:
        def __init__(self, fail_transfer: bool) -> None:
            self.fail_transfer = fail_transfer

        def __len__(self) -> int:
            return 4

        def to_host(self, host: bytearray) -> Transfer:
            if self.fail_transfer:
                raise RuntimeError("host transfer failed")
            raw = torch.tensor([[1, 2]], dtype=torch.bfloat16)
            memoryview(host)[:] = raw.view(torch.uint8).numpy().tobytes()
            return Transfer()

    class Rows:
        shape = (1, 2)

        def __init__(self, fail_transfer: bool) -> None:
            self.buffer = Buffer(fail_transfer)

    class Encoder:
        vision_input_width = 2

        fail_transfer = True
        calls = 0

        def vision(self, items: list[object]) -> list[Rows]:
            assert items == ["fresh"]
            self.calls += 1
            return [Rows(self.fail_transfer)]

    class Adapter:
        def prepare_vision(self, feature: object, _device: object) -> str:
            return feature.identifier

    @contextmanager
    def connector_output(_scheduler: object, *, encoder_cache: dict[str, torch.Tensor]):
        assert "fresh" not in encoder_cache
        yield

    encoder = Encoder()
    model = encoder_module.Encoder(encoder, object(), Adapter())
    scheduler = SimpleNamespace(
        free_encoder_mm_hashes=[],
        scheduled_encoder_inputs={"request": [0]},
    )
    requests = {
        "request": SimpleNamespace(
            mm_features=[
                SimpleNamespace(
                    identifier="fresh",
                    modality="image",
                    data=object(),
                    mm_position=SimpleNamespace(get_num_embeds=lambda: 1),
                )
            ]
        )
    }

    monkeypatch.setattr(
        encoder_module.ECConnectorModelRunnerMixin,
        "maybe_get_ec_connector_output",
        staticmethod(connector_output),
    )
    monkeypatch.setattr(
        encoder_module.ECConnectorModelRunnerMixin,
        "maybe_save_ec_to_connector",
        staticmethod(lambda *_: None),
    )
    monkeypatch.setattr(encoder_module, "make_empty_encoder_model_runner_output", lambda _: "empty")

    with pytest.raises(RuntimeError, match="host transfer failed"):
        encoder_module.step(model, scheduler, requests)

    assert model.transfers == {}

    encoder.fail_transfer = False
    assert encoder_module.step(model, scheduler, requests) == "empty"

    assert encoder.calls == 2
    assert set(model.transfers) == {"fresh"}
    assert torch.equal(model.transfers["fresh"], torch.tensor([[1, 2]], dtype=torch.bfloat16))
