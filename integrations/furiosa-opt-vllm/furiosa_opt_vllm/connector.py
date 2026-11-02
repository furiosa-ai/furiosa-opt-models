"""The engine cache as vLLM's NIXL connectors see it, when they stage blocks in host buffers.

- NIXL with `kv_buffer_device="cpu"` sizes host buffers of its own from each registered layer's
  tensor, sends and receives blocks between them, and asks the runner's copy to move blocks between
  them and the device: to the host before it sends, to the device once it received.
- `Layers` is that registration and copy. Its tensors hold no memory; its copy passes a request's
  blocks through host rows of each layer and ends before it returns, as NIXL expects, so a block
  NIXL reports received is on the device and a failed copy raises at the call that made it.
- The copy reads the cache the runner holds at that moment, which sleep mode frees and waking
  allocates anew.
- `Attention` is the backend the platform names for NIXL, which reads its name: the engine runs
  attention itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Literal, NoReturn, assert_never, final

import torch
from typing_extensions import override
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import FullAttentionSpec

import furiosa_opt_engine as engine
from furiosa_opt_vllm import residency
from furiosa_opt_vllm.cache import assigned, runs, settle, submit

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from vllm.v1.kv_cache_interface import KVCacheConfig

    from furiosa_opt_vllm.runner import Runner


@final
class Attention(AttentionBackend):
    """The attention backend of the engine's cache pages, which the engine attends to itself."""

    @staticmethod
    @override
    def get_name() -> str:
        return "FURIOSA"

    @staticmethod
    @override
    def get_impl_cls() -> NoReturn:
        raise NotImplementedError("the engine runs attention")

    @staticmethod
    @override
    def get_builder_cls() -> NoReturn:
        raise NotImplementedError("the engine runs attention")


@dataclass(frozen=True, slots=True)
class Layer:
    """A cache layer as the copy moves it: its index among the cache's layers, the shape and dtype
    of one block in the engine's order, the bytes of one block, and the host rows its copies pass
    through."""

    index: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    size: int
    rows: engine.HostBuffer

    def staged(self, blocks: int) -> torch.Tensor:
        """The first `blocks` host rows as blocks in the engine's order."""
        rows = torch.frombuffer(self.rows.slice(0, blocks * self.size), dtype=torch.uint8)
        return rows.view(self.dtype).view(blocks, *self.shape)


@final
class Layers:
    """The cache `runner` holds under vLLM's layer names of `config`, as NIXL registers and copies
    it, for requests of at most `length` tokens."""

    def __init__(self, runner: Runner, config: KVCacheConfig, length: int) -> None:
        self.runner = runner
        cache = residency.cache(runner)
        # The engine lists each layer's storage group by group.
        position = {
            layer: index
            for index, layer in enumerate(chain.from_iterable(g.layers for g in cache.spec.groups))
        }
        self.layers: dict[str, Layer] = {}
        for group in config.kv_cache_groups:
            for name in group.layer_names:
                spec = assigned(group, name)
                # The runner pages only full-history attention.
                assert isinstance(spec, FullAttentionSpec)
                # A copy moves the blocks of one request.
                rows = min(cache.blocks, -(-length // spec.block_size))
                self.layers[name] = Layer(
                    position[extract_layer_index(name)],
                    (spec.block_size, 2, spec.num_kv_heads, spec.head_size),
                    spec.dtype,
                    spec.page_size_bytes,
                    engine.HostBuffer(rows * spec.page_size_bytes),
                )
        # NIXL reads only their shape and dtype.
        self.views = {
            name: nixl_order(
                torch.empty([cache.blocks, *layer.shape], dtype=layer.dtype, device="meta")
            )
            for name, layer in self.layers.items()
        }

    def copy(
        self,
        src: Mapping[str, torch.Tensor],
        dst: Mapping[str, torch.Tensor],
        src_ids: Sequence[int],
        dst_ids: Sequence[int],
        direction: Literal["h2d", "d2h"],
    ) -> None:
        """vLLM's `CopyBlocksOp`: blocks `src_ids` of each layer `src` names to blocks `dst_ids`,
        from the device to NIXL's host buffers (`d2h`) or back (`h2d`), ended once it returns; a
        failed copy raises once every copy it submitted ended."""
        if not src_ids:
            return
        source, target = [int(i) for i in src_ids], [int(i) for i in dst_ids]
        # The device's blocks pass through host rows 0, 1, ... in turn.
        blocks = source if direction == "d2h" else target
        moved = runs(blocks, list(range(len(blocks))), 1)
        storage = residency.cache(self.runner).layers
        layers = {name: self.layers[name] for name in src}
        if any(len(blocks) * layer.size > len(layer.rows) for layer in layers.values()):
            raise ValueError(f"a copy moves the blocks of one request, not {len(blocks)}")

        def pieces(layer: Layer) -> Iterator[tuple[engine.HbmBuffer, engine.HostBuffer]]:
            for block, row, length in moved:
                yield (
                    storage[layer.index].slice(block * layer.size, length * layer.size),
                    layer.rows.slice(row * layer.size, length * layer.size),
                )

        match direction:
            case "d2h":
                copies = submit(
                    device.to_host(host)
                    for layer in layers.values()
                    for device, host in pieces(layer)
                )
                if (failure := settle(copies)) is not None:
                    raise failure
                for name, layer in layers.items():
                    dst[name][target] = nixl_order(layer.staged(len(blocks)))
            case "h2d":
                # A layer's copies start once its blocks are in its host rows, while the next
                # layer's blocks are put in theirs.
                def uploads() -> Iterator[engine.Transfer]:
                    for name, layer in layers.items():
                        _ = layer.staged(len(blocks)).copy_(engine_order(src[name][source]))
                        for device, host in pieces(layer):
                            yield host.to_hbm(device)

                if (failure := settle(submit(uploads()))) is not None:
                    raise failure
            case _:
                assert_never(direction)


def nixl_order(blocks: torch.Tensor) -> torch.Tensor:
    """A layer's `blocks` in the engine's order, `[blocks, tokens, 2, heads, width]` of keys then
    values, in NIXL's: a block holds each head's tokens, a token's key then its value,
    `[blocks, heads, tokens, 2 * width]`."""
    count, tokens, _, heads, width = blocks.shape
    return blocks.permute(0, 3, 1, 2, 4).reshape(count, heads, tokens, 2 * width)


def engine_order(blocks: torch.Tensor) -> torch.Tensor:
    """`nixl_order`'s inverse: blocks in NIXL's order, in the engine's."""
    count, heads, tokens, width = blocks.shape
    return blocks.reshape(count, heads, tokens, 2, width // 2).permute(0, 2, 3, 1, 4)
