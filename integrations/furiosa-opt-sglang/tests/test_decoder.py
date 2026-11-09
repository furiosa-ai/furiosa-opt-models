from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

import furiosa_opt_engine as engine
from furiosa_opt_sglang.decoder import Decoder

if TYPE_CHECKING:
    from furiosa_opt_sglang.decoder import Location

# One group of 40 layers, each page 16 tokens of 4 E4M3 KV heads of 128 channels.
LAYERS, PAGE, SCRATCH = 40, 16384, 256
BLOCK = LAYERS * PAGE


@pytest.mark.parametrize(
    ("memory", "blocks"),
    [
        # One byte short of an eleventh block beside the scratch pages.
        (LAYERS * SCRATCH * PAGE + 11 * BLOCK - 1, 10),
        # Past what a layer's I32 byte offsets address, its scratch pages included.
        (2**40, 2**31 // PAGE - SCRATCH),
    ],
)
def test_reserves_scratch_pages(memory: int, blocks: int) -> None:
    """The token pool's blocks fit the memory and byte offsets the decoder's scratch pages leave."""
    unread = object()
    pool = cast("object", SimpleNamespace(block=16, bytes=PAGE))
    decoder = Decoder(
        None,
        None,
        cast("Location", unread),
        cast("engine.LoadOptions", unread),
        cast("engine.Device", unread),
        0,
        cast("engine.decoder.Pool", pool),
        range(LAYERS),
        SCRATCH,
    )

    assert decoder.blocks(memory) == blocks
