"""Tokens that engine calls select on the device: the output of the call and the token's row."""

from __future__ import annotations

from typing import TYPE_CHECKING

from furiosa_opt_vllm.batch import Identity

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import furiosa_opt_engine as engine


def read(
    tokens: Mapping[Identity, Sequence[tuple[engine.HbmTensor, int]]],
) -> dict[Identity, list[int]]:
    """Each sequence's `tokens` on the host, in order; each call output is copied once."""
    outputs = {id(output): output for rows in tokens.values() for output, _ in rows}
    hosts = {key: output.to_host().numpy() for key, output in outputs.items()}
    return {
        sequence: [int(hosts[id(output)][row]) for output, row in rows]
        for sequence, rows in tokens.items()
    }
