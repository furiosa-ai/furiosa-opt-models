"""Force trace tokens and report sampling support from the host sampler's distribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.v1.outputs import SamplingMaskLists

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence, Set

    from vllm.v1.worker.gpu_input_batch import CachedRequestState


@dataclass(frozen=True, slots=True)
class Replay:
    """What a step replays: the token each `forced` batch row takes, and the rows whose supports
    it reports, `None` when vLLM asks for no sampling masks."""

    forced: Mapping[int, int]
    supported: Set[int] | None

    @classmethod
    def of(cls, requests: Sequence[CachedRequestState], sampled: Set[int], masks: bool) -> Replay:
        """What a step replays whose batch holds `requests`, of which the `sampled` rows draw a
        token; `masks` when vLLM asks for sampling masks."""
        forced = {
            row: trace[len(request.output_token_ids)]
            for row, request in enumerate(requests)
            if row in sampled
            and request.sampling_params is not None
            and (trace := request.sampling_params.trace_decode_token_ids) is not None
            and len(request.output_token_ids) < len(trace)
        }
        return cls(forced, sampled if masks else None)

    def supports(self, distribution: torch.Tensor) -> SamplingMaskLists | None:
        """In vLLM's layout over every row, the tokens a supported row could have drawn, where its
        `distribution` is finite; other rows have none."""
        if self.supported is None:
            return None
        rows = [
            torch.isfinite(scores).nonzero().flatten().numpy().astype(np.int32)
            if row in self.supported
            else np.empty(0, dtype=np.int32)
            for row, scores in enumerate(distribution)
        ]
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        _ = np.cumsum([len(tokens) for tokens in rows], out=offsets[1:])
        return SamplingMaskLists(np.concatenate(rows), offsets)
