"""Pool hidden rows while preserving SGLang's tensor-valued transport."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, pool_hidden_states

from furiosa_opt_sglang.batch import Batch
from furiosa_opt_sglang.generation import integers, sequence

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    from furiosa_opt_sglang.generation.model import Model
    from furiosa_opt_sglang.runner import Runner


def pool(
    runner: Runner, model: Model, batch: ScheduleBatch, forward_batch: ForwardBatch
) -> EmbeddingPoolerOutput:
    """Every prompt row's hidden state, pooled."""
    lengths = batch.extend_lens
    snapshot = Batch.parse(integers(batch.input_ids), lengths, integers(batch.seq_lens))
    hidden = sequence.forward(runner, batch, snapshot, every=True, feed={})
    rows = torch.cat([torch.as_tensor(block.to_host().numpy()) for _, block in hidden])
    output = model.pooler.forward(rows, forward_batch)
    if forward_batch.return_pooled_hidden_states:
        output.pooled_hidden_states = pool_hidden_states(
            model.pooler.pooling_type, rows, forward_batch
        )
    return output
