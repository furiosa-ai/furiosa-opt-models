from array import array

import torch
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams

from furiosa_opt_sglang.generation import Schedule


def test_copies_scheduled_generation() -> None:
    """Scheduler mutation cannot change a parsed generation step's inputs or score targets."""
    request = Req("request", "", array("q", [3, 4, 5, 6]), SamplingParams())
    request.full_untruncated_fill_ids = array("q", [3, 4, 5, 6])
    batch = ScheduleBatch(
        reqs=[request],
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.tensor([4, 5]),
        seq_lens=torch.tensor([3]),
        extend_lens=[2],
        extend_logprob_start_lens=[1],
        top_logprobs_nums=[2],
        token_ids_logprobs=[[6]],
        return_logprob=True,
    )

    schedule = Schedule.parse(batch)
    _ = batch.input_ids.fill_(0)
    _ = batch.seq_lens.fill_(0)
    batch.extend_lens[0] = 1
    batch.extend_logprob_start_lens[0] = 0
    assert batch.top_logprobs_nums is not None
    batch.top_logprobs_nums[0] = 0
    assert batch.token_ids_logprobs is not None
    batch.token_ids_logprobs[0][0] = 0
    request.full_untruncated_fill_ids[2] = 0

    assert schedule.inputs.tokens == ((4, 5),)
    assert schedule.inputs.cached == (1,)
    assert schedule.lengths == (2,)
    assert schedule.starts == (1,)
    assert schedule.top == (2,)
    assert schedule.ids == ((6,),)
    assert schedule.targets == ((5, 6),)


def test_sampling_state_is_classified_for_device_selection() -> None:
    """Every field of SGLang's sampling state either leaves a greedy step's argmax unchanged or is
    one the device selection checks, so a new field forces a decision."""
    from dataclasses import fields

    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    checked = {
        "is_all_greedy",
        "grammars",
        "logit_bias",
        "acc_additive_penalties",
        "acc_scaling_penalties",
        "sampling_mask_batch_indices",
        "has_custom_logit_processor",
        "has_watermark_candidates",
    }
    neutral = {
        # Sampling parameters that a greedy step does not read.
        "temperatures",
        "top_ps",
        "top_ks",
        "min_ps",
        "is_any_greedy",
        "need_top_p_sampling",
        "need_top_k_sampling",
        "need_min_p_sampling",
        "sampling_seed",
        "rids_int",
        "bootstrap_room_ids_int",
        "npu_top_k_top_p_eligible",
        "max_top_k",
        # State derived from a checked field.
        "grammar_mask",
        "penalizer_orchestrator",
        "custom_params",
        "custom_logit_processor",
        "return_sampling_masks",
        "return_sampling_support_logprobs",
        "sampling_support_logprobs_capture_indices",
        "sampling_mask_top_ks",
        "watermark_keys",
        "watermark_context_windows",
        "watermark_enabled",
        "watermark_candidates_host",
        # Shape and placement.
        "vocab_size",
        "device",
    }
    assert {field.name for field in fields(SamplingBatchInfo)} == checked | neutral
