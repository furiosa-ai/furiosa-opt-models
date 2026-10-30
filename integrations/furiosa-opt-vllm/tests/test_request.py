from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import InputBatch

from furiosa_opt_vllm.generation import draw, record, report
from furiosa_opt_vllm.request import update
from furiosa_opt_vllm.runner import Runner
from furiosa_opt_vllm.selection import Sampling


@pytest.fixture
def runner(served: Path) -> Runner:
    """A runner of a served checkpoint's configuration with the selection and batch that loading
    and cache initialization, which the tests stand in for, bind."""
    runner = Runner(
        EngineArgs(
            model=str(served), dtype="bfloat16", skip_tokenizer_init=True
        ).create_engine_config()
    )
    runner.selection = Sampling(Sampler())
    runner.input_batch = InputBatch(
        max_num_reqs=2,
        max_model_len=16,
        max_num_batched_tokens=8,
        device=torch.device("cpu"),
        vocab_size=32,
        block_sizes=[4],
        kernel_block_sizes=[4],
        max_num_blocks_per_req=[4],
    )
    return runner


def schedule(
    runner: Runner,
    new: Mapping[str, list[int]],
    cached: Mapping[str, int],
    drafts: Mapping[str, list[int]],
) -> SchedulerOutput:
    """A step that prefills each `new` prompt and decodes each `cached` request, given its output
    token count, one token and its `drafts`."""
    return SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=prompt,
                mm_features=[],
                sampling_params=SamplingParams(temperature=0),
                pooling_params=None,
                block_ids=([0],),
                num_computed_tokens=0,
                lora_request=None,
            )
            for req_id, prompt in new.items()
        ],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=list(cached),
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None for _ in cached],
            num_computed_tokens=[
                runner.requests[req_id].num_prompt_tokens + outputs - 1
                for req_id, outputs in cached.items()
            ],
            num_output_tokens=list(cached.values()),
        ),
        num_scheduled_tokens={
            **{req_id: len(prompt) for req_id, prompt in new.items()},
            **{req_id: 1 + len(drafts.get(req_id, [])) for req_id in cached},
        },
        total_num_scheduled_tokens=sum(map(len, new.values()))
        + sum(1 + len(drafts.get(req_id, [])) for req_id in cached),
        scheduled_spec_decode_tokens=dict(drafts),
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def test_refills_emptied_batch(runner: Runner) -> None:
    """Finished requests leave an empty batch ready to admit a new request."""
    update(runner, schedule(runner, {"old": [1, 2]}, {}, {}))
    update(runner, replace(schedule(runner, {}, {}, {}), finished_req_ids={"old"}))

    update(runner, schedule(runner, {"new": [1, 2]}, {}, {}))

    assert runner.input_batch.req_ids == ["new"]
    assert set(runner.requests) == {"new"}


def test_keeps_tokens(runner: Runner) -> None:
    """Known request tokens survive rejection, row movement, rollback and batch reentry."""
    update(runner, schedule(runner, {"b": [4, 5], "a": [1, 2, 3]}, {}, {}))
    runner.extend(runner.requests["b"], [7])
    runner.extend(runner.requests["a"], [6])
    # Request a's first draft is accepted, its second rejected.
    update(runner, schedule(runner, {}, {"b": 1, "a": 1}, {"a": [8, 9]}))
    runner.extend(runner.requests["b"], [11])
    runner.extend(runner.requests["a"], [8, 10])
    # Request b leaves the batch, whose next row moves into its row.
    update(runner, schedule(runner, {}, {"a": 3}, {}))
    runner.extend(runner.requests["a"], [12])

    # Request b returns, and the scheduler rolls request a back to three output tokens.
    update(runner, schedule(runner, {}, {"a": 3, "b": 2}, {}))

    batch = runner.input_batch
    rows = {
        req_id: list(map(int, batch.token_ids_cpu[index, : batch.num_tokens_no_spec[index]]))
        for req_id, index in batch.req_id_to_index.items()
    }
    assert rows == {"a": [1, 2, 3, 6, 8, 10], "b": [4, 5, 7, 11]}


def resent(
    requests: Mapping[str, tuple[list[int], list[int]]], params: SamplingParams
) -> SchedulerOutput:
    """A step in which vLLM's V2 scheduler sends each request in full: its prompt, then the tokens
    it generated before vLLM preempted it, none of them cached."""
    return SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=prompt,
                mm_features=[],
                sampling_params=params,
                pooling_params=None,
                block_ids=([1],),
                num_computed_tokens=0,
                lora_request=None,
                prefill_token_ids=prompt + generated,
            )
            for req_id, (prompt, generated) in requests.items()
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={
            req_id: len(prompt) + len(generated) for req_id, (prompt, generated) in requests.items()
        },
        total_num_scheduled_tokens=sum(
            len(prompt) + len(generated) for prompt, generated in requests.values()
        ),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


@pytest.mark.parametrize("later", [True, False], ids=["later", "next-step"])
def test_keeps_resent(runner: Runner, later: bool) -> None:
    """Resent requests retain scheduler-confirmed tokens and discard dropped outputs."""
    update(runner, schedule(runner, {"a": [1, 2]}, {}, {}))
    runner.extend(runner.requests["a"], [7, 9])
    if later:
        update(runner, replace(schedule(runner, {}, {}, {}), preempted_req_ids={"a"}))

    update(runner, resent({"a": ([1, 2], [7])}, SamplingParams(temperature=0)))

    request = runner.requests["a"]
    batch = runner.input_batch
    index = batch.req_id_to_index["a"]
    assert [request.get_token_id(position) for position in range(4)] == [1, 2, 7, -1]
    known = batch.token_ids_cpu[index, : batch.num_tokens_no_spec[index]]
    assert list(map(int, known)) == [1, 2, 7]
    assert batch.req_output_token_ids[index] == [7]


def test_continues_stream(runner: Runner) -> None:
    """A seeded request draws the same token after preemption as an equal request that ran on."""
    params = SamplingParams(temperature=1, seed=7)

    def sampled(step: SchedulerOutput) -> dict[str, int]:
        """Each scheduled request's token, drawn from even scores, so that it is the random
        stream's alone."""
        update(runner, step)
        even = {req_id: [torch.zeros(32)] for req_id in step.num_scheduled_tokens}
        drawn, _, _ = draw(runner, even, step, None)
        return {req_id: tokens[0] for req_id, tokens in drawn.items()}

    first = sampled(resent({"a": ([1, 2], []), "b": ([1, 2], [])}, params))
    for req_id, token in first.items():
        runner.extend(runner.requests[req_id], [token])
    ran_on = sampled(replace(schedule(runner, {}, {"b": 1}, {}), preempted_req_ids={"a"}))["b"]

    resumed = sampled(resent({"a": ([1, 2], [first["a"]])}, params))["a"]

    assert first["a"] != ran_on
    assert resumed == ran_on


def test_scores_prompt(runner: Runner) -> None:
    """The first output reports requested token scores from the named prompt row onward."""
    params = SamplingParams(temperature=0, prompt_logprob_token_ids=[3, 5], prompt_logprob_start=1)
    update(runner, resent({"a": ([1, 2, 3, 4], [])}, params))
    # Each prompt row but the last scores the vocabulary differently.
    rows = torch.arange(32, dtype=torch.float32) * torch.arange(1, 4, dtype=torch.float32)[:, None]
    for position, scores in enumerate(rows):
        record(runner, runner.requests["a"], position, scores[None])

    output = report(runner, {"a": [7]}, None, None)

    expected = rows[1:].log_softmax(-1)[:, [3, 5]]
    assert torch.equal(output.prompt_token_id_logprobs_dict["a"], expected)


def test_omits_cached(runner: Runner) -> None:
    """A prefix hit past the requested start omits prompt scores, as vLLM does."""
    params = SamplingParams(
        temperature=0,
        prompt_logprob_token_ids=[3, 5],
        prompt_logprob_start=1,
        skip_reading_prefix_cache=False,
    )
    step = resent({"a": ([1, 2, 3, 4], [])}, params)
    step.scheduled_new_reqs[0].num_computed_tokens = 2
    update(runner, step)

    output = report(runner, {"a": [7]}, None, None)

    assert output.prompt_token_id_logprobs_dict == {}


@pytest.mark.parametrize("resend", [True, False], ids=["new", "cached"])
def test_restores_draw(runner: Runner, resend: bool) -> None:
    """A scheduler rollback repeats the random draw at the dropped logical position."""
    params = SamplingParams(temperature=1, seed=7)
    step = resent({"a": ([1, 2], [])}, params)
    update(runner, step)
    first, _, _ = draw(runner, {"a": [torch.zeros(32)]}, step, None)
    _ = report(runner, first, None, None)
    step = resent({"a": ([1, 2], [])}, params) if resend else schedule(runner, {}, {"a": 0}, {})

    update(runner, step)
    resumed, _, _ = draw(runner, {"a": [torch.zeros(32)]}, step, None)

    assert resumed == first


def test_restores_pending(runner: Runner) -> None:
    """Rollback repeats outstanding draws; scheduler confirmation and completion release them."""
    params = SamplingParams(temperature=1, seed=7)
    step = resent({"a": ([1, 2], [])}, params)
    update(runner, step)
    expected = []
    for _ in range(3):
        drawn, _, _ = draw(runner, {"a": [torch.zeros(32)]}, step, None)
        report(runner, drawn, None, None)
        expected.extend(drawn["a"])

    update(runner, resent({"a": ([1, 2], expected[:1])}, params))
    resumed = []
    for _ in range(2):
        drawn, _, _ = draw(runner, {"a": [torch.zeros(32)]}, step, None)
        report(runner, drawn, None, None)
        resumed.extend(drawn["a"])
    assert resumed == expected[1:]
    update(runner, schedule(runner, {}, {"a": 3}, {}))
    assert runner.draws == {}
    update(runner, replace(SchedulerOutput.make_empty(), finished_req_ids={"a"}))
    assert runner.requests == {} and runner.draws == {}
