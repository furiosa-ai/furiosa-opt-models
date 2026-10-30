"""Startup refuses pipeline, context and multi-node parallelism and names the refused option, and
serves data parallelism as independent replicas on their own chips.

The engine refuses tensor degrees it has no build for when it loads the model.
"""

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from conftest import Options, Row

TASK = "text"


@pytest.mark.usefixtures("platform_env")
@pytest.mark.feature("Pipeline, expert and context parallelism")
@pytest.mark.parametrize(
    ("options", "refused"),
    [
        ({"pipeline_parallel_size": 2}, "pipeline_parallel_size"),
        ({"prefill_context_parallel_size": 2}, "prefill_context_parallel_size"),
        # vLLM itself requires several ranks per node.
        ({"tensor_parallel_size": 2, "nnodes": 2}, "nnodes"),
    ],
    ids=["pipeline", "prefill-context", "nodes"],
)
def test_refuses_parallelism_at_startup(options: dict[str, Any], refused: str) -> None:
    from vllm.engine.arg_utils import EngineArgs

    arguments = EngineArgs(model="Qwen/Qwen3-0.6B", dtype="bfloat16", **options)
    # `vllm serve` and `LLM` build this configuration, with the platform's checks, before starting.
    with pytest.raises(ValueError, match=refused):
        arguments.create_engine_config()


@pytest.mark.usefixtures("platform_env")
@pytest.mark.feature("Data parallelism")
def test_runs_each_data_parallel_replica_in_one_process() -> None:
    """A dense model's data-parallel replicas each run one engine process over its own chips."""
    from vllm.engine.arg_utils import EngineArgs

    arguments = EngineArgs(model="Qwen/Qwen3-0.6B", dtype="bfloat16", data_parallel_size=2)
    config = arguments.create_engine_config()
    assert config.parallel_config.distributed_executor_backend == "uni"


@pytest.mark.feature("Data parallelism")
def test_replicas_preserve_tokens(row: "Row", options: "Options") -> None:
    """Two data-parallel replicas answer each prompt with the greedy tokens of one replica."""
    if row["id"] != "Qwen/Qwen3-0.6B":
        pytest.skip("the inventory checkpoint for this contract runs on one chip a replica")
    import asyncio

    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    prompts = ["The capital of France is", "One plus one equals", "Water boils at", "The sky is"]
    params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    single = {"max_model_len": 1024, "max_num_seqs": 1, "max_num_batched_tokens": 32}
    llm = LLM(**options | single)
    try:
        expected = [output.outputs[0].token_ids for output in llm.generate(prompts, params)]
    finally:
        llm.llm_engine.engine_core.shutdown()

    async def generate() -> list[list[int]]:
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(**options | single, data_parallel_size=2)
        )

        async def answer(index: int, prompt: str) -> list[int]:
            tokens: list[int] = []
            async for output in engine.generate(prompt, params, request_id=str(index)):
                tokens = list(output.outputs[0].token_ids)
            return tokens

        try:
            return await asyncio.gather(*(answer(i, p) for i, p in enumerate(prompts)))
        finally:
            engine.shutdown()

    assert asyncio.run(generate()) == [list(tokens) for tokens in expected]


@pytest.mark.feature("Tensor parallelism over a compiled chip count")
def test_preserves_tokens(row: "Row", options: "Options") -> None:
    """TP2 and TP1 greedy answers are mutually top-five-close from their first differing token."""
    if row["id"] != "RedHatAI/Mistral-Nemo-Instruct-2407-FP8":
        pytest.skip("the inventory checkpoint for this contract is compiled for two chips")
    from vllm import LLM, SamplingParams

    prompt = "The capital of France is"
    outputs = []
    for tensor in (1, 2):
        llm = LLM(
            **options | {"max_model_len": 1024},
            tensor_parallel_size=tensor,
            max_num_seqs=1,
            max_num_batched_tokens=32,
        )
        try:
            outputs.append(
                llm.generate(
                    prompt, SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, logprobs=5)
                )[0].outputs[0]
            )
        finally:
            llm.llm_engine.engine_core.shutdown()
    first, second = outputs
    assert len(first.token_ids) == len(second.token_ids) == 8
    assert first.logprobs is not None and second.logprobs is not None
    differing = False
    for a, b, left, right in zip(
        first.token_ids, second.token_ids, first.logprobs, second.logprobs, strict=True
    ):
        differing |= a != b
        if differing:
            assert a in right and right[a].rank is not None and right[a].rank <= 5
            assert b in left and left[b].rank is not None and left[b].rank <= 5
