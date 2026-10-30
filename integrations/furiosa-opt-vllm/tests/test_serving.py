"""vLLM's OpenAI server answers a tool call and reports each request's timings, as the engine's
generation steps feed its frontend; a test's server ends with every process it started."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import suppress
from typing import TYPE_CHECKING, Final

import pytest
from conftest import post, stop

if TYPE_CHECKING:
    from conftest import Row

TASK = "text"
TOOL: Final = {
    "type": "function",
    "function": {
        "name": "weather",
        "description": "The current weather of a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


@pytest.fixture(scope="module")
def serving(row: Row) -> list[str]:
    """Hermes tool calls, which a checkpoint's chat template writes as `<tool_call>` blocks."""
    from transformers import AutoTokenizer

    if "<tool_call>" not in (AutoTokenizer.from_pretrained(row["id"]).chat_template or ""):
        pytest.skip("the checkpoint's chat template writes no Hermes tool calls")
    return [
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
        "--enable-per-request-metrics",
        "--max-model-len",
        "1024",
    ]


@pytest.mark.parametrize(
    "choice", ["required", {"type": "function", "function": {"name": "weather"}}]
)
@pytest.mark.feature("Tool calling and reasoning outputs")
def test_calls_tool(server: str, row: Row, choice: object) -> None:
    """A required or named tool choice returns one call of the tool, with its required arguments."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
    )

    request = ChatCompletionRequest.model_validate(
        {
            "model": row["id"],
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": [TOOL],
            "tool_choice": choice,
            "temperature": 0,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    response = post(f"{server}/v1/chat/completions", request, ChatCompletionResponse)
    (call,) = response.choices[0].message.tool_calls
    assert call.function.name == "weather"
    assert "city" in json.loads(call.function.arguments)


@pytest.mark.feature("Per-request metrics")
def test_reports_metrics(server: str, row: Row) -> None:
    """Each response carries its own first-token time and decode rate."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
    )

    request = ChatCompletionRequest.model_validate(
        {
            "model": row["id"],
            "messages": [{"role": "user", "content": "Count to five."}],
            "temperature": 0,
            "max_tokens": 8,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    metrics = post(f"{server}/v1/chat/completions", request, ChatCompletionResponse).metrics
    assert metrics is not None
    assert metrics.time_to_first_token_ms is not None and metrics.time_to_first_token_ms > 0
    assert metrics.tokens_per_second is not None and metrics.tokens_per_second > 0


def test_stops_every_process_a_server_started() -> None:
    """Stopping a server ends the processes it started within the bound, a child that ignores
    SIGTERM and outlives the server included."""
    import psutil

    child = (
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print(os.getpid(), flush=True)\n"
        "time.sleep(600)\n"
    )
    starts = f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', {child!r}])\n"
    command = [sys.executable, "-c", starts + "time.sleep(600)\n"]
    with subprocess.Popen(command, stdout=subprocess.PIPE, process_group=0) as server:
        assert server.stdout is not None
        # The child ignores SIGTERM once it prints its process ID.
        survivor = psutil.Process(int(server.stdout.readline()))
        stop(server, 1)
    with suppress(psutil.NoSuchProcess):
        assert survivor.status() == psutil.STATUS_ZOMBIE
