"""vLLM's sleep endpoints put a server's engine to sleep and wake it to the same completions."""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING, Final

import pytest
from conftest import post, serve
from pydantic import TypeAdapter
from test_sleep import CHECKPOINT, PROMPTS

if TYPE_CHECKING:
    from collections.abc import Generator
    from http.client import HTTPResponse

SLEEPING: Final = TypeAdapter(dict[str, bool])


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Generator[str]:
    """`vllm serve` of the checkpoint with sleep mode and vLLM's development endpoints, which serve
    `/sleep`."""
    arguments = [CHECKPOINT, "--dtype", "bfloat16", "--enable-sleep-mode", "--max-model-len", "512"]
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("VLLM_SERVER_DEV_MODE", "1")
        with serve(arguments, tmp_path_factory) as url:
            yield url


@pytest.mark.feature("Sleep mode")
def test_serves_the_sleep_endpoints(server: str) -> None:
    """`/sleep` at each level and `/wake_up` return the completion of before, and `/is_sleeping`
    reports the state between them."""
    from vllm.entrypoints.openai.completion.protocol import CompletionRequest, CompletionResponse

    request = CompletionRequest.model_validate(
        {"model": CHECKPOINT, "prompt": PROMPTS[0], "temperature": 0, "max_tokens": 16}
    )

    def complete() -> str:
        return post(f"{server}/v1/completions", request, CompletionResponse).choices[0].text

    def sleeping() -> bool:
        response: HTTPResponse = urllib.request.urlopen(f"{server}/is_sleeping", timeout=60)
        with response:
            return SLEEPING.validate_json(response.read())["is_sleeping"]

    def send(path: str) -> None:
        response: HTTPResponse = urllib.request.urlopen(
            urllib.request.Request(f"{server}{path}", method="POST"), timeout=300
        )
        with response:
            assert response.status == 200

    expected = complete()
    for level in (1, 2):
        send(f"/sleep?level={level}")
        assert sleeping()
        send("/wake_up")
        assert not sleeping()
        assert complete() == expected
