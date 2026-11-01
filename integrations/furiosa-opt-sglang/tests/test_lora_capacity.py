"""The device holds adapters in the entries of SGLang's own pool, taken before the cache sizes:
adapters load after startup under default cache sizing, more of them than the pool has entries,
and each request runs the adapter it names."""

from pathlib import Path

import pytest
from test_lora_lifecycle import LORA, MODEL, assert_peft_greedy, tokens, update
from test_lora_lifecycle import adapters as adapters
from test_server import Chips
from test_server import chips as chips
from test_server import plugin as plugin

pytestmark = pytest.mark.feature("LoRA")


def test_cycles_more_adapters_than_entries(
    tmp_path_factory: pytest.TempPathFactory, chips: Chips, adapters: dict[str, Path]
) -> None:
    """With no adapter at startup, the cache sized by default and two pool entries, every adapter
    loads after startup, and requests that name them in turn, each replacing an entry's adapter,
    generate the greedy tokens of Transformers with that adapter merged by PEFT."""
    flags = [*LORA, "--max-lora-rank", "32", "--lora-target-modules", "all"]
    row = {"id": MODEL, "task": "text", "flags": [*flags, "--max-loras-per-batch", "2"]}
    with chips.serve(tmp_path_factory, row) as send:
        for name, path in adapters.items():
            loaded = update(send, "/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
            assert name in loaded["loaded_adapters"], loaded
        # Each request misses: the base model keeps one entry and the last adapter the other.
        first, again = ([tokens(send, name) for name in adapters] for _ in range(2))
        assert first == again
        for name, generated in zip(adapters, first, strict=True):
            assert_peft_greedy(generated, adapters[name])
