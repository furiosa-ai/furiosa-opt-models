"""SGLang's own adapter pool decides which adapter revisions its entries hold: built by its normal
constructor over storage the plugin owns, it writes a revision into an entry only on a miss,
publishes it after the write returns, and keeps pinned entries and the base model's as SGLang
counts them."""

from collections.abc import Callable, Collection
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.lora.mem_pool import EMPTY_SLOT, LoRAMemoryPool
from sglang.srt.runtime_context import get_parallel
from typing_extensions import override

if TYPE_CHECKING:
    from transformers import AutoConfig

# The model configuration SGLang's pool reads: its layer count and hidden width.
CONFIG: Final = SimpleNamespace(num_hidden_layers=2, hidden_size=16)


@dataclass(frozen=True)
class Revision:
    """An adapter revision in host memory, which nothing changes."""

    name: str


# What an entry holds after a write that failed partway.
TORN: Final = Revision("torn")
NAMES: Final = tuple(f"adapter{index}" for index in range(20))
REVISIONS: Final = {name: Revision(name) for name in NAMES}


class Model:
    """The model SGLang's pool reads: its configuration, and nothing else."""

    __slots__: tuple[str, ...] = ("config",)

    def __init__(self) -> None:
        self.config: SimpleNamespace = CONFIG


@dataclass(frozen=True)
class Write:
    """A write of `revision` into storage `entry` for `uid`, and whether SGLang already published
    `uid` as resident when it wrote."""

    entry: int
    uid: str | None
    revision: Revision | None
    published: bool


class Pool(LoRAMemoryPool):
    """SGLang's pool over storage entries that record the revision each holds; SGLang's own code
    decides every write."""

    def __init__(self, entries: int, policy: str) -> None:
        """SGLang's pool of `entries` entries under the eviction `policy`, as SGLang builds one."""
        self.entries: list[Revision | None] = []
        self.writes: list[Write] = []
        # Whether the next write fails partway.
        self.failing: bool = False
        model = cast("torch.nn.Module", cast("object", Model()))
        config = cast("AutoConfig", cast("object", CONFIG))
        super().__init__(config, entries, torch.bfloat16, 32, {"q_proj"}, model, policy, 0)

    @override
    def init_buffers(self, base_model: torch.nn.Module) -> None:
        _ = base_model
        self.entries = [None] * self.max_loras_per_batch

    @override
    def load_lora_weight_to_buffer(
        self,
        uid: str,
        buffer_id: int,
        lora_adapter: LoRAAdapter,
        lora_modules: list[dict[str, torch.nn.Module]],
        lora_embed_tokens_module: BaseLayerWithLoRA | None,
        lora_lm_head_module: BaseLayerWithLoRA | None,
    ) -> None:
        _ = lora_modules, lora_embed_tokens_module, lora_lm_head_module
        # SGLang passes the base model's `None` for both the ID and the adapter.
        revision = cast("Revision | None", cast("object", lora_adapter))
        published = uid in self.uid_to_buffer_id
        self.writes.append(Write(buffer_id, cast("str | None", uid), revision, published))
        if self.failing:
            self.entries[buffer_id] = TORN
            raise RuntimeError("the device fails")
        self.entries[buffer_id] = revision

    def prepare(self, uids: Collection[str | None], pinned: Collection[str] = ()) -> None:
        """SGLang's residency of the batch of `uids`, of which `pinned` are pinned adapters."""
        refs = {
            name: LoRARef(lora_id=name, lora_name=name, pinned=name in pinned) for name in NAMES
        }
        adapters = cast("dict[str, LoRAAdapter]", REVISIONS)
        self.prepare_lora_batch(set(uids), adapters, [], refs, None, None)

    def holds(self, uid: str | None) -> Revision | None:
        """The revision the entry SGLang published for `uid` holds."""
        return self.entries[self.uid_to_buffer_id[uid]]


@pytest.fixture
def pool() -> Callable[..., Pool]:
    """SGLang's pools of the given entries and eviction policy, built as SGLang's manager builds
    one, in a server of one tensor-parallel rank."""

    def make(entries: int = 3, policy: str = "lru") -> Pool:
        with get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1):
            return Pool(entries, policy)

    return make


a, b, c = NAMES[:3]


def test_builds_its_entries_from_the_configuration_alone(pool: Callable[..., Pool]) -> None:
    """The pool takes `--max-loras-per-batch` entries, also past the adapters one engine call
    takes, empty and unwritten, reading of the model only its configuration."""
    built = pool(entries=8)
    assert (built.max_loras_per_batch, built.entries) == (8, [None] * 8)
    assert (built.buffer_id_to_uid, built.writes) == ([EMPTY_SLOT] * 8, [])


def test_reads_the_parallel_context_of_its_server() -> None:
    """The pool reads the tensor-parallel ranks the server publishes, so it is built in one."""
    with pytest.raises(ValueError, match="not published"):
        _ = Pool(3, "lru")


def test_writes_a_revision_on_a_miss_only(pool: Callable[..., Pool]) -> None:
    """A batch writes each adapter its entries miss once, before SGLang publishes it, and the base
    model takes an entry of no revision; a batch of resident adapters writes nothing."""
    built = pool()
    built.prepare({None, a})
    assert built.writes == [Write(0, None, None, False), Write(1, a, REVISIONS[a], False)]
    built.prepare({None, a})
    built.prepare({a})
    assert len(built.writes) == 2
    assert (built.holds(None), built.holds(a)) == (None, REVISIONS[a])


def test_cycles_more_revisions_than_entries(pool: Callable[..., Pool]) -> None:
    """Twenty adapters cycle through three entries with the base model: each batch writes its one
    missing revision into a reused entry, which then holds that revision whole."""
    built = pool()
    built.prepare({None})
    for name in NAMES:
        before = len(built.writes)
        built.prepare({None, name})
        assert len(built.writes) == before + 1
        assert built.holds(name) is REVISIONS[name]
    assert len(built.entries) == 3


@pytest.mark.parametrize(("policy", "victim"), [("lru", b), ("fifo", a)])
def test_evicts_by_its_policy(pool: Callable[..., Pool], policy: str, victim: str) -> None:
    """A miss of a full pool replaces the adapter SGLang's policy picks outside the batch, never
    the base model while an adapter is a candidate."""
    built = pool(policy=policy)
    built.prepare({None, a, b})
    built.prepare({a})
    entry = built.uid_to_buffer_id[victim]
    built.prepare({None, c})
    assert victim not in built.uid_to_buffer_id
    assert (built.uid_to_buffer_id[c], built.entries[entry]) == (entry, REVISIONS[c])
    assert None in built.uid_to_buffer_id


def test_evicts_the_base_model_for_a_batch_of_adapters(pool: Callable[..., Pool]) -> None:
    """Only a batch of adapters alone replaces the base model's entry."""
    built = pool()
    built.prepare({None, a, b})
    built.prepare({a, b, c})
    assert None not in built.uid_to_buffer_id
    assert (built.uid_to_buffer_id[c], built.holds(c)) == (0, REVISIONS[c])


def test_keeps_pinned_entries(pool: Callable[..., Pool]) -> None:
    """With one entry fewer than the pool pinned, the base model and unpinned adapters share the
    last entry and the pinned ones are never written again; a batch needing two unpinned entries
    has none, which SGLang's batch bound keeps out."""
    built = pool()
    built.prepare({None, a, b}, pinned={a, b})
    pinned = {a: built.uid_to_buffer_id[a], b: built.uid_to_buffer_id[b]}
    built.prepare({c}, pinned={a, b})
    built.prepare({None}, pinned={a, b})
    built.prepare({c}, pinned={a, b})
    assert {write.entry for write in built.writes[3:]} == {0}
    assert all(built.entries[entry] is REVISIONS[name] for name, entry in pinned.items())
    with pytest.raises(ValueError, match="No available buffer slots"):
        built.prepare({None, NAMES[3]}, pinned={a, b})


def test_publishes_after_a_whole_write(pool: Callable[..., Pool]) -> None:
    """A write that fails leaves its adapter unpublished and the victim's entry empty, raising out
    of the batch; the next batch writes the revision whole into the emptied entry."""
    built = pool(entries=2)
    built.prepare({a, b})
    entry = built.uid_to_buffer_id[a]
    built.failing = True
    with pytest.raises(RuntimeError, match="the device fails"):
        built.prepare({c})
    assert (built.entries[entry], built.buffer_id_to_uid[entry]) == (TORN, EMPTY_SLOT)
    assert a not in built.uid_to_buffer_id and c not in built.uid_to_buffer_id
    built.failing = False
    built.prepare({c})
    assert (built.uid_to_buffer_id[c], built.holds(c)) == (entry, REVISIONS[c])
    assert not any(write.published for write in built.writes)


def test_removes_without_a_write(pool: Callable[..., Pool]) -> None:
    """Removing a resident adapter empties its entry without writing it; the next miss takes it."""
    built = pool()
    built.prepare({None, a})
    assert (built.remove_lora(a), built.remove_lora(b)) == (1, None)
    assert (len(built.writes), built.buffer_id_to_uid[1]) == (2, EMPTY_SLOT)
    built.prepare({b})
    assert built.uid_to_buffer_id[b] == 1
