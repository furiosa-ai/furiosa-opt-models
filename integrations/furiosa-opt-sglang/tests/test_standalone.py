"""Draft checkpoint fields used to select sequential or parallel standalone decoding."""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

# Plugin entry points load after SGLang startup; mirror that order for standalone test runs.
import sglang  # noqa: F401

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

from furiosa_opt_engine.speculative import DraftRequest, DraftSession
from furiosa_opt_sglang.platform import Refused
from furiosa_opt_sglang.speculative import Decoding
from furiosa_opt_sglang.speculative.standalone import Config, Draft, mask_token


@pytest.mark.parametrize("room", [0, -1])
def test_decoding_requires_room_for_the_last_token(room: int) -> None:
    """A decode request must have room for its last known token before draft positions."""
    request = cast(
        "Req",
        cast(object, SimpleNamespace(rid="request", origin_input_ids=[1], output_ids=[])),
    )

    with pytest.raises(ValueError, match="decode room must include"):
        _ = Decoding.parse(request, cached=0, room=room, blocks=(0,))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dflash_config", {"mask_token_id": 7}),
        ("mask_token_id", 7),
        ("dspark_noise_token_id", 7),
        ("pard_token", 7),
        ("ptd_token_id", 7),
    ],
)
def test_reads_parallel_mask_token(field: str, value: object) -> None:
    """A recognized checkpoint mask selects parallel drafting."""
    config = Config.model_validate({"vocab_size": 10, field: value})

    assert mask_token(config) == 7


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (
            {
                "dflash_config": {"mask_token_id": 4},
                "mask_token_id": 5,
                "pard_token": 6,
            },
            4,
        ),
        ({"mask_token_id": 5, "dspark_noise_token_id": 6}, 5),
        ({"dflash_config": {"mask_token_id": None}, "mask_token_id": 5}, 5),
    ],
)
def test_uses_first_present_mask_field(fields: dict[str, object], expected: int) -> None:
    """Mask aliases keep the same priority as vLLM's parallel-draft admission."""
    config = Config.model_validate({"vocab_size": 10, **fields})

    assert mask_token(config) == expected


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (
            {
                "dspark_noise_token_id": 6,
                "pard_token": 7,
                "ptd_token_id": 8,
            },
            6,
        ),
        ({"pard_token": 7, "ptd_token_id": 8}, 7),
    ],
)
def test_keeps_priority_for_legacy_mask_aliases(fields: dict[str, object], expected: int) -> None:
    """Legacy mask aliases keep the vLLM priority when more than one is named."""
    config = Config.model_validate({"vocab_size": 10, **fields})

    assert mask_token(config) == expected


@pytest.mark.parametrize("value", [True, -1, 10, "7"])
def test_refuses_invalid_parallel_mask_token(value: object) -> None:
    """A named mask must be an integer token inside the draft vocabulary."""
    config = Config.model_validate({"vocab_size": 10, "pard_token": value})

    with pytest.raises(Refused, match="parallel drafting requires a mask token ID") as error:
        _ = mask_token(config)
    assert error.value.option == "speculative_draft_model_path"


def test_refuses_invalid_higher_priority_mask_field() -> None:
    """An invalid earlier alias must not fall through to a valid lower-priority alias."""
    config = Config.model_validate({"vocab_size": 10, "mask_token_id": 10, "pard_token": 7})

    with pytest.raises(Refused, match="parallel drafting requires a mask token ID"):
        _ = mask_token(config)


def test_keeps_sequential_drafting_without_a_mask() -> None:
    """A checkpoint without a mask field keeps standalone sequential drafting."""
    config = Config.model_validate({"vocab_size": 10})

    assert mask_token(config) is None


def test_translates_decodings_to_shared_draft_requests() -> None:
    """Adapt request IDs, token prefixes, block tables and lifecycle into the engine session."""

    class Session:
        retained: tuple[str, ...] = ()
        requests: tuple[DraftRequest[str], ...] = ()
        retired: set[str]
        resets: int

        def __init__(self) -> None:
            self.retired = set()
            self.resets = 0

        def retain(self, sequences: tuple[str, ...]) -> None:
            self.retained = sequences

        def propose(self, requests: tuple[DraftRequest[str], ...]) -> tuple[tuple[int, ...], ...]:
            self.requests = requests
            return ((31, 32), (41,))

        def retire(self, sequence: str) -> None:
            self.retired.add(sequence)

        def reset(self) -> None:
            self.resets += 1

    session = Session()
    draft = Draft.__new__(Draft)
    draft._session = cast(DraftSession[str], session)
    requests = (
        Decoding("first", (10, 11, 12), 2, (3, 4)),
        Decoding("second", (20,), 1, (7,)),
    )

    assert draft.propose(requests) == ((31, 32), (41,))
    assert session.retained == ("first", "second")
    assert session.requests == (
        DraftRequest("first", (10, 11, 12), 2, ((3, 4),), 2),
        DraftRequest("second", (20,), 0, ((7,),), 1),
    )

    draft.forget({"first"})
    draft.reset()
    assert session.retired == {"first"}
    assert session.resets == 1
