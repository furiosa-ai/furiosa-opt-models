"""Draft each request on a draft model, whose cache follows the target's block tables.

A draft config with a mask token drafts the later positions in parallel; other checkpoints draft
each position from the previous draft.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast, final

from pydantic import BaseModel
from sglang.srt.runtime_context import get_model, get_serving
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

import furiosa_opt_engine as engine
from furiosa_opt_engine.placement import draft_device
from furiosa_opt_engine.speculative import (
    DraftRequest,
    DraftSession,
    cache_memory_budget,
    same_tokenizer_vocabulary,
    validate_draft_decoder,
    validate_draft_vocabulary,
    validate_mask_token,
)
from furiosa_opt_sglang.checkpoint import checkout
from furiosa_opt_sglang.decoder import RELEASED, Decoder, Location
from furiosa_opt_sglang.platform import (
    CACHE_DTYPES,
    DRAFT_PLACEMENT_ENV,
    CacheDtype,
    Refused,
)

if TYPE_CHECKING:
    from collections.abc import Set

    from furiosa_opt_sglang.platform import Standalone
    from furiosa_opt_sglang.speculative import Decoding


class Config(BaseModel):
    """The config fields read from a draft checkpoint."""

    vocab_size: int
    dflash_config: dict[str, object] | None = None
    mask_token_id: object = None
    dspark_noise_token_id: object = None
    pard_token: object = None
    ptd_token_id: object = None


def mask_token(config: Config) -> int | None:
    """The valid parallel-drafting mask token in the draft config, if one is named."""
    dflash = config.dflash_config
    candidates = (
        dflash.get("mask_token_id") if dflash is not None else None,
        config.mask_token_id,
        config.dspark_noise_token_id,
        config.pard_token,
        config.ptd_token_id,
    )
    for token in candidates:
        if token is None:
            continue
        try:
            validated = validate_mask_token(token, config.vocab_size)
        except ValueError:
            raise Refused(
                "speculative_draft_model_path",
                "parallel drafting requires a mask token ID in the draft vocabulary, not "
                + f"{token!r} for {config.vocab_size} tokens",
            ) from None
        return validated
    return None


@final
class Draft:
    """A draft model on the selected device, cached beside the target's token pool."""

    def __init__(self, options: Standalone, target: Decoder, context: int) -> None:
        serving, model = get_serving(), get_model()
        path, revision = (
            options.speculative_draft_model_path,
            options.speculative_draft_model_revision,
        )
        draft_vocabulary = get_tokenizer(
            path,
            tokenizer_mode=serving.tokenizer_mode,
            trust_remote_code=model.trust_remote_code,
            tokenizer_revision=revision,
            tokenizer_backend=serving.tokenizer_backend,
        ).get_vocab()
        target_vocabulary = get_tokenizer(
            serving.tokenizer_path,
            tokenizer_mode=serving.tokenizer_mode,
            trust_remote_code=model.trust_remote_code,
            tokenizer_revision=model.revision,
            tokenizer_backend=serving.tokenizer_backend,
        ).get_vocab()
        if not same_tokenizer_vocabulary(target_vocabulary, draft_vocabulary):
            raise Refused(
                "speculative_draft_model_path",
                "speculative_draft_model_path requires the target's token-to-id mapping",
            )
        directory = checkout(path, revision)
        config = Config.model_validate_json((Path(directory) / "config.json").read_bytes())
        try:
            validate_draft_vocabulary(target.vocabulary, config.vocab_size)
        except ValueError as error:
            raise Refused("speculative_draft_model_path", str(error)) from None
        mask = mask_token(config)
        checkpoint = Location(directory, None)
        if target.model is None:
            raise RuntimeError(RELEASED)
        requested = options.draft_placement
        placement = (
            None
            if requested is None
            else engine.Placement(list(requested.chips), range(*requested.pes))
        )
        try:
            device = draft_device(
                target.device,
                placement,
                target.model.parallel,
            )
        except ValueError as error:
            raise Refused(DRAFT_PLACEMENT_ENV, str(error)) from None
        try:
            mapped = checkpoint.mapped()
        except (NotImplementedError, ValueError) as error:
            raise Refused(
                "speculative_draft_model_path", f"could not map the draft checkpoint: {error}"
            ) from error
        cache_dtype: CacheDtype = (
            options.speculative_draft_kv_cache_dtype
            if options.speculative_draft_kv_cache_dtype is not None
            else cast(CacheDtype, model.kv_cache_dtype)
        )
        loads = engine.LoadOptions().chips(target.model.parallel)
        if (engine_cache_dtype := CACHE_DTYPES[cache_dtype]) is not None:
            loads = loads.kv_cache(engine_cache_dtype)
        try:
            loads.parse(mapped)
        except (NotImplementedError, ValueError) as error:
            raise Refused("speculative_draft_model_path", str(error)) from error
        try:
            draft = Decoder.load(
                checkpoint,
                loads,
                device,
                config.vocab_size,
            )
        except NotImplementedError as error:
            raise Refused("speculative_draft_model_path", str(error)) from None
        draft_model = draft.model
        if draft_model is None:
            raise Refused(
                "speculative_draft_model_path",
                "speculative_draft_model_path requires a causal draft model",
            )
        try:
            validate_draft_vocabulary(target.vocabulary, draft.vocabulary)
            validate_draft_decoder(target.model, draft_model, context)
        except ValueError as error:
            raise Refused("speculative_draft_model_path", str(error)) from None
        self._decoder: Decoder = draft
        self._mask: int | None = mask
        self._session: DraftSession[str] | None = None

    def capacity(self, target: Decoder, target_memory: int, utilization: float) -> int:
        """Target blocks that fit this draft cache under colocated or disjoint placement."""
        draft = self._decoder
        if draft.device.placement == target.device.placement:
            return target.blocks(target_memory, draft)
        device = draft.device
        draft_memory = cache_memory_budget(device.available, device.capacity, utilization)
        return min(target.blocks(target_memory), draft.blocks(draft_memory))

    def allocate(self, blocks: int) -> None:
        """Allocate a shared draft session with the target token pool's block capacity."""
        model = self._decoder.model
        if model is None:
            raise RuntimeError(RELEASED)
        self._session = DraftSession(
            model,
            self._decoder.device,
            blocks,
            self._decoder.vocabulary,
            self._mask,
        )

    def propose(self, requests: tuple[Decoding, ...]) -> tuple[tuple[int, ...], ...]:
        """Draft each request from its known tokens, reusing the common draft-cache prefix."""
        session = self._session
        if session is None:
            raise RuntimeError("the target's token pool must be sized before drafting")
        session.retain(tuple(request.rid for request in requests))
        return session.propose(
            tuple(
                DraftRequest(
                    request.rid,
                    request.tokens,
                    request.cached,
                    (request.blocks,),
                    request.drafts,
                )
                for request in requests
            )
        )

    def forget(self, rids: Set[str]) -> None:
        """Drop the cached tokens of `rids`."""
        session = self._session
        if session is not None:
            for rid in rids:
                session.retire(rid)

    def reset(self) -> None:
        """Forget every cached request prefix."""
        session = self._session
        if session is not None:
            session.reset()
