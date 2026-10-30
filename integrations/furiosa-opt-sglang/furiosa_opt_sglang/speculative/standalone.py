"""Draft each request on a draft model, whose cache follows the target's block tables."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, final

from pydantic import BaseModel
from sglang.srt.runtime_context import get_model, get_serving
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

import furiosa_opt_engine as engine
from furiosa_opt_sglang.batch import operand
from furiosa_opt_sglang.checkpoint import checkout
from furiosa_opt_sglang.decoder import RELEASED, Decoder, Location
from furiosa_opt_sglang.platform import Refused

if TYPE_CHECKING:
    from collections.abc import Set

    from furiosa_opt_sglang.platform import Standalone
    from furiosa_opt_sglang.speculative import Decoding


class Config(BaseModel):
    """The config field of a draft checkpoint the draft check reads: its token embedding's rows."""

    vocab_size: int


@final
class Draft:
    """A draft model on the target's device, with a cache of as many blocks as the target's, which
    `allocate` makes once the target's token pool is sized."""

    def __init__(self, options: Standalone, target: Decoder, context: int) -> None:
        serving, model = get_serving(), get_model()
        path, revision = (
            options.speculative_draft_model_path,
            options.speculative_draft_model_revision,
        )
        vocabularies = [
            get_tokenizer(
                name,
                tokenizer_mode=serving.tokenizer_mode,
                trust_remote_code=model.trust_remote_code,
                tokenizer_revision=version,
                tokenizer_backend=serving.tokenizer_backend,
            ).get_vocab()
            for name, version in ((path, revision), (serving.tokenizer_path, model.revision))
        ]
        if vocabularies[0] != vocabularies[1]:
            raise Refused(
                "speculative_draft_model_path",
                "speculative_draft_model_path requires the target's token-to-id mapping",
            )
        directory = checkout(path, revision)
        config = Config.model_validate_json((Path(directory) / "config.json").read_bytes())
        checkpoint = Location(directory, None)
        if target.model is None:
            raise RuntimeError(RELEASED)
        try:
            loads = engine.LoadOptions().chips(target.model.parallel)
            loads.parse(checkpoint.mapped())
        except (NotImplementedError, ValueError) as error:
            raise Refused("speculative_draft_model_path", str(error)) from None
        # The draft shares the target's chips, whose memory the target's token pool then sizes.
        draft = Decoder.load(checkpoint, loads, target.device, config.vocab_size)
        if not (
            draft.model is not None
            and draft.model.output == engine.decoder.Output.TOKENS
            and draft.vocabulary == target.vocabulary
            and draft.pool.block == target.pool.block
            and draft.model.window >= context
        ):
            raise Refused(
                "speculative_draft_model_path",
                "speculative_draft_model_path requires a causal model of the target vocabulary"
                + " and blocks, whose window holds the context",
            )
        self.decoder: Decoder = draft
        self.model: engine.Decoder = draft.model
        self.cache: engine.decoder.Cache | None = None
        # How many leading tokens of each request the draft cache holds.
        self.held: dict[str, int] = {}

    def allocate(self, blocks: int) -> None:
        """The draft cache, of the target cache's `blocks` blocks."""
        self.cache = engine.decoder.Cache(self.model.cache, blocks)

    def propose(self, requests: tuple[Decoding, ...]) -> tuple[tuple[int, ...], ...]:
        """Each request's greedy drafts after its last token, caching the tokens the draft model has
        not seen first."""
        decoder, model, cache = self.decoder, self.model, self.cache
        if cache is None:
            raise RuntimeError("the target's token pool must be sized before drafting")
        self.held = {request.rid: self.held.get(request.rid, 0) for request in requests}

        def drafts(request: Decoding) -> tuple[int, ...]:
            if request.drafts == 0:
                return ()
            blocks = [list(request.blocks)]

            def chunk(start: int, keep: engine.decoder.Keep) -> list[engine.decoder.Request]:
                ids = operand(request.tokens[start : start + decoder.rows], decoder.vocabulary)
                tokens = engine.HostTensor.from_dlpack(ids).to_hbm(decoder.device)
                return [engine.decoder.Request(blocks, start, tokens, keep)]

            # Rows past the target's cache hold rejected drafts; the draft model reads them again.
            starts = range(
                min(self.held[request.rid], request.cached), len(request.tokens), decoder.rows
            )
            for start in starts[:-1]:
                _ = model.forward(chunk(start, engine.decoder.Keep.NONE), cache)
            output = model.forward(chunk(starts[-1], engine.decoder.Keep.LAST), cache)
            tokens = [int(output.to_host().numpy()[0])]
            for position in range(request.cached + 1, request.cached + request.drafts):
                output = model.forward(
                    [engine.decoder.Request(blocks, position, output[:1])], cache
                )
                tokens.append(int(output.to_host().numpy()[0]))
            self.held[request.rid] = request.cached + request.drafts
            return tuple(tokens)

        return tuple(drafts(request) for request in requests)

    def forget(self, rids: Set[str]) -> None:
        """Drop the cached tokens of `rids`."""
        self.held = {rid: count for rid, count in self.held.items() if rid not in rids}

    def reset(self) -> None:
        self.held.clear()
