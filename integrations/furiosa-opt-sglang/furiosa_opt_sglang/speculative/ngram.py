"""Draft each request from SGLang's n-gram corpus of the running requests' own tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING, final

import numpy as np
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus

if TYPE_CHECKING:
    from collections.abc import Set

    from furiosa_opt_sglang.platform import Ngram
    from furiosa_opt_sglang.speculative import Decoding


@final
class Corpus:
    """An n-gram corpus that drafts the first chain of its draft tree."""

    def __init__(self, options: Ngram, width: int) -> None:
        self.width = width
        self.depth = options.speculative_ngram_max_trie_depth
        self.corpus = NgramCorpus(
            min_bfs_breadth=options.speculative_ngram_min_bfs_breadth,
            max_bfs_breadth=options.speculative_ngram_max_bfs_breadth,
            match_type=options.speculative_ngram_match_type,
            capacity=options.speculative_ngram_capacity,
            max_trie_depth=self.depth,
            draft_token_num=width,
            # The platform refuses an external corpus.
            external_sam_budget=0,
            external_corpus_max_tokens=options.speculative_ngram_external_corpus_max_tokens,
        )
        # The requests whose match state the corpus keeps.
        self.matched: set[str] = set()

    def propose(self, requests: tuple[Decoding, ...]) -> tuple[tuple[int, ...], ...]:
        """Each request's drafts after its last token."""
        tails = [list(request.tokens[-self.depth :]) for request in requests]
        rids = [request.rid for request in requests]
        self.forget(self.matched - set(rids))
        self.matched.update(rids)
        self.corpus.batch_put(tails)
        self.corpus.synchronize()
        tokens, masks = self.corpus.batch_get(
            rids, tails, [len(request.tokens) for request in requests]
        )
        order = np.arange(self.width)
        # The corpus returns a draft tree; a position's parent is the last earlier one it attends.
        parents = np.where(
            masks.reshape(-1, self.width, self.width) & (order < order[:, None]), order, -1
        ).max(axis=-1)

        def drafts(row: np.ndarray, parent: np.ndarray, request: Decoding) -> tuple[int, ...]:
            # A chain runs while each position's parent is the one before it.
            end = next((i for i in range(1, self.width) if parent[i] != i - 1), self.width)
            return tuple(int(token) for token in row[1 : min(end, 1 + request.drafts)])

        return tuple(
            drafts(row, parent, request)
            for row, parent, request in zip(
                tokens.reshape(-1, self.width), parents, requests, strict=True
            )
        )

    def forget(self, rids: Set[str]) -> None:
        """Drop the match state of `rids`."""
        self.corpus.erase_match_state(list(rids))
        self.matched -= rids

    def reset(self) -> None:
        self.corpus.reset()
        self.matched.clear()
