"""BM25 lexical retrieval over existing chunk metadata."""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Tokenize while preserving useful automotive technical tokens."""
    return re.findall(
        r"[a-z0-9]+(?:[-./][a-z0-9]+)*",
        text.lower(),
    )


class BM25Store:
    """In-memory BM25 index built from existing chunk metadata."""

    def __init__(self, metadata: list[dict[str, Any]]) -> None:
        self.metadata = metadata

        corpus = [
            tokenize(
                f"{item.get('section', '')} {item.get('text', '')}"
            )
            for item in metadata
        ]

        self.index = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        query_tokens = tokenize(query)
        scores = self.index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: scores[idx],
            reverse=True,
        )[:top_k]

        results: list[dict[str, Any]] = []

        for rank, idx in enumerate(ranked_indices, start=1):
            item = dict(self.metadata[idx])

            item["bm25_score"] = float(scores[idx])
            item["bm25_rank"] = rank

            results.append(item)

        return results