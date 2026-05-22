import logging
from dataclasses import dataclass
from typing import Optional

from .chunker import Document
from .vector_store import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    id: str
    content: str
    score: float
    metadata: dict


def retrieve_and_rank(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    store = get_vector_store()

    if store.index is None or store.index.ntotal == 0:
        logger.warning("Vector store is empty")
        return []

    results = store.search(query, k=top_k * 2)

    ranked_results = []
    for doc, score in results:
        if score >= min_score:
            ranked_results.append(RetrievedChunk(
                id=doc.id,
                content=doc.content,
                score=score,
                metadata=doc.metadata,
            ))

    ranked_results.sort(key=lambda x: x.score, reverse=True)

    return ranked_results[:top_k]


def rerank_results(
    query: str,
    results: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    return sorted(results, key=lambda x: x.score, reverse=True)