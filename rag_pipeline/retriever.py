import logging
from dataclasses import dataclass
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

from .chunker import Document
from .vector_store import get_vector_store

logger = logging.getLogger(__name__)

_reranker: Optional[CrossEncoder] = None


@dataclass
class RetrievedChunk:
    id: str
    content: str
    score: float
    metadata: dict


def _load_reranker(model_name: str) -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _reranker = CrossEncoder(model_name, device=device)
        logger.info(f"Reranker loaded on device: {device}")
    return _reranker


def rerank_results(
    query: str,
    results: list[RetrievedChunk],
    model_name: str,
) -> list[RetrievedChunk]:
    if not results:
        return results
    reranker = _load_reranker(model_name)
    pairs = [(query, r.content) for r in results]
    scores = reranker.predict(pairs)
    for chunk, score in zip(results, scores):
        chunk.score = float(score)
    return sorted(results, key=lambda x: x.score, reverse=True)


def retrieve_and_rank(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    reranker_model: Optional[str] = None,
    reranker_enabled: bool = True,
) -> list[RetrievedChunk]:
    store = get_vector_store()

    if store.index is None or store.index.ntotal == 0:
        logger.warning("Vector store is empty")
        return []

    # Fetch extra candidates so the reranker has room to work
    candidates_raw = store.search(query, k=top_k * 2)

    candidates = [
        RetrievedChunk(
            id=doc.id,
            content=doc.content,
            score=score,
            metadata=doc.metadata,
        )
        for doc, score in candidates_raw
        if score >= min_score
    ]

    if reranker_enabled and reranker_model and candidates:
        candidates = rerank_results(query, candidates, reranker_model)
    else:
        candidates.sort(key=lambda x: x.score, reverse=True)

    return candidates[:top_k]
