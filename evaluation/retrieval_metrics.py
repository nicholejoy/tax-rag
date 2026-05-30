"""
Custom retrieval metrics: Recall@k and Hit Rate@k.

Matching strategy
-----------------
Chunk IDs from the evaluation dataset may not align with the IDs in the live
FAISS index (different chunking runs produce different IDs and chunk sizes).
Matching on ID alone is therefore unreliable.

This module supports two matching strategies, selectable at call time:

  "content"  (default)
      A hit is recorded when the ground-truth answer text appears — or
      substantially overlaps — in at least one of the top-k retrieved chunks.
      Specifically, a trigram token-overlap ratio >= `content_match_threshold`
      (default 0.5) is used so that minor wording differences don't cause
      false misses.

  "chunk_id"
      Legacy exact-match on chunk_id strings.  Only reliable when the live
      index was built from the same chunking run that produced the dataset.
      Use with caution.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

MatchStrategy = Literal["content", "chunk_id"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lower-case word tokens from a string."""
    return set(re.findall(r"\b\w+\b", text.lower()))


def _trigrams(tokens: list[str]) -> set[tuple]:
    """Character trigrams of a whitespace-joined token string."""
    s = " ".join(tokens)
    return {s[i:i+3] for i in range(len(s) - 2)}


def _content_hit(
    ground_truth: str,
    retrieved_contexts: list[str],
    threshold: float,
) -> bool:
    """
    Return True if any retrieved context overlaps with ground_truth above
    `threshold` (Jaccard overlap on word tokens).
    """
    gt_tokens = _tokenize(ground_truth)
    if not gt_tokens:
        return False

    for ctx in retrieved_contexts:
        ctx_tokens = _tokenize(ctx)
        if not ctx_tokens:
            continue
        intersection = gt_tokens & ctx_tokens
        # Use GT recall (fraction of GT tokens found in context) rather than
        # Jaccard union — ground truth answers are short; retrieved chunks are
        # long, so union-based Jaccard is always near zero even on a true hit.
        overlap = len(intersection) / len(gt_tokens)
        if overlap >= threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PerSampleRetrievalScore:
    """Retrieval metric scores for a single evaluation sample."""

    question: str
    expected_chunk_id: str          # from dataset (may not match live index IDs)
    retrieved_chunk_ids: list[str]  # top-k IDs returned by the live API
    k: int
    hit: bool
    recall_at_k: float              # 1.0 if hit, else 0.0 (binary; one ground truth)
    match_strategy: MatchStrategy


@dataclass
class AggregatedRetrievalMetrics:
    """Aggregated retrieval metrics across the full evaluation set."""

    k: int
    num_samples: int
    recall_at_k: float
    hit_rate_at_k: float
    match_strategy: MatchStrategy
    per_sample: list[PerSampleRetrievalScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "num_samples": self.num_samples,
            "match_strategy": self.match_strategy,
            f"recall_at_{self.k}": self.recall_at_k,
            f"hit_rate_at_{self.k}": self.hit_rate_at_k,
        }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_retrieval_metrics(
    questions: list[str],
    expected_chunk_ids: list[str],
    retrieved_chunk_ids_per_sample: list[list[str]],
    retrieved_contexts_per_sample: list[list[str]],
    ground_truths: list[str],
    k: int,
    match_strategy: MatchStrategy = "content",
    content_match_threshold: float = 0.5,
) -> AggregatedRetrievalMetrics:
    """
    Compute Recall@k and Hit Rate@k across a set of samples.

    Parameters
    ----------
    questions:
        The question string for each sample (for logging/output only).
    expected_chunk_ids:
        The ground-truth chunk_id for each sample as stored in the dataset.
        Used only when match_strategy="chunk_id".
    retrieved_chunk_ids_per_sample:
        The list of retrieved chunk IDs for each sample, in ranked order.
        Used only when match_strategy="chunk_id".
    retrieved_contexts_per_sample:
        The plain-text content of each retrieved chunk, in ranked order.
        Used when match_strategy="content".
    ground_truths:
        The reference answer for each sample.
        Used when match_strategy="content" to check for overlap.
    k:
        The rank cutoff — only the top-k retrieved items are considered.
    match_strategy:
        "content"  — token-overlap between ground_truth and retrieved contexts.
        "chunk_id" — exact string match on chunk IDs (fragile across re-chunking).
    content_match_threshold:
        Jaccard token-overlap ratio required for a content hit (default 0.5).

    Returns
    -------
    AggregatedRetrievalMetrics with per-sample breakdown and aggregated scores.
    """
    lengths = {
        len(questions),
        len(expected_chunk_ids),
        len(retrieved_chunk_ids_per_sample),
        len(retrieved_contexts_per_sample),
        len(ground_truths),
    }
    if len(lengths) != 1:
        raise ValueError(
            "All input lists must have the same length. Got lengths: "
            f"questions={len(questions)}, expected_chunk_ids={len(expected_chunk_ids)}, "
            f"retrieved_chunk_ids={len(retrieved_chunk_ids_per_sample)}, "
            f"retrieved_contexts={len(retrieved_contexts_per_sample)}, "
            f"ground_truths={len(ground_truths)}"
        )

    if match_strategy == "chunk_id":
        logger.warning(
            "match_strategy='chunk_id' — this is only reliable when the live "
            "index was built from the same chunking run as the evaluation dataset."
        )

    per_sample: list[PerSampleRetrievalScore] = []

    for question, expected_id, retrieved_ids, retrieved_contexts, ground_truth in zip(
        questions,
        expected_chunk_ids,
        retrieved_chunk_ids_per_sample,
        retrieved_contexts_per_sample,
        ground_truths,
    ):
        top_k_ids = retrieved_ids[:k]
        top_k_contexts = retrieved_contexts[:k]

        if match_strategy == "content":
            hit = _content_hit(ground_truth, top_k_contexts, content_match_threshold)
        else:
            hit = expected_id in top_k_ids

        recall = 1.0 if hit else 0.0

        logger.debug(
            "Retrieval | strategy=%s | hit=%s | question='%s...'",
            match_strategy,
            hit,
            question[:60],
        )

        per_sample.append(
            PerSampleRetrievalScore(
                question=question,
                expected_chunk_id=expected_id,
                retrieved_chunk_ids=top_k_ids,
                k=k,
                hit=hit,
                recall_at_k=recall,
                match_strategy=match_strategy,
            )
        )

    num_samples = len(per_sample)
    if num_samples == 0:
        logger.warning("No samples to compute retrieval metrics on")
        return AggregatedRetrievalMetrics(
            k=k,
            num_samples=0,
            recall_at_k=0.0,
            hit_rate_at_k=0.0,
            match_strategy=match_strategy,
            per_sample=[],
        )

    hits = sum(s.hit for s in per_sample)
    recall_at_k = sum(s.recall_at_k for s in per_sample) / num_samples
    hit_rate_at_k = hits / num_samples

    logger.info(
        "Retrieval metrics | strategy=%s | k=%d | recall@k=%.4f | hit_rate@k=%.4f | samples=%d",
        match_strategy,
        k,
        recall_at_k,
        hit_rate_at_k,
        num_samples,
    )

    return AggregatedRetrievalMetrics(
        k=k,
        num_samples=num_samples,
        recall_at_k=recall_at_k,
        hit_rate_at_k=hit_rate_at_k,
        match_strategy=match_strategy,
        per_sample=per_sample,
    )
