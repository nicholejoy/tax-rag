"""
Tests for custom retrieval metrics: Recall@k and Hit Rate@k.

Covers both match strategies:
  - "content": Jaccard token-overlap between ground_truth and retrieved contexts
  - "chunk_id": exact string match on chunk IDs (legacy, fragile across re-chunking)
"""

from __future__ import annotations

import pytest

from evaluation.retrieval_metrics import (
    AggregatedRetrievalMetrics,
    PerSampleRetrievalScore,
    compute_retrieval_metrics,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run(
    ground_truths: list[str],
    contexts_per_sample: list[list[str]],
    expected_ids: list[str] | None = None,
    retrieved_ids_per_sample: list[list[str]] | None = None,
    k: int = 5,
    strategy: str = "content",
    threshold: float = 0.5,
) -> AggregatedRetrievalMetrics:
    n = len(ground_truths)
    return compute_retrieval_metrics(
        questions=[f"q{i}" for i in range(n)],
        expected_chunk_ids=expected_ids or ["id"] * n,
        retrieved_chunk_ids_per_sample=retrieved_ids_per_sample or [[] * n],
        retrieved_contexts_per_sample=contexts_per_sample,
        ground_truths=ground_truths,
        k=k,
        match_strategy=strategy,  # type: ignore[arg-type]
        content_match_threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Content-based matching
# ---------------------------------------------------------------------------

class TestContentMatching:
    def test_exact_match_is_hit(self) -> None:
        result = _run(
            ground_truths=["VAT is a value-added tax on goods and services"],
            contexts_per_sample=[["VAT is a value-added tax on goods and services"]],
        )
        assert result.per_sample[0].hit is True
        assert result.hit_rate_at_k == pytest.approx(1.0)

    def test_high_overlap_is_hit(self) -> None:
        """Paraphrased but most words overlap — should still be a hit."""
        result = _run(
            ground_truths=["VAT is a value-added tax on goods and services"],
            contexts_per_sample=[["VAT is the value-added tax applied to goods and services sold"]],
            threshold=0.4,
        )
        assert result.per_sample[0].hit is True

    def test_low_overlap_is_miss(self) -> None:
        result = _run(
            ground_truths=["VAT is a value-added tax on goods and services"],
            contexts_per_sample=[["Income tax is levied on individual earnings"]],
            threshold=0.5,
        )
        assert result.per_sample[0].hit is False

    def test_hit_in_second_chunk(self) -> None:
        """The matching context does not have to be the first chunk."""
        result = _run(
            ground_truths=["The threshold is R1 million per annum"],
            contexts_per_sample=[[
                "Income tax affects all residents",
                "The registration threshold is R1 million per annum for VAT",
            ]],
            threshold=0.4,
        )
        assert result.per_sample[0].hit is True

    def test_k_cutoff_excludes_later_chunks(self) -> None:
        """Matching context is beyond the k cutoff — should be a miss."""
        result = _run(
            ground_truths=["The threshold is R1 million per annum"],
            contexts_per_sample=[[
                "Unrelated context A",
                "Unrelated context B",
                "The threshold is R1 million per annum",  # position 2, k=2 → miss
            ]],
            k=2,
            threshold=0.4,
        )
        assert result.per_sample[0].hit is False

    def test_perfect_recall_all_samples(self) -> None:
        result = _run(
            ground_truths=[
                "VAT is a consumption tax",
                "Income tax applies to residents",
            ],
            contexts_per_sample=[
                ["VAT is a consumption tax levied on goods"],
                ["Income tax applies to all South African residents"],
            ],
            threshold=0.4,
        )
        assert result.recall_at_k == pytest.approx(1.0)
        assert result.hit_rate_at_k == pytest.approx(1.0)

    def test_partial_recall(self) -> None:
        result = _run(
            ground_truths=[
                "VAT is a consumption tax",
                "Income tax applies to residents",
            ],
            contexts_per_sample=[
                ["VAT is a consumption tax levied on goods"],  # hit
                ["Completely unrelated content about other topics"],  # miss
            ],
            threshold=0.5,
        )
        assert result.recall_at_k == pytest.approx(0.5)
        assert result.hit_rate_at_k == pytest.approx(0.5)

    def test_empty_ground_truth_is_miss(self) -> None:
        result = _run(
            ground_truths=[""],
            contexts_per_sample=[["Some context"]],
        )
        assert result.per_sample[0].hit is False

    def test_match_strategy_recorded(self) -> None:
        result = _run(
            ground_truths=["VAT is a tax"],
            contexts_per_sample=[["VAT is a tax"]],
            strategy="content",
        )
        assert result.match_strategy == "content"
        assert result.per_sample[0].match_strategy == "content"

    def test_to_dict_includes_strategy(self) -> None:
        result = _run(
            ground_truths=["VAT is a tax"],
            contexts_per_sample=[["VAT is a tax"]],
        )
        d = result.to_dict()
        assert d["match_strategy"] == "content"


# ---------------------------------------------------------------------------
# Chunk-ID matching (legacy)
# ---------------------------------------------------------------------------

class TestChunkIdMatching:
    def _run_id(
        self,
        expected_ids: list[str],
        retrieved_ids: list[list[str]],
        k: int = 5,
    ) -> AggregatedRetrievalMetrics:
        n = len(expected_ids)
        return compute_retrieval_metrics(
            questions=[f"q{i}" for i in range(n)],
            expected_chunk_ids=expected_ids,
            retrieved_chunk_ids_per_sample=retrieved_ids,
            retrieved_contexts_per_sample=[["ctx"]] * n,
            ground_truths=["gt"] * n,
            k=k,
            match_strategy="chunk_id",
        )

    def test_exact_id_match_is_hit(self) -> None:
        result = self._run_id(["chunk_a"], [["chunk_a", "chunk_b"]])
        assert result.per_sample[0].hit is True

    def test_id_not_present_is_miss(self) -> None:
        result = self._run_id(["chunk_a"], [["chunk_x", "chunk_y"]])
        assert result.per_sample[0].hit is False

    def test_k_cutoff_respected(self) -> None:
        # chunk_a is at position 2 (0-indexed), k=2 → miss
        result = self._run_id(["chunk_a"], [["chunk_x", "chunk_y", "chunk_a"]], k=2)
        assert result.per_sample[0].hit is False

    def test_match_strategy_recorded(self) -> None:
        result = self._run_id(["chunk_a"], [["chunk_a"]])
        assert result.match_strategy == "chunk_id"


# ---------------------------------------------------------------------------
# Shared / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_inputs_return_zeros(self) -> None:
        result = compute_retrieval_metrics(
            questions=[],
            expected_chunk_ids=[],
            retrieved_chunk_ids_per_sample=[],
            retrieved_contexts_per_sample=[],
            ground_truths=[],
            k=5,
        )
        assert result.num_samples == 0
        assert result.recall_at_k == 0.0
        assert result.hit_rate_at_k == 0.0

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_retrieval_metrics(
                questions=["q1", "q2"],
                expected_chunk_ids=["id1"],
                retrieved_chunk_ids_per_sample=[["id1"]],
                retrieved_contexts_per_sample=[["ctx"]],
                ground_truths=["gt1"],
                k=5,
            )

    def test_returns_correct_types(self) -> None:
        result = _run(
            ground_truths=["VAT is a tax"],
            contexts_per_sample=[["VAT is a tax"]],
        )
        assert isinstance(result, AggregatedRetrievalMetrics)
        assert isinstance(result.per_sample[0], PerSampleRetrievalScore)

    def test_num_samples_correct(self) -> None:
        result = _run(
            ground_truths=["a", "b", "c"],
            contexts_per_sample=[["a"], ["b"], ["c"]],
        )
        assert result.num_samples == 3

    def test_to_dict_contains_required_keys(self) -> None:
        result = _run(
            ground_truths=["VAT is a tax"],
            contexts_per_sample=[["VAT is a tax"]],
            k=5,
        )
        d = result.to_dict()
        assert "k" in d
        assert "num_samples" in d
        assert "match_strategy" in d
        assert "recall_at_5" in d
        assert "hit_rate_at_5" in d
