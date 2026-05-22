"""
Tests for the RAGAS evaluation wrapper.

These tests validate that:
  - run_ragas_evaluation returns an AggregatedRagasMetrics object
  - all four expected metric keys are present
  - per-sample scores are returned and have the right structure
  - an empty input is handled gracefully

NOTE: These tests mock the RAGAS `evaluate` call to avoid requiring a live
Anthropic API key in CI.  An integration test that actually calls Claude is
marked with `@pytest.mark.integration` and skipped by default.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from evaluation.config import EvalSettings
from evaluation.ragas_evaluator import (
    AggregatedRagasMetrics,
    PerSampleRagasScore,
    run_ragas_evaluation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def eval_settings() -> EvalSettings:
    return EvalSettings(
        anthropic_api_key="test-key",
        ragas_llm_model="claude-sonnet-4-5",
    )


@pytest.fixture
def sample_data() -> dict:
    return {
        "questions": [
            "What is VAT?",
            "Who is liable for income tax?",
            "What is the VAT registration threshold?",
        ],
        "answers": [
            "VAT is a value-added tax levied on goods and services.",
            "Residents earning above the tax threshold are liable.",
            "The threshold is R1 million per annum.",
        ],
        "contexts": [
            ["VAT is a consumption tax.", "It is administered by SARS."],
            ["Individuals who are tax residents must pay income tax."],
            ["Registration is compulsory when turnover exceeds R1 million."],
        ],
        "ground_truths": [
            "Value-added tax on goods and services.",
            "South African tax residents above the threshold.",
            "R1 million turnover per year.",
        ],
    }


def _mock_ragas_result(questions: list[str]) -> MagicMock:
    """Build a mock RAGAS EvaluationResult that returns a plausible DataFrame."""
    df = pd.DataFrame(
        {
            "question": questions,
            "faithfulness": [0.9] * len(questions),
            "answer_relevancy": [0.85] * len(questions),
            "context_precision": [0.8] * len(questions),
            "context_recall": [0.75] * len(questions),
        }
    )
    mock_result = MagicMock()
    mock_result.to_pandas.return_value = df
    return mock_result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunRagasEvaluation:
    @patch("evaluation.ragas_evaluator.evaluate")
    @patch("evaluation.ragas_evaluator._build_ragas_llm")
    def test_returns_aggregated_metrics(
        self,
        mock_build_llm: MagicMock,
        mock_evaluate: MagicMock,
        eval_settings: EvalSettings,
        sample_data: dict,
    ) -> None:
        mock_build_llm.return_value = MagicMock()
        mock_evaluate.return_value = _mock_ragas_result(sample_data["questions"])

        result = run_ragas_evaluation(**sample_data, settings=eval_settings)

        assert isinstance(result, AggregatedRagasMetrics)

    @patch("evaluation.ragas_evaluator.evaluate")
    @patch("evaluation.ragas_evaluator._build_ragas_llm")
    def test_all_metric_keys_present(
        self,
        mock_build_llm: MagicMock,
        mock_evaluate: MagicMock,
        eval_settings: EvalSettings,
        sample_data: dict,
    ) -> None:
        mock_build_llm.return_value = MagicMock()
        mock_evaluate.return_value = _mock_ragas_result(sample_data["questions"])

        result = run_ragas_evaluation(**sample_data, settings=eval_settings)
        result_dict = result.to_dict()

        for key in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
            assert key in result_dict, f"Missing metric key: {key}"

    @patch("evaluation.ragas_evaluator.evaluate")
    @patch("evaluation.ragas_evaluator._build_ragas_llm")
    def test_per_sample_scores_returned(
        self,
        mock_build_llm: MagicMock,
        mock_evaluate: MagicMock,
        eval_settings: EvalSettings,
        sample_data: dict,
    ) -> None:
        mock_build_llm.return_value = MagicMock()
        mock_evaluate.return_value = _mock_ragas_result(sample_data["questions"])

        result = run_ragas_evaluation(**sample_data, settings=eval_settings)

        assert len(result.per_sample) == len(sample_data["questions"])
        for score in result.per_sample:
            assert isinstance(score, PerSampleRagasScore)
            assert score.faithfulness is not None
            assert score.answer_relevancy is not None
            assert score.context_precision is not None
            assert score.context_recall is not None

    @patch("evaluation.ragas_evaluator.evaluate")
    @patch("evaluation.ragas_evaluator._build_ragas_llm")
    def test_aggregated_scores_are_means(
        self,
        mock_build_llm: MagicMock,
        mock_evaluate: MagicMock,
        eval_settings: EvalSettings,
        sample_data: dict,
    ) -> None:
        mock_build_llm.return_value = MagicMock()
        mock_evaluate.return_value = _mock_ragas_result(sample_data["questions"])

        result = run_ragas_evaluation(**sample_data, settings=eval_settings)

        assert result.faithfulness == pytest.approx(0.9, abs=1e-4)
        assert result.answer_relevancy == pytest.approx(0.85, abs=1e-4)
        assert result.context_precision == pytest.approx(0.8, abs=1e-4)
        assert result.context_recall == pytest.approx(0.75, abs=1e-4)

    def test_empty_input_returns_gracefully(self, eval_settings: EvalSettings) -> None:
        result = run_ragas_evaluation(
            questions=[],
            answers=[],
            contexts=[],
            ground_truths=[],
            settings=eval_settings,
        )
        assert result.num_samples == 0
        assert result.faithfulness is None
        assert result.answer_relevancy is None

    def test_missing_api_key_raises(self) -> None:
        settings = EvalSettings(anthropic_api_key="")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            run_ragas_evaluation(
                questions=["test?"],
                answers=["test."],
                contexts=[["context"]],
                ground_truths=["test."],
                settings=settings,
            )


# ---------------------------------------------------------------------------
# Integration test (skipped unless --integration flag is passed)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_ragas_with_real_claude(eval_settings: EvalSettings) -> None:
    """
    Live integration test — requires ANTHROPIC_API_KEY in environment.
    Run with: pytest -m integration
    """
    result = run_ragas_evaluation(
        questions=["What is VAT?"],
        answers=["VAT is a value-added tax on goods and services."],
        contexts=[["VAT stands for Value-Added Tax. It is levied on the supply of goods and services."]],
        ground_truths=["Value-Added Tax levied on goods and services."],
        settings=eval_settings,
    )
    assert result.faithfulness is not None
    assert result.answer_relevancy is not None
    assert result.context_precision is not None
    assert result.context_recall is not None
