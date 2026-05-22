"""
RAGAS evaluation wrapper.

Converts pipeline outputs into the RAGAS dataset format and runs the four
required metrics:
  - faithfulness
  - answer_relevancy
  - context_precision
  - context_recall

RAGAS is used as-is; no custom metric logic is re-implemented here.
"""

import logging
from dataclasses import dataclass, field

from anthropic import Anthropic
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from .config import EvalSettings

logger = logging.getLogger(__name__)

# Column names expected by RAGAS
_RAGAS_QUESTION_COL = "question"
_RAGAS_ANSWER_COL = "answer"
_RAGAS_CONTEXTS_COL = "contexts"
_RAGAS_GROUND_TRUTH_COL = "ground_truth"


@dataclass
class PerSampleRagasScore:
    """RAGAS scores for a single sample."""

    question: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }


@dataclass
class AggregatedRagasMetrics:
    """Aggregated RAGAS metrics across the evaluation set."""

    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None
    num_samples: int
    per_sample: list[PerSampleRagasScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "num_samples": self.num_samples,
        }


def _build_ragas_llm(settings: EvalSettings):
    """Instantiate the RAGAS LLM wrapper backed by Claude."""
    if not settings.anthropic_api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is required for RAGAS evaluation. "
            "Set it in your .env file or environment."
        )
    client = Anthropic(api_key=settings.anthropic_api_key)
    llm = llm_factory(
        settings.ragas_llm_model,
        provider="anthropic",
        client=client,
    )
    logger.info("RAGAS LLM initialised: %s (anthropic)", settings.ragas_llm_model)
    return llm


def _build_ragas_dataset(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> Dataset:
    """Convert evaluation data into the HuggingFace Dataset format RAGAS expects."""
    return Dataset.from_dict(
        {
            _RAGAS_QUESTION_COL: questions,
            _RAGAS_ANSWER_COL: answers,
            _RAGAS_CONTEXTS_COL: contexts,
            _RAGAS_GROUND_TRUTH_COL: ground_truths,
        }
    )


def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    settings: EvalSettings,
) -> AggregatedRagasMetrics:
    """
    Run RAGAS evaluation on the provided data and return aggregated + per-sample scores.

    Parameters
    ----------
    questions:    The user questions.
    answers:      The generated answers from the RAG pipeline.
    contexts:     Retrieved context chunks per question (list of lists).
    ground_truths: Reference answers.
    settings:     Evaluation configuration.

    Returns
    -------
    AggregatedRagasMetrics containing mean scores and per-sample breakdown.
    """
    if not questions:
        logger.warning("No samples provided to run_ragas_evaluation")
        return AggregatedRagasMetrics(
            faithfulness=None,
            answer_relevancy=None,
            context_precision=None,
            context_recall=None,
            num_samples=0,
        )

    logger.info("Building RAGAS dataset with %d samples", len(questions))
    dataset = _build_ragas_dataset(questions, answers, contexts, ground_truths)

    llm = _build_ragas_llm(settings)

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    logger.info("Running RAGAS evaluation (this may take a while)…")
    try:
        result = evaluate(dataset, metrics=metrics)
    except Exception as exc:
        logger.error("RAGAS evaluation failed: %s", exc)
        raise

    # result is a ragas EvaluationResult; convert to a plain dict / DataFrame
    result_df = result.to_pandas()

    # Per-sample scores
    per_sample: list[PerSampleRagasScore] = []
    for _, row in result_df.iterrows():
        per_sample.append(
            PerSampleRagasScore(
                question=str(row.get(_RAGAS_QUESTION_COL, "")),
                faithfulness=_safe_float(row.get("faithfulness")),
                answer_relevancy=_safe_float(row.get("answer_relevancy")),
                context_precision=_safe_float(row.get("context_precision")),
                context_recall=_safe_float(row.get("context_recall")),
            )
        )

    # Aggregated means (RAGAS also provides these via result dict)
    def _mean(key: str) -> float | None:
        col = result_df.get(key)
        if col is None:
            return None
        valid = col.dropna()
        return float(valid.mean()) if len(valid) > 0 else None

    aggregated = AggregatedRagasMetrics(
        faithfulness=_mean("faithfulness"),
        answer_relevancy=_mean("answer_relevancy"),
        context_precision=_mean("context_precision"),
        context_recall=_mean("context_recall"),
        num_samples=len(per_sample),
        per_sample=per_sample,
    )

    logger.info(
        "RAGAS complete | faithfulness=%.4f | answer_relevancy=%.4f | "
        "context_precision=%.4f | context_recall=%.4f",
        aggregated.faithfulness or 0.0,
        aggregated.answer_relevancy or 0.0,
        aggregated.context_precision or 0.0,
        aggregated.context_recall or 0.0,
    )

    return aggregated


def _safe_float(value: object) -> float | None:
    """Convert a value to float, returning None if conversion fails."""
    try:
        f = float(value)  # type: ignore[arg-type]
        import math
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None
