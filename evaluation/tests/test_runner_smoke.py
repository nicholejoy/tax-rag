"""
Smoke tests for the evaluation runner.

These tests:
  - validate the dataset loader end-to-end with a small fixture file
  - run a minimal end-to-end pipeline with 2–3 samples using mocked
    HTTP and RAGAS calls (no live API or Claude required)
  - confirm the runner produces the expected output files
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from evaluation.config import EvalSettings
from evaluation.dataset import EvalSample, load_eval_dataset
from evaluation.query_client import QueryResult
from evaluation.ragas_evaluator import (
    AggregatedRagasMetrics,
    PerSampleRagasScore,
)
from evaluation.retrieval_metrics import compute_retrieval_metrics
from evaluation.runner import _build_detailed_rows, _run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_JSONL_RECORDS = [
    {
        "question": "What is VAT?",
        "ground_truth": "Value-added tax on goods and services.",
        "chunk_id": "vat_chunk_001",
        "url": "https://example.com/vat",
        "source_title": "VAT Guide",
    },
    {
        "title": "Who pays income tax?",   # alternate field name
        "text": "Residents above the tax threshold.",
        "chunk_id": "income_chunk_005",
        "url": "https://example.com/income",
        "source_title": "Income Tax Guide",
    },
    {
        "question": "What is the VAT registration threshold?",
        "ground_truth": "R1 million per annum.",
        "chunk_id": "vat_chunk_010",
        "url": "https://example.com/vat-reg",
        "source_title": "VAT Guide",
    },
]


@pytest.fixture
def jsonl_path(tmp_path: Path) -> Path:
    p = tmp_path / "eval_dataset.jsonl"
    with p.open("w") as fh:
        for rec in SAMPLE_JSONL_RECORDS:
            fh.write(json.dumps(rec) + "\n")
    return p


@pytest.fixture
def eval_settings(tmp_path: Path) -> EvalSettings:
    return EvalSettings(
        api_url="http://localhost:8000",
        anthropic_api_key="test-key",
        results_dir=str(tmp_path / "results"),
    )


# ---------------------------------------------------------------------------
# Dataset loader tests
# ---------------------------------------------------------------------------

class TestDatasetLoader:
    def test_loads_all_samples(self, jsonl_path: Path) -> None:
        samples = load_eval_dataset(jsonl_path)
        assert len(samples) == 3

    def test_normalises_alternate_field_names(self, jsonl_path: Path) -> None:
        samples = load_eval_dataset(jsonl_path)
        # Second record uses title/text instead of question/ground_truth
        assert samples[1].question == "Who pays income tax?"
        assert samples[1].ground_truth == "Residents above the tax threshold."

    def test_chunk_ids_preserved(self, jsonl_path: Path) -> None:
        samples = load_eval_dataset(jsonl_path)
        assert samples[0].chunk_id == "vat_chunk_001"
        assert samples[1].chunk_id == "income_chunk_005"
        assert samples[2].chunk_id == "vat_chunk_010"

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_eval_dataset("/nonexistent/path.jsonl")

    def test_malformed_line_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        with p.open("w") as fh:
            fh.write('{"question": "ok?", "ground_truth": "ok.", "chunk_id": "c1", "url": "", "source_title": ""}\n')
            fh.write("NOT VALID JSON\n")
            fh.write('{"question": "also ok?", "ground_truth": "yes.", "chunk_id": "c2", "url": "", "source_title": ""}\n')
        samples = load_eval_dataset(p)
        assert len(samples) == 2

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        samples = load_eval_dataset(p)
        assert samples == []



# ---------------------------------------------------------------------------
# End-to-end smoke test with mocked HTTP + RAGAS
# ---------------------------------------------------------------------------

def _make_query_result(question: str, chunk_id: str) -> QueryResult:
    return QueryResult(
        question=question,
        answer=f"Answer for: {question}",
        contexts=["Some retrieved context."],
        chunk_ids=[chunk_id, "other_chunk"],
        metadata=[{}],
        raw_response={},
    )


def _mock_ragas_result_df(questions: list[str]) -> MagicMock:
    df = pd.DataFrame(
        {
            "question": questions,
            "faithfulness": [0.88] * len(questions),
            "answer_relevancy": [0.82] * len(questions),
            "context_precision": [0.79] * len(questions),
            "context_recall": [0.74] * len(questions),
        }
    )
    mock = MagicMock()
    mock.to_pandas.return_value = df
    return mock


class TestEndToEndSmoke:
    @patch("evaluation.runner.run_ragas_evaluation")
    @patch("evaluation.runner.batch_query")
    def test_runner_produces_output_files(
        self,
        mock_batch_query: MagicMock,
        mock_ragas: MagicMock,
        jsonl_path: Path,
        eval_settings: EvalSettings,
        tmp_path: Path,
    ) -> None:
        samples = load_eval_dataset(jsonl_path)
        query_results = [
            _make_query_result(s.question, s.chunk_id) for s in samples
        ]
        mock_batch_query.return_value = asyncio.coroutine(lambda *a, **kw: query_results)()

        ragas_per_sample = [
            PerSampleRagasScore(
                question=s.question,
                faithfulness=0.88,
                answer_relevancy=0.82,
                context_precision=0.79,
                context_recall=0.74,
            )
            for s in samples
        ]
        mock_ragas.return_value = AggregatedRagasMetrics(
            faithfulness=0.88,
            answer_relevancy=0.82,
            context_precision=0.79,
            context_recall=0.74,
            num_samples=len(samples),
            per_sample=ragas_per_sample,
        )

        import argparse
        args = argparse.Namespace(
            dataset=str(jsonl_path),
            top_k=2,
            limit=0,
            output_dir=str(tmp_path / "results"),
            skip_ragas=False,
            log_level="WARNING",
        )

        # Patch settings so it uses our test config
        with patch("evaluation.runner.get_eval_settings", return_value=eval_settings):
            asyncio.run(_run(args))

        results_dir = tmp_path / "results"
        assert (results_dir / "retrieval_metrics.json").exists()
        assert (results_dir / "ragas_metrics.json").exists()
        assert (results_dir / "full_detailed_results.csv").exists()

    @patch("evaluation.runner.run_ragas_evaluation")
    @patch("evaluation.runner.batch_query")
    def test_retrieval_metrics_json_structure(
        self,
        mock_batch_query: MagicMock,
        mock_ragas: MagicMock,
        jsonl_path: Path,
        eval_settings: EvalSettings,
        tmp_path: Path,
    ) -> None:
        samples = load_eval_dataset(jsonl_path)
        query_results = [
            _make_query_result(s.question, s.chunk_id) for s in samples
        ]
        mock_batch_query.return_value = asyncio.coroutine(lambda *a, **kw: query_results)()
        mock_ragas.return_value = AggregatedRagasMetrics(
            faithfulness=0.9,
            answer_relevancy=0.85,
            context_precision=0.8,
            context_recall=0.75,
            num_samples=len(samples),
            per_sample=[],
        )

        import argparse
        args = argparse.Namespace(
            dataset=str(jsonl_path),
            top_k=2,
            limit=0,
            output_dir=str(tmp_path / "results"),
            skip_ragas=False,
            log_level="WARNING",
        )

        with patch("evaluation.runner.get_eval_settings", return_value=eval_settings):
            asyncio.run(_run(args))

        with (tmp_path / "results" / "retrieval_metrics.json").open() as fh:
            data = json.load(fh)

        assert "k" in data
        assert "num_samples" in data
        assert data["num_samples"] == 3
        # All chunks match (smoke fixture has chunk_id in position 0 of retrieved)
        assert data[f"hit_rate_at_{data['k']}"] == pytest.approx(1.0)

    def test_build_detailed_rows_with_failures(self) -> None:
        """_build_detailed_rows handles Exception entries in query_results."""
        samples = [
            EvalSample(
                question="What is VAT?",
                ground_truth="A tax.",
                chunk_id="vat_001",
                url="",
                source_title="",
            ),
        ]
        query_results: list = [RuntimeError("timeout")]
        retrieval = compute_retrieval_metrics(
            expected_chunk_ids=[],
            retrieved_chunk_ids_per_sample=[],
            retrieved_contexts_per_sample=[],
            ground_truths=[],
            questions=[],
            k=5,
        )
        rows = _build_detailed_rows(
            samples=samples,
            query_results=query_results,
            ragas_metrics=None,
            retrieval_metrics=retrieval,
        )
        assert len(rows) == 1
        assert rows[0]["error"] is not None
        assert rows[0]["answer"] is None

    def test_build_detailed_rows_answer_from_api(self) -> None:
        """Answer is always taken from the query result (LLM-generated response)."""
        sample1 = EvalSample(
            question="What is VAT?",
            ground_truth="A tax.",
            chunk_id="vat_001",
            url="",
            source_title="",
        )
        sample2 = EvalSample(
            question="What is income tax?",
            ground_truth="A tax on income.",
            chunk_id="inc_001",
            url="",
            source_title="",
        )
        qr1 = QueryResult(
            question="What is VAT?",
            answer="LLM answer for VAT.",
            contexts=["ctx"],
            chunk_ids=["vat_001"],
            metadata=[{}],
            raw_response={},
        )
        qr2 = QueryResult(
            question="What is income tax?",
            answer="LLM answer for income tax.",
            contexts=["ctx"],
            chunk_ids=["inc_001"],
            metadata=[{}],
            raw_response={},
        )
        retrieval = compute_retrieval_metrics(
            expected_chunk_ids=["vat_001", "inc_001"],
            retrieved_chunk_ids_per_sample=[["vat_001"], ["inc_001"]],
            retrieved_contexts_per_sample=[["ctx"], ["ctx"]],
            ground_truths=["A tax.", "A tax on income."],
            questions=["What is VAT?", "What is income tax?"],
            k=5,
        )
        rows = _build_detailed_rows(
            samples=[sample1, sample2],
            query_results=[qr1, qr2],
            ragas_metrics=None,
            retrieval_metrics=retrieval,
        )
        assert rows[0]["answer"] == "LLM answer for VAT."
        assert rows[1]["answer"] == "LLM answer for income tax."
