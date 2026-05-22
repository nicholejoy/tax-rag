"""
RAG Evaluation Runner

Entry point for the evaluation framework.

Usage:
    python -m evaluation.runner [OPTIONS]

Options:
    --dataset PATH            Path to JSONL evaluation dataset
                              (default: eval_dataset_path in config / .env)
    --top-k INT               Number of chunks to retrieve per query (default: 5)
    --limit INT               Max number of samples to evaluate (0 = all)
    --output-dir PATH         Directory to write result files
                              (default: evaluation/results/)
    --skip-ragas              Skip RAGAS evaluation (retrieval metrics only)
    --skip-retrieval-metrics  Skip Recall@k / Hit Rate@k computation entirely
    --match-strategy STR      "content" (default) or "chunk_id" — how retrieval
                              hits are determined. "content" uses token overlap
                              between the ground truth and retrieved chunks, which
                              is robust to re-chunking. "chunk_id" uses exact ID
                              matching (only reliable if the live index was built
                              from the same chunking run as the dataset).
    --match-threshold FLOAT   Token-overlap ratio required for a content hit
                              (default: 0.5, only used with --match-strategy=content)
    --log-level LEVEL         Logging level (default: INFO)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from .config import EvalSettings, get_eval_settings
from .dataset import EvalSample, load_eval_dataset
from .query_client import QueryResult, batch_query
from .ragas_evaluator import AggregatedRagasMetrics, run_ragas_evaluation
from .retrieval_metrics import AggregatedRetrievalMetrics, MatchStrategy, compute_retrieval_metrics


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _ensure_results_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", path)


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        logger.warning("No rows to write to %s", path)
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

async def _collect_query_results(
    samples: list[EvalSample],
    settings: EvalSettings,
    top_k: int,
) -> list[QueryResult | Exception]:
    """Send all questions to the API and collect responses."""
    questions = [s.question for s in samples]
    logger.info("Querying API for %d questions (top_k=%d)…", len(questions), top_k)
    return await batch_query(questions, settings, top_k=top_k)


def _build_detailed_rows(
    samples: list[EvalSample],
    query_results: list[QueryResult | Exception],
    ragas_metrics: AggregatedRagasMetrics | None,
    retrieval_metrics: AggregatedRetrievalMetrics | None,
) -> list[dict]:
    """
    Merge sample metadata, query outputs, and metric scores into one row per sample.
    """
    ragas_per_sample = (
        {s.question: s for s in ragas_metrics.per_sample}
        if ragas_metrics
        else {}
    )
    retrieval_per_sample = (
        {s.question: s for s in retrieval_metrics.per_sample}
        if retrieval_metrics
        else {}
    )

    rows: list[dict] = []
    for sample, qr in zip(samples, query_results):
        row: dict = {
            "question": sample.question,
            "ground_truth": sample.ground_truth,
            "expected_chunk_id": sample.chunk_id,
            "url": sample.url,
            "source_title": sample.source_title,
        }

        if isinstance(qr, Exception):
            row["error"] = str(qr)
            row["answer"] = None
            row["retrieved_chunk_ids"] = None
            row["num_contexts"] = 0
        else:
            row["error"] = None
            row["answer"] = qr.answer
            row["retrieved_chunk_ids"] = json.dumps(qr.chunk_ids)
            row["num_contexts"] = len(qr.contexts)

        # Retrieval metrics
        rm = retrieval_per_sample.get(sample.question)
        row["hit"] = rm.hit if rm else None
        row["recall_at_k"] = rm.recall_at_k if rm else None

        # RAGAS metrics
        rs = ragas_per_sample.get(sample.question)
        row["faithfulness"] = rs.faithfulness if rs else None
        row["answer_relevancy"] = rs.answer_relevancy if rs else None
        row["context_precision"] = rs.context_precision if rs else None
        row["context_recall"] = rs.context_recall if rs else None

        rows.append(row)

    return rows


def _print_summary(
    num_samples: int,
    retrieval: AggregatedRetrievalMetrics | None,
    ragas: AggregatedRagasMetrics | None,
) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  EVALUATION SUMMARY")
    print(sep)
    print(f"  Samples evaluated : {num_samples}")
    if retrieval:
        print(f"\n  Retrieval Metrics (k={retrieval.k}, strategy={retrieval.match_strategy})")
        print(f"    Recall@{retrieval.k}         : {retrieval.recall_at_k:.4f}")
        print(f"    Hit Rate@{retrieval.k}       : {retrieval.hit_rate_at_k:.4f}")
    if ragas:
        print("\n  RAGAS Metrics")
        print(f"    Faithfulness      : {ragas.faithfulness:.4f}" if ragas.faithfulness is not None else "    Faithfulness      : N/A")
        print(f"    Answer Relevancy  : {ragas.answer_relevancy:.4f}" if ragas.answer_relevancy is not None else "    Answer Relevancy  : N/A")
        print(f"    Context Precision : {ragas.context_precision:.4f}" if ragas.context_precision is not None else "    Context Precision : N/A")
        print(f"    Context Recall    : {ragas.context_recall:.4f}" if ragas.context_recall is not None else "    Context Recall    : N/A")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace) -> None:
    settings = get_eval_settings()

    # CLI overrides
    dataset_path = args.dataset or settings.eval_dataset_path
    top_k: int = args.top_k
    limit: int = args.limit
    output_dir = Path(args.output_dir or settings.results_dir)
    skip_ragas: bool = args.skip_ragas
    skip_retrieval: bool = args.skip_retrieval_metrics
    match_strategy: MatchStrategy = args.match_strategy
    match_threshold: float = args.match_threshold

    _ensure_results_dir(output_dir)

    # 1. Load dataset
    logger.info("Loading evaluation dataset: %s", dataset_path)
    samples: list[EvalSample] = load_eval_dataset(dataset_path)
    if limit and limit > 0:
        samples = samples[:limit]
        logger.info("Limiting to %d samples", limit)  

    if not samples:
        logger.error("No evaluation samples found. Exiting.")
        sys.exit(1)

    # 2. Query the RAG API
    query_results = await _collect_query_results(samples, settings, top_k=top_k)

    # Partition into successes and failures
    successful_indices: list[int] = []
    total_failures = 0
    for i, qr in enumerate(query_results):
        if isinstance(qr, Exception):
            total_failures += 1
        else:
            successful_indices.append(i)

    logger.info(
        "%d/%d queries succeeded", len(successful_indices), len(samples)
    )

    if total_failures >= settings.max_allowed_failures:
        logger.error(
            "Too many query failures (%d >= %d). Aborting evaluation. "
            "Check your API key, network, or Anthropic rate limits.",
            total_failures,
            settings.max_allowed_failures,
        )
        if query_results and isinstance(query_results[0], Exception):
            first_error = query_results[0]
            logger.error("First error sample: %s", str(first_error)[:200])
        sys.exit(1)

    if not successful_indices:
        logger.error(
            "All %d queries failed. Aborting evaluation. "
            "Check your API key or network connectivity.",
            len(samples),
        )
        sys.exit(1)

    successful_samples = [samples[i] for i in successful_indices]
    successful_results: list[QueryResult] = [query_results[i] for i in successful_indices]  # type: ignore[misc]

    # 3. Compute retrieval metrics (Recall@k, Hit Rate@k)
    retrieval_metrics: AggregatedRetrievalMetrics | None = None
    if not skip_retrieval:
        logger.info("Computing retrieval metrics (strategy=%s)…", match_strategy)
        retrieval_metrics = compute_retrieval_metrics(
            questions=[s.question for s in successful_samples],
            expected_chunk_ids=[s.chunk_id for s in successful_samples],
            retrieved_chunk_ids_per_sample=[qr.chunk_ids for qr in successful_results],
            retrieved_contexts_per_sample=[qr.contexts for qr in successful_results],
            ground_truths=[s.ground_truth for s in successful_samples],
            k=top_k,
            match_strategy=match_strategy,
            content_match_threshold=match_threshold,
        )
    else:
        logger.info("Skipping retrieval metrics (--skip-retrieval-metrics)")

    # 4. Run RAGAS evaluation
    ragas_metrics: AggregatedRagasMetrics | None = None
    if not skip_ragas:
        logger.info("Running RAGAS evaluation…")
        try:
            ragas_metrics = run_ragas_evaluation(
                questions=[s.question for s in successful_samples],
                answers=[qr.answer for qr in successful_results],
                contexts=[qr.contexts for qr in successful_results],
                ground_truths=[s.ground_truth for s in successful_samples],
                settings=settings,
            )
        except Exception as exc:
            logger.error("RAGAS evaluation failed — saving retrieval metrics only. Error: %s", exc)

    # 5. Save results
    # ragas_metrics.json
    if ragas_metrics:
        _write_json(
            ragas_metrics.to_dict(),
            output_dir / "ragas_metrics.json",
        )

    # retrieval_metrics.json
    if retrieval_metrics:
        _write_json(
            retrieval_metrics.to_dict(),
            output_dir / "retrieval_metrics.json",
        )

    # full_detailed_results.csv
    detailed_rows = _build_detailed_rows(
        samples=samples,
        query_results=query_results,
        ragas_metrics=ragas_metrics,
        retrieval_metrics=retrieval_metrics,
    )
    _write_csv(detailed_rows, output_dir / "full_detailed_results.csv")

    # 6. Print summary
    _print_summary(len(successful_samples), retrieval_metrics, ragas_metrics)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Runner — RAGAS + retrieval metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to JSONL evaluation dataset (overrides EVAL_DATASET_PATH env var)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="Number of chunks to retrieve per query (also used as k for Recall@k)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max samples to evaluate (0 = all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Directory to write result files",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        default=False,
        dest="skip_ragas",
        help="Skip RAGAS evaluation and only compute retrieval metrics",
    )
    parser.add_argument(
        "--skip-retrieval-metrics",
        action="store_true",
        default=False,
        dest="skip_retrieval_metrics",
        help="Skip Recall@k / Hit Rate@k computation entirely",
    )
    parser.add_argument(
        "--match-strategy",
        type=str,
        default="content",
        dest="match_strategy",
        choices=["content", "chunk_id"],
        help=(
            "How retrieval hits are determined. "
            "'content' (default) uses token overlap between ground_truth and retrieved chunks — "
            "robust to re-chunking. "
            "'chunk_id' uses exact ID matching — only reliable if the live index was built "
            "from the same chunking run as the evaluation dataset."
        ),
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.5,
        dest="match_threshold",
        help="Jaccard token-overlap ratio required for a content hit (default: 0.5)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
