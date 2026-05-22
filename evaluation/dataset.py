"""
Dataset loader for the RAG evaluation framework.

Loads JSONL evaluation samples and normalises field names so the rest of the
pipeline can work with a consistent EvalSample dataclass regardless of whether
the source file uses (question/ground_truth) or (title/text) conventions.

The answer is generated live during evaluation by an LLM using the retrieved
chunks as context. The ground_truth is the reference answer used for
correctness metrics.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample."""

    question: str
    ground_truth: str
    chunk_id: str
    url: str
    source_title: str
    # 'context' may be pre-populated in the file; runner will fill it from the
    # live query response when it is absent.
    context: str = ""
    extra: dict = field(default_factory=dict)


def _parse_record(record: dict, idx: int) -> EvalSample:
    """
    Map a raw JSONL record to an EvalSample.

    Supports two layouts:
      - Spec layout:   question / ground_truth / context / chunk_id / url / source_title
      - Dataset layout: title / text / chunk_id / url / source_title
    """
    question = record.get("question") or record.get("title", "")
    ground_truth = record.get("ground_truth") or record.get("text", "")
    context = record.get("context", "")
    chunk_id = record.get("chunk_id", f"unknown_{idx}")
    url = record.get("url", "")
    source_title = record.get("source_title", "")

    if not question:
        raise ValueError(f"Record {idx} is missing a question/title field")
    if not ground_truth:
        raise ValueError(f"Record {idx} is missing a ground_truth/text field")

    known_keys = {"question", "title", "ground_truth", "text",
                  "context", "chunk_id", "url", "source_title"}
    extra = {k: v for k, v in record.items() if k not in known_keys}

    return EvalSample(
        question=question,
        ground_truth=ground_truth,
        context=context,
        chunk_id=chunk_id,
        url=url,
        source_title=source_title,
        extra=extra,
    )


def load_eval_dataset(path: str | Path) -> list[EvalSample]:
    """
    Load all evaluation samples from a JSONL file.

    Malformed lines are skipped with a warning so a single bad record cannot
    abort the whole evaluation run.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                sample = _parse_record(record, idx)
                samples.append(sample)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                logger.warning("Skipping record %d — %s: %s", idx, type(exc).__name__, exc)

    logger.info("Loaded %d evaluation samples from %s", len(samples), path)
    return samples


def iter_eval_dataset(path: str | Path) -> Iterator[EvalSample]:
    """Lazy iterator version — yields one sample at a time."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                yield _parse_record(record, idx)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                logger.warning("Skipping record %d — %s: %s", idx, type(exc).__name__, exc)
