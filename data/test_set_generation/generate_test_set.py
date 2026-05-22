#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import fitz
import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from rag_pipeline.chunker import Document, semantic_chunk
from rag_pipeline.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
DATA_DIR = Path(__file__).parent / "data"

DOCUMENTS = [
    {"title": "VAT 404 Guide", "url": "https://www.sars.gov.za/wp-content/uploads/Ops/Guides/Legal-Pub-Guide-VAT404-VAT-404-Guide-for-Vendors.pdf"},
    {"title": "Income Tax Guide", "url": "https://www.sars.gov.za/wp-content/uploads/Ops/Guides/PAYE-GEN-01-G20-Guide-for-Employers-iro-Employees-Tax-for-2026-External-Guide.pdf"},
    {"title": "Residence Based Taxation Guide", "url": "https://www.sars.gov.za/wp-content/uploads/Ops/Guides/LAPD-IT-G02-Guide-on-the-Residence-Basis-of-Taxation-for-Individuals.pdf"},
    {"title": "income Tax Guide Individual", "url": "https://www.sars.gov.za/wp-content/uploads/Ops/Guides/Legal-Pub-Guide-IT01-Guide-on-Income-Tax-and-the-Individual.pdf"},
    {"title": "VAT 201 Guide", "url": "https://www.sars.gov.za/wp-content/uploads/Ops/Guides/GEN-ELEC-04-G01-Guide-for-completing-the-Value-Added-Tax-VAT201-Declaration-External-Guide.pdf"},
]

QA_SAMPLES_PER_DOC = 74


def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def download_pdf(url, output_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path


def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    return "\n\n".join(p for p in pages if p.strip())


# ---------------------------------------------------------------------------
# Step 1: Chunk documents
# ---------------------------------------------------------------------------

def step_chunk(force):
    output_path = DATA_DIR / "chunks.jsonl"
    if output_path.exists() and not force:
        logger.info("Step [chunk] — chunks.jsonl exists, skipping (use --force chunk to re-run)")
        return

    logger.info("Step [chunk] — downloading PDFs and chunking")
    settings = get_settings()
    seen_texts: set[str] = set()

    with open(output_path, "w") as f:
        for doc in DOCUMENTS:
            title = doc["title"]
            url = doc["url"]
            logger.info("  Processing: %s", title)

            pdf_path = f"/tmp/{title}.pdf"
            download_pdf(url, pdf_path)
            full_text = extract_pdf_text(pdf_path)
            os.remove(pdf_path)

            doc_id = title.lower().replace(" ", "_")
            chunks = semantic_chunk(
                Document(id=doc_id, content=full_text, metadata={"title": title, "url": url}),
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )

            written = 0
            for c in chunks:
                if len(c.content.split()) < 30:
                    continue
                if c.content in seen_texts:
                    continue
                seen_texts.add(c.content)
                record = {"chunk_id": c.id, "title": title, "url": url, "text": c.content}
                f.write(json.dumps(record) + "\n")
                written += 1

            logger.info("  -> %d chunks (%d filtered)", written, len(chunks) - written)


# ---------------------------------------------------------------------------
# Step 2: Generate QA pairs
# ---------------------------------------------------------------------------

def clean_json_response(text):
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def generate_qa(client, chunk_text):
    prompt = f"""You are generating evaluation data for a RAG system.

Context:
{chunk_text}

Generate 3 question-answer pairs.

Rules:
- Answers must be strictly grounded in the text
- Return ONLY valid JSON:
[
  {{"question": "...", "answer": "..."}}
]"""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def validate_qa(client, chunk_text, question, answer):
    prompt = f"""Context:
{chunk_text}

Question: {question}
Answer: {answer}

Is the answer fully supported by the context?

Reply with only YES or NO."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return "YES" in response.content[0].text.upper()


def step_qa(force):
    chunks_path = DATA_DIR / "chunks.jsonl"
    output_path = DATA_DIR / "rag_dataset.jsonl"

    if not chunks_path.exists():
        logger.error("Step [qa] — chunks.jsonl not found. Run step 'chunk' first.")
        sys.exit(1)

    if output_path.exists() and not force:
        logger.info("Step [qa] — rag_dataset.jsonl exists, skipping (use --force qa to re-run)")
        return

    logger.info("Step [qa] — generating QA pairs via LLM")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    chunks = load_jsonl(chunks_path)

    for chunk in chunks:
        logger.info("  Processing chunk: %s", chunk["chunk_id"])
        raw = generate_qa(client, chunk["text"])

        try:
            qa_pairs = json.loads(clean_json_response(raw))
        except Exception as e:
            logger.warning("  Failed to parse LLM response: %s", e)
            continue

        for qa in qa_pairs:
            if validate_qa(client, chunk["text"], qa["question"], qa["answer"]):
                record = {
                    "title": qa["question"],
                    "text": qa["answer"],
                    "url": chunk["url"],
                    "chunk_id": chunk["chunk_id"],
                    "source_title": chunk["title"],
                }
                with open(output_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

                logger.info("  Valid QA: %s", qa["question"][:60])


# ---------------------------------------------------------------------------
# Step 3: Split into test / holdout
# ---------------------------------------------------------------------------

def step_split(force):
    input_path = DATA_DIR / "rag_dataset.jsonl"
    test_path = DATA_DIR / "test_set.jsonl"
    holdout_path = DATA_DIR / "holdout_set.jsonl"

    if not input_path.exists():
        logger.error("Step [split] — rag_dataset.jsonl not found. Run step 'qa' first.")
        sys.exit(1)

    if test_path.exists() and holdout_path.exists() and not force:
        logger.info("Step [split] — test/holdout sets exist, skipping (use --force split to re-run)")
        return

    logger.info("Step [split] — splitting into test (k=%d/doc) and holdout", QA_SAMPLES_PER_DOC)
    records = load_jsonl(input_path)

    grouped = defaultdict(list)
    for row in records:
        grouped[row["url"]].append(row)

    test_set, holdout_set = [], []
    for doc, rows in grouped.items():
        random.shuffle(rows)
        test_set.extend(rows[:QA_SAMPLES_PER_DOC])
        holdout_set.extend(rows[QA_SAMPLES_PER_DOC:])

    save_jsonl(test_set, test_path)
    save_jsonl(holdout_set, holdout_path)
    logger.info("  Test: %d  |  Holdout: %d", len(test_set), len(holdout_set))


# ---------------------------------------------------------------------------
# Step 4: Merge QA with chunk context
# ---------------------------------------------------------------------------

def step_merge(force):
    test_path = DATA_DIR / "test_set.jsonl"
    chunks_path = DATA_DIR / "chunks.jsonl"
    output_path = DATA_DIR / "rag_dataset_with_context.jsonl"

    if not test_path.exists() or not chunks_path.exists():
        logger.error("Step [merge] — test_set.jsonl or chunks.jsonl not found. Run steps 'chunk' and 'split' first.")
        sys.exit(1)

    if output_path.exists() and not force:
        logger.info("Step [merge] — rag_dataset_with_context.jsonl exists, skipping (use --force merge to re-run)")
        return

    logger.info("Step [merge] — merging QA with chunk context")
    qa_data = load_jsonl(test_path)
    chunks = {c["chunk_id"]: c for c in load_jsonl(chunks_path)}

    merged, missing = [], 0
    for row in qa_data:
        chunk = chunks.get(row.get("chunk_id"))
        if not chunk:
            missing += 1
            continue
        merged.append({
            "question": row["title"],
            "ground_truth": row["text"],
            "context": chunk["text"],
            "chunk_id": row["chunk_id"],
            "url": row["url"],
            "source_title": row["source_title"],
        })

    save_jsonl(merged, output_path)
    logger.info("  Merged: %d  |  Missing chunks: %d", len(merged), missing)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

STEPS = ["chunk", "qa", "split", "merge"]


def main():
    parser = argparse.ArgumentParser(description="Generate RAG evaluation test set")
    parser.add_argument("--step", choices=STEPS, default="chunk",
                        help="Step to start from (default: chunk — run full pipeline)")
    parser.add_argument("--force", nargs="*", choices=STEPS, default=None,
                        help="Force re-run specific steps (e.g. --force chunk qa)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    force_set = set(args.force or [])
    start = STEPS.index(args.step)

    runners = [step_chunk, step_qa, step_split, step_merge]

    for i in range(start, len(STEPS)):
        step_name = STEPS[i]
        runners[i](force=step_name in force_set)


if __name__ == "__main__":
    main()
