# RAG Evaluation Framework

Evaluates a running RAG pipeline by sending questions from a test set and
computing retrieval metrics (Recall@k / Hit Rate@k) and RAGAS quality metrics
(faithfulness, answer relevancy, context precision, context recall).

## What This Evaluates

This framework evaluates **retrieval quality via proxy**, not an end-to-end RAG system:

1. **Retrieval**: Queries your running RAG API (`/query` endpoint) to get relevant chunks
2. **Answer Generation**: Uses a separate LLM (`claude-haiku-4-5`) to generate answers from the retrieved chunks
3. **Judging**: Uses a different LLM (`claude-sonnet-4-5`) as a judge for RAGAS metrics

The answer generation step simulates how a real RAG pipeline would use an LLM to answer
from retrieved context. Different models are used for generation vs. judging to avoid
circular evaluation where a model would be judging its own outputs.

If your actual RAG system includes answer generation, you would eventually want to test
that full pipeline instead.

## Prerequisites

- RAG API running at `http://localhost:8000` (or configured `API_URL`)
- Anthropic API key for RAGAS LLM-based metrics
- Evaluation dataset in JSONL format

## Quickstart

```bash
cd rag_pipeline
python -m rag_pipeline.main &

cd ../evaluation
pip install -r requirements.txt
python -m evaluation.runner --dataset ./data/test_set.jsonl --top-k 5
```

## Dataset format

Each line is a JSON record with the following fields. Two naming conventions
are supported:

| Spec layout | Alt layout | Description |
|---|---|---|
| `question` | `title` | The user query |
| `ground_truth` | `text` | The reference answer |
| `answer` | _(optional)_ | Pre-generated model answer (used directly when present) |
| `chunk_id` | | Expected ground-truth chunk ID |
| `url` | | Source URL |
| `source_title` | | Source document title |

See `data/test_set_generation/generate_test_set.py` for generating datasets.

## CLI options

```
--dataset PATH             Path to JSONL evaluation dataset
--top-k INT                Chunks to retrieve per query (default: 5)
--limit INT                Max samples to evaluate (0 = all)
--output-dir PATH          Results directory (default: evaluation/results/)
--skip-ragas               Retrieval metrics only, skip RAGAS
--skip-retrieval-metrics   Skip Recall@k / Hit Rate@k
--match-strategy STR       "content" (default) or "chunk_id"
--match-threshold FLOAT    Jaccard overlap for content hit (default: 0.5)
```

### Match strategies

- **`content`** (default) — token-overlap between ground truth and retrieved
  chunks. Robust to re-chunking since it doesn't rely on chunk IDs.
- **`chunk_id`** — exact ID match. Only reliable when the live index was built
  from the same chunking run as the dataset.

## Output

Results are written to `--output-dir`:

| File | Contents |
|---|---|
| `retrieval_metrics.json` | Aggregated Recall@k and Hit Rate@k |
| `ragas_metrics.json` | Aggregated RAGAS scores |
| `full_detailed_results.csv` | Per-sample breakdown (all metrics + metadata) |

## Configuration

All settings via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://localhost:8000` | RAG API endpoint |
| `EVAL_DATASET_PATH` | `evaluation_data/input.jsonl` | Test set location |
| `ANTHROPIC_API_KEY` | — | Required for RAGAS |
| `RAGAS_LLM_MODEL` | `claude-sonnet-4-5` | LLM judge model for RAGAS metrics |
| `ANSWER_GENERATOR_LLM_MODEL` | `claude-haiku-4-5` | LLM for answer generation (separate from judge) |
| `MAX_LLM_RETRIES` | `5` | Max retry attempts for rate-limited API calls |
| `INITIAL_RETRY_DELAY_SECONDS` | `1.0` | Initial delay before first retry (exponential backoff) |
| `MAX_ALLOWED_FAILURES` | `10` | Abort if this many queries fail |
| `RESULTS_DIR` | `evaluation/results/` | Output directory |
| `MAX_CONCURRENT_REQUESTS` | `2` | Concurrency limit for LLM API calls |

## Docker

```bash
docker build -t rag-eval -f evaluation/Dockerfile .
docker run --rm \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e API_URL=http://host.docker.internal:8000 \
  -v $(pwd)/evaluation_data:/data/evaluation_data:ro \
  -v $(pwd)/evaluation/results:/app/evaluation/results \
  rag-eval --top-k 5
```

## Testing

```bash
pytest evaluation/tests/
# Live RAGAS integration test (requires ANTHROPIC_API_KEY):
pytest evaluation/tests/ -m integration
```
