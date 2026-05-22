# tax-rag

A Retrieval-Augmented Generation (RAG) system for answering questions about South African tax law, specifically the VAT 404 Guide from SARS.

## Project Structure

```
tax-rag/
├── rag_pipeline/           # Core RAG retrieval system (FAISS + FastAPI)
│   ├── main.py            # Entry point - starts the API server
│   ├── api.py             # FastAPI endpoints (/query, /health)
│   ├── retriever.py       # Retrieval and ranking logic
│   ├── vector_store.py    # FAISS vector storage
│   ├── chunker.py         # Document chunking
│   ├── embedder.py        # Embedding generation
│   ├── config.py          # Configuration
│   └── README.md          # Detailed RAG pipeline docs
│
├── evaluation/             # RAGAS-based evaluation framework
│   ├── runner.py          # Evaluation entry point
│   ├── ragas_evaluator.py # RAGAS metrics computation
│   ├── response_generator.py # LLM answer generation
│   ├── retrieval_metrics.py # Recall@k / Hit Rate@k
│   ├── query_client.py    # API client for RAG pipeline
│   ├── dataset.py         # Test set loading
│   ├── config.py          # Evaluation configuration
│   └── README.md          # Detailed evaluation docs
│
└── data/                   # Data directory (gitignored)
    ├── input.jsonl        # Documents to index
    ├── faiss_index        # Generated FAISS index
    └── evaluation/        # Test sets for evaluation
```

## Quick Start

### Prerequisites

- Python 3.10+
- Anthropic API key (for evaluation)

### 1. Install Dependencies

```bash
pip install -r rag_pipeline/requirements.txt
pip install -r evaluation/requirements.txt
```

### 2. Prepare Documents

Place your documents in `data/input.jsonl` (JSONL format):

```json
{"id": "doc1", "content": "Document text here...", "metadata": {"url": "..."}}
```

### 3. Start the RAG Pipeline

```bash
python -m rag_pipeline.main
```

API will be available at `http://localhost:8000`.

### 4. Run Evaluation

```bash
python -m evaluation.runner --dataset ./data/evaluation/test_set.jsonl --top-k 5
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Required for LLM calls during evaluation |
| `HF_TOKEN` | Optional, for HuggingFace models |

See individual READMEs for full configuration options:
- [RAG Pipeline Config](rag_pipeline/README.md#configuration)
- [Evaluation Config](evaluation/README.md#configuration)

## Features

| Component | Description |
|-----------|-------------|
| **RAG Pipeline** | FAISS vector store, semantic chunking, FastAPI endpoints |
| **Evaluation** | RAGAS metrics (faithfulness, relevancy) + retrieval metrics (Recall@k) |
| **Rate Limiting** | Exponential backoff retry for Anthropic API |

## Testing

```bash
# Test RAG pipeline
pytest rag_pipeline/tests/

# Test evaluation framework
pytest evaluation/tests/

# Live RAGAS integration test (requires ANTHROPIC_API_KEY)
pytest evaluation/tests/ -m integration
```
