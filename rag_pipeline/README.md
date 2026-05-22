# RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline in Python.

## Features

- **Document Ingestion**: Load documents from JSONL format (`data/input.jsonl`)
- **Semantic Chunking**: Split documents into meaningful chunks using sentence-based semantic chunking
- **Embeddings**: Generate text embeddings using HuggingFace Sentence Transformers
- **Vector Storage**: Store and search vectors using FAISS (local index)
- **Retrieval & Ranking**: Retrieve relevant chunks with similarity scores
- **FastAPI Endpoint**: Query the pipeline via REST API (`/query`)
- **Logging & Error Handling**: Comprehensive logging configured

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies

## Installation

```bash
cd rag_pipeline
pip install -r requirements.txt
```


## Usage

### Start the API Server

```bash
python -m rag_pipeline.main
```

The API will be available at `http://localhost:8000`.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check - returns index status and model info |
| `/query` | POST | Query the pipeline with a request body |
| `/query?q=...` | GET | Query the pipeline with query parameters |

#### Query Examples

**POST**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "your question here", "top_k": 5}'
```

**GET**:
```bash
curl "http://localhost:8000/query?q=your%20question%20here&top_k=5"
```

## Configuration

Configuration is managed via `config.py` using Pydantic Settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding_model` | `sentence-transformers/all-mpnet-base-v2` | HuggingFace model for embeddings |
| `faiss_index_path` | `./data/faiss_index` | Path to save/load FAISS index |
| `docs_path` | `./data/input.jsonl` | Source documents file |
| `chunk_size` | 512 | Maximum tokens per chunk |
| `chunk_overlap` | 64 | Overlap between chunks |
| `default_top_k` | 5 | Default number of results to return |

## Docker Deployment

```bash
docker-compose up --build
```

## Project Structure

```
tax-rag/
├── data/
│   ├── input.jsonl            # Source documents consumed by rag_pipeline
│   ├── preprocess/            # Raw data ingestion
│   │   ├── fetch_pdfs.py     # Download PDFs → data/input.jsonl
│   ├── test_set_generation/   # One-shot test set pipeline
│   │   ├── generate_test_set.py# CLI: chunk → QA → split → merge
│   │   ├── requirements.txt
│   │   └── data/               # Generated datasets
│   │       ├── chunks.jsonl
│   │       ├── rag_dataset.jsonl
│   │       ├── test_set.jsonl
│   │       ├── holdout_set.jsonl
│   │       └── rag_dataset_with_context.jsonl
│   ├── faiss_index             # FAISS vector index
│   └── faiss_index.docs.jsonl
├── evaluation/               # Evaluation framework (standalone top-level package)
│   ├── runner.py             # Entry point for evaluation runs
│   ├── query_client.py       # Async HTTP client for /query endpoint
│   ├── dataset.py            # Dataset loader and EvalSample dataclass
│   ├── ragas_evaluator.py    # RAGAS metric computation
│   ├── retrieval_metrics.py  # Recall@k / Hit Rate@k
│   ├── config.py             # Evaluation configuration
│   ├── results/              # Evaluation output directory
│   ├── tests/                # Evaluation framework tests
│   └── Dockerfile
├── rag_pipeline/
│   ├── api.py                  # FastAPI application and endpoints
│   ├── config.py               # Configuration settings
│   ├── chunker.py              # Document loading and semantic chunking
│   ├── embedder.py              # Embedding generation
│   ├── retriever.py            # Retrieval and ranking logic
│   ├── vector_store.py          # FAISS vector store implementation
│   ├── main.py                 # Application entry point
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Docker image definition
│   ├── docker-compose.yml      # Docker Compose configuration
│   ├── prompts/
│   └── tests/                  # Pipeline test suite
```

## Testing

```bash
pytest rag_pipeline/tests/
```