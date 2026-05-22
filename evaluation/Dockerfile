# Dockerfile for the RAG Evaluation Framework
#
# Builds a self-contained image that runs the evaluation runner against a
# running RAG API endpoint.
#
# Build:
#   docker build -t rag-eval -f evaluation/Dockerfile .
#
# Run (from repo root):
#   docker run --rm \
#     -e ANTHROPIC_API_KEY=sk-ant-... \
#     -e API_URL=http://host.docker.internal:8000 \
#     -e EVAL_DATASET_PATH=/data/evaluation_data/input.jsonl \
#     -v $(pwd)/evaluation_data:/data/evaluation_data:ro \
#     -v $(pwd)/evaluation/results:/app/evaluation/results \
#     rag-eval

FROM python:3.11-slim

WORKDIR /app

# System dependencies (gcc needed by some transitive deps of sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY rag_pipeline/requirements.txt /tmp/base_requirements.txt
COPY evaluation/requirements.txt /tmp/eval_requirements.txt
RUN pip install --no-cache-dir -r /tmp/base_requirements.txt \
 && pip install --no-cache-dir -r /tmp/eval_requirements.txt

# Copy the full packages so imports work
COPY rag_pipeline/ ./rag_pipeline/
COPY evaluation/ ./evaluation/

# Results are written here; mount a host volume to persist them
RUN mkdir -p /app/evaluation/results

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default: run the evaluation runner with --help to confirm it works.
# Override CMD (or use docker run arguments) to pass real flags:
#   docker run rag-eval python -m evaluation.runner \
#       --dataset /data/evaluation_data/input.jsonl --top-k 5
ENTRYPOINT ["python", "-m", "evaluation.runner"]
CMD ["--help"]
