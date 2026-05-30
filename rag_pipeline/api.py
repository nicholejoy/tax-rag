import logging
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from .config import get_settings
from .retriever import retrieve_and_rank, RetrievedChunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="RAG Pipeline API",
    description="Production-ready RAG pipeline for document retrieval and answering",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    min_score: Optional[float] = None


class QueryResponse(BaseModel):
    query: str
    results: list[dict]
    total_results: int


class HealthResponse(BaseModel):
    status: str
    model: str
    index_loaded: bool


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    from .vector_store import get_vector_store

    store = get_vector_store()
    index_loaded = store.index is not None and store.index.ntotal > 0

    return HealthResponse(
        status="healthy" if index_loaded else "initializing",
        model=settings.embedding_model,
        index_loaded=index_loaded,
    )


@app.post("/documents/upload")
async def upload_documents(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Only .jsonl files are accepted")

    contents = await file.read()
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".jsonl", delete=False)
    try:
        tmp.write(contents)
        tmp.close()

        from .chunker import chunk_documents
        from .vector_store import get_vector_store

        chunks = chunk_documents(tmp.name, settings.chunk_size, settings.chunk_overlap)
        store = get_vector_store()
        store.add_documents(chunks)
        store.save(settings.faiss_index_path)

        return {"status": "ok", "documents_added": len(chunks)}
    finally:
        os.unlink(tmp.name)


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest) -> QueryResponse:
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    top_k = request.top_k if request.top_k is not None else settings.default_top_k
    min_score = request.min_score if request.min_score is not None else settings.min_similarity_score

    try:
        results = retrieve_and_rank(
            query=request.query,
            top_k=top_k,
            min_score=min_score,
            reranker_model=settings.reranker_model,
            reranker_enabled=settings.reranker_enabled,
        )

        return QueryResponse(
            query=request.query,
            results=[
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results
            ],
            total_results=len(results),
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/query", response_model=QueryResponse)
async def query_get(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20),
) -> QueryResponse:
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = retrieve_and_rank(
            query=q,
            top_k=top_k,
            min_score=settings.min_similarity_score,
            reranker_model=settings.reranker_model,
            reranker_enabled=settings.reranker_enabled,
        )

        return QueryResponse(
            query=q,
            results=[
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results
            ],
            total_results=len(results),
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")