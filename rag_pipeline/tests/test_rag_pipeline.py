import json
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_pipeline.chunker import Document, load_documents, semantic_chunk, chunk_documents
from rag_pipeline.embedder import generate_embeddings, get_embedding_dimension
from rag_pipeline.vector_store import VectorStore
from rag_pipeline.retriever import retrieve_and_rank, rerank_results, RetrievedChunk


@pytest.fixture
def sample_docs_path():
    docs = [
        {"id": "doc1", "content": "Python is a programming language. It is widely used."},
        {"id": "doc2", "content": "Machine learning is a subset of artificial intelligence."},
        {"id": "doc3", "content": "FastAPI is a modern web framework for building APIs."},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


class TestDocumentLoading:
    def test_load_documents(self, sample_docs_path):
        docs = list(load_documents(sample_docs_path))
        assert len(docs) == 3
        assert docs[0].id == "doc1"
        assert "Python" in docs[0].content

    def test_load_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = f.name
        try:
            docs = list(load_documents(temp_path))
            assert len(docs) == 0
        finally:
            os.unlink(temp_path)

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            list(load_documents("/nonexistent/path.jsonl"))


class TestSemanticChunking:
    def test_semantic_chunk_basic(self):
        doc = Document(
            id="test1",
            content="This is the first sentence. This is the second sentence. This is the third sentence.",
            metadata={},
        )
        chunks = semantic_chunk(doc, chunk_size=10, overlap=2)
        assert len(chunks) > 0
        assert all(c.content for c in chunks)

    def test_semantic_chunk_empty(self):
        doc = Document(id="test1", content="", metadata={})
        chunks = semantic_chunk(doc)
        assert len(chunks) == 0

    def test_chunk_documents(self, sample_docs_path):
        chunks = chunk_documents(sample_docs_path, chunk_size=10)
        assert len(chunks) > 0


class TestEmbeddings:
    def test_generate_embeddings(self):
        texts = ["Hello world", "Python is great"]
        embeddings = generate_embeddings(texts)
        assert len(embeddings) == 2
        assert all(len(emb) == get_embedding_dimension() for emb in embeddings)

    def test_generate_embeddings_empty(self):
        embeddings = generate_embeddings([])
        assert len(embeddings) == 0


class TestVectorStore:
    def test_vector_store_basic(self):
        docs = [
            Document(id="d1", content="First document", metadata={}),
            Document(id="d2", content="Second document", metadata={}),
        ]
        store = VectorStore(dimension=get_embedding_dimension())
        store.add_documents(docs)
        assert store.index is not None
        assert store.index.ntotal == 2

    def test_vector_store_search(self):
        docs = [
            Document(id="d1", content="Apple makes phones", metadata={}),
            Document(id="d2", content="Google builds search", metadata={}),
        ]
        store = VectorStore(dimension=get_embedding_dimension())
        store.add_documents(docs)

        results = store.search("phones", k=1)
        assert len(results) >= 1

    def test_vector_store_save_load(self):
        docs = [Document(id="d1", content="Test content", metadata={})]
        store = VectorStore(dimension=get_embedding_dimension())
        store.add_documents(docs)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "index"
            store.save(index_path)

            new_store = VectorStore(dimension=get_embedding_dimension())
            new_store.load(index_path)
            assert new_store.index.ntotal == 1


class TestRetriever:
    def test_rerank_results(self):
        results = [
            RetrievedChunk(id="d1", content="test", score=0.5, metadata={}),
            RetrievedChunk(id="d2", content="test", score=0.9, metadata={}),
            RetrievedChunk(id="d3", content="test", score=0.3, metadata={}),
        ]
        reranked = rerank_results("query", results)
        assert reranked[0].score == 0.9
        assert reranked[2].score == 0.3


class TestAPI:
    @pytest.fixture
    def client(self):
        from rag_pipeline.api import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model" in data

    def test_query_get_endpoint(self, client):
        response = client.get("/query", params={"q": "test query"})
        assert response.status_code == 200

    def test_query_empty_validation(self, client):
        response = client.get("/query", params={"q": ""})
        assert response.status_code == 422

    def test_query_post_endpoint(self, client):
        response = client.post("/query", json={"query": "test"})
        assert response.status_code == 200