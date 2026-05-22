import logging
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import numpy.typing as npt

from .chunker import Document
from .embedder import generate_embeddings

logger = logging.getLogger(__name__)

_index: Optional[faiss.Index] = None
_documents: list[Document] = []


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self.documents: list[Document] = []

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return

        texts = [doc.content for doc in documents]
        embeddings = generate_embeddings(texts)

        embedding_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embedding_array)

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embedding_array)
        else:
            self.index.add(embedding_array)

        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents to vector store. Total: {len(self.documents)}")

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = generate_embeddings([query])
        query_vector = np.array(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_vector)

        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                results.append((self.documents[int(idx)], float(dist)))

        return results

    def clear(self) -> None:
        self.index = None
        self.documents = []
        logger.info("Cleared vector store")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(path))
        logger.info(f"Saved index to {path}")

        docs_path = path.with_suffix(".docs.jsonl")
        import json
        with open(docs_path, "w", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(json.dumps({
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                }) + "\n")
        logger.info(f"Saved {len(self.documents)} documents to {docs_path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        self.index = faiss.read_index(str(path))
        logger.info(f"Loaded index from {path}")

        docs_path = path.with_suffix(".docs.jsonl")
        import json
        self.documents = []
        if docs_path.exists():
            with open(docs_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self.documents.append(Document(
                            id=data["id"],
                            content=data["content"],
                            metadata=data["metadata"],
                        ))
        logger.info(f"Loaded {len(self.documents)} documents")


_vector_store: Optional[VectorStore] = None


def get_vector_store(index_path: Optional[str] = None) -> VectorStore:
    global _vector_store
    if _vector_store is None:
        from .embedder import get_embedding_dimension
        dim = get_embedding_dimension()
        _vector_store = VectorStore(dimension=dim)

        if index_path and Path(index_path).exists():
            _vector_store.load(index_path)
        else:
            logger.info("Initialized new vector store")

    return _vector_store


def initialize_vector_store(
    index_path: Optional[str] = None,
) -> VectorStore:
    logger.info("Initializing vector store from existing index")
    store = get_vector_store(index_path)
    if index_path and Path(index_path).exists() and store.index is not None:
        logger.info(f"Loaded existing index with {len(store.documents)} documents")
    else:
        logger.info("No existing index found — initialized empty vector store")
    return store


def refresh_vector_store(
    docs_path: str | Path,
    index_path: Optional[str] = None,
) -> VectorStore:
    from .chunker import chunk_documents

    logger.info(f"Refreshing vector store from {docs_path}")
    chunks = chunk_documents(docs_path)

    store = get_vector_store()
    store.clear()
    store.add_documents(chunks)

    if index_path:
        store.save(index_path)

    logger.info(f"Vector store refreshed with {len(chunks)} chunks")
    return store