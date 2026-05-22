import logging
from pathlib import Path
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
_embeddings_model: Optional[SentenceTransformer] = None


def load_embedding_model(model_name: str = _DEFAULT_MODEL) -> SentenceTransformer:
    global _embeddings_model
    if _embeddings_model is None:
        logger.info(f"Loading embedding model: {model_name}")
        _embeddings_model = SentenceTransformer(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embeddings_model.to(device)
        logger.info(f"Embedding model loaded on device: {device}")
    return _embeddings_model


def generate_embeddings(texts: list[str], model_name: Optional[str] = None) -> list[list[float]]:
    if not texts:
        return []

    model = load_embedding_model(model_name or _DEFAULT_MODEL)

    logger.debug(f"Generating embeddings for {len(texts)} texts")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    return [emb.tolist() for emb in embeddings]


def get_embedding_dimension(model_name: str = _DEFAULT_MODEL) -> int:
    model = load_embedding_model(model_name)
    return model.get_sentence_embedding_dimension()