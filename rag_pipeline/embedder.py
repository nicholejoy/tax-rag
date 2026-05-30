import logging
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_embeddings_model: Optional[SentenceTransformer] = None


def _get_prefix(model_name: str, is_query: bool) -> str:
    # E5 models require explicit query/passage prefixes for correct retrieval behavior
    if "e5" in model_name.lower():
        return "query: " if is_query else "passage: "
    return ""


def load_embedding_model(model_name: str) -> SentenceTransformer:
    global _embeddings_model
    if _embeddings_model is None:
        logger.info(f"Loading embedding model: {model_name}")
        _embeddings_model = SentenceTransformer(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embeddings_model.to(device)
        logger.info(f"Embedding model loaded on device: {device}")
    return _embeddings_model


def generate_embeddings(
    texts: list[str],
    model_name: Optional[str] = None,
    is_query: bool = False,
) -> list[list[float]]:
    if not texts:
        return []

    from .config import get_settings
    effective_model = model_name or get_settings().embedding_model
    model = load_embedding_model(effective_model)

    prefix = _get_prefix(effective_model, is_query)
    prefixed = [prefix + t for t in texts] if prefix else texts

    logger.debug(f"Generating embeddings for {len(texts)} texts")
    embeddings = model.encode(prefixed, convert_to_numpy=True, show_progress_bar=False)

    return [emb.tolist() for emb in embeddings]


def get_embedding_dimension(model_name: Optional[str] = None) -> int:
    from .config import get_settings
    effective_model = model_name or get_settings().embedding_model
    model = load_embedding_model(effective_model)
    return model.get_sentence_embedding_dimension()