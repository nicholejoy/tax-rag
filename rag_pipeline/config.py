import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(
        extra="ignore",          # ✅ this is what you want
        env_file=".env",
        env_file_encoding="utf-8",
    )
    hf_token: str | None = None
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    faiss_index_path: str = "./data/faiss_index"
    docs_path: str = "./data/input.jsonl"
    log_level: str = "INFO"
    chunk_size: int = 350
    chunk_overlap: int = 64
    default_top_k: int = 5
    min_similarity_score: float = 0.0


def get_settings() -> Settings:
    return Settings()