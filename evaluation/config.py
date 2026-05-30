"""
Evaluation framework configuration.
All settings can be overridden via environment variables or the .env file.
"""

from pathlib import Path
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class EvalSettings(BaseSettings):
    model_config = ConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # API
    api_url: str = "http://localhost:8000"

    # Dataset
    eval_dataset_path: str = "evaluation_data/input.jsonl"

    # Embedding model (for RAGAS AnswerRelevancy and any custom embedding steps)
    embedding_model: str = "intfloat/e5-large-v2"

    # LLM judge for RAGAS (Anthropic Claude)
    anthropic_api_key: str = ""
    ragas_llm_model: str = "claude-sonnet-4-5"

    # LLM for answer generation (different from judge to avoid circular evaluation)
    answer_generator_llm_model: str = "claude-haiku-4-5"

    # LLM API retry settings
    max_llm_retries: int = 5
    initial_retry_delay_seconds: float = 1.0

    # Output
    results_dir: str = str(
        Path(__file__).parent / "results"
    )

    # HTTP client
    request_timeout_seconds: int = 30
    max_concurrent_requests: int = 2

    # Evaluation failure handling
    max_allowed_failures: int = 10

    # Logging
    log_level: str = "INFO"


def get_eval_settings() -> EvalSettings:
    return EvalSettings()
