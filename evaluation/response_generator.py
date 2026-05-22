import logging
import random
import time

from anthropic import APIStatusError, Anthropic, RateLimitError

from .config import EvalSettings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = "You are a helpful tax advisor answering questions about South African tax law. Answer concisely and accurately using only the provided context."


def _is_retryable_error(exc: Exception) -> bool:
    """Check if an error should be retried."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, 'status_code', None)
        if status in {429, 500, 502, 503, 504}:
            return True
    return False


def generate_answer(
    question: str,
    contexts: list[str],
    settings: EvalSettings,
) -> str:
    if not contexts:
        return ""

    context_block = "\n\n".join(contexts)
    prompt = f"""Context:
{context_block}

Question: {question}

Answer the question using only the context above. If the context does not contain enough information, say so."""

    client = Anthropic(api_key=settings.anthropic_api_key)

    last_exc: Exception | None = None

    for attempt in range(settings.max_llm_retries):
        try:
            response = client.messages.create(
                model=settings.answer_generator_llm_model,
                max_tokens=1024,
                temperature=0,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            answer = response.content[0].text
            logger.debug("Generated answer (%d chars) for question: %s", len(answer), question[:60])
            return answer

        except Exception as e:
            last_exc = e
            if not _is_retryable_error(e) or attempt == settings.max_llm_retries - 1:
                break

            delay = settings.initial_retry_delay_seconds * (2 ** attempt)
            jitter = random.uniform(0, delay * 0.1)
            sleep_time = delay + jitter

            logger.warning(
                "Anthropic API error (attempt %d/%d): %s. Retrying in %.2fs...",
                attempt + 1,
                settings.max_llm_retries,
                str(e)[:100],
                sleep_time,
            )
            time.sleep(sleep_time)

    logger.error(
        "Failed to generate answer after %d attempts for question: %s. Error: %s",
        settings.max_llm_retries,
        question[:60],
        str(last_exc)[:200] if last_exc else "unknown",
    )

    if last_exc is not None:
        raise last_exc

    raise RuntimeError("Failed to generate answer: unknown error")
