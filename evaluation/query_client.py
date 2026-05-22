"""
Async HTTP client for the RAG pipeline FastAPI /query endpoint.

Uses httpx with a semaphore-based concurrency limit so we can batch requests
without overwhelming the API server.
"""

import asyncio
import logging
from dataclasses import dataclass

import httpx

from .config import EvalSettings
from .response_generator import generate_answer

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Response from a single /query call."""

    question: str
    answer: str
    contexts: list[str]          # plain text of each retrieved chunk
    chunk_ids: list[str]         # id field of each retrieved chunk
    metadata: list[dict]         # full metadata per chunk
    raw_response: dict


class QueryClient:
    """
    Thin async wrapper around the FastAPI /query endpoint.

    Usage:
        async with QueryClient(settings) as client:
            result = await client.query("What is VAT?", top_k=5)
    """

    def __init__(self, settings: EvalSettings) -> None:
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "QueryClient":
        self._client = httpx.AsyncClient(
            base_url=self._settings.api_url,
            timeout=self._settings.request_timeout_seconds,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()

    async def query(self, question: str, top_k: int = 5) -> QueryResult:
        """
        POST to /query and return a structured QueryResult.

        Raises httpx.HTTPStatusError on non-2xx responses.
        """
        if self._client is None:
            raise RuntimeError("QueryClient must be used as an async context manager")

        async with self._semaphore:
            logger.debug("Querying API: %s", question[:80])
            response = await self._client.post(
                "/query",
                json={"query": question, "top_k": top_k},
            )
            response.raise_for_status()
            data = response.json()

            results: list[dict] = data.get("results", [])

            contexts = [r.get("content", "") for r in results]
            chunk_ids = [r.get("id", "") for r in results]
            metadata = [r.get("metadata", {}) for r in results]

            answer = await asyncio.to_thread(
                generate_answer, question, contexts, self._settings
            )

        logger.info(
            "Query complete | question='%s...' | chunks=%d | answer_len=%d",
            question[:60],
            len(results),
            len(answer),
        )

        return QueryResult(
            question=question,
            answer=answer,
            contexts=contexts,
            chunk_ids=chunk_ids,
            metadata=metadata,
            raw_response=data,
        )


async def batch_query(
    questions: list[str],
    settings: EvalSettings,
    top_k: int = 5,
) -> list[QueryResult | Exception]:
    """
    Run multiple queries concurrently respecting the concurrency limit.

    Returns a list parallel to `questions`.  Individual failures are returned
    as Exception instances rather than raising, so one bad query cannot abort
    the whole batch.
    """
    async with QueryClient(settings) as client:
        tasks = [client.query(q, top_k=top_k) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(
                "Query failed for question %d ('%s...'): %s",
                i,
                questions[i][:60],
                result,
            )

    return list(results)
