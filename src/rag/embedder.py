from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.llm.retry import is_transient_error

load_dotenv(Path(__file__).parents[2] / ".env")

logger = logging.getLogger(__name__)

MODEL = "models/gemini-embedding-001"


def _get_client():
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genai가 필요합니다: pip install google-genai") from exc

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    return genai.Client(api_key=api_key)


def _log_retry(state) -> None:
    logger.warning(
        "Gemini 임베딩 일시 오류 — 재시도 %d/5 (%s)",
        state.attempt_number,
        state.outcome.exception() if state.outcome else "unknown",
    )


@retry(
    retry=retry_if_exception(is_transient_error),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    before_sleep=_log_retry,
    reraise=True,
)
def _embed_content(client, contents: list[str]):
    return client.models.embed_content(model=MODEL, contents=contents)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """텍스트 리스트를 벡터 리스트로 변환. 503/429 일시 오류는 지수 백오프로 재시도."""
    client = _get_client()
    result = _embed_content(client, texts)
    return [e.values for e in result.embeddings]


def embed_query(query: str) -> list[float]:
    """단일 쿼리 텍스트를 벡터로 변환. 503/429 일시 오류는 지수 백오프로 재시도."""
    client = _get_client()
    result = _embed_content(client, [query])
    return result.embeddings[0].values
