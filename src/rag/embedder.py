from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")


@lru_cache(maxsize=1)
def get_embedder():
    """Google text-embedding-004 모델 싱글톤 반환."""
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "langchain-google-genai가 설치되지 않았습니다. `pip install langchain-google-genai`"
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """텍스트 리스트를 벡터 리스트로 변환."""
    embedder = get_embedder()
    return embedder.embed_documents(texts)


def embed_query(query: str) -> list[float]:
    """단일 쿼리 텍스트를 벡터로 변환."""
    embedder = get_embedder()
    return embedder.embed_query(query)
