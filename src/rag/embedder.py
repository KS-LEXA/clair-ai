from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")

MODEL = "models/gemini-embedding-001"


def _get_client():
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genai가 필요합니다: pip install google-genai") from exc

    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyB4TYXQYZOeoR8UlYCFe8duXLYZ3kv5st4")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    return genai.Client(api_key=api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """텍스트 리스트를 벡터 리스트로 변환."""
    client = _get_client()
    result = client.models.embed_content(model=MODEL, contents=texts)
    return [e.values for e in result.embeddings]


def embed_query(query: str) -> list[float]:
    """단일 쿼리 텍스트를 벡터로 변환."""
    client = _get_client()
    result = client.models.embed_content(model=MODEL, contents=[query])
    return result.embeddings[0].values
