from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(Path(__file__).parents[2] / ".env")


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
        # 503(과부하)/429(레이트리밋) 등 일시 오류 시 내부적으로 지수 백오프 재시도.
        # 기본 6회로는 Gemini 과부하 스파이크 때 소진되어 조용히 폴백되는 경우가 있어 상향.
        max_retries=8,
        timeout=120,
    )
