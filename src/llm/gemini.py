from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(Path(__file__).parents[2] / ".env")

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def get_llm(*, thinking_budget: int | None = None) -> ChatGoogleGenerativeAI:
    """Gemini LLM 인스턴스를 반환한다.

    thinking_budget이 None이면 모델 기본값(동적 thinking)을 쓴다. 0을 주면 thinking을
    끄는데, thinking 토큰도 출력 토큰으로 과금되므로 추론이 필요 없는 단계
    (OCR 오탈자 보정 등)에서 출력 토큰을 크게 줄일 수 있다.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")
    extra: dict[str, Any] = {}
    if thinking_budget is not None:
        extra["thinking_budget"] = thinking_budget
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
        # 503(과부하)/429(레이트리밋) 등 일시 오류 시 내부적으로 지수 백오프 재시도.
        # 기본 6회로는 Gemini 과부하 스파이크 때 소진되어 조용히 폴백되는 경우가 있어 상향.
        max_retries=8,
        timeout=120,
        **extra,
    )


def log_usage(label: str, response: Any) -> None:
    """LLM 응답의 토큰 사용량을 남긴다. 사용량 정보가 없으면 조용히 넘어간다.

    thinking 토큰은 output_tokens에 포함되어 과금되므로 별도로 표시한다.
    """
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return
    thinking = (usage.get("output_token_details") or {}).get("reasoning", 0)
    logger.info(
        "[gemini usage] %s: input=%s output=%s (thinking=%s) total=%s",
        label,
        usage.get("input_tokens", 0),
        usage.get("output_tokens", 0),
        thinking,
        usage.get("total_tokens", 0),
    )
