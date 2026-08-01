"""Gemini API 월간 예산 가드.

모든 Gemini 호출의 토큰 사용량을 누적 기록하고, 월 예산을 넘으면 신규 요청을
차단한다. Google Cloud 예산은 알림만 보내고 실제로 차단하지 않으므로,
지출 상한을 강제하려면 애플리케이션 레벨에서 막아야 한다.

기록 지점은 src.llm.gemini.log_usage() 한 곳이다. 모든 LLM 호출이 그곳을
지나가므로 단계가 추가되어도 자동으로 집계에 포함된다.

저장소는 SQLite(.usage_db/usage.sqlite3)다.
주의: 컨테이너처럼 파일시스템이 휘발성인 환경에서는 볼륨 마운트가 필요하다.
카운터가 사라지면 예산이 초기화되어 상한이 무력화된다.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 배포 시 볼륨 마운트 경로를 가리킬 수 있도록 환경변수로 재정의 가능하게 둔다.
# 카운터가 사라지면 예산 상한이 무력화되므로 영속 스토리지가 필수다.
_DATA_DIR = Path(os.environ.get("CLAIR_AI_DATA_DIR", Path(__file__).parents[2]))
DB_PATH = _DATA_DIR / ".usage_db" / "usage.sqlite3"

# gemini-2.5-flash 단가 (USD / 1M 토큰). thinking 토큰은 출력으로 과금된다.
# https://ai.google.dev/gemini-api/docs/pricing
_USD_PER_INPUT_TOKEN = 0.30 / 1_000_000
_USD_PER_OUTPUT_TOKEN = 2.50 / 1_000_000

_DEFAULT_BUDGET_KRW = 5000.0
_DEFAULT_USD_TO_KRW = 1400.0

_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    """월 예산을 초과해 호출을 차단했음을 알린다."""


def _budget_krw() -> float:
    try:
        return float(os.environ.get("MONTHLY_BUDGET_KRW", _DEFAULT_BUDGET_KRW))
    except ValueError:
        return _DEFAULT_BUDGET_KRW


def _usd_to_krw() -> float:
    try:
        return float(os.environ.get("USD_TO_KRW", _DEFAULT_USD_TO_KRW))
    except ValueError:
        return _DEFAULT_USD_TO_KRW


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS monthly_usage (
            month         TEXT PRIMARY KEY,
            input_tokens  INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            calls         INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    return conn


def cost_krw(input_tokens: int, output_tokens: int) -> float:
    """토큰 수를 원화 비용으로 환산한다."""
    usd = input_tokens * _USD_PER_INPUT_TOKEN + output_tokens * _USD_PER_OUTPUT_TOKEN
    return usd * _usd_to_krw()


def record(input_tokens: int, output_tokens: int) -> None:
    """호출 1건의 사용량을 누적한다. 실패해도 본 기능을 막지 않는다."""
    try:
        with _lock, _connect() as conn:
            conn.execute(
                """
                INSERT INTO monthly_usage (month, input_tokens, output_tokens, calls)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(month) DO UPDATE SET
                    input_tokens  = input_tokens  + excluded.input_tokens,
                    output_tokens = output_tokens + excluded.output_tokens,
                    calls         = calls + 1
                """,
                (_current_month(), input_tokens, output_tokens),
            )
    except Exception as exc:  # 집계 실패가 분석을 막아서는 안 된다
        logger.warning("[budget] 사용량 기록 실패 (무시): %s", exc)


def snapshot() -> dict[str, Any]:
    """이번 달 사용량과 예산 소진 현황을 반환한다."""
    month = _current_month()
    try:
        with _lock, _connect() as conn:
            row = conn.execute(
                "SELECT input_tokens, output_tokens, calls FROM monthly_usage WHERE month = ?",
                (month,),
            ).fetchone()
    except Exception as exc:
        logger.warning("[budget] 사용량 조회 실패: %s", exc)
        row = None

    inp, out, calls = row if row else (0, 0, 0)
    budget = _budget_krw()
    used = cost_krw(inp, out)
    return {
        "month": month,
        "input_tokens": inp,
        "output_tokens": out,
        "calls": calls,
        "used_krw": round(used, 2),
        "budget_krw": budget,
        "remaining_krw": round(budget - used, 2),
        "usage_ratio": round(used / budget, 4) if budget > 0 else 0.0,
        "exceeded": used >= budget,
    }


def is_exceeded() -> bool:
    return snapshot()["exceeded"]


def ensure_within_budget() -> None:
    """예산을 넘었으면 BudgetExceeded를 던진다. 비싼 작업 시작 전에 호출한다."""
    state = snapshot()
    if state["exceeded"]:
        raise BudgetExceeded(
            f"이번 달({state['month']}) Gemini API 예산을 모두 사용했습니다. "
            f"사용 {state['used_krw']}원 / 예산 {state['budget_krw']}원. "
            f"다음 달에 자동으로 초기화됩니다."
        )
