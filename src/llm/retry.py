from __future__ import annotations

"""
Gemini API 일시 오류 판별 유틸.

503(UNAVAILABLE, 과부하), 429(RESOURCE_EXHAUSTED, 레이트리밋), 500/502/504 등
서버 측 일시 오류는 재시도하면 대부분 성공하므로 transient으로 분류한다.
400(잘못된 요청)·401/403(인증)·404 등 클라이언트 오류는 재시도해도 동일하게
실패하므로 transient이 아니다 (즉시 폴백).
"""

# 재시도 대상 HTTP 상태 코드
_TRANSIENT_CODES = {429, 500, 502, 503, 504}

# 재시도 대상 상태 문자열 (google.genai가 message에 담아주는 식별자)
_TRANSIENT_MARKERS = (
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "DEADLINE_EXCEEDED",
    "INTERNAL",
    "overloaded",
    "high demand",
)


def is_transient_error(exc: BaseException) -> bool:
    """예외가 재시도 가능한 일시 오류인지 판별."""
    # google.genai.errors.APIError 계열은 .code 속성을 가짐
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in _TRANSIENT_CODES:
        return True

    # httpx.Response 기반 예외 대비
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and status in _TRANSIENT_CODES:
        return True

    text = str(exc)
    return any(marker in text for marker in _TRANSIENT_MARKERS)
