from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.chains.contract_analysis_chain import Clause

DEFAULT_CHUNK_CHARS = 250
DEFAULT_CHUNK_OVERLAP = 50
MAX_EMBEDDING_TOKENS = 2000  # gemini-embedding-001 입력 한도 대비 안전 마진
_CHARS_PER_TOKEN_KO = 1.7  # tiktoken 부재 시 한국어 대략 추정치 (경험적 비율)

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(slots=True)
class SubChunk:
    sub_chunk_id: str
    parent_clause_id: str
    parent_title: str | None
    parent_text: str
    text: str
    order: int


def estimate_tokens(text: str) -> int:
    """tiktoken이 설치되어 있으면 정확히, 없으면 글자 수 기반 근사치로 토큰 수를 추정."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return int(len(text) / _CHARS_PER_TOKEN_KO)


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_PATTERN.split(text) if p.strip()]
    return parts or [text.strip()]


def _pack_sentences(sentences: list[str], chunk_chars: int, overlap: int) -> list[str]:
    """문장을 chunk_chars 근처로 그리디하게 묶고, 청크 경계마다 overlap자를 겹쳐 문맥 단절을 줄인다."""
    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        if buf and len(buf) + 1 + len(sent) > chunk_chars:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail} {sent}".strip() if tail else sent
        else:
            buf = f"{buf} {sent}".strip() if buf else sent
    if buf:
        chunks.append(buf)
    return chunks


def _hard_split(text: str, chunk_chars: int, overlap: int) -> list[str]:
    """문장부호가 없어 패킹으로 안 잘리는 긴 나열형 텍스트(표, 목록 등)를 위한 강제 슬라이딩 분할."""
    step = max(chunk_chars - overlap, 1)
    pieces = [text[i : i + chunk_chars] for i in range(0, len(text), step)]
    return pieces or [text]


def split_into_sub_chunks(
    clause: "Clause",
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_tokens: int = MAX_EMBEDDING_TOKENS,
) -> list[SubChunk]:
    """
    조항(Clause = 부모)을 200~300자 자식 청크로 분할.

    - 짧은 조항(chunk_chars 이하)은 분할 오버헤드 없이 통째로 자식 청크 1개 유지
    - 긴 조항은 문장 단위로 우선 패킹, 문장 경계가 없는 구간은 강제 슬라이딩 분할
    - 패킹 후에도 임베딩 모델 토큰 한도를 넘는 청크는 추가로 쪼개 잘림(truncation) 방지
    """
    text = clause.text.strip()
    if not text:
        return []

    if len(text) <= chunk_chars:
        raw_chunks = [text]
    else:
        raw_chunks = _pack_sentences(_split_sentences(text), chunk_chars, overlap)

    safe_chunks: list[str] = []
    for c in raw_chunks:
        if estimate_tokens(c) <= max_tokens:
            safe_chunks.append(c)
        else:
            safe_chunks.extend(_hard_split(c, chunk_chars, overlap))

    return [
        SubChunk(
            sub_chunk_id=f"{clause.clause_id}::chunk-{i}",
            parent_clause_id=clause.clause_id,
            parent_title=clause.title,
            parent_text=clause.text,
            text=chunk_text,
            order=i,
        )
        for i, chunk_text in enumerate(safe_chunks)
    ]
