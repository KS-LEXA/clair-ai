from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

_ARTICLE_PATTERN = re.compile(r"제\s?\d+\s?조(?:의\s?\d+)?")
_AMOUNT_PATTERN = re.compile(r"\d[\d,]*(?:원|퍼센트|%|일|개월|년|회)")
_HANGUL_PATTERN = re.compile(r"[가-힣]+")
_LATIN_PATTERN = re.compile(r"[A-Za-z]+")


def tokenize_ko(text: str) -> list[str]:
    """
    형태소 분석기 없이 BM25 재현율을 높이기 위한 경량 토크나이저.

    - "제12조", "제12조의2" 같은 법률 조항 참조는 통째로 하나의 토큰으로 보존
    - 금액/기간 등 숫자+단위 표현("300만원", "6개월")도 통째로 보존
    - 나머지 한글 어절은 2-gram으로 쪼개, 조사가 붙어도("위약금은" → "위약", "약금", "금은")
      원형 키워드("위약금")와 부분적으로 겹쳐 매칭되게 함
    """
    tokens: list[str] = []
    tokens += _ARTICLE_PATTERN.findall(text)
    tokens += _AMOUNT_PATTERN.findall(text)
    for block in _HANGUL_PATTERN.findall(text):
        if len(block) <= 2:
            tokens.append(block)
        else:
            tokens.extend(block[i : i + 2] for i in range(len(block) - 1))
    tokens += [t.lower() for t in _LATIN_PATTERN.findall(text)]
    return tokens


@dataclass(slots=True)
class BM25Document:
    doc_id: str
    text: str


class BM25Index:
    """계약서 1건 단위 인메모리 BM25 인덱스. 조항 수가 수십 개 수준이라 Elasticsearch 없이 rank_bm25로 충분."""

    def __init__(self, documents: list[BM25Document]) -> None:
        from rank_bm25 import BM25Okapi

        self._ids = [d.doc_id for d in documents]
        tokenized = [tokenize_ko(d.text) for d in documents]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int) -> list[str]:
        """점수 내림차순 doc_id 리스트 반환 (점수 0인 문서는 제외)."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize_ko(query))
        ranked = sorted(zip(self._ids, scores), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in ranked[:top_k] if score > 0]


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    여러 검색 방식의 순위 리스트를 RRF로 병합.

    score(d) = sum(1 / (k + rank_i(d) + 1))  — 특정 방식의 절대 점수 스케일에 좌우되지 않고
    "어느 방식에서든 상위에 있었는가"를 기준으로 재랭킹한다. k=60은 원 논문(Cormack et al.)의
    권장값으로, 하위권 순위 변동에 과민 반응하지 않도록 완충 역할을 한다.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
