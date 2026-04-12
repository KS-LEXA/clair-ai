"""
법령 준수 검사 체인.

흐름:
  계약 조항 → 법령 벡터 검색(전체) → LLM 배치 비교 판단 → 위반/주의/적합 분류

LLM 호출 최적화:
  조항별 순차 호출(N회) 대신, 관련 법령이 있는 조항 전체를 한 번의 LLM 호출로 처리.
  vector search는 로컬이므로 조항 수에 비례해도 빠름.

출력: ComplianceResult 리스트 (조항별 법령 비교 결과)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.chains.contract_analysis_chain import Clause


@dataclass
class LawReference:
    law_name: str
    article_no: str
    article_title: str
    content: str
    similarity: float


@dataclass
class ComplianceResult:
    clause_id: str
    clause_title: str | None
    clause_text: str
    status: str                    # "위반", "주의", "적합", "검토불가"
    reason: str
    law_references: list[LawReference]


class ComplianceChain:
    """
    계약 조항과 관련 법령을 비교하여 법령 준수 여부를 판단.

    - law_store에서 모든 조항의 유사 법령을 일괄 검색 (로컬, 빠름)
    - 관련 법령이 있는 조항만 추려 LLM에 단 1번 배치 요청
    - LLM 실패 시 키워드 폴백으로 전체 처리
    """

    SIMILARITY_THRESHOLD = 0.4  # 이 이상일 때만 법령 비교 수행

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k

    def check_clauses(
        self,
        clauses: list[Clause],
        contract_type: str | None = None,
    ) -> list[ComplianceResult]:
        """모든 조항에 대해 법령 준수 검사 수행 (LLM 1회 배치 처리)."""
        from src.legal.law_store import get_law_store

        store = get_law_store()
        if not store.is_ready():
            return self._fallback_results(clauses)

        # Step 1: 모든 조항에 대해 vector search (로컬 연산, N번이어도 빠름)
        clause_refs: list[tuple[Clause, list[LawReference]]] = []
        for clause in clauses:
            query = f"{clause.title or ''} {clause.text[:300]}".strip()
            law_results = store.search(query, contract_type=contract_type, top_k=self.top_k)
            relevant = [r for r in law_results if r.score >= self.SIMILARITY_THRESHOLD]
            if not relevant:
                continue
            references = [
                LawReference(
                    law_name=r.law_name,
                    article_no=r.article_no,
                    article_title=r.article_title,
                    content=r.content,
                    similarity=round(r.score, 3),
                )
                for r in relevant
            ]
            clause_refs.append((clause, references))

        if not clause_refs:
            return []

        # Step 2: 관련 법령이 있는 조항 전체를 LLM 1번으로 배치 판단
        try:
            return self._llm_compare_batch(clause_refs)
        except Exception:
            # LLM 실패 시 키워드 폴백으로 조항별 처리
            return [self._keyword_compare(clause, refs) for clause, refs in clause_refs]

    def _llm_compare_batch(
        self,
        clause_refs: list[tuple[Clause, list[LawReference]]],
    ) -> list[ComplianceResult]:
        """관련 법령이 있는 조항 전체를 LLM 1회 호출로 배치 판단."""
        from langchain_core.messages import HumanMessage
        from src.llm.gemini import get_llm

        llm = get_llm()

        # 조항 + 관련 법령 목록 구성
        items_text = ""
        for clause, refs in clause_refs:
            refs_text = "\n".join(
                f"  - [{r.law_name} {r.article_no} {r.article_title}]: {r.content[:200]}"
                for r in refs
            )
            items_text += f"""
### [{clause.clause_id}] {clause.title or '(제목 없음)'}
조항 내용: {clause.text[:400]}
관련 법령:
{refs_text}
"""

        prompt = f"""당신은 한국 법률 전문가입니다.
아래 계약서 조항들이 각각 관련 법령을 준수하는지 분석하세요.
반드시 아래 JSON 배열 형식으로만 응답하세요. 배열 순서는 입력 조항 순서와 동일해야 합니다.

[
  {{
    "clause_id": "clause-XXX",
    "status": "위반" | "주의" | "적합",
    "reason": "판단 이유 (1-2문장, 구체적으로)"
  }},
  ...
]

판단 기준:
- "위반": 법령을 명백히 위반하거나 무효가 될 가능성이 높음
- "주의": 법령과 충돌 가능성 있거나 근로자에게 불리한 조건, 추가 검토 필요
- "적합": 법령 범위 내에서 유효한 조항

분석 대상 조항 ({len(clause_refs)}개):
{items_text}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        # LLM 응답을 clause_id로 인덱싱
        llm_map: dict[str, dict] = {item["clause_id"]: item for item in data if isinstance(item, dict)}

        results: list[ComplianceResult] = []
        for clause, refs in clause_refs:
            llm_item = llm_map.get(clause.clause_id, {})
            results.append(ComplianceResult(
                clause_id=clause.clause_id,
                clause_title=clause.title,
                clause_text=clause.text,
                status=llm_item.get("status", "검토불가"),
                reason=llm_item.get("reason", "LLM 응답에 해당 조항 없음"),
                law_references=refs,
            ))

        return results

    @staticmethod
    def _keyword_compare(
        clause: Clause,
        references: list[LawReference],
    ) -> ComplianceResult:
        """LLM 없이 키워드 기반 간이 판단."""
        violation_keywords = [
            "초과할 수 없다", "하여야 한다", "금지", "무효", "이상을 지급",
            "초과 불가", "보장하여야",
        ]
        warning_keywords = [
            "적당히 감액", "부당하게 낮은", "부당한", "공정을 잃은",
        ]

        combined_law = " ".join(r.content for r in references)
        combined_clause = clause.text

        has_violation_hint = any(kw in combined_law for kw in violation_keywords)
        has_warning_hint = any(kw in combined_law for kw in warning_keywords)

        if has_violation_hint and len(combined_clause) < 50:
            status = "주의"
            reason = f"관련 법령({references[0].law_name} {references[0].article_no})에 의무 규정이 있습니다. 계약서 조항의 구체적 내용을 확인하세요."
        elif has_warning_hint:
            status = "주의"
            reason = "관련 법령에 불공정 행위 금지 규정이 있습니다. 법률 전문가의 검토를 권장합니다."
        else:
            status = "적합"
            reason = f"관련 법령({references[0].law_name} {references[0].article_no})과 명백한 충돌이 확인되지 않았습니다."

        return ComplianceResult(
            clause_id=clause.clause_id,
            clause_title=clause.title,
            clause_text=clause.text,
            status=status,
            reason=reason,
            law_references=references,
        )

    @staticmethod
    def _fallback_results(clauses: list[Clause]) -> list[ComplianceResult]:
        """법령 DB 미초기화 시 전체 조항에 검토불가 반환."""
        return [
            ComplianceResult(
                clause_id=c.clause_id,
                clause_title=c.title,
                clause_text=c.text,
                status="검토불가",
                reason="법령 데이터베이스가 초기화되지 않았습니다. Gemini API 키를 설정하면 자동으로 초기화됩니다.",
                law_references=[],
            )
            for c in clauses
        ]


# ── 싱글톤 ────────────────────────────────────────────────────────────────────

_compliance_chain: ComplianceChain | None = None


def get_compliance_chain() -> ComplianceChain:
    global _compliance_chain
    if _compliance_chain is None:
        _compliance_chain = ComplianceChain()
    return _compliance_chain
