"""
법령 준수 검사 체인.

흐름:
  계약 조항 → 법령 벡터 검색 → LLM 비교 판단 → 위반/주의/적합 분류

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

    - law_store에서 유사 법령 top_k 검색
    - LLM이 계약 조항과 법령을 비교하여 위반/주의/적합 판단
    - LLM 실패 시 유사도만으로 키워드 폴백
    """

    SIMILARITY_THRESHOLD = 0.4  # 이 이상일 때만 비교 수행

    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k

    def check_clauses(
        self,
        clauses: list[Clause],
        contract_type: str | None = None,
    ) -> list[ComplianceResult]:
        """모든 조항에 대해 법령 준수 검사 수행."""
        from src.legal.law_store import get_law_store

        store = get_law_store()
        if not store.is_ready():
            return self._fallback_results(clauses)

        results: list[ComplianceResult] = []
        for clause in clauses:
            result = self._check_single_clause(clause, store, contract_type)
            if result:
                results.append(result)

        return results

    def _check_single_clause(
        self,
        clause: Clause,
        store,
        contract_type: str | None,
    ) -> ComplianceResult | None:
        """단일 조항에 대한 법령 준수 검사."""
        query = f"{clause.title or ''} {clause.text[:300]}".strip()
        law_results = store.search(query, contract_type=contract_type, top_k=self.top_k)

        # 유사도가 낮으면 관련 법령 없음으로 처리
        relevant = [r for r in law_results if r.score >= self.SIMILARITY_THRESHOLD]
        if not relevant:
            return None

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

        try:
            return self._llm_compare(clause, references)
        except Exception:
            return self._keyword_compare(clause, references)

    def _llm_compare(
        self,
        clause: Clause,
        references: list[LawReference],
    ) -> ComplianceResult:
        """LLM을 사용한 조항 vs 법령 비교."""
        from langchain_core.messages import HumanMessage
        from src.llm.gemini import get_llm

        llm = get_llm()
        refs_text = "\n\n".join(
            f"[{r.law_name} {r.article_no} {r.article_title}]\n{r.content}"
            for r in references
        )

        prompt = f"""당신은 한국 법률 전문가입니다.
아래 계약서 조항이 관련 법령을 준수하는지 분석하세요.
반드시 아래 JSON 형식으로만 응답하세요.

{{
  "status": "위반" | "주의" | "적합",
  "reason": "판단 이유 (2-3문장, 구체적으로)"
}}

판단 기준:
- "위반": 법령을 명백히 위반하거나 무효가 될 가능성이 높음
- "주의": 법령과 충돌 가능성 있거나 불리한 조건, 추가 검토 필요
- "적합": 법령 범위 내에서 유효한 조항

계약서 조항:
[{clause.clause_id}] {clause.title or ''}
{clause.text[:500]}

관련 법령:
{refs_text}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        return ComplianceResult(
            clause_id=clause.clause_id,
            clause_title=clause.title,
            clause_text=clause.text,
            status=data.get("status", "검토불가"),
            reason=data.get("reason", "판단 불가"),
            law_references=references,
        )

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

        # 관련 법령이 의무/금지 조항인데 계약서 조항이 짧으면 주의
        if has_violation_hint and len(combined_clause) < 50:
            status = "주의"
            reason = f"관련 법령({references[0].law_name} {references[0].article_no})에 의무 규정이 있습니다. 계약서 조항의 구체적 내용을 확인하세요."
        elif has_warning_hint:
            status = "주의"
            reason = f"관련 법령에 불공정 행위 금지 규정이 있습니다. 법률 전문가의 검토를 권장합니다."
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
            for c in clauses[:5]  # 폴백은 앞 5개만
        ]


# ── 싱글톤 ────────────────────────────────────────────────────────────────────

_compliance_chain: ComplianceChain | None = None


def get_compliance_chain() -> ComplianceChain:
    global _compliance_chain
    if _compliance_chain is None:
        _compliance_chain = ComplianceChain()
    return _compliance_chain
