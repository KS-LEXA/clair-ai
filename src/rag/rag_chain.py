from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.chains.contract_analysis_chain import Clause, QAResult


class RAGChain:
    """
    RAG 기반 Q&A 체인.

    흐름:
      1. vector_store에서 하이브리드(Dense+BM25, RRF 병합) 검색으로 관련 조항 top-k 검색
      2. 검색된 조항만 LLM 프롬프트에 포함 (전체 조항 X)
      3. LLM이 답변 + 근거 clause_id 반환

    contract_id가 없을 때 (임시 Q&A): 전달받은 clauses 전체를 LLM에 직접 넘기는 폴백 사용.
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def answer(
        self,
        question: str,
        clauses: list[Clause],
        contract_id: str | int | None = None,
    ) -> QAResult:
        from src.chains.contract_analysis_chain import QAResult

        # RAG 검색 시도
        retrieved = self._retrieve(question, clauses, contract_id)

        # LLM 답변 생성
        try:
            return self._generate_answer(question, retrieved)
        except Exception:
            return self._fallback_answer(question, retrieved or clauses[:3])

    def answer_batch(
        self,
        questions: list[str],
        clauses: list[Clause],
        contract_id: str | int | None = None,
    ) -> list[QAResult]:
        return [self.answer(q, clauses, contract_id) for q in questions]

    # ── 검색 ──────────────────────────────────────────────────────────────────

    def _retrieve(
        self,
        question: str,
        clauses: list[Clause],
        contract_id: str | int | None,
    ) -> list[Clause]:
        """
        ChromaDB에 인덱스가 있으면 벡터 검색,
        없으면 단순 키워드 매칭 폴백.
        """
        if contract_id is not None:
            try:
                from src.rag.vector_store import get_vector_store
                store = get_vector_store()
                if store.has_index(contract_id):
                    results = store.hybrid_search(contract_id, question, top_k=self.top_k)
                    clause_map = {c.clause_id: c for c in clauses}
                    retrieved = [
                        clause_map[r.clause_id]
                        for r in results
                        if r.clause_id in clause_map
                    ]
                    if retrieved:
                        return retrieved
            except Exception:
                pass

        # 폴백: 키워드 매칭
        return self._keyword_search(question, clauses)

    def _keyword_search(self, question: str, clauses: list[Clause]) -> list[Clause]:
        """질문의 핵심 단어가 포함된 조항 우선 반환 (최대 top_k개)."""
        # 조사/어미 제거 후 2글자 이상 토큰 추출
        tokens = [t for t in re.split(r"[\s,?\.!]+", question) if len(t) >= 2]
        scored: list[tuple[int, Clause]] = []
        for clause in clauses:
            score = sum(1 for t in tokens if t in clause.text)
            if score > 0:
                scored.append((score, clause))

        scored.sort(key=lambda x: x[0], reverse=True)
        result = [c for _, c in scored[: self.top_k]]
        return result or clauses[: self.top_k]

    # ── LLM 답변 생성 ─────────────────────────────────────────────────────────

    def _generate_answer(self, question: str, clauses: list[Clause]) -> QAResult:
        from langchain_core.messages import HumanMessage
        from src.chains.contract_analysis_chain import QAResult
        from src.llm.gemini import get_llm, log_usage

        llm = get_llm()
        clauses_text = "\n\n".join(
            f"[{c.clause_id}] {c.title or ''}\n{c.text[:600]}"
            for c in clauses
        )

        prompt = f"""다음은 계약서에서 질문과 관련된 조항들입니다. 이 조항들을 근거로 질문에 답하세요.
답변은 아래 JSON 형식으로만 응답하세요. 다른 설명은 출력하지 마세요.

{{
  "answer": "질문에 대한 답변 (한국어, 2-4문장)",
  "evidence_clause_ids": ["clause-001", "clause-002"]
}}

관련 조항:
{clauses_text}

질문: {question}"""

        response = llm.invoke([HumanMessage(content=prompt)])
        log_usage("qa", response)
        raw = response.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        return QAResult(
            question=question,
            answer=data.get("answer", "답변을 생성할 수 없습니다."),
            evidence_clause_ids=data.get("evidence_clause_ids", []),
        )

    @staticmethod
    def _fallback_answer(question: str, clauses: list[Clause]) -> QAResult:
        from src.chains.contract_analysis_chain import QAResult

        if clauses:
            excerpt = clauses[0].text[:300]
            answer = f"관련 조항을 찾았습니다:\n\n\"{excerpt}...\"\n\n정확한 답변을 위해 잠시 후 다시 질문해주세요."
        else:
            answer = "해당 질문과 관련된 조항을 찾지 못했습니다. 다른 방식으로 질문해보세요."

        return QAResult(
            question=question,
            answer=answer,
            evidence_clause_ids=[c.clause_id for c in clauses[:2]],
        )


# 모듈 레벨 싱글톤
_rag_chain: RAGChain | None = None


def get_rag_chain(top_k: int = 5) -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain(top_k=top_k)
    return _rag_chain
