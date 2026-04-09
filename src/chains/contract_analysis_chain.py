from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.ocr.pipeline import OCRDocumentResult, OCRPipeline


@dataclass(slots=True)
class Clause:
    clause_id: str
    title: str | None
    text: str
    page_refs: list[int]
    order: int


@dataclass(slots=True)
class FieldValue:
    value: str | float | None
    reason: str | None = None


@dataclass(slots=True)
class ExtractionResult:
    contract_type: FieldValue
    counterparty_a: FieldValue
    counterparty_b: FieldValue
    signing_date: FieldValue
    start_date: FieldValue
    end_date: FieldValue
    amount_text: FieldValue
    amount_value: FieldValue


@dataclass(slots=True)
class RiskResult:
    risk_type: str
    severity: str
    reason: str
    evidence_clause_ids: list[str]
    evidence_text: str


@dataclass(slots=True)
class QAResult:
    question: str
    answer: str
    evidence_clause_ids: list[str]


@dataclass(slots=True)
class ContractAnalysisResult:
    document_id: str
    ocr: OCRDocumentResult
    clauses: list[Clause]
    extraction: ExtractionResult
    risks: list[RiskResult]
    summary: str
    qa: list[QAResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "ocr": self.ocr.to_dict(),
            "clauses": [asdict(clause) for clause in self.clauses],
            "extraction": asdict(self.extraction),
            "risks": [asdict(risk) for risk in self.risks],
            "summary": self.summary,
            "qa": [asdict(item) for item in self.qa],
        }


class ContractAnalysisChain:
    """OCR → 조항 분리 → LLM 분석(추출/리스크/요약/Q&A) 파이프라인."""

    _clause_heading_pattern = re.compile(
        r"(?=^(제\s*\d+\s*조[^\n]*|[0-9]+\.\s+[^\n]+))",
        flags=re.MULTILINE,
    )

    def __init__(self, *, ocr_pipeline: OCRPipeline | None = None) -> None:
        self.ocr_pipeline = ocr_pipeline or OCRPipeline()

    def analyze_document(
        self,
        source: str | Path,
        *,
        document_id: str | None = None,
        questions: list[str] | None = None,
    ) -> ContractAnalysisResult:
        ocr_result = self.ocr_pipeline.extract(source, document_id=document_id)
        clauses = self._split_clauses(ocr_result)
        extraction = self._extract_fields(ocr_result.normalized_text)
        risks = self._detect_risks(clauses)
        summary = self._summarize(clauses, extraction)

        # 조항 임베딩 인덱싱 (RAG Q&A를 위해)
        contract_id = ocr_result.document_id
        self._index_clauses(contract_id, clauses)

        qa = self._answer_questions(questions or [], clauses, contract_id=contract_id)

        return ContractAnalysisResult(
            document_id=ocr_result.document_id,
            ocr=ocr_result,
            clauses=clauses,
            extraction=extraction,
            risks=risks,
            summary=summary,
            qa=qa,
        )

    # ── 조항 분리 ─────────────────────────────────────────────────────────────

    def _split_clauses(self, ocr_result: OCRDocumentResult) -> list[Clause]:
        text = ocr_result.normalized_text
        if not text:
            return []

        page_refs = [page.page_index for page in ocr_result.pages]
        chunks = [c.strip() for c in self._clause_heading_pattern.split(text) if c.strip()]
        clauses: list[Clause] = []
        index = 0
        order = 1

        while index < len(chunks):
            current = chunks[index]
            next_chunk = chunks[index + 1].strip() if index + 1 < len(chunks) else ""

            if self._looks_like_clause_heading(current):
                title = current
                body = next_chunk or current
                index += 2 if next_chunk else 1
            else:
                title = None
                body = current
                index += 1

            clauses.append(Clause(
                clause_id=f"clause-{order:03d}",
                title=title,
                text=body,
                page_refs=page_refs,
                order=order,
            ))
            order += 1

        if clauses:
            return clauses

        return [Clause(clause_id="clause-001", title=None, text=text, page_refs=page_refs, order=1)]

    # ── LLM: 핵심 정보 추출 ──────────────────────────────────────────────────

    def _extract_fields(self, text: str) -> ExtractionResult:
        try:
            from src.llm.gemini import get_llm
            llm = get_llm()
        except Exception:
            return self._extract_fields_fallback(text)

        prompt = f"""다음 계약서 본문에서 핵심 정보를 추출하여 JSON 형식으로만 응답하세요.
다른 설명 없이 JSON만 출력하세요. 값이 없으면 null을 사용하세요.

{{
  "contract_type": "계약 유형 (근로계약/용역계약/NDA/임대차계약/기타 중 하나)",
  "counterparty_a": "갑 또는 첫 번째 당사자의 이름 또는 회사명 (예: 주식회사 클레어테크)",
  "counterparty_b": "을 또는 두 번째 당사자의 이름 또는 회사명 (예: 홍길동)",
  "signing_date": "계약 체결일 YYYY-MM-DD 형식, 없으면 null",
  "start_date": "계약 시작일 YYYY-MM-DD 형식, 없으면 null",
  "end_date": "계약 종료일 YYYY-MM-DD 형식, 없으면 null",
  "amount_text": "계약 금액 원문 그대로 (예: 금 오천만원(50,000,000원)), 없으면 null",
  "amount_value": 계약 금액 숫자만 원 단위 정수 (예: 50000000), 없으면 null
}}

계약서 본문:
{text[:3000]}"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
            # 코드블록 제거
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)

            def fv(val: Any, reason: str | None = None) -> FieldValue:
                return FieldValue(value=val if val != "" else None, reason=reason)

            return ExtractionResult(
                contract_type=fv(data.get("contract_type")),
                counterparty_a=fv(data.get("counterparty_a")),
                counterparty_b=fv(data.get("counterparty_b")),
                signing_date=fv(data.get("signing_date")),
                start_date=fv(data.get("start_date")),
                end_date=fv(data.get("end_date")),
                amount_text=fv(data.get("amount_text")),
                amount_value=fv(data.get("amount_value")),
            )
        except Exception:
            return self._extract_fields_fallback(text)

    def _extract_fields_fallback(self, text: str) -> ExtractionResult:
        date_pattern = re.compile(r"(\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2})")
        amount_pattern = re.compile(r"([0-9][0-9,]*(?:원|KRW|만원))")
        dates = date_pattern.findall(text)
        amount_match = amount_pattern.search(text)
        contract_type = self._classify_contract_type(text)

        return ExtractionResult(
            contract_type=FieldValue(contract_type),
            counterparty_a=FieldValue(None, "추출 실패"),
            counterparty_b=FieldValue(None, "추출 실패"),
            signing_date=FieldValue(dates[0] if len(dates) > 0 else None),
            start_date=FieldValue(dates[1] if len(dates) > 1 else None),
            end_date=FieldValue(dates[2] if len(dates) > 2 else None),
            amount_text=FieldValue(amount_match.group(1) if amount_match else None),
            amount_value=FieldValue(self._parse_amount_value(amount_match.group(1)) if amount_match else None),
        )

    # ── LLM: 리스크 감지 ─────────────────────────────────────────────────────

    def _detect_risks(self, clauses: list[Clause]) -> list[RiskResult]:
        if not clauses:
            return []

        try:
            from src.llm.gemini import get_llm
            llm = get_llm()
        except Exception:
            return self._detect_risks_fallback(clauses)

        clauses_text = "\n\n".join(
            f"[{c.clause_id}] {c.title or ''}\n{c.text[:500]}"
            for c in clauses[:20]  # 최대 20개 조항
        )

        prompt = f"""다음 계약서 조항들에서 법적 리스크를 분석하여 아래 JSON 배열 형식으로만 응답하세요.
리스크가 없으면 빈 배열 []을 반환하세요.

[
  {{
    "risk_type": "리스크 유형 (auto_renewal/termination/liability/payment/ip/confidentiality/기타)",
    "severity": "심각도 (high/medium/low)",
    "reason": "리스크 이유 (한국어 1-2문장)",
    "evidence_clause_ids": ["clause-001"],
    "evidence_text": "관련 조항 원문 발췌 (100자 이내)"
  }}
]

계약서 조항:
{clauses_text}"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)

            return [
                RiskResult(
                    risk_type=item.get("risk_type", "unknown"),
                    severity=item.get("severity", "medium"),
                    reason=item.get("reason", ""),
                    evidence_clause_ids=item.get("evidence_clause_ids", []),
                    evidence_text=item.get("evidence_text", ""),
                )
                for item in data
                if isinstance(item, dict)
            ]
        except Exception:
            return self._detect_risks_fallback(clauses)

    def _detect_risks_fallback(self, clauses: list[Clause]) -> list[RiskResult]:
        rules = [
            ("auto_renewal", "medium", ("자동 갱신", "묵시적 갱신")),
            ("termination", "high", ("일방 해지", "즉시 해지")),
            ("liability", "high", ("손해배상", "배상 책임")),
            ("payment", "medium", ("지급 기한", "지급 조건")),
        ]
        findings: list[RiskResult] = []
        for clause in clauses:
            for risk_type, severity, keywords in rules:
                for keyword in keywords:
                    if keyword in clause.text:
                        findings.append(RiskResult(
                            risk_type=risk_type,
                            severity=severity,
                            reason=f"'{keyword}' 관련 조항이 포함되어 있어 검토가 필요합니다.",
                            evidence_clause_ids=[clause.clause_id],
                            evidence_text=clause.text[:300],
                        ))
                        break
        return findings

    # ── LLM: 요약 ────────────────────────────────────────────────────────────

    def _summarize(self, clauses: list[Clause], extraction: ExtractionResult) -> str:
        if not clauses:
            return "추출된 계약 본문이 없어 요약을 생성할 수 없습니다."

        try:
            from src.llm.gemini import get_llm
            llm = get_llm()
        except Exception:
            return self._summarize_fallback(clauses, extraction)

        full_text = "\n\n".join(
            f"{c.title or ''}\n{c.text[:400]}"
            for c in clauses[:15]
        )

        prompt = f"""다음 계약서를 읽고 핵심 내용을 3-5문장으로 요약해주세요.
- 계약 당사자(갑/을), 계약 유형, 계약 기간, 금액을 반드시 포함하세요.
- 주요 의무사항과 특이사항(자동갱신, 비밀유지 등)이 있으면 언급하세요.
- 한국어로 작성하고, 마크다운 없이 일반 텍스트로만 응답하세요.

계약서:
{full_text}"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception:
            return self._summarize_fallback(clauses, extraction)

    def _summarize_fallback(self, clauses: list[Clause], extraction: ExtractionResult) -> str:
        parts = [f"총 {len(clauses)}개 조항이 감지되었습니다."]
        if extraction.contract_type.value:
            parts.append(f"계약 유형: {extraction.contract_type.value}")
        if extraction.signing_date.value:
            parts.append(f"서명일: {extraction.signing_date.value}")
        if extraction.amount_text.value:
            parts.append(f"계약 금액: {extraction.amount_text.value}")
        return " ".join(parts)

    # ── RAG: 조항 인덱싱 ─────────────────────────────────────────────────────

    @staticmethod
    def _index_clauses(contract_id: str, clauses: list[Clause]) -> None:
        """조항을 벡터 DB에 인덱싱. 실패해도 분석 전체를 중단하지 않음."""
        if not clauses:
            return
        try:
            from src.rag.vector_store import get_vector_store
            get_vector_store().index_clauses(contract_id, clauses)
        except Exception:
            pass  # 임베딩 실패 시 RAG 폴백(키워드 검색)으로 자동 대체

    # ── LLM: Q&A (RAG) ───────────────────────────────────────────────────────

    def _answer_questions(
        self,
        questions: list[str],
        clauses: list[Clause],
        contract_id: str | None = None,
    ) -> list[QAResult]:
        if not questions:
            return []

        try:
            from src.rag.rag_chain import get_rag_chain
            rag = get_rag_chain()
            return rag.answer_batch(questions, clauses, contract_id=contract_id)
        except Exception:
            return self._answer_questions_fallback(questions, clauses)

    def _answer_questions_fallback(self, questions: list[str], clauses: list[Clause]) -> list[QAResult]:
        fallback = clauses[0] if clauses else None
        return [
            QAResult(
                question=q,
                answer="LLM 연결 실패로 답변을 생성할 수 없습니다.",
                evidence_clause_ids=[fallback.clause_id] if fallback else [],
            )
            for q in questions
        ]

    # ── 유틸 ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _looks_like_clause_heading(text: str) -> bool:
        return bool(re.match(r"^(제\s*\d+\s*조|[0-9]+\.)", text.strip()))

    @staticmethod
    def _classify_contract_type(text: str) -> str:
        if "비밀유지" in text or "nda" in text.lower():
            return "NDA"
        if "근로" in text or "고용" in text:
            return "근로계약"
        if "용역" in text or "서비스" in text:
            return "용역계약"
        return "unknown"

    @staticmethod
    def _parse_amount_value(amount_text: str) -> float | None:
        digits = re.sub(r"[^0-9]", "", amount_text)
        return float(digits) if digits else None
