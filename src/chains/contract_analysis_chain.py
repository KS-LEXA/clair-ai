from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    """OCR 결과와 후속 분석 단계를 연결하는 계약 분석 체인 골격."""

    _clause_heading_pattern = re.compile(
        r"(?=^(제\s*\d+\s*조[^\n]*|[0-9]+\.\s+[^\n]+))",
        flags=re.MULTILINE,
    )
    _date_pattern = re.compile(r"(\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2})")
    _amount_pattern = re.compile(r"([0-9][0-9,]*(?:원|KRW|만원))")

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
        extraction = self._build_extraction(ocr_result.normalized_text)
        risks = self._detect_risks(clauses)
        summary = self._build_summary(clauses, extraction)
        qa = self._answer_questions(questions or [], clauses)

        return ContractAnalysisResult(
            document_id=ocr_result.document_id,
            ocr=ocr_result,
            clauses=clauses,
            extraction=extraction,
            risks=risks,
            summary=summary,
            qa=qa,
        )

    def _split_clauses(self, ocr_result: OCRDocumentResult) -> list[Clause]:
        text = ocr_result.normalized_text
        if not text:
            return []

        page_refs = [page.page_index for page in ocr_result.pages]
        chunks = [chunk.strip() for chunk in self._clause_heading_pattern.split(text) if chunk.strip()]
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

            clauses.append(
                Clause(
                    clause_id=f"clause-{order:03d}",
                    title=title,
                    text=body,
                    page_refs=page_refs,
                    order=order,
                )
            )
            order += 1

        if clauses:
            return clauses

        return [
            Clause(
                clause_id="clause-001",
                title=None,
                text=text,
                page_refs=page_refs,
                order=1,
            )
        ]

    def _build_extraction(self, text: str) -> ExtractionResult:
        dates = self._date_pattern.findall(text)
        amount_match = self._amount_pattern.search(text)
        contract_type = self._classify_contract_type(text)

        return ExtractionResult(
            contract_type=FieldValue(contract_type, None if contract_type != "unknown" else "계약 유형 분류 미구현"),
            counterparty_a=FieldValue(None, "당사자 추출 미구현"),
            counterparty_b=FieldValue(None, "당사자 추출 미구현"),
            signing_date=FieldValue(dates[0] if len(dates) > 0 else None, None if len(dates) > 0 else "날짜 패턴 미검출"),
            start_date=FieldValue(dates[1] if len(dates) > 1 else None, None if len(dates) > 1 else "시작일 추출 미구현"),
            end_date=FieldValue(dates[2] if len(dates) > 2 else None, None if len(dates) > 2 else "종료일 추출 미구현"),
            amount_text=FieldValue(amount_match.group(1) if amount_match else None, None if amount_match else "금액 패턴 미검출"),
            amount_value=FieldValue(self._parse_amount_value(amount_match.group(1)) if amount_match else None, None if amount_match else "금액 정규화 미구현"),
        )

    def _detect_risks(self, clauses: list[Clause]) -> list[RiskResult]:
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
                        findings.append(
                            RiskResult(
                                risk_type=risk_type,
                                severity=severity,
                                reason=f"'{keyword}' 관련 조항이 포함되어 있어 검토가 필요하다.",
                                evidence_clause_ids=[clause.clause_id],
                                evidence_text=clause.text[:300],
                            )
                        )
                        break
        return findings

    def _build_summary(self, clauses: list[Clause], extraction: ExtractionResult) -> str:
        if not clauses:
            return "추출된 계약 본문이 없어 요약을 생성할 수 없다."

        summary_parts = [
            f"총 {len(clauses)}개 조항이 감지되었다.",
            f"계약 유형 추정: {extraction.contract_type.value or 'unknown'}",
        ]

        if extraction.signing_date.value:
            summary_parts.append(f"서명일 후보: {extraction.signing_date.value}")
        if extraction.amount_text.value:
            summary_parts.append(f"금액 후보: {extraction.amount_text.value}")

        return " ".join(summary_parts)

    def _answer_questions(self, questions: list[str], clauses: list[Clause]) -> list[QAResult]:
        if not questions:
            return []

        fallback_clause = clauses[0] if clauses else None
        answers: list[QAResult] = []
        for question in questions:
            if fallback_clause is None:
                answers.append(
                    QAResult(
                        question=question,
                        answer="문서 본문이 없어 답변할 수 없다.",
                        evidence_clause_ids=[],
                    )
                )
                continue

            answers.append(
                QAResult(
                    question=question,
                    answer=f"현재는 규칙 기반 임시 답변만 제공한다. 우선 {fallback_clause.clause_id}를 검토하라.",
                    evidence_clause_ids=[fallback_clause.clause_id],
                )
            )
        return answers

    @staticmethod
    def _looks_like_clause_heading(text: str) -> bool:
        stripped = text.strip()
        return bool(re.match(r"^(제\s*\d+\s*조|[0-9]+\.)", stripped))

    @staticmethod
    def _classify_contract_type(text: str) -> str:
        lowered = text.lower()
        if "비밀유지" in text or "nda" in lowered:
            return "nda"
        if "근로" in text or "고용" in text:
            return "employment"
        if "용역" in text or "서비스" in text:
            return "service"
        return "unknown"

    @staticmethod
    def _parse_amount_value(amount_text: str) -> float | None:
        digits = re.sub(r"[^0-9]", "", amount_text)
        if not digits:
            return None
        return float(digits)
