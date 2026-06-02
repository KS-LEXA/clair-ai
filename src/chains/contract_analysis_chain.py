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
    severity_score: int = 5       # 1~10 심각도 수치
    confidence: float = 0.7       # 0.0~1.0 신뢰도
    title: str = ""               # 사용자 친화적 위험 조항명 (한국어)
    problematic_text: str = ""    # 위험 판단 근거 원문


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
    compliance: list[Any] | None = None  # list[ComplianceResult] — 지연 임포트

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "ocr": self.ocr.to_dict(),
            "clauses": [asdict(clause) for clause in self.clauses],
            "extraction": asdict(self.extraction),
            "risks": [asdict(risk) for risk in self.risks],
            "summary": self.summary,
            "qa": [asdict(item) for item in self.qa],
            "compliance": [asdict(c) for c in self.compliance] if self.compliance else [],
        }


class ContractAnalysisChain:
    """OCR → 조항 분리 → LLM 분석(추출/리스크/요약/Q&A) 파이프라인."""

    _clause_heading_pattern = re.compile(
        r"(?=^(제\s*\d+\s*조[^\n]*|[0-9]{1,2}\.\s*[가-힣A-Za-z][^\n]*))",
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
        import time
        t0 = time.time()
        doc_id = document_id or str(source)
        print(f"\n{'='*48}")
        print(f"🚀 [AI] 분석 시작 (ID: {doc_id})")
        print(f"📂 [AI] 파일: {source}")

        print("1️⃣  OCR 시작")
        ocr_result = self.ocr_pipeline.extract(source, document_id=document_id)
        print(f"2️⃣  OCR 완료 — 텍스트 {len(ocr_result.raw_text)}자")

        print("3️⃣  조항 분리 시작")
        clauses = self._split_clauses(ocr_result)
        print(f"4️⃣  조항 분리 완료 — {len(clauses)}개")

        print("5️⃣  핵심 정보 추출 시작")
        extraction = self._extract_fields(ocr_result.normalized_text)
        print(f"6️⃣  핵심 정보 추출 완료 — 계약유형: {extraction.contract_type.value or 'unknown'}")

        print("7️⃣  리스크 분석 시작")
        risks = self._detect_risks(clauses)
        print(f"8️⃣  리스크 분석 완료 — {len(risks)}개")

        print("9️⃣  요약 생성 시작")
        summary = self._summarize(clauses, extraction)
        print(f"🔟  요약 생성 완료")

        print("1️⃣1️⃣ 벡터 인덱싱 시작")
        contract_id = ocr_result.document_id
        self._index_clauses(contract_id, clauses)
        print("1️⃣2️⃣ 벡터 인덱싱 완료")

        print("1️⃣3️⃣ Q&A 처리 시작")
        qa = self._answer_questions(questions or [], clauses, contract_id=contract_id)
        print("1️⃣4️⃣ Q&A 처리 완료")

        print("1️⃣5️⃣ 법령 준수 검사 시작")
        contract_type = extraction.contract_type.value if extraction.contract_type.value else None
        compliance = self._check_compliance(clauses, contract_type=contract_type)
        print("1️⃣6️⃣ 법령 준수 검사 완료")

        elapsed = time.time() - t0
        print(f"✅ [AI] 분석 완료 — 소요 시간: {elapsed:.2f}초")
        print(f"{'='*48}\n")

        return ContractAnalysisResult(
            document_id=ocr_result.document_id,
            ocr=ocr_result,
            clauses=clauses,
            extraction=extraction,
            risks=risks,
            summary=summary,
            qa=qa,
            compliance=compliance,
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
            print("   ↳ Gemini 핵심 정보 추출 호출")
        except Exception as e:
            print(f"   ↳ Gemini 초기화 실패, fallback 사용: {e}")
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
        # 서양식(2024-01-15, 2024.01.15)과 한국어식(2024년 01월 15일) 모두 지원
        date_pattern = re.compile(
            r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일|\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}"
        )
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
            print("   ↳ Gemini 리스크 분석 호출")
        except Exception as e:
            print(f"   ↳ Gemini 초기화 실패, fallback 사용: {e}")
            return self._detect_risks_fallback(clauses)

        clauses_text = "\n\n".join(
            f"[{c.clause_id}] {c.title or ''}\n{c.text[:500]}"
            for c in clauses[:40]  # 최대 40개 조항
        )

        prompt = f"""계약서를 분석하여 위험 조항만 추출하세요.

최종 safety_score는 계산하지 마세요.
점수 계산은 백엔드에서 수행합니다.

각 위험 조항마다 아래 정보를 반드시 포함하세요.

- title
- category
- risk_level (low | medium | high)
- severity (1~10 정수)
- confidence (0.0~1.0)
- problematic_text
- reason

규칙:
- severity는 실제 위험도를 세밀하게 판단하세요.
- 모든 조항에 비슷한 점수를 주지 마세요.
- reason은 사용자 친화적으로 작성하세요.
- problematic_text에는 위험 판단 근거가 된 계약서 원문을 넣으세요.
- 위험하지 않은 일반 조항은 제외하세요.

category는 아래 중 하나만 사용:
payment, liability, termination, confidentiality, renewal, penalty, ip, dispute, warranty, privacy, etc

좋은 title 예시:
- "손해배상 무제한"
- "자동 갱신 조항"
- "일방적 계약 해지"

반드시 JSON 배열만 반환하세요.
마크다운, 설명, 코드블록 없이 순수 JSON만 출력하세요.

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
                    title=item.get("title", ""),
                    risk_type=item.get("category", "unknown"),
                    severity=item.get("risk_level", "medium"),
                    severity_score=max(1, min(10, int(item.get("severity", 5)))),
                    confidence=max(0.0, min(1.0, float(item.get("confidence", 0.7)))),
                    reason=item.get("reason", ""),
                    problematic_text=item.get("problematic_text", ""),
                    evidence_clause_ids=[],
                    evidence_text=item.get("problematic_text", ""),
                )
                for item in data
                if isinstance(item, dict)
            ]
        except Exception:
            return self._detect_risks_fallback(clauses)

    def _detect_risks_fallback(self, clauses: list[Clause]) -> list[RiskResult]:
        # (risk_type, severity, keywords, severity_score)
        rules = [
            ("auto_renewal",    "medium", ("자동 갱신", "묵시적 갱신", "자동으로 연장"),        5),
            ("termination",     "high",   ("일방 해지", "즉시 해지", "일방적으로 해지", "사전 통보 없이"), 8),
            ("liability",       "high",   ("손해배상", "배상 책임", "무한책임", "연대보증"),     9),
            ("payment",         "medium", ("지급 기한", "지급 조건", "지체상금", "연체이자"),    5),
            ("confidentiality", "medium", ("비밀유지", "기밀유지", "영업비밀", "정보 유출"),     5),
            ("ip",              "high",   ("지식재산권", "저작권 귀속", "특허", "발명의 권리"), 8),
            ("non_compete",     "high",   ("경업금지", "전직 금지", "동종업계", "경쟁사 취업"), 8),
            ("penalty",         "high",   ("위약금", "위약벌", "손해배상액 예정"),              9),
            ("unilateral",      "high",   ("갑의 재량", "을의 동의 없이", "일방적으로 변경"),   7),
        ]
        findings: list[RiskResult] = []
        for clause in clauses:
            for risk_type, severity, keywords, severity_score in rules:
                for keyword in keywords:
                    if keyword in clause.text:
                        findings.append(RiskResult(
                            risk_type=risk_type,
                            severity=severity,
                            severity_score=severity_score,
                            confidence=0.6,
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
            print("   ↳ Gemini 요약 호출")
        except Exception as e:
            print(f"   ↳ Gemini 초기화 실패, fallback 사용: {e}")
            return self._summarize_fallback(clauses, extraction)

        full_text = "\n\n".join(
            f"{c.title or ''}\n{c.text[:400]}"
            for c in clauses[:30]
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

    # ── 법령 준수 검사 ────────────────────────────────────────────────────────

    @staticmethod
    def _check_compliance(
        clauses: list[Clause],
        contract_type: str | None = None,
    ) -> list[Any]:
        """법령 벡터 DB와 비교하여 조항별 준수 여부 판단. 실패 시 빈 리스트 반환."""
        try:
            from src.legal.compliance_chain import get_compliance_chain
            chain = get_compliance_chain()
            return chain.check_clauses(clauses, contract_type=contract_type)
        except Exception:
            return []

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
        # 구체적인 계약서 명칭을 먼저 확인 (비밀유지 조항이 포함된 근로계약서 오분류 방지)
        if "근로계약" in text or "근로" in text or "고용" in text:
            return "근로계약"
        if "비밀유지계약" in text or "기밀유지계약" in text or "nda" in text.lower():
            return "NDA"
        if "용역" in text or "서비스" in text:
            return "용역계약"
        if "임대차" in text or "임대" in text:
            return "임대차계약"
        return "unknown"

    @staticmethod
    def _parse_amount_value(amount_text: str) -> float | None:
        digits = re.sub(r"[^0-9]", "", amount_text)
        return float(digits) if digits else None
