"""ContractAnalysisChain 단위 테스트.

LLM(Gemini), EasyOCR 등 외부 의존성은 모두 모킹.
조항 분리, 정보 추출 폴백, 리스크 감지 폴백, 요약 폴백, 전체 흐름을 검증.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.chains.contract_analysis_chain import (
    Clause,
    ContractAnalysisChain,
    ContractAnalysisResult,
    ExtractionResult,
    FieldValue,
    RiskResult,
)
from src.ocr.pipeline import OCRDocumentResult, OCRPageResult


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _make_ocr_result(text: str, document_id: str = "test-doc") -> OCRDocumentResult:
    return OCRDocumentResult(
        document_id=document_id,
        source_type="text",
        raw_text=text,
        normalized_text=text,
        pages=[OCRPageResult(page_index=0, text=text, lines=text.splitlines())],
    )


# ── _looks_like_clause_heading ────────────────────────────────────────────────

class TestLooksLikeClauseHeading:
    @pytest.mark.parametrize("heading", [
        "제1조", "제 1 조", "제12조", "1. 제목", "10. 항목",
    ])
    def test_valid_headings(self, heading):
        assert ContractAnalysisChain._looks_like_clause_heading(heading)

    @pytest.mark.parametrize("text", [
        "일반 문장입니다.", "갑은 을에게", "  ", "",
    ])
    def test_non_headings(self, text):
        assert not ContractAnalysisChain._looks_like_clause_heading(text)


# ── _classify_contract_type ───────────────────────────────────────────────────

class TestClassifyContractType:
    def test_nda(self):
        assert ContractAnalysisChain._classify_contract_type("비밀유지계약") == "NDA"

    def test_nda_english(self):
        assert ContractAnalysisChain._classify_contract_type("nda agreement") == "NDA"

    def test_employment(self):
        assert ContractAnalysisChain._classify_contract_type("근로계약서") == "근로계약"

    def test_employment_goyong(self):
        assert ContractAnalysisChain._classify_contract_type("고용 계약") == "근로계약"

    def test_service(self):
        assert ContractAnalysisChain._classify_contract_type("용역 제공 계약") == "용역계약"

    def test_unknown(self):
        assert ContractAnalysisChain._classify_contract_type("임의의 문서") == "unknown"


# ── _parse_amount_value ───────────────────────────────────────────────────────

class TestParseAmountValue:
    def test_plain_number(self):
        assert ContractAnalysisChain._parse_amount_value("50,000,000원") == 50000000.0

    def test_no_digits_returns_none(self):
        assert ContractAnalysisChain._parse_amount_value("없음") is None

    def test_mixed(self):
        result = ContractAnalysisChain._parse_amount_value("금 5,000원")
        assert result == 5000.0


# ── _split_clauses ────────────────────────────────────────────────────────────

class TestSplitClauses:
    def test_splits_korean_articles(self, sample_contract_text):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        ocr = _make_ocr_result(sample_contract_text)
        clauses = chain._split_clauses(ocr)

        assert len(clauses) >= 5  # 제1조~제6조
        ids = [c.clause_id for c in clauses]
        assert "clause-001" in ids

    def test_clause_ids_sequential(self, sample_contract_text):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        ocr = _make_ocr_result(sample_contract_text)
        clauses = chain._split_clauses(ocr)
        for i, c in enumerate(clauses, start=1):
            assert c.clause_id == f"clause-{i:03d}"

    def test_empty_text_returns_empty(self):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        ocr = _make_ocr_result("")
        assert chain._split_clauses(ocr) == []

    def test_no_headings_returns_single_clause(self):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        ocr = _make_ocr_result("그냥 텍스트입니다. 조항 없음.")
        clauses = chain._split_clauses(ocr)
        assert len(clauses) == 1
        assert clauses[0].clause_id == "clause-001"

    def test_page_refs_populated(self, sample_contract_text):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        ocr = _make_ocr_result(sample_contract_text)
        clauses = chain._split_clauses(ocr)
        assert all(isinstance(c.page_refs, list) for c in clauses)


# ── _extract_fields_fallback ──────────────────────────────────────────────────

class TestExtractFieldsFallback:
    def test_detects_contract_type(self, sample_contract_text):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        result = chain._extract_fields_fallback(sample_contract_text)
        assert result.contract_type.value == "근로계약"

    def test_detects_dates(self, sample_contract_text):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        result = chain._extract_fields_fallback(sample_contract_text)
        # 날짜 형식 2024.01.15 또는 2024-01-15 등 감지
        assert result.signing_date.value is not None or result.start_date.value is not None

    def test_detects_amount(self, sample_contract_text):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        result = chain._extract_fields_fallback(sample_contract_text)
        assert result.amount_text.value is not None
        assert "원" in result.amount_text.value

    def test_returns_extraction_result_type(self, sample_contract_text):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        result = chain._extract_fields_fallback(sample_contract_text)
        assert isinstance(result, ExtractionResult)
        assert isinstance(result.contract_type, FieldValue)


# ── _detect_risks_fallback ────────────────────────────────────────────────────

class TestDetectRisksFallback:
    def _make_clauses(self, texts: list[str]) -> list[Clause]:
        return [
            Clause(clause_id=f"clause-{i+1:03d}", title=None, text=t, page_refs=[0], order=i+1)
            for i, t in enumerate(texts)
        ]

    def test_detects_auto_renewal(self):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        clauses = self._make_clauses(["자동 갱신 조항이 있습니다."])
        risks = chain._detect_risks_fallback(clauses)
        assert any(r.risk_type == "auto_renewal" for r in risks)

    def test_detects_liability(self):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        clauses = self._make_clauses(["손해배상 책임을 부담한다."])
        risks = chain._detect_risks_fallback(clauses)
        assert any(r.risk_type == "liability" for r in risks)

    def test_no_risk_keywords(self):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        clauses = self._make_clauses(["목적물을 인도한다.", "대금을 지급한다."])
        risks = chain._detect_risks_fallback(clauses)
        assert risks == []

    def test_risk_contains_evidence(self):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        clauses = self._make_clauses(["일방 해지 가능하다."])
        risks = chain._detect_risks_fallback(clauses)
        assert risks[0].evidence_clause_ids == ["clause-001"]
        assert len(risks[0].evidence_text) > 0


# ── _summarize_fallback ───────────────────────────────────────────────────────

class TestSummarizeFallback:
    def _make_extraction(self, **kwargs) -> ExtractionResult:
        defaults = dict(
            contract_type=FieldValue("근로계약"),
            counterparty_a=FieldValue(None),
            counterparty_b=FieldValue(None),
            signing_date=FieldValue("2024-01-15"),
            start_date=FieldValue(None),
            end_date=FieldValue(None),
            amount_text=FieldValue("50,000,000원"),
            amount_value=FieldValue(50000000),
        )
        defaults.update(kwargs)
        return ExtractionResult(**defaults)

    def test_includes_clause_count(self):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        clauses = [Clause(f"clause-{i:03d}", None, "본문", [0], i) for i in range(1, 4)]
        extraction = self._make_extraction()
        summary = chain._summarize_fallback(clauses, extraction)
        assert "3" in summary

    def test_includes_contract_type(self):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        clauses = [Clause("clause-001", None, "본문", [0], 1)]
        extraction = self._make_extraction(contract_type=FieldValue("NDA"))
        summary = chain._summarize_fallback(clauses, extraction)
        assert "NDA" in summary

    def test_includes_amount(self):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        clauses = [Clause("clause-001", None, "본문", [0], 1)]
        extraction = self._make_extraction()
        summary = chain._summarize_fallback(clauses, extraction)
        assert "50,000,000원" in summary


# ── ContractAnalysisResult.to_dict ───────────────────────────────────────────

class TestContractAnalysisResultToDict:
    def test_to_dict_structure(self, sample_contract_text):
        chain = ContractAnalysisChain.__new__(ContractAnalysisChain)
        ocr = _make_ocr_result(sample_contract_text)
        clauses = chain._split_clauses(ocr)
        extraction = chain._extract_fields_fallback(sample_contract_text)
        risks = chain._detect_risks_fallback(clauses)
        summary = chain._summarize_fallback(clauses, extraction)

        result = ContractAnalysisResult(
            document_id="test",
            ocr=ocr,
            clauses=clauses,
            extraction=extraction,
            risks=risks,
            summary=summary,
            qa=[],
        )
        d = result.to_dict()

        assert "document_id" in d
        assert "clauses" in d
        assert "extraction" in d
        assert "risks" in d
        assert "summary" in d
        assert "compliance" in d
        assert isinstance(d["clauses"], list)
        assert isinstance(d["compliance"], list)


# ── analyze_document 통합 (OCRPipeline 모킹) ──────────────────────────────────

class TestAnalyzeDocument:
    def test_full_pipeline_with_text_file(self, sample_text_file):
        """LLM 없이 텍스트 파일로 전체 파이프라인 실행."""
        chain = ContractAnalysisChain(
            ocr_pipeline=MagicMock(
                **{"extract.return_value": _make_ocr_result(
                    sample_text_file.read_text(encoding="utf-8")
                )}
            )
        )

        # LLM은 RuntimeError 발생 → 폴백 경로 실행
        with patch("src.chains.contract_analysis_chain.ContractAnalysisChain._index_clauses"):
            with patch("src.chains.contract_analysis_chain.ContractAnalysisChain._check_compliance", return_value=[]):
                result = chain.analyze_document(sample_text_file)

        assert isinstance(result, ContractAnalysisResult)
        assert len(result.clauses) >= 1
        assert isinstance(result.extraction, ExtractionResult)
        assert isinstance(result.summary, str)
        assert isinstance(result.risks, list)

    def test_returns_correct_document_id(self, sample_text_file):
        chain = ContractAnalysisChain(
            ocr_pipeline=MagicMock(
                **{"extract.return_value": _make_ocr_result(
                    sample_text_file.read_text(encoding="utf-8"),
                    document_id="my-doc",
                )}
            )
        )
        with patch("src.chains.contract_analysis_chain.ContractAnalysisChain._index_clauses"):
            with patch("src.chains.contract_analysis_chain.ContractAnalysisChain._check_compliance", return_value=[]):
                result = chain.analyze_document(sample_text_file, document_id="my-doc")

        assert result.document_id == "my-doc"

    def test_empty_questions_returns_no_qa(self, sample_text_file):
        chain = ContractAnalysisChain(
            ocr_pipeline=MagicMock(
                **{"extract.return_value": _make_ocr_result(
                    sample_text_file.read_text(encoding="utf-8")
                )}
            )
        )
        with patch("src.chains.contract_analysis_chain.ContractAnalysisChain._index_clauses"):
            with patch("src.chains.contract_analysis_chain.ContractAnalysisChain._check_compliance", return_value=[]):
                result = chain.analyze_document(sample_text_file, questions=[])

        assert result.qa == []
