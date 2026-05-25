"""OCRPipeline 단위 테스트.

PaddleOCR v5와 PyMuPDF(fitz)는 무거운 모델을 로드하므로 모킹 처리.
텍스트 파일 입력은 실제 I/O를 사용한다.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ocr.pipeline import OCRDocumentResult, OCRPageResult, OCRPipeline


# ── 순수 유틸 메서드 ─────────────────────────────────────────────────────────

class TestNormalizeText:
    def test_removes_excess_newlines(self):
        text = "a\n\n\n\nb"
        assert OCRPipeline._normalize_text(text) == "a\n\nb"

    def test_collapses_spaces_and_tabs(self):
        assert OCRPipeline._normalize_text("hello  \t  world") == "hello world"

    def test_normalizes_crlf(self):
        assert OCRPipeline._normalize_text("a\r\nb") == "a\nb"

    def test_strips_leading_trailing(self):
        assert OCRPipeline._normalize_text("  hello  ") == "hello"

    def test_empty_string(self):
        assert OCRPipeline._normalize_text("") == ""


class TestSplitLines:
    def test_basic_split(self):
        assert OCRPipeline._split_lines("a\nb\nc") == ["a", "b", "c"]

    def test_filters_empty_lines(self):
        assert OCRPipeline._split_lines("a\n\nb") == ["a", "b"]

    def test_strips_whitespace_per_line(self):
        assert OCRPipeline._split_lines("  hello  \n  world  ") == ["hello", "world"]


class TestDetectSourceType:
    def test_txt(self, tmp_path):
        p = tmp_path / "f.txt"
        p.touch()
        assert OCRPipeline._detect_source_type(p) == "text"

    def test_pdf(self, tmp_path):
        p = tmp_path / "f.pdf"
        p.touch()
        assert OCRPipeline._detect_source_type(p) == "pdf"

    @pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"])
    def test_image_extensions(self, tmp_path, ext):
        p = tmp_path / f"f{ext}"
        p.touch()
        assert OCRPipeline._detect_source_type(p) == "image"

    def test_unsupported_raises(self, tmp_path):
        p = tmp_path / "f.docx"
        p.touch()
        with pytest.raises(ValueError, match="Unsupported"):
            OCRPipeline._detect_source_type(p)


class TestParseOcrLines:
    def test_normal_result(self):
        # PaddleOCR v5 predict() 출력 형식: list[dict]
        result = [{"rec_texts": ["안녕하세요", "계약서"], "rec_scores": [0.99, 0.95]}]
        assert OCRPipeline._parse_ocr_lines(result) == ["안녕하세요", "계약서"]

    def test_empty_result(self):
        assert OCRPipeline._parse_ocr_lines([]) == []

    def test_filters_whitespace_only(self):
        result = [{"rec_texts": ["   ", "텍스트"], "rec_scores": [0.9, 0.9]}]
        assert OCRPipeline._parse_ocr_lines(result) == ["텍스트"]

    def test_malformed_items_skipped(self):
        # dict가 아닌 항목은 건너뜀
        result = [None, [], "string", {"rec_texts": ["ok"], "rec_scores": [0.9]}]
        lines = OCRPipeline._parse_ocr_lines(result)
        assert lines == ["ok"]

    def test_filters_low_score(self):
        result = [{"rec_texts": ["낮은신뢰도", "높은신뢰도"], "rec_scores": [0.3, 0.9]}]
        assert OCRPipeline._parse_ocr_lines(result) == ["높은신뢰도"]


class TestChunkText:
    def test_short_text_not_split(self):
        text = "abc"
        assert OCRPipeline._chunk_text(text, max_chars=100) == ["abc"]

    def test_splits_on_line_boundaries(self):
        text = "a\n" * 100  # 200자
        chunks = OCRPipeline._chunk_text(text, max_chars=20)
        assert len(chunks) > 1
        # 재결합 시 원본 복원
        assert "".join(chunks) == text

    def test_single_long_line_not_split(self):
        text = "a" * 200
        chunks = OCRPipeline._chunk_text(text, max_chars=50)
        # 줄이 하나뿐이라 분리 불가 → 청크 1개
        assert len(chunks) == 1


# ── 텍스트 파일 I/O (실제 실행) ───────────────────────────────────────────────

class TestExtractTextFile:
    def test_returns_ocr_document_result(self, sample_text_file):
        pipeline = OCRPipeline(correct_with_llm=False)
        result = pipeline.extract(sample_text_file)

        assert isinstance(result, OCRDocumentResult)
        assert result.source_type == "text"
        assert result.document_id == "contract"
        assert len(result.pages) == 1

    def test_text_content_preserved(self, sample_text_file):
        pipeline = OCRPipeline(correct_with_llm=False)
        result = pipeline.extract(sample_text_file)

        assert "제1조" in result.normalized_text
        assert "홍길동" in result.normalized_text
        assert "오천만원" in result.normalized_text

    def test_custom_document_id(self, sample_text_file):
        pipeline = OCRPipeline(correct_with_llm=False)
        result = pipeline.extract(sample_text_file, document_id="doc-001")
        assert result.document_id == "doc-001"

    def test_file_not_found_raises(self, tmp_path):
        pipeline = OCRPipeline(correct_with_llm=False)
        with pytest.raises(FileNotFoundError):
            pipeline.extract(tmp_path / "nonexistent.txt")

    def test_pages_have_lines(self, sample_text_file):
        pipeline = OCRPipeline(correct_with_llm=False)
        result = pipeline.extract(sample_text_file)
        assert len(result.pages[0].lines) > 0


# ── 이미지 입력 (PaddleOCR v5 모킹) ─────────────────────────────────────────

class TestExtractImage:
    def _make_fake_image(self, tmp_path: Path) -> Path:
        from PIL import Image
        img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
        p = tmp_path / "sample.png"
        img.save(p)
        return p

    def test_image_calls_paddleocr(self, tmp_path):
        img_path = self._make_fake_image(tmp_path)
        mock_engine = MagicMock()
        mock_engine.predict.return_value = [
            {"rec_texts": ["제1조"], "rec_scores": [0.99]},
        ]

        pipeline = OCRPipeline(correct_with_llm=False)
        with patch.object(pipeline, "_get_ocr_engine", return_value=mock_engine):
            result = pipeline.extract(img_path)

        assert result.source_type == "image"
        assert "제1조" in result.raw_text
        mock_engine.predict.assert_called_once()

    def test_image_empty_ocr_result(self, tmp_path):
        img_path = self._make_fake_image(tmp_path)
        mock_engine = MagicMock()
        mock_engine.predict.return_value = []

        pipeline = OCRPipeline(correct_with_llm=False)
        with patch.object(pipeline, "_get_ocr_engine", return_value=mock_engine):
            result = pipeline.extract(img_path)

        assert result.raw_text == ""
        assert result.pages[0].lines == []


# ── PDF 입력 (PyMuPDF/fitz + PaddleOCR v5 모킹) ──────────────────────────────

class TestExtractPdf:
    def _make_fake_pdf(self, tmp_path: Path) -> Path:
        p = tmp_path / "sample.pdf"
        p.write_bytes(b"%PDF-1.4 fake")  # 실제 파싱은 모킹
        return p

    def _make_pdf_mocks(self, num_pages: int, ocr_lines: list, native_text: str = ""):
        """fitz(PyMuPDF), PaddleOCR v5 engine mock 세트 반환.

        native_text: 각 페이지의 get_text() 반환값.
        50자 미만이면 OCR 경로, 이상이면 텍스트 레이어 경로.
        """
        fake_array = np.zeros((100, 200, 3), dtype=np.uint8)

        mock_pix = MagicMock()
        mock_pix.samples = fake_array.tobytes()
        mock_pix.height = 100
        mock_pix.width = 200
        mock_pix.n = 3

        mock_page = MagicMock()
        mock_page.get_text.return_value = native_text
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=num_pages)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        mock_engine = MagicMock()
        mock_engine.predict.return_value = ocr_lines

        return mock_fitz, mock_engine

    def test_pdf_processes_each_page(self, tmp_path):
        pdf_path = self._make_fake_pdf(tmp_path)
        mock_fitz, mock_engine = self._make_pdf_mocks(
            num_pages=2,
            ocr_lines=[{"rec_texts": ["페이지 텍스트"], "rec_scores": [0.97]}],
        )

        pipeline = OCRPipeline(correct_with_llm=False)
        with patch.object(pipeline, "_import_fitz", return_value=mock_fitz), \
             patch.object(pipeline, "_get_ocr_engine", return_value=mock_engine):
            result = pipeline.extract(pdf_path)

        assert result.source_type == "pdf"
        assert len(result.pages) == 2

    def test_pdf_joins_page_texts(self, tmp_path):
        pdf_path = self._make_fake_pdf(tmp_path)
        mock_fitz, mock_engine = self._make_pdf_mocks(
            num_pages=1,
            ocr_lines=[{"rec_texts": ["계약서 본문"], "rec_scores": [0.95]}],
        )

        pipeline = OCRPipeline(correct_with_llm=False)
        with patch.object(pipeline, "_import_fitz", return_value=mock_fitz), \
             patch.object(pipeline, "_get_ocr_engine", return_value=mock_engine):
            result = pipeline.extract(pdf_path)

        assert "계약서 본문" in result.raw_text

    def test_pdf_uses_text_layer_when_available(self, tmp_path):
        """텍스트 레이어가 충분한 페이지는 OCR을 호출하지 않아야 한다."""
        pdf_path = self._make_fake_pdf(tmp_path)
        long_text = "이것은 텍스트 레이어가 있는 페이지입니다. " * 5  # 50자 이상
        mock_fitz, mock_engine = self._make_pdf_mocks(
            num_pages=1,
            ocr_lines=[],
            native_text=long_text,
        )

        pipeline = OCRPipeline(correct_with_llm=False)
        with patch.object(pipeline, "_import_fitz", return_value=mock_fitz), \
             patch.object(pipeline, "_get_ocr_engine", return_value=mock_engine):
            result = pipeline.extract(pdf_path)

        assert long_text.strip() in result.raw_text
        mock_engine.predict.assert_not_called()
