"""OCRPipeline 단위 테스트.

EasyOCR과 pypdfium2는 무거운 모델을 로드하므로 모킹 처리.
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
        result = [([0, 0], "안녕하세요", 0.99), ([0, 0], "계약서", 0.95)]
        assert OCRPipeline._parse_ocr_lines(result) == ["안녕하세요", "계약서"]

    def test_empty_result(self):
        assert OCRPipeline._parse_ocr_lines([]) == []

    def test_filters_whitespace_only(self):
        result = [([0, 0], "   ", 0.9), ([0, 0], "텍스트", 0.9)]
        assert OCRPipeline._parse_ocr_lines(result) == ["텍스트"]

    def test_malformed_items_skipped(self):
        result = [None, [], [1], ([0], "ok", 0.9)]
        lines = OCRPipeline._parse_ocr_lines(result)
        assert lines == ["ok"]


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


# ── 이미지 입력 (EasyOCR 모킹) ───────────────────────────────────────────────

class TestExtractImage:
    def _make_fake_image(self, tmp_path: Path) -> Path:
        from PIL import Image
        img = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
        p = tmp_path / "sample.png"
        img.save(p)
        return p

    def test_image_calls_easyocr(self, tmp_path):
        img_path = self._make_fake_image(tmp_path)
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "제1조", 0.99),
        ]

        pipeline = OCRPipeline(correct_with_llm=False)
        # easyocr가 venv에 없을 수 있으므로 _get_ocr_engine 직접 패치
        with patch.object(pipeline, "_get_ocr_engine", return_value=mock_reader):
            result = pipeline.extract(img_path)

        assert result.source_type == "image"
        assert "제1조" in result.raw_text
        mock_reader.readtext.assert_called_once()

    def test_image_empty_ocr_result(self, tmp_path):
        img_path = self._make_fake_image(tmp_path)
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = []

        pipeline = OCRPipeline(correct_with_llm=False)
        with patch.object(pipeline, "_get_ocr_engine", return_value=mock_reader):
            result = pipeline.extract(img_path)

        assert result.raw_text == ""
        assert result.pages[0].lines == []


# ── PDF 입력 (pypdfium2 + EasyOCR 모킹) ──────────────────────────────────────

class TestExtractPdf:
    def _make_fake_pdf(self, tmp_path: Path) -> Path:
        p = tmp_path / "sample.pdf"
        p.write_bytes(b"%PDF-1.4 fake")  # 실제 파싱은 모킹
        return p

    def _make_pdf_mocks(self, num_pages: int, ocr_lines: list):
        """pypdfium2, OCR reader mock 세트 반환."""
        fake_image = np.zeros((100, 200, 3), dtype=np.uint8)
        from PIL import Image

        mock_page = MagicMock()
        pil_img = Image.fromarray(fake_image)
        mock_page.render.return_value.to_pil.return_value = pil_img

        mock_pdf = MagicMock()
        mock_pdf.__len__ = MagicMock(return_value=num_pages)
        mock_pdf.get_page.return_value = mock_page

        mock_pdfium = MagicMock()
        mock_pdfium.PdfDocument.return_value = mock_pdf

        mock_reader = MagicMock()
        mock_reader.readtext.return_value = ocr_lines

        return mock_pdfium, mock_reader

    def test_pdf_processes_each_page(self, tmp_path):
        pdf_path = self._make_fake_pdf(tmp_path)
        mock_pdfium, mock_reader = self._make_pdf_mocks(
            num_pages=2,
            ocr_lines=[([[0, 0], [10, 0], [10, 10], [0, 10]], "페이지 텍스트", 0.97)],
        )

        pipeline = OCRPipeline(correct_with_llm=False)
        with patch.object(pipeline, "_import_pdfium", return_value=mock_pdfium), \
             patch.object(pipeline, "_get_ocr_engine", return_value=mock_reader):
            result = pipeline.extract(pdf_path)

        assert result.source_type == "pdf"
        assert len(result.pages) == 2

    def test_pdf_joins_page_texts(self, tmp_path):
        pdf_path = self._make_fake_pdf(tmp_path)
        mock_pdfium, mock_reader = self._make_pdf_mocks(
            num_pages=1,
            ocr_lines=[([[0, 0], [1, 0], [1, 1], [0, 1]], "계약서 본문", 0.95)],
        )

        pipeline = OCRPipeline(correct_with_llm=False)
        with patch.object(pipeline, "_import_pdfium", return_value=mock_pdfium), \
             patch.object(pipeline, "_get_ocr_engine", return_value=mock_reader):
            result = pipeline.extract(pdf_path)

        assert "계약서 본문" in result.raw_text
