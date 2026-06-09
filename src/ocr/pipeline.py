from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


SourceType = Literal["text", "pdf", "image"]


@dataclass(slots=True)
class OCRPageResult:
    page_index: int
    text: str
    lines: list[str]


@dataclass(slots=True)
class OCRDocumentResult:
    document_id: str
    source_type: SourceType
    raw_text: str
    normalized_text: str
    pages: list[OCRPageResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_type": self.source_type,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "pages": [asdict(page) for page in self.pages],
        }


class OCRPipeline:
    """PaddleOCR v5 기반 텍스트 추출 파이프라인 (Gemini 보정 포함).

    PDF는 PyMuPDF로 텍스트 레이어를 먼저 시도하고,
    텍스트가 부족한 페이지만 PaddleOCR로 폴백한다.
    """

    # 페이지당 텍스트가 이 글자 수 이상이면 OCR 스킵
    # 5 이하로 설정: "6 / 7" 같은 페이지 번호만 있는 페이지는 OCR 불필요
    _TEXT_LAYER_MIN_CHARS = 5

    def __init__(
        self,
        *,
        dpi: int = 150,
        correct_with_llm: bool = True,
    ) -> None:
        self.dpi = dpi
        self.correct_with_llm = correct_with_llm
        self._ocr_engine: Any | None = None

    def extract(self, source: str | Path, *, document_id: str | None = None) -> OCRDocumentResult:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"OCR source not found: {path}")

        source_type = self._detect_source_type(path)
        resolved_document_id = document_id or path.stem

        if source_type == "text":
            text = path.read_text(encoding="utf-8")
            page = OCRPageResult(page_index=0, text=text, lines=self._split_lines(text))
            return OCRDocumentResult(
                document_id=resolved_document_id,
                source_type=source_type,
                raw_text=text,
                normalized_text=self._normalize_text(text),
                pages=[page],
            )

        if source_type == "pdf":
            pages = self._extract_from_pdf(path)
        else:
            pages = [self._extract_page(self._load_image(path), page_index=0)]

        raw_text = "\n\n".join(page.text for page in pages).strip()

        corrected_text = self._correct_with_gemini(raw_text) if self.correct_with_llm else raw_text

        return OCRDocumentResult(
            document_id=resolved_document_id,
            source_type=source_type,
            raw_text=raw_text,
            normalized_text=self._normalize_text(corrected_text),
            pages=pages,
        )

    def _extract_from_pdf(self, path: Path) -> list[OCRPageResult]:
        fitz = self._import_fitz()
        doc = fitz.open(str(path))
        scale = self.dpi / 72
        pages: list[OCRPageResult] = []

        for page_index in range(len(doc)):
            pdf_page = doc[page_index]

            # 텍스트 레이어 우선 시도 — 스캔본이 아니면 OCR 불필요
            native_text = pdf_page.get_text().strip()
            if len(native_text) >= self._TEXT_LAYER_MIN_CHARS:
                lines = self._split_lines(native_text)
                pages.append(OCRPageResult(page_index=page_index, text=native_text, lines=lines))
                continue

            # 텍스트 레이어 없음 → 이미지로 렌더링 후 OCR
            pix = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = img[:, :, :3]
            pages.append(self._extract_page(img, page_index=page_index))

        doc.close()
        return pages

    def _extract_page(self, image: np.ndarray, *, page_index: int) -> OCRPageResult:
        try:
            engine = self._get_ocr_engine()
            result = engine.predict(image)
            lines = self._parse_ocr_lines(result)
            text = "\n".join(lines).strip()
        except Exception:
            text, lines = "", []
        return OCRPageResult(page_index=page_index, text=text, lines=lines)

    def _get_ocr_engine(self) -> Any:
        if self._ocr_engine is None:
            PaddleOCR = self._import_paddleocr()
            self._ocr_engine = PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="cpu",
            )
        return self._ocr_engine

    def _correct_with_gemini(self, raw_text: str) -> str:
        """OCR 결과를 Gemini로 보정: 오탈자 수정, 줄 정렬, 계약서 형식 복원."""
        if not raw_text.strip():
            return raw_text

        try:
            from langchain_core.messages import HumanMessage
            from src.llm.gemini import get_llm
            llm = get_llm()
        except Exception:
            return raw_text

        chunks = self._chunk_text(raw_text, max_chars=3000)
        corrected_chunks: list[str] = []

        for chunk in chunks:
            prompt = f"""다음은 OCR로 추출된 한국어 계약서 텍스트입니다. 아래 규칙에 따라 보정하여 출력하세요.

규칙:
1. OCR 오탈자(예: "제 1 조" → "제1조", "갑 은" → "갑은")를 수정하세요.
2. 잘못 분리된 단어를 붙이고, 잘못 합쳐진 단어를 분리하세요.
3. 계약서의 조항 구조(제N조, 번호 목록)를 보존하세요.
4. 내용을 추가하거나 삭제하지 마세요. 원문의 의미를 바꾸지 마세요.
5. 보정된 텍스트만 출력하세요. 설명이나 주석을 추가하지 마세요.

OCR 원문:
{chunk}"""

            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                corrected_chunks.append(response.content.strip())
            except Exception:
                corrected_chunks.append(chunk)

        return "\n\n".join(corrected_chunks)

    @staticmethod
    def _chunk_text(text: str, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        current_lines: list[str] = []
        current_len = 0

        for line in text.splitlines(keepends=True):
            if current_len + len(line) > max_chars and current_lines:
                chunks.append("".join(current_lines))
                current_lines = []
                current_len = 0
            current_lines.append(line)
            current_len += len(line)

        if current_lines:
            chunks.append("".join(current_lines))

        return chunks

    @staticmethod
    def _detect_source_type(path: Path) -> SourceType:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return "text"
        if suffix == ".pdf":
            return "pdf"
        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}:
            return "image"
        raise ValueError(f"Unsupported OCR source type: {path.suffix}")

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        from PIL import Image
        image = Image.open(path).convert("RGB")
        return np.array(image)

    @staticmethod
    def _parse_ocr_lines(result: Any) -> list[str]:
        """PaddleOCR v3.x predict() 결과 파싱: list[dict] 형식."""
        if not result:
            return []

        lines: list[str] = []
        for res in result:
            if not isinstance(res, dict):
                continue
            rec_texts = res.get("rec_texts", [])
            rec_scores = res.get("rec_scores", [])
            for text, score in zip(rec_texts, rec_scores):
                text = str(text).strip()
                if text and score >= 0.5:
                    lines.append(text)

        return lines

    @staticmethod
    def _split_lines(text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip()]

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    @staticmethod
    def _import_paddleocr() -> Any:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "paddleocr is not installed. Install with `pip install paddleocr`."
            ) from exc
        return PaddleOCR

    @staticmethod
    def _import_fitz() -> Any:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError(
                "pymupdf is required for PDF processing. Install with `pip install pymupdf`."
            ) from exc
        return fitz
