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
    """PaddleOCR 기반 텍스트 추출 파이프라인."""

    def __init__(
        self,
        *,
        use_angle_cls: bool = True,
        lang: str = "korean",
        dpi: int = 180,
    ) -> None:
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.dpi = dpi
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
        return OCRDocumentResult(
            document_id=resolved_document_id,
            source_type=source_type,
            raw_text=raw_text,
            normalized_text=self._normalize_text(raw_text),
            pages=pages,
        )

    def _extract_from_pdf(self, path: Path) -> list[OCRPageResult]:
        pdfium = self._import_pdfium()
        pdf = pdfium.PdfDocument(str(path))
        scale = self.dpi / 72
        pages: list[OCRPageResult] = []

        for page_index in range(len(pdf)):
            page = pdf.get_page(page_index)
            pil_image = page.render(scale=scale).to_pil()
            image = np.array(pil_image)
            pages.append(self._extract_page(image, page_index=page_index))
            page.close()

        pdf.close()
        return pages

    def _extract_page(self, image: np.ndarray, *, page_index: int) -> OCRPageResult:
        engine = self._get_ocr_engine()
        result = engine.ocr(image)
        lines = self._parse_ocr_lines(result)
        text = "\n".join(lines).strip()
        return OCRPageResult(page_index=page_index, text=text, lines=lines)

    def _get_ocr_engine(self) -> Any:
        if self._ocr_engine is None:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            paddleocr = self._import_paddleocr()
            self._ocr_engine = paddleocr.PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                show_log=False,
            )
        return self._ocr_engine

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
        if not result:
            return []

        parsed_lines: list[str] = []
        for page_result in result:
            if not page_result:
                continue
            for item in page_result:
                if not item or len(item) < 2:
                    continue
                text_info = item[1]
                if not text_info:
                    continue
                line_text = str(text_info[0]).strip()
                if line_text:
                    parsed_lines.append(line_text)
        return parsed_lines

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
            import paddleocr
        except ImportError as exc:
            raise RuntimeError(
                "paddleocr is not installed in the active environment. "
                "Activate .venv and install dependencies with `python -m pip install -e .`."
            ) from exc
        return paddleocr

    @staticmethod
    def _import_pdfium() -> Any:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RuntimeError(
                "pypdfium2 is required for PDF OCR. Install dependencies with "
                "`python -m pip install -e .`."
            ) from exc
        return pdfium
