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
    """EasyOCR 기반 텍스트 추출 파이프라인 (맥북 환경 최적화 및 Gemini 보정 포함)."""

    def __init__(
        self,
        *,
        lang: list[str] | None = None,
        dpi: int = 180,
        # 수정: 맥북 M시리즈 가속 충돌 방지를 위해 기본값을 False로 설정
        use_gpu: bool = False,
        correct_with_llm: bool = True,
    ) -> None:
        self.lang = lang or ["ko", "en"]
        self.dpi = dpi
        self.use_gpu = use_gpu
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

        # 모든 페이지의 텍스트 병합
        raw_text = "\n\n".join(page.text for page in pages).strip()

        # Gemini로 OCR 결과 보정 (형식 복원 및 오타 수정)
        corrected_text = self._correct_with_gemini(raw_text) if self.correct_with_llm else raw_text

        return OCRDocumentResult(
            document_id=resolved_document_id,
            source_type=source_type,
            raw_text=raw_text,
            normalized_text=self._normalize_text(corrected_text),
            pages=pages,
        )

    def _extract_from_pdf(self, path: Path) -> list[OCRPageResult]:
        pdfium = self._import_pdfium()
        pdf = pdfium.PdfDocument(str(path))
        scale = self.dpi / 72
        pages: list[OCRPageResult] = []

        for page_index in range(len(pdf)):
            page = pdf.get_page(page_index)
            # PDF 페이지를 고해상도 이미지로 렌더링
            pil_image = page.render(scale=scale).to_pil()
            image = np.array(pil_image)
            
            # OCR 수행
            pages.append(self._extract_page(image, page_index=page_index))
            page.close()

        pdf.close()
        return pages

    def _extract_page(self, image: np.ndarray, *, page_index: int) -> OCRPageResult:
        engine = self._get_ocr_engine()
        # EasyOCR: [(bbox, text, confidence), ...] 반환
        result = engine.readtext(image)
        lines = self._parse_ocr_lines(result)
        text = "\n".join(lines).strip()
        return OCRPageResult(page_index=page_index, text=text, lines=lines)

    def _get_ocr_engine(self) -> Any:
        if self._ocr_engine is None:
            # macOS 환경의 SSL 인증서 및 경로 이슈 해결
            try:
                import certifi
                os.environ.setdefault("SSL_CERT_FILE", certifi.where())
                os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
            except ImportError:
                pass

            easyocr = self._import_easyocr()
            
            # 엔진 초기화 (여기서 모델 다운로드 및 로드가 발생함)
            self._ocr_engine = easyocr.Reader(
                self.lang,
                gpu=self.use_gpu,
                verbose=True, # 모델 다운로드 상황을 보기 위해 True로 설정 추천
            )
        return self._ocr_engine

    def _correct_with_gemini(self, raw_text: str) -> str:
        """Gemini 모델을 사용해 OCR의 물리적 한계를 보완."""
        if not raw_text.strip():
            return raw_text

        try:
            from langchain_core.messages import HumanMessage
            from src.llm.gemini import get_llm
            llm = get_llm()
        except Exception:
            return raw_text

        # 텍스트가 너무 길면 토큰 제한을 위해 청크로 분리
        chunks = self._chunk_text(raw_text, max_chars=3000)
        corrected_chunks: list[str] = []

        for chunk in chunks:
            prompt = f"""다음은 OCR로 추출된 계약서 원문입니다. 가독성과 정확성을 위해 보정하세요.

규칙:
1. '제 1 조'와 같은 조항 번호의 띄어쓰기를 '제1조'로 수정하세요.
2. OCR 과정에서 발생한 한글 오탈자를 문맥에 맞게 수정하세요. (예: '갑 은' -> '갑은')
3. 계약서 특유의 계층 구조(1. 가. (1))를 최대한 유지하세요.
4. 내용을 임의로 요약하거나 변경하지 마세요.
5. 결과값은 보정된 텍스트만 출력하세요.

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
        raise ValueError(f"지원하지 않는 파일 형식입니다: {path.suffix}")

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        from PIL import Image
        image = Image.open(path).convert("RGB")
        return np.array(image)

    @staticmethod
    def _parse_ocr_lines(result: Any) -> list[str]:
        if not result:
            return []
        return [str(item[1]).strip() for item in result if item and len(item) >= 2]

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
    def _import_easyocr() -> Any:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("pip install easyocr 가 필요합니다.") from exc
        return easyocr

    @staticmethod
    def _import_pdfium() -> Any:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RuntimeError("pip install pypdfium2 가 필요합니다.") from exc
        return pdfium