"""
clair-ai FastAPI 서버.
clair-backend가 localhost HTTP로 호출하는 내부 AI 서비스.
기본 포트: 8001 (uvicorn src.api.main:app --port 8001)
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from src.chains.contract_analysis_chain import Clause, ContractAnalysisChain

app = FastAPI(title="clair-ai", docs_url="/docs")
_chain = ContractAnalysisChain()


# ── 요청/응답 스키마 ──────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    contract_id: int
    file_path: str
    file_type: str
    document_id: str


class QARequest(BaseModel):
    contract_id: int
    question: str
    clauses: list[dict[str, Any]]   # [{clause_id, title, text}]


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["시스템"])
def health():
    return {"status": "ok", "service": "clair-ai"}


@app.post("/analyze", tags=["분석"])
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    path = Path(req.file_path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"파일을 찾을 수 없습니다: {req.file_path}")

    try:
        result = _chain.analyze_document(path, document_id=req.document_id)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    # clair-backend AIAnalysisResponse 형태로 변환
    return {
        "document_id": result.document_id,
        "clauses": [asdict(c) for c in result.clauses],
        "extraction": asdict(result.extraction),
        "risks": [asdict(r) for r in result.risks],
        "summary": result.summary,
        "detected_objects": [],         # Vision 파이프라인 연동 시 채울 것
        "ocr_raw_text": result.ocr.raw_text,
        "ocr_pages": [asdict(p) for p in result.ocr.pages],
    }


@app.post("/qa", tags=["질의응답"])
def qa(req: QARequest) -> dict[str, Any]:
    """
    clair-backend가 이미 DB에 저장한 조항 리스트를 받아 QA 수행.
    OCR 재실행 없이 빠르게 답변 가능.
    """
    clauses = [
        Clause(
            clause_id=c["clause_id"],
            title=c.get("title"),
            text=c["text"],
            page_refs=[],
            order=idx,
        )
        for idx, c in enumerate(req.clauses)
    ]

    try:
        results = _chain._answer_questions([req.question], clauses)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    if not results:
        return {"question": req.question, "answer": "답변을 생성할 수 없습니다.", "evidence_clause_ids": []}

    qa_result = results[0]
    return {
        "question": qa_result.question,
        "answer": qa_result.answer,
        "evidence_clause_ids": qa_result.evidence_clause_ids,
    }
