"""
clair-ai FastAPI 서버.
clair-backend가 localhost HTTP로 호출하는 내부 AI 서비스.
기본 포트: 8001 (uvicorn src.api.main:app --port 8001)
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv(Path(__file__).parents[2] / ".env")

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from src.chains.contract_analysis_chain import Clause, ContractAnalysisChain


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 법령 DB 초기화 (Gemini 키 있을 때만)
    try:
        from src.legal.law_store import ensure_law_db_initialized
        count = ensure_law_db_initialized()
        if count > 0:
            print(f"[clair-ai] 법령 DB 초기화 완료: {count}개 조항")
        else:
            print("[clair-ai] 법령 DB 초기화 건너뜀 (GEMINI_API_KEY 없음)")
    except Exception as e:
        print(f"[clair-ai] 법령 DB 초기화 실패 (무시): {e}")
    yield


app = FastAPI(title="clair-ai", docs_url="/docs", lifespan=lifespan)
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
    use_rag: bool = True             # False이면 전체 조항 직접 LLM 전달


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
        "compliance": [asdict(c) for c in result.compliance] if result.compliance else [],
        "detected_objects": [],         # Vision 파이프라인 연동 시 채울 것
        "ocr_raw_text": result.ocr.raw_text,
        "ocr_pages": [asdict(p) for p in result.ocr.pages],
    }


class ComplianceRequest(BaseModel):
    contract_id: int
    contract_type: str | None = None   # "근로계약", "용역계약", "NDA", "임대차계약"
    clauses: list[dict[str, Any]]      # [{clause_id, title, text}]


@app.post("/compliance", tags=["법령준수검사"])
def compliance(req: ComplianceRequest) -> dict[str, Any]:
    """
    계약서 조항을 법령 DB와 비교하여 위반/주의/적합 여부 판단.
    - Gemini API 키 없으면 "검토불가" 반환
    - 서버 시작 시 법령 DB가 자동 초기화됨
    """
    from dataclasses import asdict as _asdict
    from src.legal.compliance_chain import get_compliance_chain

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
        chain = get_compliance_chain()
        results = chain.check_clauses(clauses, contract_type=req.contract_type)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return {
        "contract_id": req.contract_id,
        "contract_type": req.contract_type,
        "total_clauses_checked": len(results),
        "violation_count": sum(1 for r in results if r.status == "위반"),
        "warning_count": sum(1 for r in results if r.status == "주의"),
        "results": [_asdict(r) for r in results],
    }


@app.get("/compliance/laws", tags=["법령준수검사"])
def list_laws() -> dict[str, Any]:
    """현재 법령 DB에 인덱싱된 법령 목록 조회."""
    from src.legal.fetcher import get_all_articles
    articles = get_all_articles()
    law_summary: dict[str, list[str]] = {}
    for a in articles:
        law_summary.setdefault(a.law_name, []).append(f"{a.article_no} {a.article_title}")
    return {"laws": law_summary, "total_articles": len(articles)}


@app.post("/qa", tags=["질의응답"])
def qa(req: QARequest) -> dict[str, Any]:
    """
    clair-backend가 이미 DB에 저장한 조항 리스트를 받아 RAG Q&A 수행.
    - use_rag=True (기본): 벡터 검색으로 관련 조항만 찾아 LLM 전달
    - use_rag=False: 전체 조항을 LLM에 직접 전달 (조항 수가 적을 때 유용)
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
        if req.use_rag:
            from src.rag.rag_chain import get_rag_chain
            from src.rag.vector_store import get_vector_store

            store = get_vector_store()
            # 인덱스가 없으면 요청 조항으로 즉시 인덱싱
            if not store.has_index(req.contract_id):
                store.index_clauses(req.contract_id, clauses)

            rag = get_rag_chain()
            result = rag.answer(req.question, clauses, contract_id=req.contract_id)
        else:
            results = _chain._answer_questions([req.question], clauses)
            result = results[0] if results else None

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    if not result:
        return {"question": req.question, "answer": "답변을 생성할 수 없습니다.", "evidence_clause_ids": []}

    return {
        "question": result.question,
        "answer": result.answer,
        "evidence_clause_ids": result.evidence_clause_ids,
    }
