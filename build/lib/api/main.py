import os
import time
import requests
import traceback

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from src.chains.contract_analysis_chain import ContractAnalysisChain
from src.ocr.pipeline import OCRPipeline


app = FastAPI(title="Clair AI Server")

_chain = None


def get_chain():
    global _chain

    if _chain is None:
        print("🔧 [AI-SERVER] OCRPipeline 초기화 시작")
        ocr_pipeline = OCRPipeline(use_gpu=False)
        print("✅ [AI-SERVER] OCRPipeline 초기화 완료")

        print("🔧 [AI-SERVER] ContractAnalysisChain 초기화 시작")
        _chain = ContractAnalysisChain(ocr_pipeline=ocr_pipeline)
        print("✅ [AI-SERVER] ContractAnalysisChain 초기화 완료")

    return _chain


class AnalysisRequest(BaseModel):
    document_id: Optional[int] = Field(None, alias="contract_id")
    contract_id: Optional[int] = None
    file_path: str

    class Config:
        populate_by_name = True


@app.on_event("startup")
async def startup_event():
    get_chain()
    print("✅ [SUCCESS] AI 분석 엔진 로드 완료 (Port: 8001)")


@app.post("/analyze")
async def analyze_endpoint(req: AnalysisRequest, background_tasks: BackgroundTasks):
    final_id = req.document_id or req.contract_id

    if not final_id:
        return JSONResponse(
            status_code=422,
            content={"message": "contract_id 또는 document_id가 필요합니다."},
        )

    req.document_id = final_id

    print(f"📩 [AI-SERVER] /analyze 요청 수신")
    print(f"📩 [AI-SERVER] document_id: {req.document_id}")
    print(f"📩 [AI-SERVER] file_path: {req.file_path}")

    background_tasks.add_task(run_analysis_task, req)

    return {
        "status": "processing",
        "document_id": final_id,
    }


def run_analysis_task(req: AnalysisRequest):
    start_time = time.time()

    try:
        print("\n================ AI 분석 작업 시작 ================")
        print(f"🚀 [AI-SERVER] 분석 작업을 시작합니다 (ID: {req.document_id})")
        print(f"📂 [AI-SERVER] 파일 경로 확인: {req.file_path}")

        if not os.path.exists(req.file_path):
            print("❌ [AI-SERVER] 파일을 찾을 수 없습니다.")
            print(f"❌ [AI-SERVER] 존재하지 않는 경로: {req.file_path}")
            return

        file_size = os.path.getsize(req.file_path)
        print(f"📄 [AI-SERVER] 파일 존재 확인 완료")
        print(f"📄 [AI-SERVER] 파일 크기: {file_size} bytes")

        print("🔍 [AI-SERVER] Chain 객체 가져오기 시작")
        chain = get_chain()
        print("✅ [AI-SERVER] Chain 객체 가져오기 완료")

        print("⚙️ [AI-SERVER] chain.analyze_document 호출 직전")
        analyze_start = time.time()

        result = chain.analyze_document(
            req.file_path,
            document_id=str(req.document_id),
        )

        analyze_elapsed = time.time() - analyze_start
        print(f"✅ [AI-SERVER] chain.analyze_document 호출 완료")
        print(f"⏱️ [AI-SERVER] 분석 소요 시간: {analyze_elapsed:.2f}초")

        print("📦 [AI-SERVER] result.to_dict() 변환 시작")
        result_payload = result.to_dict()
        print("✅ [AI-SERVER] result.to_dict() 변환 완료")

        print(f"📦 [AI-SERVER] 전송 데이터 미리보기: {str(result_payload)[:500]}...")

        target_url = f"http://127.0.0.1:8000/api/v1/contracts/{req.document_id}/analyze"

        print(f"📡 [AI-SERVER] 백엔드 결과 저장 요청 시작")
        print(f"📡 [AI-SERVER] target_url: {target_url}")

        response = requests.post(
            target_url,
            json=result_payload,
            timeout=180,
        )

        print(f"📡 [AI-SERVER] 백엔드 응답 상태 코드: {response.status_code}")

        if response.status_code == 200:
            total_elapsed = time.time() - start_time
            print(f"🏁 [AI-SERVER] 최종 전송 성공! (ID: {req.document_id})")
            print(f"⏱️ [AI-SERVER] 전체 소요 시간: {total_elapsed:.2f}초")
        else:
            print(f"⚠️ [AI-SERVER] 백엔드 응답 거부")
            print(f"⚠️ [AI-SERVER] Status: {response.status_code}")
            print(f"⚠️ [AI-SERVER] Response: {response.text}")

    except Exception as e:
        print("🔥 [AI-SERVER] 분석 중 치명적 에러 발생!")
        print(f"🔥 [AI-SERVER] 에러 내용: {str(e)}")
        traceback.print_exc()

    finally:
        print("================ AI 분석 작업 종료 ================\n")
