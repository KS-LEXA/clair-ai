# clair-ai

CLAIR의 AI 마이크로서비스 (port 8001). OCR, 계약서 분석, RAG 기반 Q&A, 법령 준수 검사를 담당한다.

## 실행

```bash
# python3.12 필수 (anaconda 환경)
python3.12 -m uvicorn src.api.main:app --port 8001

# Swagger: http://localhost:8001/docs
```

### 테스트

```bash
python3.12 -m pytest tests/ -v
python3.12 -m pytest tests/test_ocr_pipeline.py -v        # OCR 단위 테스트
python3.12 -m pytest tests/test_contract_analysis_chain.py -v  # 분석 체인 단위 테스트
```

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 상태 확인 |
| POST | `/analyze` | 계약서 파일 분석 (파일 경로 수신) |
| POST | `/qa` | 계약서 Q&A |
| POST | `/compliance` | 법령 준수 검사 |
| GET | `/compliance/laws` | 등록된 법령 목록 조회 |

## 아키텍처

### OCR 파이프라인 (`src/ocr/pipeline.py`)

- **PDF**: PyMuPDF로 텍스트 레이어 우선 추출 (50자 이상이면 OCR 스킵). 스캔본/이미지 PDF만 PaddleOCR v5 실행
- **이미지**: PaddleOCR v5 (`PP-OCRv5_mobile_det` + `korean_PP-OCRv5_mobile_rec`)
- **텍스트 파일**: 직접 읽기
- OCR 결과는 Gemini로 오탈자 보정 (`correct_with_llm=True` 시)

### 계약서 분석 체인 (`src/chains/contract_analysis_chain.py`)

`ContractAnalysisChain.analyze_document()` 진입점 기준 실행 순서:

1. **OCR** — 텍스트 추출
2. **조항 분리** — `제N조` 패턴 기준 분리. 첫 `제N조` 이전 텍스트는 `clause-000(전문)`으로 보존
3. **핵심 정보 추출** — Gemini → 계약유형/당사자/기간/금액. Gemini 실패 시 정규식 폴백
4. **리스크 감지** — Gemini → severity(1~10), confidence(0~1) 포함 위험 조항 분류. Gemini 실패 시 9가지 키워드 폴백 (auto_renewal, termination, liability, payment, confidentiality, ip, non_compete, penalty, unilateral)
5. **요약** — Gemini → 3~5문장 한국어 요약. Gemini 실패 시 폴백 요약
6. **RAG 인덱싱** — ChromaDB + Gemini 임베딩
7. **Q&A** — RAG 검색 후 Gemini 답변. Gemini 실패 시 관련 조항 원문 발췌로 폴백
8. **법령 준수 검사** — 법령 DB 비교

모든 단계가 예외를 캐치하고 폴백 처리 — Gemini 없이도 파이프라인 완료된다.

## 환경 변수 (`.env`)

```
GEMINI_API_KEY
LAW_API_KEY    # law.go.kr API (없으면 하드코딩 법령 데이터 사용)
```

## 주의사항

- `python3.12`로 실행해야 한다 (`python3` 또는 venv python은 패키지 경로가 다를 수 있음)
- PaddleOCR은 NumPy 1.x 기준으로 컴파일됨 → `numpy<2.0.0` 제약 필수
- Apple Silicon CPU에서 PDF 스캔본 처리 시 페이지당 8~30초 소요
- Vision 파이프라인 (`src/vision/`) 은 미구현 — `detected_objects`는 항상 빈 배열
