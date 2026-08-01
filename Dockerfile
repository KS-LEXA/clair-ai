# clair-ai — OCR + LLM 분석 마이크로서비스
#
# 이미지 크기 주의: paddlepaddle이 400MB대라 베이스가 크다.
# torch/lightning/ultralytics는 실제로 쓰이지 않아 의존성에서 제외했다
# (Vision 탐지는 미사용 — detected_objects는 항상 빈 배열).
FROM python:3.12-slim

# PaddleOCR가 내부적으로 opencv를 쓰므로 libgl/libglib이 필요하다.
# PyMuPDF는 휠에 네이티브 코드가 포함되어 별도 시스템 패키지가 없다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성을 먼저 설치해 레이어 캐시를 살린다 (소스만 바뀌면 재설치 안 함).
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY src/ ./src/

# ChromaDB(.law_db, .contract_db)와 예산 카운터(.usage_db)가 저장되는 경로.
# 반드시 영속 볼륨을 마운트할 것 — 사라지면 RAG 인덱스가 날아가고
# 월 예산 상한이 초기화되어 무력화된다.
ENV CLAIR_AI_DATA_DIR=/data
RUN mkdir -p /data
VOLUME ["/data"]

# PaddleOCR 모델(PP-OCRv5)은 최초 실행 시 내려받아 캐시한다.
ENV PADDLE_PDX_CACHE_HOME=/data/.paddlex

EXPOSE 8001

# 프리 티어는 메모리가 빠듯하므로 워커 1개로 고정한다.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
