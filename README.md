# clair-ai

OCR, LLM 오케스트레이션, 객체 탐지, 학습 자산을 관리하는 AI 저장소다.

## 책임 범위

- LangChain 기반 계약서 분석 체인
- PaddleOCR 기반 문서 텍스트 추출
- YOLOv8 기반 인감, 서명 탐지
- Roboflow 데이터셋 버전 관리
- PyTorch Lightning 기반 학습 실험

## 권장 구조

```text
src/
  chains/
  ocr/
  vision/
  prompts/
  evaluation/
training/
models/
datasets/
```
