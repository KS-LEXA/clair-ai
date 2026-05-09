"""공통 픽스처."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


SAMPLE_CONTRACT_TEXT = textwrap.dedent("""\
    근로계약서

    주식회사 클레어테크(이하 "갑")와 홍길동(이하 "을")은 다음과 같이 근로계약을 체결한다.

    제1조 (계약 기간)
    계약 기간은 2024년 01월 15일부터 2025년 01월 14일까지로 한다.

    제2조 (근무 장소 및 업무)
    을의 근무 장소는 서울특별시 강남구 테헤란로 123이며, 담당 업무는 소프트웨어 개발이다.

    제3조 (근로 시간)
    소정 근로 시간은 1일 8시간, 주 40시간으로 하며, 법정 기준을 초과할 수 없다.

    제4조 (임금)
    월 급여는 금 오천만원(50,000,000원)으로 하며, 매월 25일에 지급한다.

    제5조 (비밀유지)
    을은 재직 중 및 퇴직 후 2년간 갑의 영업비밀을 외부에 공개하거나 사용하여서는 아니 된다.

    제6조 (자동 갱신)
    계약 기간 만료 30일 전까지 서면으로 이의를 제기하지 않을 경우 동일한 조건으로 자동 갱신된다.

    체결일: 2024년 01월 15일
    갑: 주식회사 클레어테크
    을: 홍길동
""")


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """임시 텍스트 계약서 파일."""
    f = tmp_path / "contract.txt"
    f.write_text(SAMPLE_CONTRACT_TEXT, encoding="utf-8")
    return f


@pytest.fixture
def sample_contract_text() -> str:
    return SAMPLE_CONTRACT_TEXT
