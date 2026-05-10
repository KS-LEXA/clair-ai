"""
법령 데이터 수집 모듈.

우선순위:
1. law.go.kr Open API (API 키 필요 — https://open.law.go.kr 에서 무료 발급)
2. 정적 폴백 데이터 (주요 위반 조항 하드코딩)

환경변수: LAW_API_KEY (law.go.kr OC 파라미터)
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LawArticle:
    law_name: str        # 법령명 (예: 근로기준법)
    article_no: str      # 조문번호 (예: 제56조)
    article_title: str   # 조문제목 (예: 연장·야간 및 휴일 근로)
    content: str         # 조문 내용
    contract_types: list[str]  # 관련 계약 유형 (예: ["근로계약"])


# ── law.go.kr API 클라이언트 ───────────────────────────────────────────────────

def fetch_from_law_api(law_name: str, max_articles: int = 50) -> list[LawArticle]:
    """
    law.go.kr Open API에서 법령 조문을 가져옴.
    2단계: lawSearch.do → 법령일련번호(MST) 조회 → lawService.do → 조문 파싱.
    API 키는 환경변수 LAW_API_KEY에 설정.
    """
    api_key = os.environ.get("LAW_API_KEY", "")
    if not api_key:
        raise RuntimeError("LAW_API_KEY가 설정되지 않았습니다.")

    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx가 필요합니다: pip install httpx") from exc

    # Step 1: 법령 검색 → MST(법령일련번호) 획득
    search_resp = httpx.get(
        "https://www.law.go.kr/DRF/lawSearch.do",
        params={"OC": api_key, "target": "law", "type": "XML", "query": law_name, "display": 1},
        timeout=10,
    )
    search_resp.raise_for_status()
    mst = _extract_mst(search_resp.text)
    if not mst:
        return []

    # Step 2: MST로 전체 조문 조회
    content_resp = httpx.get(
        "https://www.law.go.kr/DRF/lawService.do",
        params={"OC": api_key, "target": "law", "MST": mst, "type": "XML"},
        timeout=15,
    )
    content_resp.raise_for_status()

    return _parse_law_xml(content_resp.text, law_name, max_articles)


def _extract_mst(xml_text: str) -> str:
    """lawSearch.do 응답에서 법령일련번호(MST) 추출."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
        return root.findtext(".//법령일련번호", default="")
    except Exception:
        return ""


def _parse_law_xml(xml_text: str, law_name: str, max_articles: int) -> list[LawArticle]:
    """lawService.do XML 응답 파싱. <조문단위> 태그 기준."""
    import xml.etree.ElementTree as ET
    import re

    articles: list[LawArticle] = []
    try:
        root = ET.fromstring(xml_text)
        contract_types = _infer_contract_types(law_name)

        for jo in root.findall(".//조문단위"):
            # 전문(장 제목 등) 제외, 실제 조문만 파싱
            if jo.findtext("조문여부", default="") != "조문":
                continue

            no = jo.findtext("조문번호", default="").strip()
            title = jo.findtext("조문제목", default="").strip()
            content = jo.findtext("조문내용", default="").strip()

            # 조문내용이 없으면 항 내용을 합쳐서 사용
            if not content:
                parts = [h.findtext("항내용", default="").strip() for h in jo.findall("항")]
                content = " ".join(p for p in parts if p)

            # 개정 메타태그 제거 (<개정 ...>)
            content = re.sub(r"<[^>]+>", "", content).strip()

            if no and content:
                articles.append(LawArticle(
                    law_name=law_name,
                    article_no=f"제{no}조",
                    article_title=title,
                    content=content[:500],  # 너무 긴 조문은 잘라서 저장
                    contract_types=contract_types,
                ))

            if len(articles) >= max_articles:
                break

    except Exception:
        pass

    return articles


def _infer_contract_types(law_name: str) -> list[str]:
    mapping = {
        "근로기준법": ["근로계약"],
        "최저임금법": ["근로계약"],
        "민법": ["용역계약", "근로계약", "임대차계약", "NDA"],
        "하도급법": ["용역계약"],
        "주택임대차보호법": ["임대차계약"],
        "상가건물임대차보호법": ["임대차계약"],
        "부정경쟁방지법": ["NDA"],
    }
    for key, types in mapping.items():
        if key in law_name:
            return types
    return ["기타"]


# ── 정적 폴백 데이터 ──────────────────────────────────────────────────────────
# law.go.kr API 키 없이도 핵심 조항 비교 가능하도록 주요 위반 조항 하드코딩

STATIC_LAW_ARTICLES: list[LawArticle] = [

    # ── 근로기준법 ────────────────────────────────────────────────────────────
    LawArticle(
        law_name="근로기준법",
        article_no="제17조",
        article_title="근로조건의 명시",
        content="사용자는 근로계약을 체결할 때 근로자에게 임금, 소정근로시간, 휴일, 연차 유급휴가, 취업 장소, 종사 업무 등을 명시하여야 한다.",
        contract_types=["근로계약"],
    ),
    LawArticle(
        law_name="근로기준법",
        article_no="제35조",
        article_title="수습 사용 중인 근로자의 해고 예고",
        content="수습 사용한 날부터 3개월 이내인 자는 해고 예고 규정을 적용하지 아니한다. 수습 기간은 3개월을 초과할 수 없으며, 수습 기간 중 임금은 최저임금액의 100분의 90 이상을 지급하여야 한다.",
        contract_types=["근로계약"],
    ),
    LawArticle(
        law_name="근로기준법",
        article_no="제36조",
        article_title="금품 청산",
        content="사용자는 근로자가 사망 또는 퇴직한 경우에는 그 지급 사유가 발생한 때부터 14일 이내에 임금, 보상금, 그 밖에 일체의 금품을 지급하여야 한다.",
        contract_types=["근로계약"],
    ),
    LawArticle(
        law_name="근로기준법",
        article_no="제50조",
        article_title="근로시간",
        content="1주간의 근로시간은 휴게시간을 제외하고 40시간을 초과할 수 없다. 1일의 근로시간은 휴게시간을 제외하고 8시간을 초과할 수 없다.",
        contract_types=["근로계약"],
    ),
    LawArticle(
        law_name="근로기준법",
        article_no="제53조",
        article_title="연장 근로의 제한",
        content="당사자 간에 합의하면 1주간에 12시간을 한도로 근로시간을 연장할 수 있다. 1주간의 총 근로시간은 52시간을 초과할 수 없다.",
        contract_types=["근로계약"],
    ),
    LawArticle(
        law_name="근로기준법",
        article_no="제54조",
        article_title="휴게",
        content="사용자는 근로시간이 4시간인 경우에는 30분 이상, 8시간인 경우에는 1시간 이상의 휴게시간을 근로시간 도중에 주어야 한다.",
        contract_types=["근로계약"],
    ),
    LawArticle(
        law_name="근로기준법",
        article_no="제55조",
        article_title="휴일",
        content="사용자는 근로자에게 1주에 평균 1회 이상의 유급휴일을 보장하여야 한다. 사용자는 근로자에게 대통령령으로 정하는 휴일을 유급으로 보장하여야 한다.",
        contract_types=["근로계약"],
    ),
    LawArticle(
        law_name="근로기준법",
        article_no="제56조",
        article_title="연장·야간 및 휴일 근로",
        content="사용자는 연장근로(50%), 야간근로(오후 10시~오전 6시, 50%), 휴일근로(8시간 이내 50%, 8시간 초과 100%)에 대해 통상임금의 100분의 50 이상을 가산하여 지급하여야 한다.",
        contract_types=["근로계약"],
    ),
    LawArticle(
        law_name="근로기준법",
        article_no="제60조",
        article_title="연차 유급휴가",
        content="사용자는 1년간 80퍼센트 이상 출근한 근로자에게 15일의 유급휴가를 주어야 한다. 3년 이상 계속하여 근로한 근로자에게는 최초 1년을 초과하는 계속 근로 연수 매 2년에 대하여 1일을 가산한 유급휴가를 주어야 한다.",
        contract_types=["근로계약"],
    ),
    LawArticle(
        law_name="근로기준법",
        article_no="제23조",
        article_title="해고 등의 제한",
        content="사용자는 근로자에게 정당한 이유 없이 해고, 휴직, 정직, 전직, 감봉, 그 밖의 징벌을 하지 못한다.",
        contract_types=["근로계약"],
    ),

    # ── 최저임금법 ────────────────────────────────────────────────────────────
    LawArticle(
        law_name="최저임금법",
        article_no="제6조",
        article_title="최저임금의 효력",
        content="사용자는 최저임금의 적용을 받는 근로자에게 최저임금액 이상의 임금을 지급하여야 한다. 최저임금액보다 낮은 임금을 정한 근로계약은 그 부분에 한하여 이 법으로 정한 최저임금액과 동일한 임금을 지급하기로 한 것으로 본다. (2025년 최저임금: 시간당 10,030원)",
        contract_types=["근로계약"],
    ),

    # ── 민법 ──────────────────────────────────────────────────────────────────
    LawArticle(
        law_name="민법",
        article_no="제103조",
        article_title="반사회질서의 법률행위",
        content="선량한 풍속 기타 사회질서에 위반한 사항을 내용으로 하는 법률행위는 무효로 한다.",
        contract_types=["용역계약", "근로계약", "NDA", "임대차계약"],
    ),
    LawArticle(
        law_name="민법",
        article_no="제104조",
        article_title="불공정한 법률행위",
        content="당사자의 궁박, 경솔 또는 무경험으로 인하여 현저하게 공정을 잃은 법률행위는 무효로 한다.",
        contract_types=["용역계약", "근로계약"],
    ),
    LawArticle(
        law_name="민법",
        article_no="제398조",
        article_title="배상액의 예정",
        content="당사자는 채무불이행에 관한 손해배상액을 예정할 수 있다. 손해배상의 예정액이 부당히 과다한 경우에는 법원은 적당히 감액할 수 있다.",
        contract_types=["용역계약", "근로계약"],
    ),
    LawArticle(
        law_name="민법",
        article_no="제544조",
        article_title="이행지체와 해제",
        content="당사자 일방이 그 채무를 이행하지 아니하는 때에는 상대방은 상당한 기간을 정하여 그 이행을 최고하고 그 기간 내에 이행하지 아니한 때에는 계약을 해제할 수 있다.",
        contract_types=["용역계약", "근로계약", "임대차계약"],
    ),

    # ── 하도급법 ──────────────────────────────────────────────────────────────
    LawArticle(
        law_name="하도급거래 공정화에 관한 법률",
        article_no="제8조",
        article_title="부당한 하도급대금의 결정 금지",
        content="원사업자는 수급사업자에게 제조 등의 위탁을 하는 경우 부당하게 낮은 수준으로 하도급대금을 결정하거나 감액하여서는 아니 된다.",
        contract_types=["용역계약"],
    ),
    LawArticle(
        law_name="하도급거래 공정화에 관한 법률",
        article_no="제13조",
        article_title="하도급대금의 지급",
        content="원사업자는 수급사업자에게 제조 등의 위탁을 한 경우에는 목적물 등의 수령일(용역의 경우에는 수행일)부터 60일 이내의 가능한 짧은 기한으로 정한 지급기일까지 하도급대금을 지급하여야 한다.",
        contract_types=["용역계약"],
    ),

    # ── 부정경쟁방지법 ────────────────────────────────────────────────────────
    LawArticle(
        law_name="부정경쟁방지 및 영업비밀보호에 관한 법률",
        article_no="제2조",
        article_title="정의",
        content="영업비밀이란 공공연히 알려져 있지 아니하고 독립된 경제적 가치를 가지는 것으로서, 비밀로 관리된 생산방법, 판매방법, 그 밖에 영업활동에 유용한 기술상 또는 경영상의 정보를 말한다.",
        contract_types=["NDA"],
    ),
    LawArticle(
        law_name="부정경쟁방지 및 영업비밀보호에 관한 법률",
        article_no="제10조",
        article_title="영업비밀 침해행위에 대한 금지청구권 등",
        content="영업비밀의 보유자는 영업비밀 침해행위를 하거나 하려는 자에 대하여 그 행위에 의하여 영업상의 이익이 침해되거나 침해될 우려가 있는 경우에는 법원에 그 행위의 금지 또는 예방을 청구할 수 있다.",
        contract_types=["NDA"],
    ),

    # ── 주택임대차보호법 ──────────────────────────────────────────────────────
    LawArticle(
        law_name="주택임대차보호법",
        article_no="제4조",
        article_title="임대차기간 등",
        content="기간을 정하지 아니하거나 2년 미만으로 정한 임대차는 그 기간을 2년으로 본다. 다만, 임차인은 2년 미만으로 정한 기간이 유효함을 주장할 수 있다.",
        contract_types=["임대차계약"],
    ),
    LawArticle(
        law_name="주택임대차보호법",
        article_no="제6조",
        article_title="계약의 갱신",
        content="임대인이 임대차기간이 끝나기 6개월 전부터 2개월 전까지의 기간에 임차인에게 갱신거절의 통지 또는 조건을 변경하지 아니하면 갱신하지 아니한다는 뜻의 통지를 하지 아니한 경우에는 그 기간이 끝난 때에 전 임대차와 동일한 조건으로 다시 임대차한 것으로 본다.",
        contract_types=["임대차계약"],
    ),
]


def get_articles_for_contract_type(contract_type: str) -> list[LawArticle]:
    """계약 유형에 맞는 법령 조항 필터링."""
    return [a for a in STATIC_LAW_ARTICLES if contract_type in a.contract_types]


def get_all_articles() -> list[LawArticle]:
    """모든 정적 법령 조항 반환."""
    return STATIC_LAW_ARTICLES


_LAW_NAMES_BY_TYPE: dict[str, list[str]] = {
    "근로계약": ["근로기준법", "최저임금법"],
    "용역계약": ["민법", "하도급거래 공정화에 관한 법률"],
    "NDA": ["부정경쟁방지 및 영업비밀보호에 관한 법률"],
    "임대차계약": ["주택임대차보호법"],
}


def fetch_articles(contract_type: str | None = None) -> list[LawArticle]:
    """
    법령 조항 로드.
    LAW_API_KEY가 있으면 API 호출, 없으면 정적 데이터 사용.
    contract_type=None이면 모든 유형의 법령을 가져옴.
    """
    api_key = os.environ.get("LAW_API_KEY", "")
    if api_key:
        if contract_type:
            target_laws = _LAW_NAMES_BY_TYPE.get(contract_type, [])
        else:
            # 모든 유형의 법령을 중복 없이 수집
            seen: set[str] = set()
            target_laws = []
            for laws in _LAW_NAMES_BY_TYPE.values():
                for law in laws:
                    if law not in seen:
                        seen.add(law)
                        target_laws.append(law)

        articles: list[LawArticle] = []
        for law in target_laws:
            try:
                articles.extend(fetch_from_law_api(law))
            except Exception:
                pass
        if articles:
            return articles

    # 폴백: 정적 데이터
    if contract_type:
        return get_articles_for_contract_type(contract_type)
    return get_all_articles()
