from __future__ import annotations

import io
import math
import zipfile
from datetime import datetime
from typing import Any
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

st.set_page_config(
    page_title="OpenDART DOL 분석",
    page_icon="📊",
    layout="wide",
)

DART_BASE_URL = "https://opendart.fss.or.kr/api"
REPORT_CODE_ANNUAL = "11011"
# OpenDART 서버가 일시적으로 느릴 때를 대비해 연결/응답 timeout을 분리합니다.
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 90
MAX_RETRIES = 4
BACKOFF_FACTOR = 1.5

# 너무 작은 변화율은 DOL을 비정상적으로 크게 만들 수 있으므로 주의 처리합니다.
MIN_SALES_CHANGE_FOR_NORMAL_INTERPRETATION = 0.01  # 1%
NEAR_ZERO_OPERATING_PROFIT_RATIO = 0.001  # 매출액 대비 0.1%
EXTREME_DOL_ABS = 10.0


# -----------------------------
# 공통 유틸리티
# -----------------------------
def get_api_key() -> str:
    """Streamlit Secrets에서 OpenDART API 키를 가져옵니다."""
    key_names = ("DART_API_KEY", "OPEN_DART_API_KEY", "OPENDART_API_KEY")
    for key_name in key_names:
        try:
            value = st.secrets.get(key_name, "")
        except Exception:
            value = ""
        if value:
            return str(value).strip()
    return ""


def parse_amount(value: Any) -> float | None:
    """OpenDART 금액 문자열을 숫자로 변환합니다."""
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "None", "nan"}:
        return None
    # 일부 공시에서 괄호로 음수를 표시하는 경우에 대비합니다.
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    try:
        return float(text)
    except ValueError:
        return None


def format_won(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    if abs_value >= 1_0000_0000_0000:
        return f"{sign}{abs_value / 1_0000_0000_0000:,.2f}조 원"
    if abs_value >= 1_0000_0000:
        return f"{sign}{abs_value / 1_0000_0000:,.1f}억 원"
    if abs_value >= 1_0000:
        return f"{sign}{abs_value / 1_0000:,.1f}만 원"
    return f"{value:,.0f}원"


def format_pct(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value * 100:+.2f}%"


def safe_growth(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return (current - previous) / abs(previous)


# -----------------------------
# OpenDART 데이터 조회
# -----------------------------
def build_retry_session() -> requests.Session:
    """
    OpenDART의 일시적인 연결 지연/5xx 오류에 자동 재시도하는 세션을 만듭니다.
    API 키가 포함된 실제 요청 URL은 화면에 출력하지 않습니다.
    """
    retry = Retry(
        total=MAX_RETRIES,
        connect=MAX_RETRIES,
        read=MAX_RETRIES,
        status=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; OpenDART-DOL-Analyzer/1.0)",
            "Accept": "*/*",
        }
    )
    return session


def dart_get(endpoint: str, params: dict[str, Any]) -> requests.Response:
    """
    OpenDART GET 요청 공통 함수.
    오류 시 API 키가 포함된 원문 예외를 그대로 노출하지 않고 일반화된 메시지만 발생시킵니다.
    """
    session = build_retry_session()
    try:
        response = session.get(
            f"{DART_BASE_URL}/{endpoint}",
            params=params,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
    except requests.exceptions.ConnectTimeout as exc:
        raise RuntimeError(
            "OpenDART 서버 연결 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."
        ) from exc
    except requests.exceptions.ReadTimeout as exc:
        raise RuntimeError(
            "OpenDART 서버 응답이 지연되고 있습니다. 잠시 후 다시 시도해 주세요."
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            "OpenDART 서버에 연결하지 못했습니다. 네트워크 상태 또는 OpenDART 서버 상태를 확인해 주세요."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            "OpenDART 통신 중 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
        ) from exc
    finally:
        session.close()

    if response.status_code >= 400:
        raise RuntimeError(
            f"OpenDART 서버가 HTTP {response.status_code} 오류를 반환했습니다. 잠시 후 다시 시도해 주세요."
        )

    return response


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_corp_codes(api_key: str) -> pd.DataFrame:
    """
    OpenDART 고유번호 ZIP/XML을 내려받아 기업 목록으로 변환합니다.
    성공한 결과는 24시간 캐시되므로 Streamlit 재실행 때마다 다시 다운로드하지 않습니다.
    """
    response = dart_get(
        "corpCode.xml",
        {"crtfc_key": api_key},
    )

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            xml_name = next(
                name for name in zf.namelist()
                if name.lower().endswith(".xml")
            )
            xml_bytes = zf.read(xml_name)
    except (zipfile.BadZipFile, StopIteration) as exc:
        raise RuntimeError(
            "기업 고유번호 파일을 해석하지 못했습니다. OpenDART 응답 또는 API 키 상태를 확인해 주세요."
        ) from exc

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise RuntimeError(
            "기업 고유번호 XML을 해석하지 못했습니다."
        ) from exc

    rows: list[dict[str, str]] = []
    for item in root.findall("list"):
        row = {child.tag: (child.text or "").strip() for child in item}
        if row.get("corp_code") and row.get("corp_name"):
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("기업 고유번호 목록이 비어 있습니다.")

    for column in ("corp_code", "corp_name", "stock_code", "modify_date"):
        if column not in df.columns:
            df[column] = ""

    df["corp_code"] = df["corp_code"].fillna("").astype(str).str.strip()
    df["stock_code"] = df["stock_code"].fillna("").astype(str).str.strip()
    df["corp_name"] = df["corp_name"].fillna("").astype(str).str.strip()

    return df[["corp_code", "corp_name", "stock_code", "modify_date"]]


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_financial_statement(
    api_key: str,
    corp_code: str,
    business_year: int,
    fs_div: str,
) -> dict[str, Any]:
    """선택 연도의 사업보고서 전체 재무제표를 조회합니다."""
    response = dart_get(
        "fnlttSinglAcntAll.json",
        {
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bsns_year": str(business_year),
            "reprt_code": REPORT_CODE_ANNUAL,
            "fs_div": fs_div,
        },
    )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("OpenDART 재무제표 응답을 JSON으로 해석하지 못했습니다.") from exc

    return data


def get_statement_with_fallback(
    api_key: str,
    corp_code: str,
    business_year: int,
) -> tuple[list[dict[str, Any]], str]:
    """연결재무제표(CFS)를 우선 조회하고 없으면 별도(OFS)로 전환합니다."""
    messages: list[str] = []

    for fs_div, label in (("CFS", "연결재무제표"), ("OFS", "별도재무제표")):
        data = fetch_financial_statement(
            api_key,
            corp_code,
            business_year,
            fs_div,
        )

        status = str(data.get("status", ""))
        if status == "000" and data.get("list"):
            return list(data["list"]), label

        # OpenDART가 반환한 공개 메시지만 사용하고 요청 URL/키는 노출하지 않습니다.
        message = str(data.get("message", "조회 실패"))
        messages.append(f"{label}: {message}")

    raise RuntimeError(" / ".join(messages))


# -----------------------------
# 계정 추출
# -----------------------------
REVENUE_ACCOUNT_IDS = {
    "ifrs-full_Revenue",
    "dart_Revenue",
    "ifrs-full_RevenueFromContractsWithCustomers",
}
OPERATING_PROFIT_ACCOUNT_IDS = {
    "dart_OperatingIncomeLoss",
    "ifrs-full_ProfitLossFromOperatingActivities",
    "ifrs-full_OperatingIncomeLoss",
}

REVENUE_NAMES_EXACT = {
    "매출액",
    "수익(매출액)",
    "영업수익",
    "수익",
    "매출",
}
OPERATING_PROFIT_NAMES_EXACT = {
    "영업이익",
    "영업이익(손실)",
    "영업손익",
    "영업손실",
}


def normalize_name(name: Any) -> str:
    return "".join(str(name or "").split()).replace("·", "")


def select_account(
    rows: list[dict[str, Any]],
    account_ids: set[str],
    exact_names: set[str],
    keyword_groups: list[tuple[str, ...]],
) -> dict[str, Any] | None:
    """계정 ID를 우선 사용하고, 계정명으로 보완합니다."""
    income_rows = [
        row for row in rows
        if str(row.get("sj_div", "")).upper() in {"IS", "CIS"}
    ]
    candidates = income_rows or rows

    # 1순위: 표준 계정 ID
    for row in candidates:
        if str(row.get("account_id", "")) in account_ids and parse_amount(row.get("thstrm_amount")) is not None:
            return row

    # 2순위: 정확한 한글 계정명
    normalized_exact = {normalize_name(name) for name in exact_names}
    for row in candidates:
        if normalize_name(row.get("account_nm")) in normalized_exact and parse_amount(row.get("thstrm_amount")) is not None:
            return row

    # 3순위: 키워드 조합
    for keywords in keyword_groups:
        for row in candidates:
            name = normalize_name(row.get("account_nm"))
            if all(keyword in name for keyword in keywords) and parse_amount(row.get("thstrm_amount")) is not None:
                return row

    return None


def extract_financial_values(rows: list[dict[str, Any]]) -> dict[str, Any]:
    revenue_row = select_account(
        rows,
        REVENUE_ACCOUNT_IDS,
        REVENUE_NAMES_EXACT,
        [("매출", "액"), ("영업", "수익")],
    )
    op_row = select_account(
        rows,
        OPERATING_PROFIT_ACCOUNT_IDS,
        OPERATING_PROFIT_NAMES_EXACT,
        [("영업", "이익"), ("영업", "손익")],
    )

    if revenue_row is None:
        raise RuntimeError("재무제표에서 매출액 계정을 찾지 못했습니다.")
    if op_row is None:
        raise RuntimeError("재무제표에서 영업이익 계정을 찾지 못했습니다.")

    result = {
        "revenue_current": parse_amount(revenue_row.get("thstrm_amount")),
        "revenue_previous": parse_amount(revenue_row.get("frmtrm_amount")),
        "op_current": parse_amount(op_row.get("thstrm_amount")),
        "op_previous": parse_amount(op_row.get("frmtrm_amount")),
        "revenue_account_name": revenue_row.get("account_nm", "매출액"),
        "op_account_name": op_row.get("account_nm", "영업이익"),
        "current_period_name": revenue_row.get("thstrm_nm", "당기"),
        "previous_period_name": revenue_row.get("frmtrm_nm", "전기"),
    }

    if any(result[key] is None for key in ("revenue_current", "revenue_previous", "op_current", "op_previous")):
        raise RuntimeError("당기 또는 전기 매출액·영업이익 값이 없어 DOL을 계산할 수 없습니다.")
    return result


# -----------------------------
# DOL 분석 로직
# -----------------------------
def classify_dol(dol: float) -> tuple[str, str]:
    if dol < 0:
        return (
            "일반적 해석 곤란",
            "매출액과 영업이익이 서로 반대 방향으로 움직였습니다. 원가율, 판관비 또는 일회성 비용 변화를 추가로 확인해야 합니다.",
        )
    if dol < 1:
        return (
            "매우 낮은 민감도",
            "영업이익의 변화폭이 매출액 변화폭보다 작았습니다. 매출 변화에 대한 이익 반응이 제한적으로 나타난 기간입니다.",
        )
    if dol < 2:
        return (
            "낮은 민감도",
            "매출 변화에 대한 영업이익의 반응이 비교적 작았습니다. 이익 안정성은 상대적으로 높을 수 있지만 성장 시 이익 확대 효과도 제한될 수 있습니다.",
        )
    if dol < 3:
        return (
            "중간 민감도",
            "매출 변화보다 영업이익이 더 크게 움직였습니다. 성장기에는 이익 확대 효과가 나타나지만 매출 감소 시 이익 감소폭도 커질 수 있습니다.",
        )
    return (
        "높은 민감도",
        "매출 변화에 비해 영업이익이 크게 반응했습니다. 성장기에는 이익 확대 효과가 크지만 매출 감소 시 영업이익이 더 큰 폭으로 줄어들 수 있습니다.",
    )


def build_reason_sentence(sales_growth: float, op_growth: float, dol: float) -> str:
    if sales_growth > 0 and op_growth > 0:
        if dol >= 3:
            return "영업이익이 매출액보다 훨씬 큰 폭으로 증가하여 높은 DOL이 계산되었습니다."
        if dol >= 1:
            return "영업이익이 매출액보다 큰 폭으로 증가하여 매출 확대에 따른 이익 증폭 효과가 나타났습니다."
        return "매출액은 증가했지만 영업이익 증가폭은 상대적으로 작아 낮은 DOL이 계산되었습니다."

    if sales_growth < 0 and op_growth < 0:
        if dol >= 3:
            return "매출액 감소보다 영업이익 감소폭이 훨씬 커 높은 하락 민감도가 나타났습니다."
        if dol >= 1:
            return "매출액 감소에 따라 영업이익도 더 큰 폭으로 감소했습니다."
        return "매출액이 감소했지만 영업이익 감소폭은 상대적으로 작아 이익 방어력이 나타났습니다."

    if sales_growth > 0 and op_growth < 0:
        return "매출액은 증가했지만 영업이익은 감소하여 일반적인 영업레버리지 해석이 어렵습니다."
    if sales_growth < 0 and op_growth > 0:
        return "매출액은 감소했지만 영업이익은 증가하여 비용구조 변화 또는 비용 절감 요인을 추가로 확인해야 합니다."
    return "매출액 또는 영업이익 변화가 거의 없어 일반적인 DOL 해석에 주의가 필요합니다."


def evaluate_reliability(
    revenue_previous: float,
    op_previous: float,
    op_current: float,
    sales_growth: float,
    dol: float | None,
) -> tuple[str, list[str]]:
    warnings: list[str] = []

    if op_previous == 0:
        warnings.append("전기 영업이익이 0이어서 영업이익 변화율을 계산할 수 없습니다.")
    elif abs(op_previous) / max(abs(revenue_previous), 1.0) < NEAR_ZERO_OPERATING_PROFIT_RATIO:
        warnings.append("전기 영업이익이 매출액에 비해 매우 작아 변화율이 과도하게 계산될 수 있습니다.")

    if abs(sales_growth) < MIN_SALES_CHANGE_FOR_NORMAL_INTERPRETATION:
        warnings.append("매출액 변화율이 1% 미만이어서 DOL이 과도하게 계산될 수 있습니다.")

    if op_previous * op_current < 0:
        warnings.append("영업손익이 흑자와 적자 사이에서 전환되어 일반적인 DOL 해석이 어렵습니다.")

    if dol is not None and abs(dol) >= EXTREME_DOL_ABS:
        warnings.append("DOL 절댓값이 10 이상으로 매우 커 결과 해석에 주의가 필요합니다.")

    if op_previous * op_current < 0 or op_previous == 0:
        return "해석 어려움", warnings
    if warnings:
        return "주의", warnings
    return "양호", ["매출액과 영업이익의 변화가 일반적인 DOL 계산 범위에 있습니다."]


def analyze_dol(values: dict[str, Any]) -> dict[str, Any]:
    revenue_current = float(values["revenue_current"])
    revenue_previous = float(values["revenue_previous"])
    op_current = float(values["op_current"])
    op_previous = float(values["op_previous"])

    sales_growth = safe_growth(revenue_current, revenue_previous)
    op_growth = safe_growth(op_current, op_previous)

    dol: float | None = None
    if sales_growth is not None and op_growth is not None and sales_growth != 0:
        dol = op_growth / sales_growth

    reliability, warnings = evaluate_reliability(
        revenue_previous,
        op_previous,
        op_current,
        sales_growth if sales_growth is not None else 0.0,
        dol,
    )

    if dol is None:
        level = "계산 불가"
        interpretation = "전기 영업이익 또는 매출액 변화율이 0이어서 추정 DOL을 계산할 수 없습니다."
        reason = interpretation
    else:
        level, interpretation = classify_dol(dol)
        reason = build_reason_sentence(sales_growth, op_growth, dol)

    return {
        **values,
        "sales_growth": sales_growth,
        "op_growth": op_growth,
        "dol": dol,
        "level": level,
        "interpretation": interpretation,
        "reason": reason,
        "reliability": reliability,
        "warnings": warnings,
    }


# -----------------------------
# 화면 구성
# -----------------------------
def render_intro() -> None:
    st.title("OpenDART 재무제표 기반 영업레버리지도(DOL) 분석")
    st.caption("공개 재무제표를 활용하여 매출 변화에 대한 영업이익의 민감도를 분석합니다.")

    st.info(
        "외부 공시 재무제표는 비용을 고정비와 변동비로 구분하지 않습니다.\n\n"
        "따라서 공개 재무제표만으로는 공헌이익을 이용한 이론적 DOL을 직접 계산하기 어렵습니다.\n\n"
        "본 도구는 매출액 변화율과 영업이익 변화율을 이용하여 **추정 DOL**을 계산합니다.\n\n"
        "추정 DOL은 매출이 1% 변화할 때 영업이익이 몇 % 변화했는지를 보여주는 지표입니다.\n\n"
        "이를 통해 기업의 영업이익 민감도를 손쉽게 확인할 수 있습니다."
    )

    with st.expander("계산 방식 자세히 보기"):
        st.markdown(
            """
외부에 공시되는 손익계산서는 일반적으로 다음 구조로 표시됩니다.

**매출액 - 매출원가 - 판매비와관리비 = 영업이익**

하지만 매출원가와 판매비와관리비에는 고정비와 변동비가 함께 포함되어 있습니다.
예를 들어 매출원가에는 원재료비, 직접노무비, 공장 감가상각비, 공장 임차료 등이 포함될 수 있고,
판매비와관리비에는 판매수수료, 운반비, 본사 인건비, 광고비, 감가상각비 등이 포함될 수 있습니다.

원재료비와 판매수수료는 변동비 성격이 강하고, 감가상각비와 임차료는 고정비 성격이 강하지만,
외부 공시 재무제표에서는 이러한 비용을 원가행태에 따라 고정비와 변동비로 구분하여 제공하지 않습니다.

따라서 OpenDART 공개 재무제표만으로는 다음과 같은 이론적 영업레버리지도를 정확하게 계산하기 어렵습니다.

**DOL = 공헌이익 ÷ 영업이익**

공헌이익을 계산하려면 변동비 총액이 필요합니다.

**공헌이익 = 매출액 - 변동비**

그러나 OpenDART에서는 변동비 총액이 별도로 공시되지 않기 때문에,
본 도구는 다음 산식을 사용합니다.

**추정 DOL = 영업이익 변화율 ÷ 매출액 변화율**

예를 들어 매출액이 10% 증가하고 영업이익이 30% 증가했다면 추정 DOL은 3입니다.
이는 해당 기간의 실적에서 매출액이 1% 변화할 때 영업이익이 약 3% 변화했다는 의미입니다.

본 도구의 DOL은 기업 내부의 고정비와 변동비 자료를 이용한 구조적 DOL이 아니라,
두 기간의 공개 재무제표 실적 변화를 바탕으로 계산한 **기간별 실현 추정치**입니다.
            """
        )

    st.subheader("DOL은 어떻게 해석할까요?")
    left, right = st.columns(2)
    with left:
        with st.expander("DOL이 높다면?"):
            st.markdown(
                """
**매출 변화보다 영업이익이 더 크게 움직이는 구조입니다.**

- 성장기에는 영업이익 확대 효과가 크게 나타날 수 있습니다.
- 매출이 감소하면 영업이익 감소폭도 더 커질 수 있습니다.
- 수요 변화에 대한 이익 민감도가 높으므로 경기와 가동률 관리가 중요합니다.
- 고정비 부담이 상대적으로 큰 구조일 가능성이 있습니다.
                """
            )
    with right:
        with st.expander("DOL이 낮다면?"):
            st.markdown(
                """
**매출 변화에 대한 영업이익의 반응이 상대적으로 작습니다.**

- 매출 감소 시 영업이익의 방어력이 비교적 높을 수 있습니다.
- 매출이 증가해도 영업이익 확대 효과는 제한적일 수 있습니다.
- 변동비 비중이 상대적으로 높은 구조일 가능성이 있습니다.
- 변동비 절감과 규모의 경제 발생 여부를 함께 살펴볼 필요가 있습니다.
                """
            )

    st.warning(
        "**DOL의 높고 낮음은 절대적인 우열을 의미하지 않습니다.** "
        "높은 DOL은 성장기에 유리하지만 매출 감소기에 위험이 커질 수 있고, "
        "낮은 DOL은 안정성이 높을 수 있지만 성장에 따른 이익 확대 효과는 제한될 수 있습니다."
    )


def render_result(result: dict[str, Any], company_name: str, stock_code: str, year: int, fs_label: str) -> None:
    st.divider()
    st.header("분석 결과")
    stock_text = f" · 종목코드 {stock_code}" if stock_code else ""
    st.caption(f"{company_name}{stock_text} · {year}년 사업보고서 · {fs_label}")

    dol = result["dol"]
    dol_text = "계산 불가" if dol is None else f"{dol:,.2f}배"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("추정 DOL", dol_text)
    c2.metric("매출액 변화율", format_pct(result["sales_growth"]))
    c3.metric("영업이익 변화율", format_pct(result["op_growth"]))
    c4.metric("분석 신뢰도", result["reliability"])

    st.subheader("계산 근거")
    table = pd.DataFrame(
        [
            {
                "항목": result["revenue_account_name"],
                result["previous_period_name"]: format_won(result["revenue_previous"]),
                result["current_period_name"]: format_won(result["revenue_current"]),
                "변화율": format_pct(result["sales_growth"]),
            },
            {
                "항목": result["op_account_name"],
                result["previous_period_name"]: format_won(result["op_previous"]),
                result["current_period_name"]: format_won(result["op_current"]),
                "변화율": format_pct(result["op_growth"]),
            },
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    if dol is not None:
        st.code(
            f"추정 DOL = {result['op_growth'] * 100:.2f}% ÷ "
            f"{result['sales_growth'] * 100:.2f}% = {dol:.2f}배",
            language=None,
        )

    st.subheader("DOL 구간 해석")
    if result["level"] == "높은 민감도":
        st.error(f"**{result['level']}**  \n{result['interpretation']}")
    elif result["level"] in {"일반적 해석 곤란", "계산 불가"}:
        st.warning(f"**{result['level']}**  \n{result['interpretation']}")
    else:
        st.info(f"**{result['level']}**  \n{result['interpretation']}")

    st.subheader("왜 이렇게 계산되었을까요?")
    st.success(result["reason"])

    st.subheader("의사결정 관점")
    if dol is None or dol < 0:
        st.markdown(
            "- 일반적인 DOL 판단보다 원가율과 판관비 변화를 먼저 확인할 필요가 있습니다.\n"
            "- 매출액과 영업이익이 다른 방향으로 움직인 원인을 별도로 분석해야 합니다."
        )
    elif dol >= 3:
        st.markdown(
            "- 매출 성장 시 영업이익 확대 효과가 크게 나타날 수 있습니다.\n"
            "- 매출 감소 시 영업이익 감소폭도 확대될 수 있습니다.\n"
            "- 경기 변화, 수요 변동, 고정비 및 가동률 관리가 중요합니다."
        )
    elif dol >= 1:
        st.markdown(
            "- 매출 변화가 영업이익에 확대되어 반영되는 구조입니다.\n"
            "- 성장성과 이익 안정성 사이의 균형을 함께 살펴볼 필요가 있습니다.\n"
            "- 동종기업 또는 과거 기간과 비교하면 해석력이 높아집니다."
        )
    else:
        st.markdown(
            "- 매출 변화에 대한 영업이익 반응이 상대적으로 작습니다.\n"
            "- 매출 감소 시 이익 방어력이 나타날 수 있습니다.\n"
            "- 변동비 절감과 규모의 경제 발생 여부를 점검할 필요가 있습니다."
        )

    st.subheader("분석 유의사항")
    st.caption(
        "본 도구의 DOL은 공개 재무제표를 이용한 기간별 실현 추정치이며, "
        "기업 내부 관리회계에서 사용하는 구조적 DOL과 차이가 있을 수 있습니다."
    )

    if result["reliability"] == "양호":
        st.success("분석 신뢰도: 양호")
    elif result["reliability"] == "주의":
        st.warning("분석 신뢰도: 주의")
    else:
        st.error("분석 신뢰도: 해석 어려움")

    for warning in result["warnings"]:
        st.write(f"- {warning}")

    with st.expander("DOL 구간 기준 보기"):
        st.markdown(
            """
| 추정 DOL | 본 도구의 해석 |
|---:|---|
| 0 미만 | 일반적인 영업레버리지 해석 곤란 |
| 0 이상 1 미만 | 매우 낮은 영업이익 민감도 |
| 1 이상 2 미만 | 낮은 영업이익 민감도 |
| 2 이상 3 미만 | 중간 수준의 영업이익 민감도 |
| 3 이상 | 높은 영업이익 민감도 |

※ 위 구간은 절대적인 업계 기준이 아니라 결과를 직관적으로 설명하기 위한 본 도구의 분류 기준입니다.
            """
        )


def main() -> None:
    render_intro()

    api_key = get_api_key()
    if not api_key:
        st.error(
            "OpenDART API 키가 설정되지 않았습니다. "
            "`.streamlit/secrets.toml`에 `DART_API_KEY = \"발급받은_키\"`를 저장해 주세요."
        )
        st.stop()

    st.divider()
    st.subheader("분석 조건")

    try:
        with st.spinner("기업 목록을 불러오고 있습니다..."):
            corp_df = load_corp_codes(api_key)
    except Exception as exc:
        st.error("기업 목록을 불러오지 못했습니다.")
        st.info(str(exc))
        if st.button("기업 목록 다시 불러오기"):
            load_corp_codes.clear()
            st.rerun()
        st.stop()

    query = st.text_input(
        "기업명 또는 6자리 종목코드",
        placeholder="예: 삼성전자 또는 005930",
    ).strip()

    selected_row: pd.Series | None = None
    if query:
        normalized_query = query.replace(" ", "").lower()
        matches = corp_df[
            corp_df["corp_name"].str.replace(" ", "", regex=False).str.lower().str.contains(normalized_query, na=False)
            | corp_df["stock_code"].eq(query.zfill(6) if query.isdigit() else query)
        ].copy()

        # 상장사를 먼저 보여주고 이름순으로 정렬합니다.
        matches["is_listed"] = matches["stock_code"].ne("")
        matches = matches.sort_values(["is_listed", "corp_name"], ascending=[False, True]).head(50)

        if matches.empty:
            st.warning("검색 결과가 없습니다. 기업명 또는 종목코드를 다시 확인해 주세요.")
        else:
            options = matches.index.tolist()

            def company_label(index: int) -> str:
                row = matches.loc[index]
                stock = f" ({row['stock_code']})" if row["stock_code"] else ""
                return f"{row['corp_name']}{stock}"

            selected_index = st.selectbox(
                "분석 대상 기업",
                options=options,
                format_func=company_label,
            )
            selected_row = matches.loc[selected_index]

    current_year = datetime.now().year
    available_years = list(range(current_year - 1, 2015, -1))
    business_year = st.selectbox("분석 사업연도", available_years, index=0)
    st.caption(f"선택한 {business_year}년 사업보고서의 당기와 전기 수치를 비교합니다.")

    analyze_clicked = st.button(
        "분석 시작",
        type="primary",
        use_container_width=True,
        disabled=selected_row is None,
    )

    if analyze_clicked and selected_row is not None:
        try:
            with st.spinner("OpenDART 재무제표를 불러오고 DOL을 계산하고 있습니다..."):
                rows, fs_label = get_statement_with_fallback(
                    api_key,
                    str(selected_row["corp_code"]),
                    int(business_year),
                )
                values = extract_financial_values(rows)
                result = analyze_dol(values)

            render_result(
                result,
                str(selected_row["corp_name"]),
                str(selected_row["stock_code"]),
                int(business_year),
                fs_label,
            )
        except Exception as exc:
            st.error("분석을 완료하지 못했습니다.")
            st.info(str(exc))


if __name__ == "__main__":
    main()
