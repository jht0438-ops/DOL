from __future__ import annotations

import io
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DART_BASE_URL = "https://opendart.fss.or.kr/api"
OUTPUT_FILE = Path(__file__).with_name("corp_codes.json")

CONNECT_TIMEOUT = 15
READ_TIMEOUT = 90
MAX_RETRIES = 4
BACKOFF_FACTOR = 1.5


def build_retry_session() -> requests.Session:
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
    return session


def download_corp_codes(api_key: str) -> list[dict[str, str]]:
    session = build_retry_session()

    try:
        response = session.get(
            f"{DART_BASE_URL}/corpCode.xml",
            params={"crtfc_key": api_key},
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        response.raise_for_status()
    finally:
        session.close()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        xml_name = next(
            name for name in zf.namelist()
            if name.lower().endswith(".xml")
        )
        xml_bytes = zf.read(xml_name)

    root = ET.fromstring(xml_bytes)

    rows: list[dict[str, str]] = []
    for item in root.findall("list"):
        row = {child.tag: (child.text or "").strip() for child in item}

        if row.get("corp_code") and row.get("corp_name"):
            rows.append(
                {
                    "corp_code": row.get("corp_code", ""),
                    "corp_name": row.get("corp_name", ""),
                    "stock_code": row.get("stock_code", ""),
                    "modify_date": row.get("modify_date", ""),
                }
            )

    return rows


def main() -> None:
    api_key = input("OpenDART API Key: ").strip()
    if not api_key:
        raise SystemExit("API Key가 입력되지 않았습니다.")

    rows = download_corp_codes(api_key)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, separators=(",", ":"))

    listed_count = sum(bool(row["stock_code"]) for row in rows)

    print(f"저장 완료: {OUTPUT_FILE}")
    print(f"전체 기업 수: {len(rows):,}")
    print(f"종목코드가 있는 기업 수: {listed_count:,}")


if __name__ == "__main__":
    main()
