"""Synthetic fixtures following official field names; these are NOT real financial data."""

from io import BytesIO
from zipfile import ZipFile

import pytest

from finsight.models import Corporation

TEST_API_KEY = "t" * 40
DIRECTORY_XML = """<?xml version="1.0" encoding="UTF-8"?>
<result>
  <list><corp_code>00126380</corp_code><corp_name>삼성전자</corp_name>
    <stock_code>005930</stock_code><modify_date>20250101</modify_date></list>
  <list><corp_code>00164779</corp_code><corp_name>SK하이닉스</corp_name>
    <stock_code>000660</stock_code><modify_date>20250101</modify_date></list>
  <list><corp_code>00999999</corp_code><corp_name>테스트비상장</corp_name>
    <stock_code> </stock_code><modify_date>20250101</modify_date></list>
</result>""".encode()


def zip_xml(content: bytes = DIRECTORY_XML, name: str = "CORPCODE.xml") -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(name, content)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def isolate_secrets(tmp_path, monkeypatch):
    """Unit tests must never read the workspace .env or use real API credentials."""
    monkeypatch.chdir(tmp_path)
    for name in ("OPENDART_API_KEY", "DART_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def company():
    return Corporation("00126380", "삼성전자", "005930", "20250101")


@pytest.fixture
def accounts_factory():
    def make(
        *,
        stock="005930",
        year="2024",
        report="11011",
        basis="CFS",
        revenue="1,000",
        profit="100",
        net="80",
        currency="KRW",
    ):
        return [
            {
                "rcept_no": "20250317000001",
                "bsns_year": year,
                "stock_code": stock,
                "reprt_code": report,
                "fs_div": basis,
                "fs_nm": "연결재무제표" if basis == "CFS" else "재무제표",
                "sj_div": "IS",
                "sj_nm": "손익계산서",
                "account_nm": name,
                "thstrm_nm": "제 1 기",
                "thstrm_dt": "2024.01.01 ~ 2024.12.31",
                "thstrm_amount": amount,
                "thstrm_add_amount": amount,
                "currency": currency,
                "ord": str(index),
            }
            for index, (name, amount) in enumerate(
                [("매출액", revenue), ("영업이익", profit), ("당기순이익", net)], start=1
            )
        ]

    return make
