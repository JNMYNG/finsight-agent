"""Synthetic fixtures following official field names; these are NOT real financial data."""

from io import BytesIO
from zipfile import ZipFile

import httpx
import pytest

from finsight.dart.client import DartClient
from finsight.models import Corporation
from finsight.services import FinancialService

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
    for name in (
        "OPENDART_API_KEY",
        "DART_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_BASE_URL",
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
    ):
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


@pytest.fixture
def agent_service(accounts_factory):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("corpCode.xml"):
            return httpx.Response(200, content=zip_xml())
        stock = "005930" if request.url.params["corp_code"] == "00126380" else "000660"
        year = request.url.params["bsns_year"]
        report = request.url.params["reprt_code"]
        rows = accounts_factory(
            stock=stock, year=year, report=report, revenue="1000" if stock == "005930" else "500"
        )
        rows += accounts_factory(stock=stock, year=year, report=report, basis="OFS", revenue="2000")
        for row in rows:
            row["thstrm_dt"] = f"{year}.01.01 ~ {year}.12.31"
        return httpx.Response(200, json={"status": "000", "list": rows})

    with DartClient(TEST_API_KEY, transport=httpx.MockTransport(handler)) as client:
        yield FinancialService(client), requests
