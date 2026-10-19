from dataclasses import replace
from decimal import Decimal

import httpx
import pytest

from finsight.dart.client import DartClient
from finsight.models import ReportType
from finsight.services import FinancialService, compare_summaries

from .conftest import TEST_API_KEY, zip_xml
from .test_financials import summary


def test_full_comparison_through_mocked_http(accounts_factory):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith("corpCode.xml"):
            return httpx.Response(200, content=zip_xml())
        code = request.url.params["corp_code"]
        if code == "00126380":
            rows = accounts_factory(revenue="1000", profit="100", net="80")
        else:
            assert code == "00164779"
            rows = accounts_factory(stock="000660", revenue="500", profit="100", net="50")
        return httpx.Response(200, json={"status": "000", "list": rows})

    with DartClient(TEST_API_KEY, transport=httpx.MockTransport(handler)) as client:
        result = FinancialService(client).compare_semiconductors(2024)
    assert result.left.company.corp_name == "삼성전자"
    assert result.right.company.corp_name == "SK하이닉스"
    assert result.revenue.left_value == Decimal("1000")
    assert result.revenue.right_value == Decimal("500")
    assert result.revenue.difference == Decimal("500")
    assert result.operating_profit.difference == Decimal("0")
    assert result.net_income.difference == Decimal("30")
    assert result.operating_margin_pct.difference == Decimal("-10.00")
    assert result.operating_margin_pct.unit == "percentage_points"
    assert paths.count("/api/corpCode.xml") == 1
    assert paths.count("/api/fnlttSinglAcnt.json") == 2


@pytest.mark.parametrize("basis", ["CFS", "OFS"])
def test_matching_bases_are_comparable(accounts_factory, company, basis):
    left = summary(accounts_factory(basis=basis), company)
    right = summary(accounts_factory(basis=basis, revenue="500"), company)
    assert compare_summaries(left, right).revenue.difference == Decimal("500")


def test_mixed_statement_basis_does_not_produce_differences(accounts_factory, company):
    left = summary(accounts_factory(), company)
    right = summary(accounts_factory(basis="OFS"), company)
    result = compare_summaries(left, right)
    assert result.revenue.difference is None
    assert "STATEMENT_BASIS_MISMATCH" in result.revenue.issues
    assert result.operating_margin_pct.difference is None
    assert result.revenue.left_value == Decimal("1000")
    assert result.revenue.right_value == Decimal("1000")


@pytest.mark.parametrize(
    "change",
    [{"business_year": 2023}, {"report_type": ReportType.HALF}, {"period_basis": "different"}],
)
def test_mismatching_report_period_not_compared(accounts_factory, company, change):
    left = summary(accounts_factory(), company)
    right = replace(left, **change)
    result = compare_summaries(left, right)
    assert result.revenue.difference is None
    assert result.operating_margin_pct.difference is None


def test_currency_mismatch_not_compared(accounts_factory, company):
    left = summary(accounts_factory(), company)
    right = summary(accounts_factory(currency="USD"), company)
    result = compare_summaries(left, right)
    assert result.revenue.difference is None
    assert "CURRENCY_MISMATCH_OR_MISSING" in result.revenue.issues
    assert result.operating_margin_pct.difference is None


def test_different_account_period_not_compared(accounts_factory, company):
    left = summary(accounts_factory(), company)
    rows = accounts_factory()
    for row in rows:
        row["thstrm_dt"] = "2024.04.01 ~ 2025.03.31"
    right = summary(rows, company)
    result = compare_summaries(left, right)
    assert result.revenue.difference is None
    assert "ACCOUNT_PERIOD_MISMATCH_OR_MISSING" in result.revenue.issues


def test_equivalent_date_formats_are_comparable(accounts_factory, company):
    left = summary(accounts_factory(), company)
    rows = accounts_factory()
    for row in rows:
        row["thstrm_dt"] = "2024-1-1~2024-12-31"
    assert compare_summaries(left, summary(rows, company)).revenue.difference == Decimal("0")


def test_missing_net_income_only_blocks_that_comparison(accounts_factory, company):
    left = summary(accounts_factory(), company)
    right = summary(accounts_factory(net="-"), company)
    result = compare_summaries(left, right)
    assert result.net_income.difference is None
    assert result.revenue.difference == Decimal("0")
    assert result.operating_margin_pct.difference == Decimal("0.00")


def test_api_failure_propagates_instead_of_fabricating_comparison():
    from finsight.errors import NoDataError

    def handler(request):
        if request.url.path.endswith("corpCode.xml"):
            return httpx.Response(200, content=zip_xml())
        return httpx.Response(200, json={"status": "013", "message": "no data"})

    with DartClient(TEST_API_KEY, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(NoDataError):
            FinancialService(client).compare_semiconductors(2024)
