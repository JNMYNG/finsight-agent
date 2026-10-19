from datetime import UTC, datetime
from decimal import Decimal

import pytest

from finsight.errors import InvalidResponseError
from finsight.financials import calculate_operating_margin, extract_summary, parse_amount
from finsight.models import ReportType, StatementBasis

NOW = datetime(2026, 10, 19, tzinfo=UTC)


def summary(rows, company, report=ReportType.ANNUAL, **kwargs):
    return extract_summary(rows, company, 2024, report, retrieved_at=NOW, **kwargs)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1,234,567", "1234567"),
        ("0", "0"),
        ("-12,345", "-12345"),
        ("(1,234)", "-1234"),
        ("−123", "-123"),
        (" 1,234.50 ", "1234.50"),
        ("+100", "100"),
        ("１２３", "123"),
    ],
)
def test_amount_parsing(raw, expected):
    assert parse_amount(raw) == Decimal(expected)


@pytest.mark.parametrize("raw", [None, "", " ", "-", "—", "N/A"])
def test_missing_amount_is_not_zero(raw):
    assert parse_amount(raw) is None


@pytest.mark.parametrize("raw", ["NaN", "Infinity", "12,34", "1억원", "abc", "1e3", "(-10)"])
def test_malformed_amount_rejected(raw):
    with pytest.raises(ValueError):
        parse_amount(raw)


def test_annual_summary_prefers_cfs_and_keeps_evidence(accounts_factory, company):
    rows = accounts_factory(basis="OFS", revenue="999") + accounts_factory()
    result = summary(rows, company)
    assert result.basis == StatementBasis.CONSOLIDATED
    assert result.revenue.value == Decimal("1000")
    assert result.operating_profit.value == Decimal("100")
    assert result.net_income.value == Decimal("80")
    assert result.operating_margin_pct == Decimal("10.00")
    assert result.retrieved_at == NOW
    evidence = result.revenue.evidence[0]
    assert evidence.raw_amount == "1,000"
    assert evidence.amount_field == "thstrm_amount"
    assert evidence.receipt_no in evidence.filing_url
    assert "crtfc_key" not in result.source


def test_ofs_fallback_is_explicit(accounts_factory, company):
    result = summary(accounts_factory(basis="OFS"), company)
    assert result.basis == StatementBasis.SEPARATE
    assert "FALLBACK_TO_OFS" in result.issues


def test_basis_can_be_requested_explicitly(accounts_factory, company):
    rows = accounts_factory() + accounts_factory(basis="OFS", revenue="2000")
    result = summary(rows, company, basis=StatementBasis.SEPARATE)
    assert result.revenue.value == Decimal("2000")
    assert "FALLBACK_TO_OFS" not in result.issues
    with pytest.raises(InvalidResponseError):
        summary(accounts_factory(basis="OFS"), company, basis=StatementBasis.CONSOLIDATED)


def test_missing_cfs_account_is_not_filled_from_ofs(accounts_factory, company):
    rows = accounts_factory()[1:] + accounts_factory(basis="OFS")
    result = summary(rows, company)
    assert result.basis == StatementBasis.CONSOLIDATED
    assert result.revenue.value is None
    assert result.revenue.issues == ("ACCOUNT_MISSING",)
    assert result.operating_margin_pct is None


def test_cfs_balance_sheet_only_does_not_trigger_ofs_fallback(accounts_factory, company):
    cfs = accounts_factory()[0]
    cfs.update(sj_div="BS", account_nm="자산총계")
    result = summary([cfs] + accounts_factory(basis="OFS"), company)
    assert result.basis == StatementBasis.CONSOLIDATED
    assert result.revenue.value is None


@pytest.mark.parametrize("report", [ReportType.Q1, ReportType.HALF, ReportType.Q3])
def test_interim_uses_cumulative_amount_consistently(accounts_factory, company, report):
    rows = accounts_factory(report=report)
    for row in rows:
        row["thstrm_amount"] = "1"
        row["thstrm_dt"] = "2024.01.01 ~ 2024.09.30"
    result = summary(rows, company, report)
    assert result.revenue.value == Decimal("1000")
    assert result.operating_margin_pct == Decimal("10.00")
    assert result.period_basis == "year_to_date"
    assert result.revenue.evidence[0].amount_field == "thstrm_add_amount"


def test_interim_never_falls_back_to_quarter_amount(accounts_factory, company):
    rows = accounts_factory(report="11012")
    del rows[0]["thstrm_add_amount"]
    result = summary(rows, company, ReportType.HALF)
    assert result.revenue.value is None
    assert "AMOUNT_MISSING" in result.revenue.issues


@pytest.mark.parametrize(
    "names",
    [
        ("매 출 액", "영업이익(손실)", "당기순이익(손실)"),
        ("수익(매출액)", "영업손익", "분기순이익(손실)"),
        ("영업수익", "영업손실", "반기순이익(손실)"),
        ("Ⅰ. 매출액", "2. 영업이익", "3) 당기순손익"),
    ],
)
def test_account_aliases(accounts_factory, company, names):
    rows = accounts_factory(profit="-100", net="(80)")
    for row, name in zip(rows, names, strict=True):
        row["account_nm"] = name
    result = summary(rows, company)
    assert result.operating_margin_pct == Decimal("-10.00")
    assert result.net_income.value == Decimal("-80")


def test_does_not_select_attributable_net_income_or_balance_sheet(accounts_factory, company):
    rows = accounts_factory()
    rows[2]["account_nm"] = "지배기업 소유주지분 당기순이익"
    rows[0]["sj_div"] = "BS"
    result = summary(rows, company)
    assert result.net_income.value is None
    assert result.revenue.value is None


def test_identical_duplicate_is_not_summed(accounts_factory, company):
    rows = accounts_factory()
    rows.append({**rows[0], "account_nm": "수익(매출액)"})
    result = summary(rows, company)
    assert result.revenue.value == Decimal("1000")
    assert len(result.revenue.evidence) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"thstrm_amount": "2000"},
        {"currency": "USD"},
        {"thstrm_dt": "2024.01.01 ~ 2024.09.30"},
        {"rcept_no": "20250318000001"},
        {"thstrm_amount": "-"},
    ],
)
def test_conflicting_duplicate_is_unavailable(accounts_factory, company, change):
    rows = accounts_factory()
    rows.append({**rows[0], **change})
    result = summary(rows, company)
    assert result.revenue.value is None
    assert "AMBIGUOUS_ACCOUNT" in result.revenue.issues


@pytest.mark.parametrize(
    "change,issue",
    [
        ({"thstrm_amount": "not a number"}, "INVALID_AMOUNT"),
        ({"thstrm_amount": "-"}, "AMOUNT_MISSING"),
        ({"currency": ""}, "CURRENCY_MISSING"),
        ({"thstrm_dt": ""}, "PERIOD_MISSING"),
        ({"thstrm_dt": "2024.99.99"}, "PERIOD_MISSING"),
        ({"thstrm_dt": "2024.12.31 ~ 2024.01.01"}, "PERIOD_MISSING"),
    ],
)
def test_invalid_or_incomplete_value_is_not_fabricated(accounts_factory, company, change, issue):
    rows = accounts_factory()
    rows[0].update(change)
    result = summary(rows, company)
    assert result.revenue.value is None
    assert issue in result.revenue.issues
    assert result.operating_margin_pct is None


@pytest.mark.parametrize(
    "change,issue",
    [
        ({"currency": "USD"}, "MARGIN_CURRENCY_MISMATCH"),
        ({"thstrm_dt": "2024.01.01 ~ 2024.09.30"}, "MARGIN_PERIOD_OR_FILING_MISMATCH"),
        ({"rcept_no": "20250318000001"}, "MARGIN_PERIOD_OR_FILING_MISMATCH"),
    ],
)
def test_margin_requires_matching_evidence(accounts_factory, company, change, issue):
    rows = accounts_factory()
    rows[1].update(change)
    result = summary(rows, company)
    assert result.operating_margin_pct is None
    assert issue in result.issues


@pytest.mark.parametrize(
    "revenue,profit,expected",
    [
        ("3", "1", "33.33"),
        ("100000", "12345", "12.35"),
        ("100", "-3", "-3.00"),
        ("0", "5", None),
        ("-100", "5", None),
    ],
)
def test_margin_calculation_and_rounding(revenue, profit, expected):
    actual = calculate_operating_margin(Decimal(revenue), Decimal(profit))
    assert actual == (Decimal(expected) if expected is not None else None)


@pytest.mark.parametrize(
    "change",
    [
        {"bsns_year": "2023"},
        {"reprt_code": "11012"},
        {"stock_code": "000660"},
        {"corp_code": "99999999"},
        {"fs_div": "UNKNOWN"},
        {"rcept_no": "bad"},
    ],
)
def test_wrong_identity_or_metadata_rejected(accounts_factory, company, change):
    rows = accounts_factory()
    rows[0].update(change)
    with pytest.raises(InvalidResponseError):
        summary(rows, company)
