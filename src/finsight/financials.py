"""Pure extraction and Decimal calculations; no API, filesystem, or LLM calls."""

import re
import unicodedata
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal, localcontext
from typing import Any

from finsight.dart.client import ACCOUNTS_URL
from finsight.errors import InvalidResponseError
from finsight.models import (
    Corporation,
    Evidence,
    FinancialMetric,
    FinancialSummary,
    ReportType,
    StatementBasis,
)

ACCOUNT_ALIASES = {
    "revenue": {"매출액", "매출", "수익(매출액)", "영업수익"},
    "operating_profit": {"영업이익", "영업손실", "영업이익(손실)", "영업손익"},
    "net_income": {
        "당기순이익",
        "당기순손실",
        "당기순이익(손실)",
        "당기순손익",
        "분기순이익",
        "분기순손실",
        "분기순이익(손실)",
        "반기순이익",
        "반기순손실",
        "반기순이익(손실)",
    },
}
MISSING_AMOUNTS = {"", "-", "—", "–", "N/A", "NA"}


def normalize_account(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"^\s*(?:[IVX]+|[0-9]+)[.)]\s*", "", value)
    return "".join(value.split())


def parse_amount(raw: str | None) -> Decimal | None:
    """Parse explicit numeric strings; missing values are never interpreted as zero."""
    if raw is None:
        return None
    value = unicodedata.normalize("NFKC", raw).strip()
    if value.upper() in MISSING_AMOUNTS:
        return None
    value = value.replace("−", "-")
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1].strip()
    if not re.fullmatch(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?", value):
        raise ValueError("해석할 수 없는 재무금액 형식입니다.")
    return Decimal(value.replace(",", ""))


def period_key(value: str | None) -> tuple[str, ...]:
    """Ignore formatting differences, but retain all period dates in the response."""
    if not value:
        return ()
    dates = re.findall(r"([0-9]{4})[.\-/]([0-9]{1,2})[.\-/]([0-9]{1,2})", value)
    try:
        parsed = tuple(date(int(year), int(month), int(day)) for year, month, day in dates)
    except ValueError:
        return ()
    if len(parsed) == 2 and parsed[0] > parsed[1]:
        return ()
    return tuple(item.isoformat() for item in parsed)


def calculate_operating_margin(
    revenue: Decimal | None, operating_profit: Decimal | None
) -> Decimal | None:
    if revenue is None or operating_profit is None or revenue <= 0:
        return None
    with localcontext() as context:
        context.prec = 50
        return (operating_profit / revenue * 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _optional_text(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidResponseError(f"주요계정 {key}는 문자열이어야 합니다.")
    return value.strip() or None


def _validate_rows(
    rows: list[dict[str, Any]], company: Corporation, year: int, report: ReportType
) -> None:
    for row in rows:
        # corp_code is not a documented response field for this endpoint.
        # Validate it when present, and always validate the documented stock code.
        if (
            row.get("bsns_year") != str(year)
            or row.get("reprt_code") != report.value
            or ("corp_code" in row and row["corp_code"] != company.corp_code)
            or (company.stock_code is not None and row.get("stock_code") != company.stock_code)
        ):
            raise InvalidResponseError("요청 기업·사업연도·보고서와 응답 메타데이터가 다릅니다.")
        if row.get("fs_div") not in {"CFS", "OFS"}:
            raise InvalidResponseError("응답에 유효한 연결/별도 구분(fs_div)이 없습니다.")
        if not re.fullmatch(r"[0-9]{14}", str(row.get("rcept_no", ""))):
            raise InvalidResponseError("응답에 유효한 접수번호(rcept_no)가 없습니다.")


def _extract_metric(rows: list[dict[str, Any]], name: str, amount_field: str) -> FinancialMetric:
    candidates = [
        row
        for row in rows
        if row.get("sj_div") in {"IS", "CIS"}
        and normalize_account(_optional_text(row, "account_nm") or "") in ACCOUNT_ALIASES[name]
    ]
    if not candidates:
        return FinancialMetric(None, None, issues=("ACCOUNT_MISSING",))

    evidence: list[Evidence] = []
    values: list[Decimal | None] = []
    issues: list[str] = []
    for row in candidates:
        raw = _optional_text(row, amount_field)
        currency = _optional_text(row, "currency")
        evidence.append(
            Evidence(
                account_name=row["account_nm"],
                raw_amount=raw,
                amount_field=amount_field,
                statement=row["sj_div"],
                currency=currency,
                period_name=_optional_text(row, "thstrm_nm"),
                period_date=_optional_text(row, "thstrm_dt"),
                receipt_no=row["rcept_no"],
            )
        )
        try:
            value = parse_amount(raw)
        except ValueError:
            value = None
            issues.append("INVALID_AMOUNT")
        else:
            if value is None:
                issues.append("AMOUNT_MISSING")
        values.append(value)
        if not currency:
            issues.append("CURRENCY_MISSING")
        if not period_key(evidence[-1].period_date):
            issues.append("PERIOD_MISSING")

    # Identical duplicates are allowed. Conflicting amounts, currencies, periods,
    # or filings are not resolved by arbitrary ordering or summation.
    signatures = {
        (value, item.currency, period_key(item.period_date), item.receipt_no)
        for value, item in zip(values, evidence, strict=True)
    }
    if len(signatures) != 1:
        issues.append("AMBIGUOUS_ACCOUNT")
    unique_issues = tuple(dict.fromkeys(issues))
    currency = evidence[0].currency if len({item.currency for item in evidence}) == 1 else None
    return FinancialMetric(
        value=None if unique_issues else values[0],
        currency=currency,
        evidence=tuple(evidence),
        issues=unique_issues,
    )


def extract_summary(
    rows: list[dict[str, Any]],
    company: Corporation,
    business_year: int,
    report_type: ReportType,
    *,
    retrieved_at: datetime,
    basis: StatementBasis | None = None,
) -> FinancialSummary:
    report_type = ReportType(report_type)
    _validate_rows(rows, company, business_year, report_type)
    issues: list[str] = []
    if basis is None:
        basis = StatementBasis.CONSOLIDATED
        if not any(row["fs_div"] == basis for row in rows):
            basis = StatementBasis.SEPARATE
            issues.append("FALLBACK_TO_OFS")
    basis = StatementBasis(basis)
    selected = [row for row in rows if row["fs_div"] == basis]
    if not selected:
        raise InvalidResponseError(f"요청한 재무제표 구분({basis})의 데이터가 없습니다.")
    # Always use one period policy across all three metrics. Never fall back from
    # cumulative amounts to standalone quarter amounts on a per-account basis.
    amount_field = "thstrm_amount" if report_type == ReportType.ANNUAL else "thstrm_add_amount"
    metrics = {name: _extract_metric(selected, name, amount_field) for name in ACCOUNT_ALIASES}
    revenue = metrics["revenue"]
    profit = metrics["operating_profit"]
    margin = None
    if revenue.value is None or profit.value is None:
        issues.append("MARGIN_INPUT_MISSING")
    elif revenue.currency != profit.currency:
        issues.append("MARGIN_CURRENCY_MISMATCH")
    elif (
        period_key(revenue.evidence[0].period_date) != period_key(profit.evidence[0].period_date)
        or revenue.evidence[0].receipt_no != profit.evidence[0].receipt_no
    ):
        issues.append("MARGIN_PERIOD_OR_FILING_MISMATCH")
    elif revenue.value <= 0:
        issues.append("MARGIN_NON_POSITIVE_REVENUE")
    else:
        margin = calculate_operating_margin(revenue.value, profit.value)
    return FinancialSummary(
        company=company,
        business_year=business_year,
        report_type=report_type,
        basis=basis,
        period_basis="annual" if report_type == ReportType.ANNUAL else "year_to_date",
        **metrics,
        operating_margin_pct=margin,
        issues=tuple(issues),
        source=ACCOUNTS_URL,
        retrieved_at=retrieved_at,
    )
