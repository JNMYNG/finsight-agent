"""Application services reusable from CLI, future API handlers, and agent tools."""

from datetime import UTC, datetime
from decimal import Decimal, localcontext

from finsight.dart.client import DartClient
from finsight.dart.corporations import CorporationDirectory
from finsight.financials import extract_summary, period_key
from finsight.models import (
    ComparisonMetric,
    FinancialComparison,
    FinancialMetric,
    FinancialSummary,
    ReportType,
    StatementBasis,
)


def _subtract(left: Decimal, right: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = max(50, len(left.as_tuple().digits) + len(right.as_tuple().digits) + 2)
        return left - right


def compare_summaries(left: FinancialSummary, right: FinancialSummary) -> FinancialComparison:
    shared_issues: list[str] = []
    if left.business_year != right.business_year or left.report_type != right.report_type:
        shared_issues.append("REPORT_PERIOD_MISMATCH")
    if left.period_basis != right.period_basis:
        shared_issues.append("PERIOD_BASIS_MISMATCH")
    if left.basis != right.basis:
        shared_issues.append("STATEMENT_BASIS_MISMATCH")

    def compare_metric(a: FinancialMetric, b: FinancialMetric) -> ComparisonMetric:
        issues = shared_issues.copy()
        if a.value is None or b.value is None:
            issues.append("VALUE_MISSING")
        if a.currency is None or b.currency is None or a.currency != b.currency:
            issues.append("CURRENCY_MISMATCH_OR_MISSING")
        if (
            not a.evidence
            or not b.evidence
            or not period_key(a.evidence[0].period_date)
            or period_key(a.evidence[0].period_date) != period_key(b.evidence[0].period_date)
        ):
            issues.append("ACCOUNT_PERIOD_MISMATCH_OR_MISSING")
        difference = None
        if not issues and a.value is not None and b.value is not None:
            difference = _subtract(a.value, b.value)
        return ComparisonMetric(
            a.value,
            b.value,
            difference,
            a.currency if a.currency == b.currency else None,
            tuple(issues),
        )

    revenue = compare_metric(left.revenue, right.revenue)
    profit = compare_metric(left.operating_profit, right.operating_profit)
    net = compare_metric(left.net_income, right.net_income)
    margin_issues = list(dict.fromkeys(revenue.issues + profit.issues))
    a, b = left.operating_margin_pct, right.operating_margin_pct
    if a is None or b is None:
        margin_issues.append("MARGIN_MISSING")
    margin = ComparisonMetric(
        a,
        b,
        _subtract(a, b) if not margin_issues and a is not None and b is not None else None,
        "percentage_points",
        tuple(margin_issues),
    )
    return FinancialComparison(left, right, revenue, profit, net, margin)


class FinancialService:
    def __init__(self, client: DartClient) -> None:
        self.client = client
        self._directory: CorporationDirectory | None = None

    @property
    def directory(self) -> CorporationDirectory:
        if self._directory is None:
            self._directory = CorporationDirectory.from_client(self.client)
        return self._directory

    def get_financials(
        self,
        company: str,
        business_year: int,
        report_type: ReportType = ReportType.ANNUAL,
        *,
        basis: StatementBasis | None = None,
    ) -> FinancialSummary:
        corporation = self.directory.find(company)
        rows = self.client.get_major_accounts(corporation.corp_code, business_year, report_type)
        return extract_summary(
            rows,
            corporation,
            business_year,
            report_type,
            retrieved_at=datetime.now(UTC),
            basis=basis,
        )

    def compare(
        self,
        left: str,
        right: str,
        business_year: int,
        report_type: ReportType = ReportType.ANNUAL,
        *,
        basis: StatementBasis | None = None,
    ) -> FinancialComparison:
        return compare_summaries(
            self.get_financials(left, business_year, report_type, basis=basis),
            self.get_financials(right, business_year, report_type, basis=basis),
        )

    def compare_semiconductors(
        self,
        business_year: int,
        report_type: ReportType = ReportType.ANNUAL,
        *,
        basis: StatementBasis | None = None,
    ) -> FinancialComparison:
        # Stock codes identify the intended listed companies even if their names change.
        # Corporate DART codes are always resolved from corpCode.xml, never hardcoded.
        return self.compare("005930", "000660", business_year, report_type, basis=basis)
