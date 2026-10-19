"""Immutable outputs shared by services and future tool adapters."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class ReportType(StrEnum):
    ANNUAL = "11011"
    HALF = "11012"
    Q1 = "11013"
    Q3 = "11014"

    @property
    def label(self) -> str:
        return {
            self.ANNUAL: "사업보고서",
            self.HALF: "반기보고서",
            self.Q1: "1분기보고서",
            self.Q3: "3분기보고서",
        }[self]


class StatementBasis(StrEnum):
    CONSOLIDATED = "CFS"
    SEPARATE = "OFS"


@dataclass(frozen=True)
class Corporation:
    corp_code: str
    corp_name: str
    stock_code: str | None
    modify_date: str


@dataclass(frozen=True)
class Evidence:
    account_name: str
    raw_amount: str | None
    amount_field: str
    statement: str
    currency: str | None
    period_name: str | None
    period_date: str | None
    receipt_no: str

    @property
    def filing_url(self) -> str:
        return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={self.receipt_no}"


@dataclass(frozen=True)
class FinancialMetric:
    value: Decimal | None
    currency: str | None
    evidence: tuple[Evidence, ...] = ()
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class FinancialSummary:
    company: Corporation
    business_year: int
    report_type: ReportType
    basis: StatementBasis
    period_basis: str
    revenue: FinancialMetric
    operating_profit: FinancialMetric
    net_income: FinancialMetric
    operating_margin_pct: Decimal | None
    issues: tuple[str, ...]
    source: str
    retrieved_at: datetime


@dataclass(frozen=True)
class ComparisonMetric:
    left_value: Decimal | None
    right_value: Decimal | None
    difference: Decimal | None
    unit: str | None
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class FinancialComparison:
    left: FinancialSummary
    right: FinancialSummary
    revenue: ComparisonMetric
    operating_profit: ComparisonMetric
    net_income: ComparisonMetric
    operating_margin_pct: ComparisonMetric
    difference_direction: str = "left_minus_right"
