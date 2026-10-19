"""Build the answer exclusively from successful Python tool results, never model prose."""

from decimal import Decimal
from typing import Any

from finsight.agent.tools import ToolOutcome
from finsight.models import Corporation, FinancialComparison, FinancialMetric, FinancialSummary


def _number(value: Decimal | None, unit: str = "") -> str:
    return "확인 불가" if value is None else f"{value:,}{unit}"


def _metric(label: str, metric: FinancialMetric) -> str:
    value = _number(metric.value)
    if metric.currency:
        value += f" {metric.currency}"
    issues = f" (사유: {', '.join(metric.issues)})" if metric.issues else ""
    return f"{label}: {value}{issues}"


def _summary(data: FinancialSummary) -> str:
    basis = "연결" if data.basis == "CFS" else "별도"
    period = "연간" if data.period_basis == "annual" else "당기 누적"
    lines = [
        f"{data.company.corp_name} ({data.company.corp_code}) / {data.business_year}년 "
        f"{data.report_type.label} / {basis}({data.basis}) / {period}",
        _metric("매출액", data.revenue),
        _metric("영업이익", data.operating_profit),
        _metric("당기순이익", data.net_income),
        f"영업이익률: {_number(data.operating_margin_pct, '%')}",
    ]
    if data.issues:
        lines.append(f"주의: {', '.join(data.issues)}")
    references = dict.fromkeys(
        evidence.filing_url
        for metric in (data.revenue, data.operating_profit, data.net_income)
        for evidence in metric.evidence
    )
    lines.extend(f"공시 근거: {url}" for url in references)
    lines.append(f"데이터 출처: {data.source} / 조회시각: {data.retrieved_at.isoformat()}")
    return "\n".join(lines)


def _comparison(data: FinancialComparison) -> str:
    lines = [_summary(data.left), _summary(data.right), "차이 (왼쪽 기업 - 오른쪽 기업):"]
    for label, metric in (
        ("매출액", data.revenue),
        ("영업이익", data.operating_profit),
        ("당기순이익", data.net_income),
        ("영업이익률", data.operating_margin_pct),
    ):
        unit = "%p" if metric.unit == "percentage_points" else f" {metric.unit or ''}"
        value = _number(metric.difference, unit) if metric.difference is not None else "확인 불가"
        issues = f" (사유: {', '.join(metric.issues)})" if metric.issues else ""
        lines.append(f"{label}: {value}{issues}")
    delta = data.operating_margin_pct.difference
    if delta is not None:
        if delta == 0:
            lines.append("두 기업의 영업이익률은 표시 정밀도에서 같습니다.")
        else:
            higher = data.left if delta > 0 else data.right
            lines.append(f"영업이익률이 더 높은 기업: {higher.company.corp_name}.")
    return "\n\n".join(lines[:2]) + "\n\n" + "\n".join(lines[2:])


def render_results(outcomes: list[ToolOutcome], stop_reason: str) -> str:
    sections = []
    seen: set[str] = set()
    for outcome in outcomes:
        if not outcome.ok:
            error = outcome.error or {}
            text = f"처리 실패: {error.get('code')} — {error.get('message')}"
        elif isinstance(outcome.data, Corporation):
            data = outcome.data
            text = (
                f"기업 조회: {data.corp_name} / 고유번호 {data.corp_code} / "
                f"종목코드 {data.stock_code or '없음'}"
            )
        elif isinstance(outcome.data, FinancialComparison):
            text = _comparison(outcome.data)
        elif isinstance(outcome.data, FinancialSummary):
            text = _summary(outcome.data)
        else:
            fields = (outcome.data or {}).get("missing_fields", [])
            labels = {
                "company": "기업명",
                "business_year": "사업연도",
                "report_type": "보고서 유형",
                "basis": "연결·별도 구분",
            }
            text = "추가 입력이 필요합니다: " + ", ".join(labels[field] for field in fields)
        if text not in seen:
            sections.append(text)
            seen.add(text)
    if not sections:
        sections.append("검증된 조회 결과가 없습니다. 기업명과 사업연도를 지정해 질문해 주세요.")
    if stop_reason != "finished":
        sections.append(
            f"실행 종료 사유: {stop_reason}. 위 결과는 완료된 Tool의 결과만 포함합니다."
        )
    return "\n\n".join(sections)


def has_data_issues(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value.get("issues")) or any(has_data_issues(item) for item in value.values())
    if isinstance(value, list):
        return any(has_data_issues(item) for item in value)
    return False
