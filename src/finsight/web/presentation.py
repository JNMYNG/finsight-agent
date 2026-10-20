"""Evidence-only projection and display units, never financial metric calculation."""

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from finsight.agent.graph import AgentResult
from finsight.web.service import COMPANIES, AnalysisRequest

METRICS = {
    "revenue": "매출액",
    "operating_profit": "영업이익",
    "net_income": "당기순이익",
}
TOOLS = {
    "lookup_company",
    "get_financials",
    "calculate_metrics",
    "compare_companies",
    "search_disclosures",
    "request_clarification",
}


def number(value) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def format_number(value, *, suffix: str = "") -> str:
    parsed = number(value)
    return "데이터 없음" if parsed is None else f"{parsed:,}{suffix}"


def format_amount(value, currency: str | None, *, compact: bool = True) -> str:
    """Round only the label; never replace or mutate the source amount."""
    parsed = number(value)
    if parsed is None:
        return "데이터 없음"
    if currency != "KRW":
        return f"{format_number(value)} {currency or '(통화 미확인)'}"
    if compact:
        for divisor, label in ((10**12, "조 원"), (10**8, "억 원"), (10**4, "만 원")):
            if abs(parsed) >= divisor:
                display = (parsed / Decimal(divisor)).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
                return f"{display:,}{label}"
    return f"{format_number(value)} 원"


def filing_url(receipt) -> str | None:
    if isinstance(receipt, str) and re.fullmatch(r"[0-9]{14}", receipt):
        return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}"
    return None


@dataclass(frozen=True)
class Dashboard:
    status: str
    summaries: tuple[dict, ...]
    hits: tuple[dict, ...]
    calls: tuple[dict, ...]
    notices: tuple[str, ...]
    comparisons: tuple[dict, ...] = ()


def project(result: AgentResult, request: AnalysisRequest) -> Dashboard:
    summaries, hits, calls, notices, comparisons = {}, {}, [], [], {}
    for record in result.tool_calls:
        tool = record.get("tool")
        outcome = record.get("result", {})
        calls.append(
            {
                "순서": len(calls) + 1,
                "Tool": tool if tool in TOOLS else "알 수 없는 도구",
                "결과": "성공" if outcome.get("ok") else "실패",
                "소요 시간 (ms)": record.get("elapsed_ms"),
            }
        )
        if not outcome.get("ok"):
            code = (outcome.get("error") or {}).get("code", "TOOL_ERROR")
            code = (
                code if isinstance(code, str) and re.fullmatch(r"[A-Z_]+", code) else "TOOL_ERROR"
            )
            notices.append(f"도구 결과를 확보하지 못했습니다 ({code}).")
            continue
        data = outcome.get("data") or {}
        candidates = (
            [data.get("left", {}), data.get("right", {})]
            if tool == "compare_companies"
            else [data]
            if tool in {"get_financials", "calculate_metrics"}
            else []
        )
        accepted = []
        for summary in candidates:
            company = summary.get("company", {})
            name, year = company.get("corp_name"), summary.get("business_year")
            if (
                name not in request.companies
                or company.get("corp_code") != COMPANIES[name][1]
                or year not in request.years
                or summary.get("basis") != request.basis
                or summary.get("report_type") != "11011"
            ):
                notices.append("선택 조건과 일치하지 않는 재무 결과를 표시에서 제외했습니다.")
                continue
            summaries[(name, year)] = summary
            accepted.append(summary)
        if tool == "compare_companies" and len(accepted) == 2:
            left, right = accepted
            margin = data.get("operating_margin_pct", {})
            if (
                left["company"]["corp_code"] != right["company"]["corp_code"]
                and left["business_year"] == right["business_year"]
                and data.get("difference_direction") == "left_minus_right"
                and margin.get("unit") == "percentage_points"
                and not margin.get("issues")
                and margin.get("left_value") == left.get("operating_margin_pct")
                and margin.get("right_value") == right.get("operating_margin_pct")
                and number(margin.get("left_value")) is not None
                and number(margin.get("right_value")) is not None
                and number(margin.get("difference")) is not None
            ):
                comparisons[
                    (
                        left["company"]["corp_code"],
                        right["company"]["corp_code"],
                        left["business_year"],
                    )
                ] = data
        if tool == "search_disclosures":
            company = data.get("company", {})
            name = company.get("corp_name")
            if (
                not request.rag
                or name not in request.companies
                or company.get("corp_code") != COMPANIES[name][1]
                or data.get("business_year") != 2024
                or data.get("report_type") != "11011"
            ):
                notices.append("선택 조건과 일치하지 않는 문서 검색을 표시에서 제외했습니다.")
                continue
            for hit in data.get("hits", []):
                meta = hit.get("metadata", {})
                source = filing_url(meta.get("receipt_no"))
                if (
                    meta.get("corp_code") != company["corp_code"]
                    or meta.get("corp_name") != name
                    or meta.get("business_year") != 2024
                    or meta.get("report_type") != "11011"
                    or not source
                    or meta.get("source") != source
                    or not hit.get("excerpt")
                ):
                    notices.append("기업·연도·출처가 일치하지 않는 문서 근거를 제외했습니다.")
                    continue
                hits[(name, hit["chunk_id"])] = hit
    expected = {(name, year) for name in request.companies for year in request.years}
    for name, year in sorted(expected - summaries.keys()):
        notices.append(
            f"{name} {year}년 재무 조회 결과가 없습니다. 차트와 수치를 생성하지 않습니다."
        )
    for summary in summaries.values():
        if (
            summary.get("issues")
            or any(
                summary.get(key, {}).get("issues") or summary.get(key, {}).get("value") is None
                for key in METRICS
            )
            or summary.get("operating_margin_pct") is None
        ):
            notices.append("누락되거나 충돌한 재무 항목이 있습니다. 데이터 없음으로 표시합니다.")
    if request.rag:
        retrieved = {hit["metadata"]["corp_name"] for hit in hits.values()}
        for name in request.companies:
            if name not in retrieved:
                notices.append(
                    f"{name} 2024년 사업보고서 근거 부족: 관련 문장을 확보하지 못했습니다."
                )
    useful = bool(summaries or hits)
    status = result.status
    if notices and status == "completed":
        status = "partial" if useful else "error"
    return Dashboard(
        status,
        tuple(summaries.values()),
        tuple(hits.values()),
        tuple(calls),
        tuple(dict.fromkeys(notices)),
        tuple(comparisons.values()),
    )


def comparison_rows(dashboard: Dashboard, *, compact: bool = True) -> list[dict]:
    rows = []
    for summary in sorted(
        dashboard.summaries, key=lambda item: (item["business_year"], item["company"]["corp_name"])
    ):
        row = {
            "기업": summary["company"]["corp_name"],
            "사업연도": str(summary["business_year"]),
            "재무제표": summary["basis"],
        }
        currencies = {summary.get(key, {}).get("currency") or "미확인" for key in METRICS}
        row["통화"] = next(iter(currencies)) if len(currencies) == 1 else "혼합 / 항목별 표시"
        for key, label in METRICS.items():
            metric = summary.get(key, {})
            row[label] = format_amount(metric.get("value"), metric.get("currency"), compact=compact)
        row["영업이익률"] = format_number(summary.get("operating_margin_pct"), suffix="%")
        rows.append(row)
    return rows


def margin_gap_lines(dashboard: Dashboard) -> list[str]:
    """Explain the existing comparison's signed %p value, without subtracting ratios."""
    lines = []
    for comparison in sorted(dashboard.comparisons, key=lambda item: item["left"]["business_year"]):
        left, right = comparison["left"], comparison["right"]
        left_name, right_name = left["company"]["corp_name"], right["company"]["corp_name"]
        difference = number(comparison["operating_margin_pct"]["difference"])
        if difference is None:
            continue
        lead = (
            "두 기업의 영업이익률이 같습니다 (0%p)."
            if difference == 0
            else f"{left_name if difference > 0 else right_name}가 "
            f"{format_number(abs(difference), suffix='%p')} 높습니다."
        )
        lines.append(
            f"{left['business_year']}년 영업이익률 차이 · {lead} "
            f"({left_name} − {right_name}: {format_number(difference, suffix='%p')}, "
            f"{left['basis']})"
        )
    return lines


def financial_sources(summary: dict) -> list[tuple[str, str]]:
    receipts = {
        evidence.get("receipt_no")
        for key in METRICS
        for evidence in summary.get(key, {}).get("evidence", [])
    }
    return [
        (receipt, filing_url(receipt))
        for receipt in sorted(item for item in receipts if filing_url(item))
    ]


def chart_rows(dashboard: Dashboard, metric: str) -> list[dict]:
    rows = []
    for summary in dashboard.summaries:
        value = (
            summary.get("operating_margin_pct")
            if metric == "operating_margin_pct"
            else summary.get(metric, {}).get("value")
        )
        parsed = number(value)
        if metric != "operating_margin_pct" and summary.get(metric, {}).get("currency") != "KRW":
            continue  # Never combine currencies on one axis.
        rows.append(
            {
                "사업연도": str(summary["business_year"]),
                "기업": summary["company"]["corp_name"],
                "값": float(parsed) if parsed is not None else None,
                "표시값": (
                    format_number(value, suffix="%")
                    if metric == "operating_margin_pct"
                    else format_amount(value, summary.get(metric, {}).get("currency"))
                ),
                "원본": (
                    format_number(value, suffix="%")
                    if metric == "operating_margin_pct"
                    else format_amount(
                        value, summary.get(metric, {}).get("currency"), compact=False
                    )
                ),
            }
        )
    return sorted(rows, key=lambda row: (row["사업연도"], row["기업"]))


def chart_spec(metric: str, *, multiyear: bool) -> dict:
    """Vega-Lite layout only; the quantitative values remain the tool's values."""
    margin = metric == "operating_margin_pct"
    axis = {"title": "영업이익률 (%)" if margin else f"{METRICS[metric]} (원)", "tickCount": 5}
    if margin:
        axis["format"] = ".2f"
    else:
        axis["labelExpr"] = (
            "abs(datum.value) >= 1e12 ? format(datum.value / 1e12, ',.1f') + '조' : "
            "abs(datum.value) >= 1e8 ? format(datum.value / 1e8, ',.1f') + '억' : "
            "abs(datum.value) >= 1e4 ? format(datum.value / 1e4, ',.1f') + '만' : "
            "format(datum.value, ',.0f')"
        )
    return {
        "height": 240,
        "padding": {"left": 4, "top": 4, "right": 8, "bottom": 4},
        "mark": {"type": "line", "point": True}
        if multiyear
        else {
            "type": "bar",
            "size": 64,
            "cornerRadiusEnd": 4,
        },
        "encoding": {
            "x": {
                "field": "사업연도" if multiyear else "기업",
                "type": "ordinal",
                "sort": "ascending",
                "axis": {"labelAngle": 0, "title": None, "labelPadding": 8},
            },
            "y": {"field": "값", "type": "quantitative", "axis": axis, "scale": {"zero": True}},
            "color": {
                "field": "기업",
                "type": "nominal",
                "legend": {
                    "orient": "top",
                    "direction": "horizontal",
                    "title": None,
                    "offset": 8,
                }
                if multiyear
                else None,
            },
            "tooltip": [
                {"field": field, "type": "nominal"}
                for field in ("기업", "사업연도", "표시값", "원본")
            ],
        },
        "config": {"view": {"stroke": None}, "axis": {"domain": False, "labelFontSize": 12}},
    }
