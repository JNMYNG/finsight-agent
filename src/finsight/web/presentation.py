"""Evidence-only dashboard projection. No financial arithmetic lives here."""

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

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


def project(result: AgentResult, request: AnalysisRequest) -> Dashboard:
    summaries, hits, calls, notices = {}, {}, [], []
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
    )


def comparison_rows(dashboard: Dashboard) -> list[dict]:
    rows = []
    for summary in sorted(
        dashboard.summaries, key=lambda item: (item["business_year"], item["company"]["corp_name"])
    ):
        row = {
            "기업": summary["company"]["corp_name"],
            "사업연도": str(summary["business_year"]),
            "재무제표": summary["basis"],
        }
        for key, label in METRICS.items():
            metric = summary.get(key, {})
            row[label] = format_number(metric.get("value"))
            row[f"{label} 통화"] = metric.get("currency") or "미확인"
        row["영업이익률"] = format_number(summary.get("operating_margin_pct"), suffix="%")
        rows.append(row)
    return rows


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
            }
        )
    return sorted(rows, key=lambda row: (row["사업연도"], row["기업"]))
