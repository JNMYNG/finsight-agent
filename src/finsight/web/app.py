"""Streamlit UI. No clients, keys, or financial calculations are cached globally."""

from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

from finsight.errors import ConfigurationError, FinSightError
from finsight.web.presentation import (
    METRICS,
    chart_rows,
    comparison_rows,
    filing_url,
    format_number,
    project,
)
from finsight.web.service import BASIS_LABELS, COMPANIES, AnalysisRequest, run_analysis


def render_result(request, result):
    dashboard = project(result, request)
    labels = {
        "completed": "정상 완료",
        "partial": "일부 성공",
        "error": "실패",
        "needs_input": "조건 확인 필요",
        "limit_reached": "호출 한도 도달 · 일부 결과",
    }
    message = labels.get(dashboard.status, "실패")
    display = (
        st.success
        if dashboard.status == "completed"
        else (st.error if dashboard.status == "error" else st.warning)
    )
    display(message)
    st.caption(
        f"실행 조건: {', '.join(request.companies)} · "
        f"{', '.join(map(str, sorted(request.years)))}년 · "
        f"{BASIS_LABELS[request.basis]}({request.basis}) · 연간 사업보고서(11011)"
    )
    for notice in dashboard.notices:
        st.warning(notice)
    if result.status == "error":
        st.info(
            "API 설정·네트워크·조회 조건을 확인하세요. 검증된 도구 결과 없이 분석하지 않습니다."
        )
    financial_tab, evidence_tab, execution_tab = st.tabs(
        ["재무 대시보드", "사업보고서 근거", "Agent 실행"]
    )
    with financial_tab:
        st.subheader("기업별 주요 지표")
        st.caption("금액은 원문 통화 단위 그대로, 영업이익률은 기존 Python 계산값입니다.")
        latest = max(request.years)
        for summary in dashboard.summaries:
            if summary["business_year"] != latest:
                continue
            st.markdown(f"#### {summary['company']['corp_name']} · {latest}년")
            columns = st.columns(4)
            for column, (key, label) in zip(columns, METRICS.items(), strict=False):
                metric = summary.get(key, {})
                column.metric(
                    f"{label} ({metric.get('currency') or '통화 미확인'})",
                    format_number(metric.get("value")),
                )
            columns[3].metric(
                "영업이익률", format_number(summary.get("operating_margin_pct"), suffix="%")
            )
        rows = comparison_rows(dashboard)
        if rows:
            st.subheader("재무지표 비교표")
            st.dataframe(rows, hide_index=True, width="stretch")
            st.subheader("연도별 실적")
            if len(request.years) == 1:
                st.info("여러 연도 추이를 보려면 사업연도를 2~3개 선택해 다시 분석하세요.")
            label = st.selectbox(
                "차트 지표",
                ["영업이익률 (%)", "매출액 (KRW)", "영업이익 (KRW)", "당기순이익 (KRW)"],
                key="chart_metric",
            )
            key = {
                "영업이익률 (%)": "operating_margin_pct",
                "매출액 (KRW)": "revenue",
                "영업이익 (KRW)": "operating_profit",
                "당기순이익 (KRW)": "net_income",
            }[label]
            chart = chart_rows(dashboard, key)
            if any(row["값"] is not None for row in chart):
                if len(request.years) > 1:
                    st.line_chart(chart, x="사업연도", y="값", color="기업", y_label=label)
                else:
                    st.bar_chart(chart, x="기업", y="값", color="기업", y_label=label)
                st.caption("차트만 부동소수점으로 표시합니다. 누락은 0으로 대체하지 않습니다.")
            else:
                st.warning("차트에 표시할 검증된 수치가 없습니다.")
            with st.expander("재무 데이터 공시 출처"):
                for summary in dashboard.summaries:
                    st.text(
                        f"{summary['company']['corp_name']} · {summary['business_year']}년 · "
                        f"{summary['basis']} · 고유번호 {summary['company']['corp_code']}"
                    )
                    receipts = {
                        evidence.get("receipt_no")
                        for key in METRICS
                        for evidence in summary.get(key, {}).get("evidence", [])
                    }
                    for receipt in sorted(item for item in receipts if filing_url(item)):
                        st.link_button(f"OpenDART 공시 {receipt}", filing_url(receipt))
                    st.caption("출처: 금융감독원 OpenDART 단일회사 주요계정 API")
        else:
            st.info("표시할 재무 조회 결과가 없습니다.")
    with evidence_tab:
        st.subheader("검색된 사업보고서 문장")
        st.info(
            "지원 범위: 삼성전자·SK하이닉스 2024년 사업보고서의 사업 내용·경영진단. "
            "문서 서술은 연결/별도 수치를 분리한 인과 분석이 아닙니다."
        )
        st.caption(
            "검색 문장은 원문 데이터입니다. "
            "원문 수치·계획을 위의 재무 계산값으로 사용하지 않습니다. "
            "관련 문장만으로 이익률 차이의 인과관계나 기여도를 확정하지 않습니다."
        )
        if not request.rag:
            st.info("이번 실행은 RAG 검색을 요청하지 않았습니다.")
        elif not dashboard.hits:
            st.warning(
                "사업보고서 근거 부족: 관련 문장을 확보하지 못했습니다. 원인을 추론하지 않습니다."
            )
        for hit in dashboard.hits:
            meta = hit["metadata"]
            with st.container(border=True):
                st.text(
                    f"{meta['corp_name']} · {meta['business_year']}년 사업보고서 · "
                    f"접수번호 {meta['receipt_no']}"
                )
                st.text(hit["excerpt"])
                st.caption(
                    f"문서 위치: {meta.get('section', '미확인')} | "
                    f"{meta.get('member', '미확인')} | XML {meta.get('line', '?')}행 · "
                    f"블록 {meta.get('block', '?')}"
                )
                st.link_button("금융감독원 DART 원문", meta["source"])
                with st.expander("검색 문단 전체 (원문 데이터)"):
                    st.text(hit["text"])
    with execution_tab:
        st.subheader("실제 Tool 호출 순서")
        if dashboard.calls:
            st.dataframe(list(dashboard.calls), hide_index=True, width="stretch")
        else:
            st.info("실행된 Tool이 없습니다.")
        st.caption(f"실행 ID: {result.run_id}. 인증 정보·원시 요청·민감 로그는 표시하지 않습니다.")
        with st.expander("근거 기반 Agent 보고서"):
            st.text(result.answer)


def main():
    st.set_page_config(page_title="FinSight | 공시 기반 재무분석", page_icon="◈", layout="wide")
    # Static CSS only: keep exact KRW amounts visible without UI-side unit arithmetic.
    st.markdown(
        "<style>[data-testid='stMetricValue'], [data-testid='stMetricValue'] > div {"
        "font-size: clamp(0.8rem, 1.2vw, 1.1rem) !important; font-variant-numeric: tabular-nums;"
        "}</style>",
        unsafe_allow_html=True,
    )
    st.markdown("FINANCIAL INTELLIGENCE / OPENDART")
    st.title("FinSight")
    st.caption("실제 공시에서 숫자를 확인하고, 사업보고서에서 근거를 찾습니다.")
    with st.sidebar:
        st.header("분석 조건")
        st.caption("연간 사업보고서 · 숫자는 Python 계산")
        companies = st.multiselect(
            "기업", list(COMPANIES), default=list(COMPANIES), key="companies"
        )
        latest = datetime.now(ZoneInfo("Asia/Seoul")).year - 1
        years = st.multiselect(
            "사업연도 (최대 3개)",
            list(range(latest, 2014, -1)),
            default=[2024],
            max_selections=3,
            key="years",
        )
        basis = st.selectbox(
            "재무제표 기준",
            ["CFS", "OFS"],
            format_func=lambda value: f"{BASIS_LABELS[value]} ({value})",
            key="basis",
        )
        rag = st.checkbox(
            "2024년 사업보고서 근거 검색",
            value=True,
            key="rag",
            help="선택 연도에 2024년이 있어야 합니다. 다른 연도 RAG는 미지원입니다.",
        )
        st.divider()
        st.caption(
            "RAG: 삼성전자·SK하이닉스 / 2024년만 지원.\n"
            "API 키는 로컬 .env에서만 읽습니다. 입력창이나 로그로 노출하지 않습니다."
        )
    with st.form("analysis_form"):
        question = st.text_area(
            "자연어 질문",
            value="선택한 기업의 재무 실적과 영업이익률을 분석해 주세요.",
            max_chars=2000,
            height=100,
            key="question",
        )
        st.caption(
            "예: 삼성전자와 SK하이닉스의 2024년 영업이익률을 비교하고, "
            "사업보고서에서 관련 근거를 찾아 주세요."
        )
        submitted = st.form_submit_button("분석 실행", type="primary", width="stretch")
    if submitted:
        st.session_state.pop("analysis", None)
        request = AnalysisRequest(question, tuple(companies), tuple(years), basis, rag)
        try:
            request.validate()
        except ValueError as exc:
            st.warning(str(exc))
        else:
            with st.status("FinSight Agent 실행 중…", expanded=True) as status:
                progress = st.empty()
                try:
                    result = run_analysis(request, progress.info)
                except ConfigurationError:
                    status.update(label="API 설정 확인 필요", state="error")
                    st.error(
                        "로컬 .env의 OPENDART_API_KEY (또는 DART_API_KEY)와 "
                        "OPENAI_API_KEY 설정을 확인하세요. 키 값은 화면에 표시하지 않습니다."
                    )
                except (FinSightError, OSError, ValueError):
                    status.update(label="분석 실패", state="error")
                    st.error(
                        "데이터·API·로컬 색인을 확인하지 못했습니다. 설정과 네트워크를 확인하세요. "
                        "RAG 색인이 없으면 터미널에서 finsight rag-index를 실행하세요."
                    )
                except Exception:
                    # No SDK/Chroma traceback, headers, or credentials may reach the browser.
                    status.update(label="분석 실패", state="error")
                    st.error("예상하지 못한 실행 오류입니다. 결과를 생성하지 않았습니다.")
                else:
                    st.session_state.analysis = (request, result)
                    state = (
                        "complete" if project(result, request).status == "completed" else "error"
                    )
                    status.update(label="Agent 실행 종료", state=state, expanded=False)
    if "analysis" in st.session_state:
        previous, _ = st.session_state.analysis
        # Streamlit hot reload may replace the dataclass type while keeping session data.
        previous_scope = (
            previous.question,
            previous.companies,
            previous.years,
            previous.basis,
            previous.rag,
        )
        if previous_scope != (question, tuple(companies), tuple(years), basis, rag):
            st.info("아래는 이전 실행 결과입니다. 변경한 조건을 적용하려면 분석 실행을 누르세요.")
        render_result(*st.session_state.analysis)
    else:
        st.info("기업·연도·재무제표 기준을 선택하고 질문을 입력해 분석을 실행하세요.")
        st.caption(
            "재무 조회: 2015년 이후 연간 자료 (기업·연도별 제공 여부에 따라 누락 가능). "
            "색인은 UI에서 자동 수집하지 않습니다. 최초 사용 전 finsight rag-index가 필요합니다."
        )
