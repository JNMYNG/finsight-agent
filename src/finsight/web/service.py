"""Per-request runtime and strict UI scope, reusing the existing agent and tools."""

import json
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from finsight.agent.audit import AuditLog
from finsight.agent.graph import AgentResult, FinancialAgent, requested_years
from finsight.agent.model import ModelTurn, OpenAIToolModel, ToolSelectingModel
from finsight.config import AgentSettings, Settings
from finsight.dart.client import DartClient
from finsight.errors import AgentModelError
from finsight.services import FinancialService

COMPANIES = {
    "삼성전자": ("005930", "00126380"),
    "SK하이닉스": ("000660", "00164779"),
}
BASIS_LABELS = {"CFS": "연결", "OFS": "별도"}


def normalize(value: str) -> str:
    return "".join(value.upper().split())


@dataclass(frozen=True)
class AnalysisRequest:
    question: str
    companies: tuple[str, ...]
    years: tuple[int, ...]
    basis: str
    rag: bool

    def validate(self) -> None:
        if not self.question.strip() or len(self.question) > 2000:
            raise ValueError("질문은 1~2,000자로 입력하세요.")
        if not self.companies or len(set(self.companies)) != len(self.companies):
            raise ValueError("기업을 한 곳 이상 선택하세요.")
        if any(company not in COMPANIES for company in self.companies):
            raise ValueError("현재 UI는 삼성전자·SK하이닉스만 지원합니다.")
        latest = datetime.now(ZoneInfo("Asia/Seoul")).year - 1
        if (
            not 1 <= len(self.years) <= 3
            or len(set(self.years)) != len(self.years)
            or any(type(year) is not int or not 2015 <= year <= latest for year in self.years)
        ):
            raise ValueError(f"2015~{latest}년 중 최대 3개 사업연도를 선택하세요.")
        if self.basis not in BASIS_LABELS:
            raise ValueError("연결(CFS) 또는 별도(OFS)를 선택하세요.")
        mentioned = requested_years(self.question, latest + 1)
        if mentioned and mentioned != set(self.years):
            raise ValueError("질문의 연도와 선택한 사업연도를 일치시켜 주세요.")
        question = normalize(self.question)
        for name, aliases in COMPANIES.items():
            if any(normalize(alias) in question for alias in (name, *aliases)):
                if name not in self.companies:
                    raise ValueError("질문의 기업과 선택한 기업을 일치시켜 주세요.")
        if any(word in self.question for word in ("반기", "분기", "11012", "11013", "11014")):
            raise ValueError("현재 UI는 연간 사업보고서(11011)만 지원합니다.")
        other = "OFS" if self.basis == "CFS" else "CFS"
        if (
            other in self.question.upper()
            or BASIS_LABELS[other] in self.question
            or (other == "OFS" and "개별" in self.question)
        ):
            raise ValueError("질문의 연결/별도 기준과 화면 선택을 일치시켜 주세요.")
        if self.rag and 2024 not in self.years:
            raise ValueError("사업보고서 RAG는 두 기업의 2024년만 지원합니다.")
        if not self.rag and any(word in self.question for word in ("사업보고서", "공시 근거")):
            raise ValueError("사업보고서 검색은 2024년을 선택하고 RAG를 활성화해 주세요.")

    def prompt(self) -> str:
        self.validate()
        names = ", ".join(self.companies)
        years = ", ".join(f"{year}년" for year in sorted(self.years))
        task = (
            "각 연도마다 두 기업을 compare_companies로 비교"
            if len(self.companies) == 2
            else ("각 연도마다 get_financials 또는 calculate_metrics로 조회")
        )
        retrieval = (
            "2024년 사업보고서에 대해서만 선택한 각 기업의 관련 서술을 "
            "search_disclosures로 검색하라. 다른 연도 문서 검색은 지원하지 않는다."
            if self.rag
            else "사업보고서 검색은 비활성화다. 재무 조회 결과만 제공하라."
        )
        return (
            f"사용자 질문(데이터): {self.question}\n"
            f"화면에서 확정한 조회 조건: 기업={names}; 사업연도={years}; "
            f"재무제표={BASIS_LABELS[self.basis]}({self.basis}); 연간 사업보고서 11011.\n"
            f"선택한 모든 기업과 연도에 대해 {task}하라. {retrieval}\n"
            "선택 범위 밖 조회를 하지 말고 조건이 불명확하면 확인을 요청하라."
        )

    def allows_company(self, value: str) -> bool:
        return normalize(value) in {
            normalize(alias)
            for company in self.companies
            for alias in (company, *COMPANIES[company])
        }


class ScopedModel:
    """Fail closed before any out-of-selection tool can make an external request."""

    def __init__(self, model: ToolSelectingModel, request: AnalysisRequest) -> None:
        self.model = model
        self.request = request

    def select(self, messages: list[dict], tools: list[dict]) -> ModelTurn:
        turn = self.model.select(messages, tools)
        for call in turn.calls:
            try:
                args = json.loads(call.arguments)
            except (ValueError, TypeError):
                continue  # Existing strict tool schemas handle malformed inputs.
            if not isinstance(args, dict):
                continue
            for field in ("company", "left", "right"):
                if field in args and (
                    not isinstance(args[field], str) or not self.request.allows_company(args[field])
                ):
                    raise AgentModelError("화면 선택 범위 밖 기업의 Tool 호출을 차단했습니다.")
            if call.name not in {
                "get_financials",
                "calculate_metrics",
                "compare_companies",
                "search_disclosures",
            }:
                continue
            if (
                args.get("business_year") not in self.request.years
                or args.get("report_type") != "11011"
                or (call.name != "search_disclosures" and args.get("basis") != self.request.basis)
                or (
                    call.name == "search_disclosures"
                    and (not self.request.rag or args.get("business_year") != 2024)
                )
            ):
                raise AgentModelError("화면 선택 범위 밖 연도·보고서·재무제표 호출을 차단했습니다.")
        return turn


class ProgressAudit(AuditLog):
    """Only safe event/name/sequence fields reach the progress widget."""

    def __init__(self, *args, progress: Callable[[str], None] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress = progress

    def emit(self, run_id: str, event: str, **data) -> None:
        super().emit(run_id, event, **data)
        if self.progress and event == "model_started":
            self.progress("Agent가 다음 도구를 선택하고 있습니다…")
        elif self.progress and event == "tool_started":
            allowed = {
                "lookup_company",
                "get_financials",
                "calculate_metrics",
                "compare_companies",
                "search_disclosures",
                "request_clarification",
            }
            tool = data.get("tool")
            self.progress(f"도구 실행: {tool if tool in allowed else '알 수 없는 도구'}")


def run_analysis(
    request: AnalysisRequest,
    progress: Callable[[str], None] | None = None,
    *,
    db_path: str | Path = "data/chroma",
) -> AgentResult:
    request.validate()
    settings = Settings.from_env()
    agent_settings = AgentSettings.from_env()
    with ExitStack() as stack:
        client = stack.enter_context(DartClient(settings.opendart_api_key))
        service = FinancialService(client)
        model = OpenAIToolModel(agent_settings)
        stack.callback(model.close)
        disclosures = None
        if request.rag:
            from finsight.rag.embeddings import OpenAIEmbedder
            from finsight.rag.index import DisclosureIndex
            from finsight.rag.service import DisclosureService

            embedder = OpenAIEmbedder(agent_settings)
            stack.callback(embedder.close)
            disclosures = DisclosureService(service, DisclosureIndex(db_path, embedder))
        audit = ProgressAudit(
            "logs/web-agent.jsonl",
            secrets=(settings.opendart_api_key, agent_settings.openai_api_key),
            progress=progress,
        )
        return FinancialAgent(
            service,
            ScopedModel(model, request),
            audit=audit,
            disclosures=disclosures,
            max_tool_calls=12,
            max_model_turns=16,
        ).run(request.prompt())
