"""Typed function tools. Amounts can only come from the trusted Day 1 service."""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from finsight.errors import (
    AmbiguousCompanyError,
    CompanyNotFoundError,
    DartAPIError,
    DartTransportError,
    DisclosureError,
    EmbeddingError,
    InvalidResponseError,
    NoDataError,
)
from finsight.models import (
    Corporation,
    FinancialComparison,
    FinancialSummary,
    ReportType,
    StatementBasis,
)
from finsight.rag.models import DisclosureSearch
from finsight.services import FinancialService, compare_summaries

if TYPE_CHECKING:
    from finsight.rag.service import DisclosureService


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class CompanyInput(ToolInput):
    company: str = Field(min_length=1, max_length=200, description="정식 기업명 또는 종목코드")


class FinancialInput(CompanyInput):
    business_year: int = Field(ge=2015, le=9999, description="사용자가 요청한 사업연도")
    report_type: ReportType = Field(description="사업 11011, 반기 11012, 1분기 11013, 3분기 11014")
    basis: StatementBasis | None = Field(
        description="연결 CFS, 별도 OFS, 미지정 시 null(연결 우선)"
    )


class ComparisonInput(ToolInput):
    left: str = Field(min_length=1, max_length=200, description="왼쪽 기업명 또는 종목코드")
    right: str = Field(min_length=1, max_length=200, description="오른쪽 기업명 또는 종목코드")
    business_year: int = Field(ge=2015, le=9999)
    report_type: ReportType
    basis: StatementBasis | None


class ClarificationInput(ToolInput):
    missing_fields: list[Literal["company", "business_year", "report_type", "basis"]] = Field(
        min_length=1, max_length=4
    )


class DisclosureInput(CompanyInput):
    business_year: int = Field(ge=2015, le=9999)
    report_type: ReportType
    query: str = Field(min_length=3, max_length=2000, description="사용자 질문의 검색 주제")


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[ToolInput]

    def schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        definitions = schema.get("$defs", {})

        def inline(value: Any) -> Any:
            # OpenAI rejects a Pydantic enum $ref with a sibling description.
            # Inline local enum definitions while keeping server-side validation intact.
            if isinstance(value, dict):
                if "$ref" in value:
                    referenced = definitions[value["$ref"].removeprefix("#/$defs/")]
                    return inline(
                        {
                            **referenced,
                            **{key: item for key, item in value.items() if key != "$ref"},
                        }
                    )
                return {key: inline(item) for key, item in value.items() if key != "$defs"}
            if isinstance(value, list):
                return [inline(item) for item in value]
            return value

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": inline(schema),
                "strict": True,
            },
        }


TOOL_DEFINITIONS = (
    ToolDefinition(
        "lookup_company", "기업명/종목코드로 OpenDART 기업 고유번호를 조회한다.", CompanyInput
    ),
    ToolDefinition(
        "get_financials",
        "실제 재무제표의 매출액·영업이익·순이익과 근거를 조회한다.",
        FinancialInput,
    ),
    ToolDefinition(
        "calculate_metrics",
        "기업 재무제표를 조회하고 Python 계산 영업이익률을 반환한다. "
        "금액을 인자로 받거나 모델이 계산하지 않는다.",
        FinancialInput,
    ),
    ToolDefinition(
        "compare_companies",
        "동일 사업연도·보고서의 두 기업 재무정보와 Python 계산 차이를 "
        "반환한다. 여러 연도 비교는 연도마다 한 번씩 호출한다.",
        ComparisonInput,
    ),
    ToolDefinition(
        "request_clarification",
        "사용자 질문에 기업·사업연도가 없거나 조건이 모호하면 "
        "추정하지 말고 필요한 필드를 요청한다.",
        ClarificationInput,
    ),
)
TOOLS_BY_NAME = {definition.name: definition for definition in TOOL_DEFINITIONS}
DISCLOSURE_TOOL = ToolDefinition(
    "search_disclosures",
    "삼성전자 또는 SK하이닉스 2024년 사업보고서 서술 근거를 검색한다. "
    "실적 관련 설명 요청 시 각 기업마다 호출한다. 재무 수치 계산은 기존 재무 Tool을 사용한다.",
    DisclosureInput,
)


@dataclass(frozen=True)
class ToolOutcome:
    ok: bool
    data: (
        Corporation
        | FinancialSummary
        | FinancialComparison
        | DisclosureSearch
        | dict[str, Any]
        | None
    ) = None
    error: dict[str, Any] | None = None


def tool_error(code: str, message: str, **details: Any) -> ToolOutcome:
    return ToolOutcome(False, error={"code": code, "message": message, **details})


class FinancialTools:
    """One executor per agent run; no cross-question or cross-user financial cache."""

    def __init__(
        self, service: FinancialService, disclosures: "DisclosureService | None" = None
    ) -> None:
        self.service = service
        self.disclosures = disclosures
        self._summaries: dict[tuple[Any, ...], FinancialSummary] = {}

    def _financials(
        self,
        company: str,
        business_year: int,
        report_type: ReportType,
        basis: StatementBasis | None,
    ) -> FinancialSummary:
        corporation = self.service.directory.find(company)
        key = (corporation.corp_code, business_year, report_type, basis)
        if key not in self._summaries:
            self._summaries[key] = self.service.get_financials(
                corporation.stock_code or corporation.corp_name,
                business_year,
                report_type,
                basis=basis,
            )
        return self._summaries[key]

    def execute(self, name: str, arguments: str) -> tuple[dict[str, Any], ToolOutcome]:
        definition = (
            DISCLOSURE_TOOL
            if name == DISCLOSURE_TOOL.name and self.disclosures
            else TOOLS_BY_NAME.get(name)
        )
        if definition is None:
            return {}, tool_error("UNKNOWN_TOOL", "허용되지 않은 도구입니다.")
        try:
            parsed = definition.input_model.model_validate_json(arguments)
        except ValidationError as exc:
            # Validation errors can echo raw user input; expose only field paths/types.
            fields = [
                {"field": ".".join(map(str, err["loc"])), "type": err["type"]}
                for err in exc.errors(include_input=False, include_url=False)
            ]
            return {}, tool_error("INVALID_TOOL_INPUT", "도구 입력을 확인하세요.", fields=fields)
        args = parsed.model_dump(mode="json")
        try:
            if isinstance(parsed, ClarificationInput):
                return args, ToolOutcome(True, {"missing_fields": parsed.missing_fields})
            if isinstance(parsed, DisclosureInput):
                data = self.disclosures.search(
                    parsed.company, parsed.business_year, parsed.report_type, parsed.query
                )
            elif isinstance(parsed, ComparisonInput):
                data = compare_summaries(
                    self._financials(
                        parsed.left, parsed.business_year, parsed.report_type, parsed.basis
                    ),
                    self._financials(
                        parsed.right, parsed.business_year, parsed.report_type, parsed.basis
                    ),
                )
            elif isinstance(parsed, FinancialInput):
                # get_financials and calculate_metrics share the audited Day 1 calculation.
                data = self._financials(
                    parsed.company, parsed.business_year, parsed.report_type, parsed.basis
                )
            else:
                data = self.service.directory.find(parsed.company)
            return args, ToolOutcome(True, data)
        except CompanyNotFoundError:
            outcome = tool_error(
                "COMPANY_NOT_FOUND", "기업을 찾을 수 없습니다. 정식명칭/종목코드를 확인하세요."
            )
        except AmbiguousCompanyError:
            outcome = tool_error("AMBIGUOUS_COMPANY", "기업명이 중복됩니다. 종목코드로 지정하세요.")
        except NoDataError:
            outcome = tool_error("NO_DATA", "요청 기업·연도·보고서의 재무정보가 없습니다.")
        except DartAPIError as exc:
            outcome = tool_error(
                "DART_API_ERROR", "OpenDART API 요청에 실패했습니다.", api_status=exc.status
            )
        except DartTransportError:
            outcome = tool_error("DART_TRANSPORT_ERROR", "OpenDART 통신 실패 또는 시간 초과입니다.")
        except InvalidResponseError:
            outcome = tool_error(
                "INVALID_DART_RESPONSE", "OpenDART 응답 형식이나 메타데이터가 잘못됐습니다."
            )
        except DisclosureError:
            outcome = tool_error("DISCLOSURE_ERROR", "원문 범위·출처 또는 색인 상태를 확인하세요.")
        except EmbeddingError:
            outcome = tool_error("EMBEDDING_ERROR", "검색 임베딩 생성에 실패했습니다.")
        return args, outcome


def outcome_message(outcome: ToolOutcome) -> str:
    from finsight.serialization import to_json_value

    return json.dumps(to_json_value(outcome), ensure_ascii=False, allow_nan=False)
