import json
from decimal import Decimal

import httpx
import pytest

from finsight.agent.tools import TOOL_DEFINITIONS, FinancialTools
from finsight.dart.client import DartClient
from finsight.models import FinancialSummary
from finsight.services import FinancialService

from .conftest import TEST_API_KEY, zip_xml


def financial_args(**changes):
    return {
        "company": "삼성전자",
        "business_year": 2024,
        "report_type": "11011",
        "basis": "CFS",
        **changes,
    }


def test_all_tool_schemas_are_strict_and_have_no_amount_inputs():
    names = set()
    for definition in TOOL_DEFINITIONS:
        schema = definition.schema()["function"]
        names.add(schema["name"])
        assert schema["strict"] is True
        params = schema["parameters"]
        assert params["additionalProperties"] is False
        assert set(params["required"]) == set(params["properties"])
        assert "revenue" not in params["properties"]
        assert "operating_profit" not in params["properties"]
        assert '"$ref"' not in json.dumps(params)
        if "report_type" in params["properties"]:
            assert params["properties"]["report_type"]["enum"] == [
                "11011",
                "11012",
                "11013",
                "11014",
            ]
    assert names == {
        "lookup_company",
        "get_financials",
        "calculate_metrics",
        "compare_companies",
        "request_clarification",
    }


def test_lookup_uses_real_directory_adapter(agent_service):
    service, _ = agent_service
    args, outcome = FinancialTools(service).execute("lookup_company", '{"company":"SK하이닉스"}')
    assert args == {"company": "SK하이닉스"}
    assert outcome.ok
    assert outcome.data.corp_code == "00164779"


@pytest.mark.parametrize("tool", ["get_financials", "calculate_metrics"])
def test_financial_context_inputs_and_python_calculation(agent_service, tool):
    service, requests = agent_service
    args, result = FinancialTools(service).execute(tool, json.dumps(financial_args()))
    assert args == financial_args()
    assert result.ok and isinstance(result.data, FinancialSummary)
    assert result.data.company.corp_name == "삼성전자"
    assert result.data.business_year == 2024 and result.data.basis == "CFS"
    assert result.data.operating_margin_pct == Decimal("10.00")
    assert dict(requests[-1].url.params) == {
        "crtfc_key": TEST_API_KEY,
        "corp_code": "00126380",
        "bsns_year": "2024",
        "reprt_code": "11011",
    }


def test_ofs_and_report_code_are_preserved(agent_service):
    service, requests = agent_service
    _, result = FinancialTools(service).execute(
        "get_financials",
        json.dumps(financial_args(basis="OFS", report_type="11012", business_year=2023)),
    )
    assert result.data.basis == "OFS"
    assert result.data.report_type == "11012"
    assert result.data.period_basis == "year_to_date"
    assert result.data.revenue.value == Decimal("2000")
    assert result.data.operating_margin_pct == Decimal("5.00")
    assert requests[-1].url.params["bsns_year"] == "2023"


def test_same_data_reused_within_run_but_not_another_executor(agent_service):
    service, requests = agent_service
    tools = FinancialTools(service)
    args = json.dumps(financial_args())
    tools.execute("get_financials", args)
    tools.execute("calculate_metrics", args)
    assert len(requests) == 2  # one directory + one financial query
    FinancialTools(service).execute("get_financials", args)
    assert len(requests) == 3


def test_comparison_uses_day1_decimal_difference(agent_service):
    service, _ = agent_service
    args = {
        "left": "005930",
        "right": "000660",
        "business_year": 2024,
        "report_type": "11011",
        "basis": "CFS",
    }
    _, result = FinancialTools(service).execute("compare_companies", json.dumps(args))
    assert result.ok
    assert result.data.revenue.difference == Decimal("500")
    assert result.data.operating_margin_pct.difference == Decimal("-10.00")


@pytest.mark.parametrize(
    "changes",
    [
        {"business_year": "2024"},
        {"business_year": True},
        {"business_year": []},
        {"business_year": 2014},
        {"basis": "bad"},
        {"report_type": "bad"},
        {"company": "  "},
        {"revenue": "100000000"},
    ],
)
def test_invalid_tool_args_do_not_reach_api(agent_service, changes):
    service, requests = agent_service
    _, result = FinancialTools(service).execute(
        "calculate_metrics", json.dumps(financial_args(**changes))
    )
    assert not result.ok and result.error["code"] == "INVALID_TOOL_INPUT"
    assert not requests


@pytest.mark.parametrize("raw", ["{", "[]", "null", '{"company":"삼성전자"}'])
def test_malformed_and_missing_fields(agent_service, raw):
    service, requests = agent_service
    _, result = FinancialTools(service).execute("get_financials", raw)
    assert result.error["code"] == "INVALID_TOOL_INPUT"
    assert not requests


def test_unknown_tool_and_missing_company(agent_service):
    service, requests = agent_service
    tools = FinancialTools(service)
    assert tools.execute("eval", "{}")[1].error["code"] == "UNKNOWN_TOOL"
    assert not requests
    result = tools.execute("lookup_company", '{"company":"없는기업"}')[1]
    assert result.error["code"] == "COMPANY_NOT_FOUND"


@pytest.mark.parametrize("status,code", [("013", "NO_DATA"), ("020", "DART_API_ERROR")])
def test_api_errors_are_structured(status, code):
    def handler(request):
        if request.url.path.endswith("corpCode.xml"):
            return httpx.Response(200, content=zip_xml())
        return httpx.Response(200, json={"status": status, "message": "error with secret"})

    with DartClient(TEST_API_KEY, transport=httpx.MockTransport(handler)) as client:
        _, result = FinancialTools(FinancialService(client)).execute(
            "get_financials", json.dumps(financial_args())
        )
    assert not result.ok and result.error["code"] == code
    assert "secret" not in str(result.error)


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("network", "DART_TRANSPORT_ERROR"),
        ("malformed", "INVALID_DART_RESPONSE"),
        ("ambiguous", "AMBIGUOUS_COMPANY"),
    ],
)
def test_other_lookup_and_data_failures_are_structured(failure, expected):
    from .conftest import DIRECTORY_XML

    def handler(request):
        if request.url.path.endswith("corpCode.xml"):
            content = DIRECTORY_XML
            if failure == "ambiguous":
                content = content.replace("SK하이닉스".encode(), "삼성전자".encode())
            return httpx.Response(200, content=zip_xml(content))
        if failure == "network":
            raise httpx.ConnectError("sensitive diagnostics", request=request)
        return httpx.Response(200, json={"status": "000", "list": {}})

    with DartClient(TEST_API_KEY, transport=httpx.MockTransport(handler)) as client:
        result = FinancialTools(FinancialService(client)).execute(
            "get_financials", json.dumps(financial_args())
        )[1]
    assert not result.ok and result.error["code"] == expected
