import json
from copy import deepcopy

import httpx
import pytest

from finsight.agent.audit import AuditLog
from finsight.agent.graph import FinancialAgent, requested_years
from finsight.agent.model import ModelTurn, ToolCall
from finsight.dart.client import DartClient
from finsight.errors import AgentModelError
from finsight.services import FinancialService

from .conftest import TEST_API_KEY, zip_xml
from .test_agent_tools import financial_args


class ScriptedModel:
    """Deterministic routing fixture; it does not test real LLM semantic accuracy."""

    def __init__(self, turns):
        self.turns = iter(turns)
        self.requests = []

    def select(self, messages, tools):
        self.requests.append((deepcopy(messages), deepcopy(tools)))
        turn = next(self.turns)
        if isinstance(turn, Exception):
            raise turn
        return turn


def turn(name, args, call_id="call_1"):
    return ModelTurn((ToolCall(call_id, name, json.dumps(args, ensure_ascii=False)),))


def test_graph_routes_model_tools_and_back_with_observations(agent_service):
    service, _ = agent_service
    model = ScriptedModel(
        [
            turn("lookup_company", {"company": "삼성전자"}),
            turn("calculate_metrics", financial_args(), "call_2"),
            ModelTurn(content="영업이익률은 99999%이고 매출은 123456789조원이다."),
        ]
    )
    result = FinancialAgent(service, model).run("삼성전자 2024년 연결 영업이익률 계산해 줘")
    assert result.status == "completed"
    assert [item["tool"] for item in result.tool_calls] == ["lookup_company", "calculate_metrics"]
    assert result.tool_calls[1]["arguments"] == financial_args()
    assert "10.00%" in result.answer and "2024년" in result.answer and "연결(CFS)" in result.answer
    assert "99999" not in result.answer and "123456789" not in result.answer
    tool_message = model.requests[2][0][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "call_2"
    assert json.loads(tool_message["content"])["data"]["operating_margin_pct"] == "10.00"


def test_no_evidence_rejects_free_text_numbers(agent_service):
    service, requests = agent_service
    result = FinancialAgent(service, ScriptedModel([ModelTurn(content="매출은 999999999원")])).run(
        "삼성전자 2024년 매출액"
    )
    assert result.status == "error"
    assert "999999999" not in result.answer
    assert "검증된 조회 결과가 없습니다" in result.answer
    assert not requests


def test_multi_year_context_is_preserved(agent_service):
    service, requests = agent_service
    model = ScriptedModel(
        [
            turn("get_financials", financial_args(business_year=2023)),
            turn("get_financials", financial_args(), "call_2"),
            ModelTurn(),
        ]
    )
    result = FinancialAgent(service, model).run("삼성전자 2023년과 2024년 연결 재무제표 조회")
    assert result.status == "completed"
    assert [item["result"]["data"]["business_year"] for item in result.tool_calls] == [2023, 2024]
    assert [request.url.params["bsns_year"] for request in requests[1:]] == ["2023", "2024"]


@pytest.mark.parametrize(
    "question,args,code",
    [
        ("삼성전자 2024년 연결", financial_args(business_year=2023), "YEAR_NOT_REQUESTED"),
        ("삼성전자 영업이익률", financial_args(), "BUSINESS_YEAR_REQUIRED"),
        ("삼성전자 2024년 별도", financial_args(), "BASIS_NOT_REQUESTED"),
        ("삼성전자 2024년 OFS", financial_args(), "BASIS_NOT_REQUESTED"),
        ("삼성전자 2024년 개별", financial_args(), "BASIS_NOT_REQUESTED"),
        ("삼성전자 2024년 반기", financial_args(), "REPORT_NOT_REQUESTED"),
        ("삼성전자 2024년 11012", financial_args(), "REPORT_NOT_REQUESTED"),
    ],
)
def test_model_cannot_change_explicit_context(agent_service, question, args, code):
    service, requests = agent_service
    result = FinancialAgent(
        service,
        ScriptedModel(
            [
                turn("get_financials", args),
                ModelTurn(),
            ]
        ),
    ).run(question)
    assert result.tool_calls[0]["result"]["error"]["code"] == code
    assert result.status == "error" and not requests


def test_bad_input_is_returned_to_model_without_crash(agent_service):
    service, requests = agent_service
    result = FinancialAgent(
        service,
        ScriptedModel(
            [
                turn("get_financials", financial_args(business_year=[])),
                ModelTurn(),
            ]
        ),
    ).run("삼성전자 2024년 연결")
    assert result.tool_calls[0]["result"]["error"]["code"] == "INVALID_TOOL_INPUT"
    assert not requests


def test_missing_year_clarification_terminates_without_network(agent_service):
    service, requests = agent_service
    result = FinancialAgent(
        service,
        ScriptedModel([turn("request_clarification", {"missing_fields": ["business_year"]})]),
    ).run("삼성전자 매출액 알려 줘")
    assert result.status == "needs_input"
    assert "사업연도" in result.answer
    assert not requests


def test_comparison_report_and_log_have_results_and_order(agent_service, tmp_path):
    service, _ = agent_service
    path = tmp_path / "trace.jsonl"
    args = {
        "left": "삼성전자",
        "right": "SK하이닉스",
        "business_year": 2024,
        "report_type": "11011",
        "basis": "CFS",
    }
    model = ScriptedModel([turn("compare_companies", args), ModelTurn()])
    result = FinancialAgent(service, model, audit=AuditLog(path, secrets=(TEST_API_KEY,))).run(
        "삼성전자와 SK하이닉스 2024년 연결 비교"
    )
    assert result.status == "completed"
    assert "-10.00%p" in result.answer and "더 높은 기업: SK하이닉스" in result.answer
    assert "공시 근거:" in result.answer
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[0]["event"] == "run_started"
    assert events[-1]["event"] == "run_finished"
    completed = [item for item in events if item["event"] == "tool_finished"]
    assert completed[0]["sequence"] == 1 and completed[0]["arguments"] == args
    assert completed[0]["result"]["data"]["revenue"]["difference"] == "500"
    assert {item["run_id"] for item in events} == {result.run_id}
    assert TEST_API_KEY not in path.read_text()


def test_audit_redacts_values_keys_and_newlines(tmp_path, caplog):
    path = tmp_path / "audit.jsonl"
    audit = AuditLog(path, secrets=(TEST_API_KEY,))
    with caplog.at_level("INFO", logger="finsight.agent"):
        audit.emit(
            "test",
            "check",
            arguments={"api_key": "secret", "company": TEST_API_KEY},
            note="sk-unknownSecret\ncrtfc_key=hidden&ok=1",
        )
    content = path.read_text()
    assert TEST_API_KEY not in content + caplog.text
    assert "sk-unknownSecret" not in content + caplog.text
    assert "hidden" not in content + caplog.text
    assert len(content.splitlines()) == 1
    assert json.loads(content)["arguments"]["api_key"] == "[REDACTED]"


@pytest.mark.parametrize(
    "max_tools,max_turns,expected", [(2, 10, "tool_limit"), (8, 1, "model_turn_limit")]
)
def test_loop_limits_are_enforced(agent_service, max_tools, max_turns, expected):
    service, _ = agent_service
    model = ScriptedModel([turn("unknown", {}, f"call_{index}") for index in range(10)])
    result = FinancialAgent(
        service, model, max_tool_calls=max_tools, max_model_turns=max_turns
    ).run("삼성전자 2024년")
    assert result.status == "limit_reached" and result.stop_reason == expected
    assert len(result.tool_calls) <= max_tools and len(model.requests) <= max_turns


def test_duplicate_call_ids_fail_safely(agent_service):
    service, _ = agent_service
    model = ScriptedModel(
        [
            turn("lookup_company", {"company": "삼성전자"}),
            turn("lookup_company", {"company": "삼성전자"}),
        ]
    )
    result = FinancialAgent(service, model).run("삼성전자 고유번호")
    assert result.status == "partial" and result.stop_reason == "model_error"
    assert len(result.tool_calls) == 1


def test_batch_tool_calls_cannot_exceed_budget(agent_service):
    service, _ = agent_service
    calls = tuple(
        ToolCall(f"call_{i}", "lookup_company", '{"company":"삼성전자"}') for i in range(5)
    )
    model = ScriptedModel([ModelTurn(calls)])
    result = FinancialAgent(service, model, max_tool_calls=2).run("삼성전자 고유번호")
    assert result.status == "limit_reached" and len(result.tool_calls) == 2
    assert len(model.requests) == 1


@pytest.mark.parametrize("has_result", [False, True])
def test_model_failure_preserves_completed_evidence(agent_service, has_result):
    service, _ = agent_service
    turns = [turn("get_financials", financial_args())] if has_result else []
    turns.append(AgentModelError("safe error"))
    result = FinancialAgent(service, ScriptedModel(turns)).run("삼성전자 2024년 연결")
    assert result.status == ("partial" if has_result else "error")
    assert result.stop_reason == "model_error"
    if has_result:
        assert "10.00%" in result.answer


def test_missing_account_is_partial_not_fabricated(accounts_factory):
    def handler(request):
        if request.url.path.endswith("corpCode.xml"):
            return httpx.Response(200, content=zip_xml())
        return httpx.Response(200, json={"status": "000", "list": accounts_factory(net="-")})

    with DartClient(TEST_API_KEY, transport=httpx.MockTransport(handler)) as client:
        result = FinancialAgent(
            FinancialService(client),
            ScriptedModel(
                [
                    turn("get_financials", financial_args()),
                    ModelTurn(),
                ]
            ),
        ).run("삼성전자 2024년 연결 순이익")
    assert result.status == "partial"
    assert "당기순이익: 확인 불가" in result.answer
    assert "AMOUNT_MISSING" in result.answer
    assert result.tool_calls[0]["result"]["data"]["net_income"]["value"] is None


@pytest.mark.parametrize(
    "question,years",
    [
        ("2023년과 2024년", {2023, 2024}),
        ("작년", {2025}),
        ("재작년", {2024}),
        ("올해와 지난해", {2025, 2026}),
        ("005930 기업", set()),
        ("2022년부터 2024년까지", {2022, 2023, 2024}),
        ("2022~2024년", {2022, 2023, 2024}),
    ],
)
def test_requested_years(question, years):
    assert requested_years(question, 2026) == years
