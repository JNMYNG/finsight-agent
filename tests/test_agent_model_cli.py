import json
import traceback

import httpx
import pytest

from finsight import cli
from finsight.agent.graph import AgentResult
from finsight.agent.model import OpenAIToolModel
from finsight.agent.tools import TOOL_DEFINITIONS
from finsight.config import AgentSettings
from finsight.errors import AgentModelError, ConfigurationError

from .conftest import TEST_API_KEY


def completion(*, calls=None, content=None, finish="tool_calls", refusal=None):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": calls,
                    "refusal": refusal,
                },
            }
        ],
    }


def test_real_sdk_request_serialization_and_model_tool_selection():
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.host == "api.openai.com"
        assert payload["parallel_tool_calls"] is False and payload["store"] is False
        assert payload["model"] == "gpt-4.1-mini"
        assert payload["tools"][0]["function"]["strict"] is True
        return httpx.Response(
            200,
            json=completion(
                calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_company",
                            "arguments": '{"company":"삼성전자"}',
                        },
                    }
                ]
            ),
        )

    model = OpenAIToolModel(AgentSettings("sk-test-secret"), transport=httpx.MockTransport(handler))
    try:
        result = model.select(
            [{"role": "user", "content": "삼성전자 고유번호"}],
            [definition.schema() for definition in TOOL_DEFINITIONS],
        )
    finally:
        model.close()
    assert result.calls[0].name == "lookup_company"
    assert json.loads(result.calls[0].arguments) == {"company": "삼성전자"}
    assert len(requests) == 1


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_openai_error_body_never_leaks_to_public_exception(status):
    secret = "sk-test-secret"
    model = OpenAIToolModel(
        AgentSettings(secret),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(status, json={"error": {"message": secret}})
        ),
    )
    try:
        with pytest.raises(AgentModelError) as exc:
            model.select([], [])
    finally:
        model.close()
    assert secret not in "".join(traceback.format_exception(exc.value))
    assert exc.value.__context__ is None


@pytest.mark.parametrize(
    "response",
    [
        completion(finish="length"),
        completion(finish="content_filter"),
        completion(finish="stop", refusal="refused"),
    ],
)
def test_incomplete_or_refused_model_response(response):
    model = OpenAIToolModel(
        AgentSettings("sk-test"),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)),
    )
    try:
        with pytest.raises(AgentModelError):
            model.select([], [])
    finally:
        model.close()


def test_agent_settings_does_not_expose_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    settings = AgentSettings.from_env(tmp_path / "missing.env")
    assert settings.model == "test-model" and "sk-test-secret" not in repr(settings)


def test_agent_settings_missing_key(tmp_path):
    with pytest.raises(ConfigurationError):
        AgentSettings.from_env(tmp_path / "missing.env")


def test_schema_error_detail_is_sanitized():
    secret = "sk-test-secret"
    model = OpenAIToolModel(
        AgentSettings(secret),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                400,
                json={
                    "error": {
                        "code": "invalid_function_parameters",
                        "message": "Invalid schema " + secret,
                    }
                },
            )
        ),
    )
    try:
        with pytest.raises(AgentModelError) as exc:
            model.select([], [])
    finally:
        model.close()
    assert "Invalid schema" in str(exc.value)
    assert secret not in "".join(traceback.format_exception(exc.value))


def test_model_timeout_is_safe_and_not_retried():
    requests = []

    def handler(request):
        requests.append(request)
        raise httpx.ReadTimeout("sk-test-secret", request=request)

    model = OpenAIToolModel(AgentSettings("sk-test-secret"), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(AgentModelError) as exc:
            model.select([], [])
    finally:
        model.close()
    assert len(requests) == 1
    assert "sk-test-secret" not in "".join(traceback.format_exception(exc.value))


@pytest.mark.parametrize(
    "status,exit_code",
    [("completed", 0), ("needs_input", 0), ("partial", 1), ("error", 1), ("limit_reached", 1)],
)
def test_cli_agent_output_and_exit_status(monkeypatch, capsys, status, exit_code):
    import finsight.agent.graph as graph_module
    import finsight.agent.model as model_module

    monkeypatch.setenv("DART_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    closed = []

    class FakeModel:
        def __init__(self, settings):
            assert settings.openai_api_key == "sk-test-secret"

        def close(self):
            closed.append(True)

    class FakeAgent:
        def __init__(self, service, model, **kwargs):
            assert kwargs["max_tool_calls"] == 8

        def run(self, question):
            assert question == "삼성전자 2024년 연결 영업이익률"
            return AgentResult("test-run", status, "검증된 답변", "finished", ())

    monkeypatch.setattr(model_module, "OpenAIToolModel", FakeModel)
    monkeypatch.setattr(graph_module, "FinancialAgent", FakeAgent)
    assert cli.main(["agent", "삼성전자 2024년 연결 영업이익률"]) == exit_code
    assert json.loads(capsys.readouterr().out)["status"] == status
    assert closed == [True]
