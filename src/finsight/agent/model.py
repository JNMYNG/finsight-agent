"""Injectable model boundary. Credentials never enter graph state or tool arguments."""

from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from openai import OpenAI, OpenAIError

from finsight.agent.audit import AuditLog
from finsight.config import AgentSettings
from finsight.errors import AgentModelError


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelTurn:
    calls: tuple[ToolCall, ...] = ()
    # Deliberately discarded when producing the financial answer.
    content: str | None = None


class ToolSelectingModel(Protocol):
    def select(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn: ...


class OpenAIToolModel:
    def __init__(
        self, settings: AgentSettings, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.model = settings.model
        self._redact = AuditLog(secrets=(settings.openai_api_key,)).redact
        # Explicit official endpoint; ignore accidental OPENAI_BASE_URL overrides.
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url="https://api.openai.com/v1",
            max_retries=0,
            timeout=30.0,
            http_client=httpx.Client(transport=transport, timeout=30.0),
        )

    def close(self) -> None:
        self._client.close()

    def select(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        failure = None
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=False,
                store=False,
            )
        except OpenAIError as exc:
            code = getattr(exc, "status_code", None)
            failure = (
                f"OpenAI 요청 실패 (HTTP {code})." if code else "OpenAI 통신 실패 또는 시간 초과."
            )
            error_code = getattr(exc, "code", None)
            if error_code in {
                "insufficient_quota",
                "invalid_api_key",
                "model_not_found",
                "rate_limit_exceeded",
                "invalid_function_parameters",
                "invalid_json_schema",
            }:
                failure += f" 오류 유형: {error_code}."
            if error_code in {"invalid_function_parameters", "invalid_json_schema"}:
                body = getattr(exc, "body", None)
                if isinstance(body, dict) and isinstance(body.get("message"), str):
                    failure += " " + self._redact(body["message"])[:600]
        if failure:
            raise AgentModelError(failure)
        if not response.choices or response.choices[0].finish_reason not in {"stop", "tool_calls"}:
            raise AgentModelError("OpenAI 응답이 거부되거나 완료되지 않았습니다.")
        message = response.choices[0].message
        if message.refusal:
            raise AgentModelError("OpenAI가 요청 처리를 거부했습니다.")
        calls = []
        for call in message.tool_calls or ():
            if call.type != "function":
                raise AgentModelError("지원하지 않는 OpenAI Tool 호출 형식입니다.")
            calls.append(ToolCall(call.id, call.function.name, call.function.arguments))
        return ModelTurn(tuple(calls), message.content)
