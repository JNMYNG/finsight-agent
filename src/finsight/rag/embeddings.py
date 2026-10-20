"""Explicit OpenAI embeddings; no Chroma default model downloads or hidden API calls."""

import math
from typing import Protocol

import httpx
from openai import OpenAI, OpenAIError

from finsight.config import AgentSettings
from finsight.errors import EmbeddingError


class Embedder(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    model = "text-embedding-3-small"
    dimensions = 1536

    def __init__(self, settings: AgentSettings, *, transport: httpx.BaseTransport | None = None):
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url="https://api.openai.com/v1",
            max_retries=0,
            timeout=45,
            http_client=httpx.Client(transport=transport, timeout=45),
        )

    def close(self) -> None:
        self._client.close()

    def embed(self, texts: list[str]) -> list[list[float]]:
        # UTF-8 bytes conservatively bound token count without a hidden tokenizer download.
        # 24 * 8000 bytes also stays below the total tokens/request limit.
        if not texts or any(not text.strip() or len(text.encode()) > 8000 for text in texts):
            raise EmbeddingError("임베딩 입력이 비어 있거나 허용 크기를 초과했습니다.")
        result: list[list[float]] = []
        for start in range(0, len(texts), 24):
            batch = texts[start : start + 24]
            failure = None
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    dimensions=self.dimensions,
                    input=batch,
                    encoding_format="float",
                )
            except OpenAIError as exc:
                status = getattr(exc, "status_code", None)
                failure = f"OpenAI Embeddings 요청 실패 (HTTP {status})."
            if failure:
                raise EmbeddingError(failure)
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise EmbeddingError("임베딩 응답 개수/순서가 잘못됐습니다.")
            for item in ordered:
                vector = item.embedding
                if (
                    len(vector) != self.dimensions
                    or not all(math.isfinite(value) for value in vector)
                    or sum(value * value for value in vector) == 0
                ):
                    raise EmbeddingError("임베딩 응답 차원/벡터가 잘못됐습니다.")
                result.append(vector)
        return result
