import json
import traceback

import httpx
import pytest

from finsight.config import AgentSettings
from finsight.errors import EmbeddingError
from finsight.rag.embeddings import OpenAIEmbedder


def test_openai_embedding_request_ordering_dimensions_and_batches():
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == "/v1/embeddings"
        assert payload["model"] == "text-embedding-3-small"
        assert payload["dimensions"] == 1536 and payload["encoding_format"] == "float"
        assert len(payload["input"]) <= 24
        data = [
            {"index": i, "object": "embedding", "embedding": [float(i + 1)] + [0.0] * 1535}
            for i in range(len(payload["input"]))
        ]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": payload["model"],
                "data": list(reversed(data)),
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    embedder = OpenAIEmbedder(
        AgentSettings("sk-test-secret"), transport=httpx.MockTransport(handler)
    )
    try:
        vectors = embedder.embed(["합성 문서"] * 25)
    finally:
        embedder.close()
    assert len(vectors) == 25 and vectors[0][0] == 1 and vectors[23][0] == 24
    assert [len(item["input"]) for item in requests] == [24, 1]


@pytest.mark.parametrize("status", [401, 429, 500])
def test_embedding_errors_do_not_expose_key_or_retry(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "sk-test-secret"}})

    embedder = OpenAIEmbedder(
        AgentSettings("sk-test-secret"), transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(EmbeddingError) as exc:
            embedder.embed(["합성 문서"])
    finally:
        embedder.close()
    assert len(calls) == 1
    assert "sk-test-secret" not in "".join(traceback.format_exception(exc.value))


@pytest.mark.parametrize("texts", [[], [" "], ["가" * 3000]])
def test_invalid_embedding_input_does_not_send_request(texts):
    def unexpected_request(_):
        pytest.fail("Invalid text must not be sent")

    embedder = OpenAIEmbedder(
        AgentSettings("sk-test"), transport=httpx.MockTransport(unexpected_request)
    )
    try:
        with pytest.raises(EmbeddingError):
            embedder.embed(texts)
    finally:
        embedder.close()
