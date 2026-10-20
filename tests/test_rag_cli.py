import json

import httpx
import pytest

from finsight import cli
from finsight.dart.client import DartClient

from .conftest import TEST_API_KEY, zip_xml
from .rag_helpers import FakeEmbedder, archive, reference


@pytest.fixture
def mocked_rag_cli(monkeypatch, accounts_factory):
    import finsight.rag.embeddings as embedding_module

    monkeypatch.setenv("DART_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    state = {"code": "00126380"}

    def handler(request):
        if request.url.path.endswith("corpCode.xml"):
            return httpx.Response(200, content=zip_xml())
        if request.url.path.endswith("document.xml"):
            return httpx.Response(200, content=archive(reference(code=state["code"])))
        state["code"] = request.url.params["corp_code"]
        stock = "005930" if state["code"] == "00126380" else "000660"
        return httpx.Response(200, json={"status": "000", "list": accounts_factory(stock=stock)})

    def embedder(_):
        fake = FakeEmbedder()
        fake.close = lambda: None
        return fake

    monkeypatch.setattr(
        cli, "DartClient", lambda key: DartClient(key, transport=httpx.MockTransport(handler))
    )
    monkeypatch.setattr(embedding_module, "OpenAIEmbedder", embedder)


def test_cli_index_both_companies_search_and_reindex(mocked_rag_cli, tmp_path, capsys):
    path = str(tmp_path / "index")
    assert cli.main(["rag-index", "--db-path", path]) == 0
    initial = json.loads(capsys.readouterr().out)
    assert len(initial) == 2 and all(item["embedded_chunks"] > 0 for item in initial)
    for company, code in (("삼성전자", "00126380"), ("SK하이닉스", "00164779")):
        assert cli.main(["rag-search", company, "메모리 HBM 수익성", "--db-path", path]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["hits"] and all(hit["metadata"]["corp_code"] == code for hit in data["hits"])
        assert data["retrieval_query"].startswith(company + " 2024 사업보고서")
    assert cli.main(["rag-index", "--db-path", path]) == 0
    repeated = json.loads(capsys.readouterr().out)
    assert all(item["embedded_chunks"] == 0 and item["reused"] for item in repeated)


def test_cli_unindexed_search_returns_nonzero_and_no_evidence(mocked_rag_cli, tmp_path, capsys):
    assert (
        cli.main(["rag-search", "삼성전자", "메모리 수익성", "--db-path", str(tmp_path / "empty")])
        == 1
    )
    data = json.loads(capsys.readouterr().out)
    assert not data["hits"] and data["issues"] == ["NOT_INDEXED"]


def test_cli_rag_agent_uses_the_same_persisted_index(mocked_rag_cli, tmp_path, capsys, monkeypatch):
    import finsight.agent.model as model_module
    from finsight.agent.model import ModelTurn

    from .test_agent_graph import ScriptedModel, turn
    from .test_rag_agent import comparison_turn, search_args

    path = str(tmp_path / "index")
    assert cli.main(["rag-index", "--db-path", path]) == 0
    capsys.readouterr()
    model = ScriptedModel(
        [
            comparison_turn(),
            turn("search_disclosures", search_args(), "call_2"),
            turn("search_disclosures", search_args("SK하이닉스"), "call_3"),
            ModelTurn(),
        ]
    )
    model.close = lambda: None
    monkeypatch.setattr(model_module, "OpenAIToolModel", lambda _: model)
    assert (
        cli.main(
            [
                "agent",
                "--rag",
                "--db-path",
                path,
                "삼성전자와 SK하이닉스 2024년 영업이익률 비교와 사업보고서 설명을 찾아줘",
            ]
        )
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "completed" and "공시 근거" in data["answer"]
    assert [item["tool"] for item in data["tool_calls"]].count("search_disclosures") == 2
