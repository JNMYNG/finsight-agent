from dataclasses import replace

import pytest

from finsight.errors import DisclosureError, EmbeddingError
from finsight.rag.index import DisclosureIndex
from finsight.rag.parser import parse_disclosure
from finsight.rag.service import DisclosureService, evidence_excerpt, retrieval_query

from .rag_helpers import FakeEmbedder, archive, reference


def publish(index, ref=None, **kwargs):
    ref = ref or reference()
    digest, chunks = parse_disclosure(archive(ref, **kwargs), ref)
    return index.publish(ref, digest, chunks)


def test_real_local_chroma_filters_company_year_report_and_receipt(tmp_path):
    index = DisclosureIndex(tmp_path / "index", FakeEmbedder())
    targets = [
        reference(),
        reference(code="00164779"),
        reference(year=2023),
        reference(report="11012"),
    ]
    for ref in targets:
        publish(index, ref)
    for ref in targets:
        hits = index.candidates(ref, "메모리 수요 수익성")
        assert hits
        assert all(hit.metadata["corp_code"] == ref.company.corp_code for hit in hits)
        assert all(hit.metadata["business_year"] == ref.business_year for hit in hits)
        assert all(hit.metadata["report_type"] == ref.report_type for hit in hits)
        assert all(hit.metadata["receipt_no"] == ref.receipt_no for hit in hits)
        assert all(hit.metadata["source"] == ref.source for hit in hits)
        assert all(hit.metadata["kind"] == "paragraph" for hit in hits)
    assert not index.candidates(replace(reference(), receipt_no="20250318000001"), "메모리")


def test_duplicate_ingestion_does_not_reembed_or_duplicate_records(tmp_path):
    embedder = FakeEmbedder()
    index = DisclosureIndex(tmp_path / "index", embedder)
    first = publish(index)
    count = index.collection.count()
    calls = len(embedder.calls)
    second = publish(index)
    assert second.reused and second.embedded_chunks == 0
    assert first.generation == second.generation
    assert index.collection.count() == count and len(embedder.calls) == calls
    reopened = DisclosureIndex(tmp_path / "index", embedder)
    assert reopened.active(reference().key)["generation"] == first.generation


def test_reindex_fewer_chunks_does_not_return_stale_version(tmp_path):
    index = DisclosureIndex(tmp_path / "index", FakeEmbedder())
    first = publish(index, paragraphs=[f"메모리 수요와 수익성 OLD_VERSION_{i}" for i in range(8)])
    second = publish(index, paragraphs=["메모리 수요와 수익성 NEW_VERSION"], table=False)
    assert first.generation != second.generation
    assert index.active(reference().key)["chunks"] == second.chunks < first.chunks
    hits = index.candidates(reference(), "메모리 수요 수익성")
    assert all("OLD_VERSION" not in hit.text for hit in hits)
    assert any("NEW_VERSION" in hit.text for hit in hits)


def test_failed_reindex_keeps_previous_generation_and_can_resume(tmp_path):
    embedder = FakeEmbedder()
    index = DisclosureIndex(tmp_path / "index", embedder)
    original = publish(index, paragraphs=["메모리 수익성 PREVIOUS"], table=False)
    embedder.fail_after = len(embedder.calls) + 1  # first new batch succeeds, second fails
    paragraphs = [f"메모리 수요와 수익성 REPLACEMENT_{i}" for i in range(30)]
    with pytest.raises(EmbeddingError):
        publish(index, paragraphs=paragraphs, table=False)
    assert index.active(reference().key)["generation"] == original.generation
    embedder.fail_after = None
    assert any("PREVIOUS" in hit.text for hit in index.candidates(reference(), "메모리 수익성"))
    resumed = publish(index, paragraphs=paragraphs, table=False)
    assert 0 < resumed.embedded_chunks < resumed.chunks
    assert index.active(reference().key)["generation"] == resumed.generation
    assert all("PREVIOUS" not in hit.text for hit in index.candidates(reference(), "메모리 수익성"))


def test_embedding_model_or_dimension_cannot_be_mixed(tmp_path):
    index = DisclosureIndex(tmp_path / "index", FakeEmbedder())
    publish(index)
    changed = FakeEmbedder()
    changed.model = "different-embedding-model"
    with pytest.raises(DisclosureError):
        DisclosureIndex(tmp_path / "index", changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("corp_code", "00164779"),
        ("business_year", 2023),
        ("report_type", "11012"),
        ("receipt_no", "20250318000001"),
        ("source", "https://untrusted.example/"),
    ],
)
def test_cannot_attach_wrong_source_metadata(tmp_path, field, value):
    index = DisclosureIndex(tmp_path / "index", FakeEmbedder())
    ref = reference()
    digest, chunks = parse_disclosure(archive(ref), ref)
    chunks[0] = replace(chunks[0], metadata={**chunks[0].metadata, field: value})
    with pytest.raises(DisclosureError):
        index.publish(ref, digest, chunks)
    assert index.active(ref.key) is None


def test_no_index_or_unrelated_results_never_fabricate_evidence(agent_service, tmp_path):
    financials, _ = agent_service
    embedder = FakeEmbedder()
    index = DisclosureIndex(tmp_path / "index", embedder)
    service = DisclosureService(financials, index)
    empty = service.search("삼성전자", 2024, "11011", "메모리 수익성")
    assert not empty.hits and empty.issues == ("NOT_INDEXED",)
    assert not embedder.calls
    publish(index)
    unrelated = service.search("삼성전자", 2024, "11011", "우주유니콘 양자중력")
    assert not unrelated.hits and unrelated.issues == ("INSUFFICIENT_EVIDENCE",)
    found = service.search("삼성전자", 2024, "11011", "메모리 HBM 수요 수익성")
    assert found.hits and not found.issues
    assert all(hit.metadata["corp_code"] == "00126380" for hit in found.hits)


@pytest.mark.parametrize(
    "company,year,report",
    [
        ("SK하이닉스", 2023, "11011"),
        ("삼성전자", 2024, "11012"),
        ("테스트비상장", 2024, "11011"),
    ],
)
def test_service_rejects_out_of_scope_sources(agent_service, tmp_path, company, year, report):
    financials, _ = agent_service
    service = DisclosureService(financials, DisclosureIndex(tmp_path / "index", FakeEmbedder()))
    with pytest.raises(DisclosureError):
        service.search(company, year, report, "메모리 수요")


def test_low_similarity_is_rejected_even_when_words_match(agent_service, tmp_path):
    financials, _ = agent_service
    index = DisclosureIndex(tmp_path / "index", FakeEmbedder())
    publish(index)
    # A candidate can match a topic keyword but remain far from the overall query.
    candidates = index.candidates(reference(), "메모리 수익성")
    index.candidates = lambda *args, **kwargs: [replace(hit, distance=0.9) for hit in candidates]
    result = DisclosureService(financials, index).search("삼성전자", 2024, "11011", "메모리 수익성")
    assert not result.hits and result.issues == ("INSUFFICIENT_EVIDENCE",)


def test_future_outlook_is_not_a_cause_of_previous_year_results():
    query = "메모리 HBM 수요와 영업이익 수익성"
    past = "메모리 HBM 판매 확대로 수익성이 개선되었습니다."
    future = "2025년 메모리 수요 개선이 전망됩니다."
    assert evidence_excerpt(query, future, 2024) is None
    assert evidence_excerpt(query, past + " " + future, 2024) == past
    assert evidence_excerpt(query, "HBM 수요는 확대될 것으로 예상됩니다.", 2024) is None
    assert evidence_excerpt(query, "NAND 제품 경쟁력을 제고하겠습니다.", 2024) is None


def test_other_year_sentences_and_continuations_are_not_current_year_evidence():
    query = "메모리 수익성 실적"
    old = "2021년 메모리 수요가 증가하였습니다. 메모리 시장 가격도 상승하였습니다."
    current = "2024년 메모리 수요와 수익성이 개선되었습니다."
    assert evidence_excerpt(query, old + " " + current, 2024) == current
    assert evidence_excerpt(query, old, 2024) is None


def test_generic_query_expansion_is_explicit_and_specific_queries_are_preserved():
    assert "HBM" in retrieval_query("실적 차이")
    assert retrieval_query("메모리 HBM 수익성") == "메모리 HBM 수익성"
    assert retrieval_query("원가 수익성") == "원가 수익성"
    assert retrieval_query("우주유니콘 양자중력") == "우주유니콘 양자중력"
    assert "HBM" in retrieval_query("수익성 및 제품 수요 변화에 따른 실적 요인")


@pytest.mark.parametrize("text", ["(3) 판매경로별 매출액 비중", "판매경로 등", "(4) 판매 전략"])
def test_subheadings_cannot_be_used_as_performance_explanations(text):
    assert evidence_excerpt("수익성 및 제품 수요 변화에 따른 실적 요인", text, 2024) is None


def test_present_tense_without_year_is_not_assumed_to_describe_2024():
    assert (
        evidence_excerpt(
            "메모리 HBM 수익성", "메모리 HBM 공급 확대를 통해 제품 수익성을 높이고 있습니다.", 2024
        )
        is None
    )
