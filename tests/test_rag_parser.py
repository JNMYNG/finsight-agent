from io import BytesIO
from zipfile import ZipFile

import httpx
import pytest

from finsight.dart.client import DartClient
from finsight.errors import DartAPIError, DisclosureError, InvalidResponseError
from finsight.rag.parser import parse_disclosure

from .conftest import TEST_API_KEY
from .rag_helpers import archive, reference


def test_actual_format_contract_and_body_attachment_selection():
    ref = reference()
    digest, chunks = parse_disclosure(archive(ref), ref)
    assert len(digest) == 64 and chunks
    assert {chunk.metadata["kind"] for chunk in chunks} == {"paragraph", "table"}
    assert all(chunk.metadata["member"] == f"{ref.receipt_no}.xml" for chunk in chunks)
    assert all(chunk.metadata["corp_code"] == ref.company.corp_code for chunk in chunks)
    assert all(chunk.metadata["business_year"] == 2024 for chunk in chunks)
    assert all(chunk.metadata["source"] == ref.source for chunk in chunks)
    assert all(chunk.metadata["line"] > 0 for chunk in chunks)
    assert not any("재무 주석 제외" in chunk.text for chunk in chunks)
    assert not any("색인하지 않는 항목" in chunk.text for chunk in chunks)
    table = next(chunk for chunk in chunks if chunk.metadata["kind"] == "table")
    assert "구분 | 가격" in table.text and "rowspan=2" in table.text
    assert "II. 사업의 내용 > 1. 사업의 개요" in table.metadata["section"]


def test_deterministic_chunks_and_long_table_row_location_regression():
    ref = reference()
    content = archive(
        ref,
        paragraphs=["긴 문단 " * 500],
        table=False,
        extra="<TABLE><TR><TD>" + "메모리 " * 500 + "</TD></TR></TABLE>",
    )
    first = parse_disclosure(content, ref, max_chars=300)[1]
    second = parse_disclosure(content, ref, max_chars=300)[1]
    assert first == second
    assert len({chunk.id for chunk in first}) == len(first)
    assert all(len(chunk.text) <= 300 for chunk in first)
    tables = [chunk for chunk in first if chunk.metadata["kind"] == "table"]
    assert len(tables) > 1
    assert all(chunk.metadata["row_start"] == chunk.metadata["row_end"] == 1 for chunk in tables)


@pytest.mark.parametrize(
    "other", [reference(code="00164779"), reference(year=2023), reference(report="11012")]
)
def test_wrong_company_year_or_report_cannot_be_relabelled(other):
    with pytest.raises(DisclosureError):
        parse_disclosure(archive(other), reference())


def test_paths_and_entity_declarations_rejected():
    ref = reference()
    with pytest.raises(DisclosureError):
        parse_disclosure(archive(ref, member=f"../{ref.receipt_no}.xml"), ref)
    with pytest.raises(DisclosureError):
        parse_disclosure(
            archive(ref, extra="<!DOCTYPE x [<!ENTITY x SYSTEM 'file:///etc/passwd'>]>"), ref
        )
    with pytest.raises(DisclosureError):
        parse_disclosure(b"not zip", ref)


def test_script_text_removed_without_executing_or_fetching():
    _, chunks = parse_disclosure(archive(extra="<SCRIPT>never execute this</SCRIPT>"), reference())
    assert not any("never execute" in chunk.text for chunk in chunks)


def test_tolerates_xml_like_unescaped_ampersand():
    _, chunks = parse_disclosure(
        archive(extra="<P>메모리 수요 & 공급이 수익성에 관련됨</P>"), reference()
    )
    assert any("& 공급" in chunk.text for chunk in chunks)


def test_document_endpoint_parameters_and_error_handling():
    ref = reference()
    content = archive(ref)

    def handler(request):
        assert request.url.path == "/api/document.xml"
        assert dict(request.url.params) == {"crtfc_key": TEST_API_KEY, "rcept_no": ref.receipt_no}
        return httpx.Response(200, content=content)

    with DartClient(TEST_API_KEY, transport=httpx.MockTransport(handler)) as client:
        assert client.download_disclosure(ref.receipt_no) == content
        with pytest.raises(ValueError):
            client.download_disclosure("bad")


@pytest.mark.parametrize(
    "body,error",
    [
        (b"<result><status>014</status><message>no file</message></result>", DartAPIError),
        (b"broken", InvalidResponseError),
        (b"<result><status>000</status></result>", InvalidResponseError),
    ],
)
def test_document_error_is_not_parsed_as_source(body, error):
    with DartClient(
        TEST_API_KEY, transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body))
    ) as client:
        with pytest.raises(error):
            client.download_disclosure(reference().receipt_no)


def test_missing_main_member_rejected():
    output = BytesIO()
    with ZipFile(output, "w") as zipped:
        zipped.writestr("other.xml", "not a disclosure")
    with pytest.raises(DisclosureError):
        parse_disclosure(output.getvalue(), reference())
