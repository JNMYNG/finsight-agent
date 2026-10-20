"""Clearly synthetic DART-like sources and deterministic embeddings for offline tests."""

from html import escape
from io import BytesIO
from zipfile import ZipFile

from finsight.errors import EmbeddingError
from finsight.models import Corporation
from finsight.rag.models import DisclosureRef


def reference(*, code="00126380", year=2024, report="11011", receipt="20250317000001"):
    name, stock = ("삼성전자", "005930") if code == "00126380" else ("SK하이닉스", "000660")
    return DisclosureRef(Corporation(code, name, stock, "20250101"), year, report, receipt)


def archive(ref=None, *, paragraphs=None, table=True, extra="", member=None):
    ref = ref or reference()
    paragraphs = (
        paragraphs
        if paragraphs is not None
        else [
            f"{ref.business_year}년 합성 테스트 문단: "
            "메모리 수요 및 HBM 판매 증가가 수익성 개선에 관련됩니다."
        ]
    )
    rows = (
        (
            "<TABLE><THEAD><TR><TH>구분</TH><TH>가격</TH></TR></THEAD>"
            "<TBODY><TR><TD ROWSPAN='2'>메모리</TD><TE>100</TE></TR>"
            "<TR><TU>합성</TU><TD>200</TD></TR></TBODY></TABLE>"
        )
        if table
        else ""
    )
    markup = (
        f"<?xml version='1.0' encoding='utf-8'?><DOCUMENT>"
        f"<DOCUMENT-NAME ACODE='{ref.report_type}'>사업보고서</DOCUMENT-NAME>"
        f"<COMPANY-NAME AREGCIK='{ref.company.corp_code}'>{ref.company.corp_name}</COMPANY-NAME>"
        f"<P>사업연도 {ref.business_year}년 01월 01일 부터 {ref.business_year}년 12월 31일</P>"
        "<SECTION-1><TITLE>I. 회사의 개요</TITLE><P>색인하지 않는 항목</P></SECTION-1>"
        "<SECTION-1><TITLE>II. 사업의 내용</TITLE>\n"
        "<SECTION-2><TITLE>1. 사업의 개요</TITLE>\n"
        + "\n".join(f"<P>{escape(text)}</P>" for text in paragraphs)
        + rows
        + extra
        + "</SECTION-2></SECTION-1>"
        "<SECTION-1><TITLE>III. 재무에 관한 사항</TITLE><P>재무 주석 제외</P></SECTION-1>"
        "<SECTION-1><TITLE>IV. 이사의 경영진단 및 분석의견</TITLE>"
        "<P>합성 테스트 문단: 반도체 판매 가격과 메모리 수요 변화.</P></SECTION-1></DOCUMENT>"
    )
    output = BytesIO()
    with ZipFile(output, "w") as zipped:
        zipped.writestr(member or f"{ref.receipt_no}.xml", markup.encode())
        zipped.writestr(f"{ref.receipt_no}_00760.xml", b"audit attachment, not the report")
    return output.getvalue()


class FakeEmbedder:
    model = "test-only-embedding"
    dimensions = 3

    def __init__(self):
        self.calls = []
        self.fail_after = None

    def embed(self, texts):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise EmbeddingError("synthetic interrupted embedding")
        self.calls.append(list(texts))
        return [
            [1.0, 0.0, 0.0]
            if any(word in text for word in ("메모리", "HBM", "영업이익", "수익성"))
            else [0.0, 1.0, 0.0]
            for text in texts
        ]
