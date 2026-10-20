"""Parse DART's XML-like markup without executing scripts or resolving entities."""

import hashlib
import re
import warnings
from io import BytesIO
from zipfile import BadZipFile, ZipFile

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from finsight.errors import DisclosureError
from finsight.rag.models import PARSER_VERSION, DisclosureRef, DocumentChunk

MAX_XML_BYTES = 30 * 1024 * 1024
SECTION_PATTERN = re.compile(r"^(?:II\.|IV\.)\s")


def _text(node: Tag) -> str:
    return " ".join(node.get_text(" ", strip=True).replace("\xa0", " ").split())


def parse_disclosure(
    archive_bytes: bytes, reference: DisclosureRef, *, max_chars: int = 1400
) -> tuple[str, list[DocumentChunk]]:
    if not 300 <= max_chars <= 1800:
        raise ValueError("문서 조각 크기는 300~1800자여야 합니다.")
    member_name = f"{reference.receipt_no}.xml"
    try:
        with ZipFile(BytesIO(archive_bytes)) as archive:
            members = [item for item in archive.infolist() if item.filename == member_name]
            if len(members) != 1 or members[0].file_size > MAX_XML_BYTES:
                raise DisclosureError("원문 ZIP에 크기가 적절한 단일 본문 XML이 없습니다.")
            raw = archive.read(members[0])
    except (BadZipFile, RuntimeError, NotImplementedError):
        raise DisclosureError("공시 원문 ZIP을 읽을 수 없습니다.") from None
    # html.parser never fetches network resources. Reject declarations explicitly too.
    if re.search(rb"<!\s*(?:DOCTYPE|ENTITY)\b", raw, re.IGNORECASE):
        raise DisclosureError("외부 선언/엔티티가 있는 원문은 처리하지 않습니다.")
    try:
        markup = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise DisclosureError("이번 범위는 실제 확인한 UTF-8 DART 본문만 지원합니다.") from None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(markup, "html.parser")
    document = soup.find("document-name")
    company = soup.find("company-name")
    if (
        document is None
        or document.get("acode") != reference.report_type
        or company is None
        or company.get("aregcik") != reference.company.corp_code
    ):
        raise DisclosureError("원문의 기업 고유번호/사업보고서 코드가 요청 조건과 다릅니다.")
    # Verify the fiscal year in the cover, not a number appearing in the body or receipt year.
    prefix = " ".join(soup.stripped_strings)[:5000]
    fiscal_year = re.search(r"사업연도\s*([0-9]{4})\s*년", prefix)
    if fiscal_year is None or int(fiscal_year[1]) != reference.business_year:
        raise DisclosureError("원문 표지의 사업연도를 확인할 수 없거나 요청 연도와 다릅니다.")
    for tag in soup.find_all(["script", "style", "iframe", "object"]):
        tag.decompose()

    digest = hashlib.sha256(raw).hexdigest()
    chunks: list[DocumentChunk] = []
    block_number = 0

    def emit(
        text: str,
        *,
        section: str,
        kind: str,
        line: int,
        block: int,
        part: int = 0,
        row_start: int = 0,
        row_end: int = 0,
    ) -> None:
        if not text.strip():
            return
        identity = f"{reference.key}:{digest}:{PARSER_VERSION}:{max_chars}:{block}:{part}"
        metadata = {
            "document_key": reference.key,
            "corp_code": reference.company.corp_code,
            "corp_name": reference.company.corp_name,
            "business_year": reference.business_year,
            "report_type": reference.report_type,
            "receipt_no": reference.receipt_no,
            "source": reference.source,
            "download_endpoint": "https://opendart.fss.or.kr/api/document.xml",
            "member": member_name,
            "section": section[:600],
            "kind": kind,
            "line": line,
            "block": block,
            "part": part,
            "row_start": row_start,
            "row_end": row_end,
            "source_sha256": digest,
            "parser_version": PARSER_VERSION,
            "indexed_scope": "business_and_management",
            "max_chars": max_chars,
        }
        chunks.append(DocumentChunk(hashlib.sha256(identity.encode()).hexdigest(), text, metadata))

    selected = []
    for section in soup.find_all("section-1"):
        title = section.find("title", recursive=False)
        if title is not None and SECTION_PATTERN.match(_text(title)):
            selected.append(section)
    if not selected:
        raise DisclosureError("사업 내용/경영진단 본문 섹션을 찾을 수 없습니다.")
    for section in selected:
        for node in section.find_all(["title", "p", "table"]):
            if node.find_parent("table") is not None or node.find_parent("p") is not None:
                continue
            if node.name == "title":
                continue
            headings = []
            for parent in reversed(list(node.parents)):
                if parent.name in {"section-1", "section-2", "section-3"}:
                    title = parent.find("title", recursive=False)
                    if title is not None:
                        headings.append(_text(title))
            heading = " > ".join(headings)
            line = node.sourceline or 0
            block_number += 1
            if node.name == "table":
                rows = [row for row in node.find_all("tr") if row.find_parent("table") is node]
                buffer: list[str] = []
                first_row = 1
                last_row = 0
                part = 0
                for row_index, row in enumerate(rows, 1):
                    cells = row.find_all(["td", "th", "te", "tu"], recursive=False)
                    cells_text = []
                    for cell in cells:
                        span = " ".join(
                            f"{key}={cell[key]}"
                            for key in ("rowspan", "colspan")
                            if cell.has_attr(key)
                        )
                        cells_text.append(_text(cell) + (f" [{span}]" if span else ""))
                    row_text = " | ".join(cells_text)
                    if not row_text.strip(" |"):
                        continue
                    # Long rows remain explicitly marked as fragments of the same row.
                    fragments = [
                        row_text[i : i + max_chars] for i in range(0, len(row_text), max_chars)
                    ]
                    for fragment in fragments:
                        if buffer and len("\n".join(buffer)) + len(fragment) + 1 > max_chars:
                            emit(
                                "\n".join(buffer),
                                section=heading,
                                kind="table",
                                line=line,
                                block=block_number,
                                part=part,
                                row_start=first_row,
                                row_end=last_row,
                            )
                            part += 1
                            buffer = []
                        if not buffer:
                            first_row = row_index
                        buffer.append(fragment)
                        last_row = row_index
                if buffer:
                    emit(
                        "\n".join(buffer),
                        section=heading,
                        kind="table",
                        line=line,
                        block=block_number,
                        part=part,
                        row_start=first_row,
                        row_end=last_row,
                    )
            else:
                text = _text(node)
                for part, start in enumerate(range(0, len(text), max_chars)):
                    emit(
                        text[start : start + max_chars],
                        section=heading,
                        kind="paragraph",
                        line=line,
                        block=block_number,
                        part=part,
                    )
    if not chunks:
        raise DisclosureError("선택한 본문에서 색인 가능한 문단/표를 찾지 못했습니다.")
    return digest, chunks
