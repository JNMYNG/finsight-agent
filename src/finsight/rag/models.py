"""Source identity and immutable retrieval evidence."""

from dataclasses import dataclass

from finsight.models import Corporation

TARGET_CORP_CODES = {"00126380", "00164779"}
PARSER_VERSION = "dart-narrative-v2"


@dataclass(frozen=True)
class DisclosureRef:
    company: Corporation
    business_year: int
    report_type: str
    receipt_no: str

    @property
    def key(self) -> str:
        return f"{self.company.corp_code}:{self.business_year}:{self.report_type}"

    @property
    def source(self) -> str:
        return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={self.receipt_no}"


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class DisclosureHit:
    chunk_id: str
    text: str
    distance: float
    metadata: dict[str, str | int]
    excerpt: str | None = None


@dataclass(frozen=True)
class DisclosureSearch:
    company: Corporation
    business_year: int
    report_type: str
    query: str
    hits: tuple[DisclosureHit, ...]
    issues: tuple[str, ...] = ()
    retrieval_query: str | None = None


@dataclass(frozen=True)
class IndexResult:
    reference: DisclosureRef
    generation: str
    chunks: int
    embedded_chunks: int
    reused: bool
    source_sha256: str
