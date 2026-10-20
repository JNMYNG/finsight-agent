"""Collect, verify, index, and search only the two requested 2024 annual reports."""

import re
from dataclasses import replace
from pathlib import Path

from finsight.errors import DisclosureError
from finsight.models import ReportType
from finsight.rag.index import DisclosureIndex
from finsight.rag.models import TARGET_CORP_CODES, DisclosureRef, DisclosureSearch, IndexResult
from finsight.rag.parser import parse_disclosure
from finsight.services import FinancialService


def relevant(query: str, text: str) -> bool:
    """Conservative topical guard in addition to cosine distance; not a causal claim."""
    groups = (
        (
            ("영업이익", "수익성", "실적", "수익", "영업손실", "이익률"),
            ("수요", "가격", "고부가", "HBM", "판매", "매출", "이익", "수익", "실적", "원가"),
        ),
        (("메모리", "반도체", "HBM", "DRAM", "NAND"), ("메모리", "반도체", "HBM", "DRAM", "NAND")),
        (("위험", "리스크"), ("위험", "리스크", "불확실")),
    )
    applicable = [
        words
        for triggers, words in groups
        if any(word.casefold() in query.casefold() for word in triggers)
    ]
    if applicable:
        return all(
            any(word.casefold() in text.casefold() for word in words) for words in applicable
        )
    terms = set(re.findall(r"[가-힣A-Za-z]{2,}", query)) - {
        "삼성전자",
        "하이닉스",
        "SK하이닉스",
        "사업보고서",
        "내용",
        "관련",
        "설명",
        "검색",
        "찾아줘",
    }
    return any(term.casefold() in text.casefold() for term in terms)


def evidence_excerpt(query: str, text: str, year: int) -> str | None:
    """Keep source sentences; future outlook is not evidence of historical performance."""
    performance = any(word in query for word in ("영업이익", "수익", "실적", "이익률", "원인"))
    sentences = re.split(r"(?<=[.!?])\s+|(?<=다\.)(?=[가-힣A-Za-z])", text)
    selected = []
    context_year = year
    anchored_year = False
    for sentence in sentences:
        sentence = sentence.strip()
        explicit_years = [int(value) for value in re.findall(r"(20[0-9]{2})\s*년", sentence)]
        if explicit_years:
            context_year = explicit_years[-1]
            anchored_year = True
        if not sentence or not relevant(query, sentence):
            continue
        if performance and (len(sentence) < 20 or not re.search(r"다[.!?]?$", sentence)):
            # DART often encodes short subheadings as P tags. They are not explanations.
            continue
        if (
            performance
            and not anchored_year
            and not any(word in sentence for word in ("하였", "되었", "했", "보였", "이었", "있었"))
        ):
            # Present-tense Q1 commentary in a report filed next year is ambiguous.
            continue
        if performance and (
            context_year != year
            or any(value > year for value in explicit_years)
            or any(
                word in sentence
                for word in ("전망", "예상", "계획", "예정", "것입니다", "겠습니다", "준비 중")
            )
        ):
            continue
        selected.append(sentence)
        if len(selected) == 3:
            break
    return " […] ".join(selected) if selected else None


def retrieval_query(query: str) -> str:
    """Expand underspecified performance queries into search terms, never causal facts."""
    generic = len(query) <= 40 and any(word in query for word in ("실적", "수익성", "이익률"))
    specific = any(
        word.casefold() in query.casefold()
        for word in (
            "메모리",
            "HBM",
            "DRAM",
            "NAND",
            "반도체",
            "원가",
            "위험",
            "DX",
            "SDC",
            "디스플레이",
            "스마트폰",
            "가전",
        )
    )
    if generic and not specific:
        return query + " 메모리 HBM 수요와 고부가 제품 판매 가격 및 영업이익 수익성 개선 요인"
    return query


class DisclosureService:
    def __init__(
        self,
        financials: FinancialService,
        index: DisclosureIndex,
        *,
        raw_path: str | Path = "data/disclosures/raw",
        max_distance: float = 0.55,
    ):
        if not 0 <= max_distance <= 2:
            raise ValueError("코사인 거리 한도는 0~2여야 합니다.")
        self.financials = financials
        self.index = index
        self.raw_path = Path(raw_path)
        self.max_distance = max_distance

    def _company(self, name: str, year: int, report_type: str):
        company = self.financials.directory.find(name)
        if company.corp_code not in TARGET_CORP_CODES or year != 2024 or report_type != "11011":
            raise DisclosureError("이번 RAG 범위는 삼성전자·SK하이닉스 2024년 사업보고서입니다.")
        return company

    def ingest(self, company: str, year: int = 2024, report_type: str = "11011") -> IndexResult:
        corporation = self._company(company, year, report_type)
        summary = self.financials.get_financials(company, year, ReportType(report_type))
        receipts = {
            item.receipt_no
            for metric in (summary.revenue, summary.operating_profit, summary.net_income)
            for item in metric.evidence
        }
        if len(receipts) != 1:
            raise DisclosureError("재무정보에서 단일 사업보고서 접수번호를 확인할 수 없습니다.")
        reference = DisclosureRef(corporation, year, report_type, receipts.pop())
        archive = self.financials.client.download_disclosure(reference.receipt_no)
        digest, chunks = parse_disclosure(archive, reference)
        self.raw_path.mkdir(parents=True, exist_ok=True)
        # Immutable content-named archives preserve earlier fetched versions.
        (self.raw_path / f"{reference.receipt_no}-{digest[:16]}.zip").write_bytes(archive)
        return self.index.publish(reference, digest, chunks)

    def search(
        self, company: str, business_year: int, report_type: str, query: str, *, top_k: int = 3
    ) -> DisclosureSearch:
        corporation = self._company(company, business_year, report_type)
        if not query.strip() or len(query) > 2000 or not 1 <= top_k <= 5:
            raise ValueError("검색어(1~2000자)와 결과 수(1~5)를 확인하세요.")
        key = f"{corporation.corp_code}:{business_year}:{report_type}"
        active = self.index.active(key)
        if active is None:
            return DisclosureSearch(
                corporation, business_year, report_type, query, (), ("NOT_INDEXED",)
            )
        reference = DisclosureRef(corporation, business_year, report_type, active["receipt_no"])
        effective_query = f"{corporation.corp_name} {business_year} 사업보고서\n" + retrieval_query(
            query
        )
        candidates = self.index.candidates(reference, effective_query)
        # Explanations use prose, not standalone financial tables; tables remain indexed.
        selected = []
        for hit in candidates:
            if hit.distance > self.max_distance or hit.metadata["kind"] != "paragraph":
                continue
            excerpt = evidence_excerpt(effective_query, hit.text, business_year)
            if excerpt:
                selected.append(replace(hit, excerpt=excerpt))
        unique = []
        seen = set()
        for hit in selected:
            if hit.text not in seen:
                unique.append(hit)
                seen.add(hit.text)
        hits = tuple(unique[:top_k])
        return DisclosureSearch(
            corporation,
            business_year,
            report_type,
            query,
            hits,
            () if hits else ("INSUFFICIENT_EVIDENCE",),
            effective_query,
        )
