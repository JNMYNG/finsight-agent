"""Opt-in verification of actual retrieval provenance against saved OpenDART originals."""

import argparse
import json
from pathlib import Path

from finsight.config import AgentSettings, Settings
from finsight.dart.client import DartClient
from finsight.errors import DisclosureError, FinSightError
from finsight.rag.embeddings import OpenAIEmbedder
from finsight.rag.index import DisclosureIndex
from finsight.rag.models import DisclosureRef
from finsight.rag.parser import parse_disclosure
from finsight.rag.service import DisclosureService
from finsight.serialization import to_json_value
from finsight.services import FinancialService

QUERY = "메모리 HBM 수요와 고부가 제품 판매 가격 및 영업이익 수익성 개선 요인"


def verify(disclosures, financials, company):
    summary = financials.get_financials(company, 2024)
    receipts = {
        item.receipt_no
        for metric in (summary.revenue, summary.operating_profit, summary.net_income)
        for item in metric.evidence
    }
    result = disclosures.search(company, 2024, "11011", QUERY)
    if result.issues or not result.hits or len(receipts) != 1:
        raise DisclosureError("실제 원문 검색 근거를 충분히 확인하지 못했습니다.")
    receipt = receipts.pop()
    ref = DisclosureRef(summary.company, 2024, "11011", receipt)
    active = disclosures.index.active(ref.key)
    path = Path(disclosures.raw_path) / f"{receipt}-{active['source_sha256'][:16]}.zip"
    digest, chunks = parse_disclosure(path.read_bytes(), ref)
    originals = {chunk.id: chunk for chunk in chunks}
    verified = []
    for hit in result.hits:
        meta = hit.metadata
        source = originals.get(hit.chunk_id)
        if (
            source is None
            or source.text != hit.text
            or digest != meta["source_sha256"]
            or meta["corp_code"] != ref.company.corp_code
            or meta["business_year"] != 2024
            or meta["report_type"] != "11011"
            or meta["receipt_no"] != receipt
            or meta["source"] != ref.source
            or meta["member"] != f"{receipt}.xml"
            or meta["line"] != source.metadata["line"]
            or meta["block"] != source.metadata["block"]
            or not hit.excerpt
            or not all(part in source.text for part in hit.excerpt.split(" […] "))
        ):
            raise DisclosureError("검색 결과가 원문·기업·연도·접수번호·위치와 일치하지 않습니다.")
        verified.append(
            {
                "chunk_id": hit.chunk_id,
                "distance": hit.distance,
                "excerpt": hit.excerpt,
                "source": meta["source"],
                "receipt_no": receipt,
                "section": meta["section"],
                "member": meta["member"],
                "line": meta["line"],
                "block": meta["block"],
            }
        )
    unrelated = disclosures.search(company, 2024, "11011", "우주유니콘 양자중력")
    if unrelated.hits or unrelated.issues != ("INSUFFICIENT_EVIDENCE",):
        raise DisclosureError("무관한 검색어가 근거로 승인되었습니다.")
    return {
        "company": summary.company,
        "business_year": 2024,
        "report_type": "11011",
        "operating_margin_pct": summary.operating_margin_pct,
        "provenance_verified": True,
        "unrelated_query_rejected": True,
        "hits": verified,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="두 기업 실제 RAG 검색·원문 대조")
    parser.add_argument("--db-path", default="data/chroma")
    args = parser.parse_args()
    embedder = OpenAIEmbedder(AgentSettings.from_env())
    try:
        with DartClient(Settings.from_env().opendart_api_key) as client:
            financials = FinancialService(client)
            disclosures = DisclosureService(financials, DisclosureIndex(args.db_path, embedder))
            results = [verify(disclosures, financials, company) for company in ("005930", "000660")]
        print(
            json.dumps(
                to_json_value({"query": QUERY, "results": results}),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    except (FinSightError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1
    finally:
        embedder.close()


if __name__ == "__main__":
    raise SystemExit(main())
