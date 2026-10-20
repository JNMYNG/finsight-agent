"""Opt-in REAL API verification through Streamlit AppTest (not a browser screenshot)."""

import argparse
import json
import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from finsight.config import Settings
from finsight.dart.client import DartClient
from finsight.models import StatementBasis
from finsight.rag.models import DisclosureRef
from finsight.rag.parser import parse_disclosure
from finsight.serialization import to_json_value
from finsight.services import FinancialService
from finsight.web.presentation import project

ENTRY = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def verify(years: list[int], basis: str, rag: bool) -> dict:
    at = AppTest.from_file(ENTRY, default_timeout=240).run()
    assert not at.exception and at.title[0].value == "FinSight", "INITIAL_RENDER_FAILED"
    at.multiselect(key="years").set_value(years)
    at.selectbox(key="basis").select(basis)
    at.checkbox(key="rag").set_value(rag)
    at.button[0].click().run()
    assert not at.exception and "analysis" in at.session_state, "UI_EXECUTION_FAILED"
    request, result = at.session_state.analysis
    dashboard = project(result, request)
    assert dashboard.status == "completed", "ANALYSIS_INCOMPLETE"
    assert len(dashboard.summaries) == len(years) * 2, "MISSING_FINANCIALS"
    assert len(at.metric) == 8 and len(at.dataframe) == 3, "MISSING_UI_RESULTS"
    assert at.get("vega_lite_chart"), "MISSING_CHART"
    with DartClient(Settings.from_env().opendart_api_key) as client:
        service = FinancialService(client)
        for summary in dashboard.summaries:
            expected = service.get_financials(
                summary["company"]["corp_name"],
                summary["business_year"],
                basis=StatementBasis(basis),
            )
            expected = to_json_value(expected)
            for key in (
                "company",
                "business_year",
                "report_type",
                "basis",
                "revenue",
                "operating_profit",
                "net_income",
                "operating_margin_pct",
            ):
                assert summary[key] == expected[key], "FINANCIAL_SOURCE_MISMATCH"
    verified_hits, originals = [], {}
    for hit in dashboard.hits:
        meta = hit["metadata"]
        receipt = meta["receipt_no"]
        if receipt not in originals:
            company = next(
                item["company"]
                for item in dashboard.summaries
                if item["company"]["corp_code"] == meta["corp_code"]
            )
            from finsight.models import Corporation

            ref = DisclosureRef(Corporation(**company), 2024, "11011", receipt)
            path = (
                ENTRY.parent
                / "data/disclosures/raw"
                / (f"{receipt}-{meta['source_sha256'][:16]}.zip")
            )
            digest, chunks = parse_disclosure(path.read_bytes(), ref)
            assert digest == meta["source_sha256"], "ORIGINAL_HASH_MISMATCH"
            originals[receipt] = {chunk.id: chunk for chunk in chunks}
        original = originals[receipt][hit["chunk_id"]]
        assert hit["text"] == original.text, "ORIGINAL_TEXT_MISMATCH"
        assert all(meta.get(key) == value for key, value in original.metadata.items()), (
            "ORIGINAL_METADATA_MISMATCH"
        )
        assert all(part in original.text for part in hit["excerpt"].split(" […] "))
        assert any(link.proto.url == meta["source"] for link in at.get("link_button")), (
            "SOURCE_LINK_NOT_RENDERED"
        )
        assert any(f"XML {meta['line']}행" in element.value for element in at.caption)
        verified_hits.append(
            {
                key: meta[key]
                for key in (
                    "corp_name",
                    "corp_code",
                    "business_year",
                    "receipt_no",
                    "source",
                    "section",
                    "line",
                )
            }
        )
    if rag:
        assert {hit["corp_name"] for hit in verified_hits} == set(request.companies), "RAG_MISSING"
    run_id = result.run_id
    at.selectbox(key="chart_metric").select("매출액 (KRW)").run()
    assert not at.exception and at.session_state.analysis[1].run_id == run_id, "REQUERY_ON_RERENDER"
    return {
        "verification": "real OpenDART + OpenAI via Streamlit AppTest (headless)",
        "run_id": run_id,
        "status": dashboard.status,
        "years": years,
        "basis": basis,
        "tool_order": [call["Tool"] for call in dashboard.calls],
        "financials": [
            {
                "company": item["company"]["corp_name"],
                "business_year": item["business_year"],
                "operating_margin_pct": item["operating_margin_pct"],
            }
            for item in dashboard.summaries
        ],
        "financial_source_verified": True,
        "rendered_chart": True,
        "rag_original_provenance_verified": bool(verified_hits),
        "sources": verified_hits,
        "rerender_did_not_requery": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="실제 API를 사용하는 Streamlit 화면·근거 검증")
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    parser.add_argument("--basis", choices=["CFS", "OFS"], default="CFS")
    parser.add_argument("--no-rag", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.years, args.basis, not args.no_rag)
    except Exception as exc:
        # Never print SDK/config exceptions or raw AppTest tracebacks with real credentials.
        code = str(exc) if isinstance(exc, AssertionError) else "VERIFICATION_ERROR"
        code = code if re.fullmatch(r"[A-Z_]+", code) else "VERIFICATION_ERROR"
        print(json.dumps({"verified": False, "error_type": type(exc).__name__, "code": code}))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
