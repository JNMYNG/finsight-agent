"""Explicit, opt-in real OpenDART verification. Never executed by pytest."""

import argparse
import json
import sys

from finsight.agent.reporting import has_data_issues
from finsight.config import Settings
from finsight.dart.client import DartClient
from finsight.errors import FinSightError
from finsight.models import ReportType, StatementBasis
from finsight.serialization import to_json_value
from finsight.services import FinancialService


def main() -> int:
    parser = argparse.ArgumentParser(description="삼성전자·SK하이닉스 복수 연도 실제 조회")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    if any(not 2015 <= year <= 9999 for year in args.years):
        parser.error("사업연도는 2015년 이후 4자리 정수여야 합니다.")
    try:
        settings = Settings.from_env(args.env_file)
        reports = []
        with DartClient(settings.opendart_api_key) as client:
            service = FinancialService(client)
            for year in dict.fromkeys(args.years):
                comparison = service.compare_semiconductors(
                    year, ReportType.ANNUAL, basis=StatementBasis.CONSOLIDATED
                )
                for summary in (comparison.left, comparison.right):
                    reports.append(
                        {
                            "company": summary.company,
                            "business_year": summary.business_year,
                            "report_type": summary.report_type,
                            "basis": summary.basis,
                            "revenue": summary.revenue,
                            "operating_profit": summary.operating_profit,
                            "net_income": summary.net_income,
                            "operating_margin_pct": summary.operating_margin_pct,
                            "issues": summary.issues,
                            "source": summary.source,
                            "retrieved_at": summary.retrieved_at,
                        }
                    )
        payload = to_json_value(reports)
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if has_data_issues(payload) else 0
    except FinSightError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
