"""Day 1 command line interface. Output contains data and evidence, not LLM prose."""

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from finsight.config import Settings
from finsight.dart.client import DartClient
from finsight.errors import FinSightError
from finsight.models import FinancialSummary, ReportType, StatementBasis
from finsight.services import FinancialService


def to_json_value(value: Any) -> Any:
    if isinstance(value, FinancialSummary):
        result = {name: to_json_value(item) for name, item in vars(value).items()}
        result["report_name"] = value.report_type.label
        return result
    if is_dataclass(value) and not isinstance(value, type):
        # Traverse fields before asdict so nested FinancialSummary keeps its report label.
        return {name: to_json_value(item) for name, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FinSight OpenDART 재무 조회·비교")
    parser.add_argument("--env-file", default=".env", help="환경변수 파일 경로")
    subparsers = parser.add_subparsers(dest="command", required=True)
    lookup = subparsers.add_parser("lookup", help="기업명/종목코드로 고유번호 조회")
    lookup.add_argument("company")
    financials = subparsers.add_parser("financials", help="한 기업의 주요 재무정보 조회")
    financials.add_argument("company")
    comparison = subparsers.add_parser("compare", help="두 기업 비교 (기본: 삼성전자·SK하이닉스)")
    comparison.add_argument("--left", default="005930")
    comparison.add_argument("--right", default="000660")
    for command in (financials, comparison):
        command.add_argument("--year", type=int, required=True, help="사업연도 (2015년 이후)")
        command.add_argument("--report", choices=list(ReportType), default=ReportType.ANNUAL)
        command.add_argument(
            "--basis",
            choices=list(StatementBasis),
            default=None,
            help="생략 시 연결 우선, 연결 부재 시 별도로 전환",
        )
    args = parser.parse_args(argv)
    if hasattr(args, "year") and not 2015 <= args.year <= 9999:
        parser.error("--year는 2015년 이후 4자리 사업연도여야 합니다.")
    try:
        settings = Settings.from_env(args.env_file)
        with DartClient(settings.opendart_api_key) as client:
            service = FinancialService(client)
            if args.command == "lookup":
                result = asdict(service.directory.find(args.company))
            else:
                report = ReportType(args.report)
                basis = StatementBasis(args.basis) if args.basis else None
                if args.command == "financials":
                    result = service.get_financials(args.company, args.year, report, basis=basis)
                else:
                    result = service.compare(args.left, args.right, args.year, report, basis=basis)
        print(json.dumps(to_json_value(result), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (FinSightError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
