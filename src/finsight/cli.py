"""Day 1 data commands and Day 2 evidence-grounded agent command."""

import argparse
import json
import sys
from dataclasses import asdict

from finsight.config import Settings
from finsight.dart.client import DartClient
from finsight.errors import FinSightError
from finsight.models import ReportType, StatementBasis
from finsight.serialization import to_json_value
from finsight.services import FinancialService


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
    agent = subparsers.add_parser("agent", help="자연어 질문으로 LangGraph Agent 실행")
    agent.add_argument("question")
    agent.add_argument("--log-file", default="logs/agent.jsonl", help="순서·입력·결과 JSONL 로그")
    agent.add_argument("--max-tool-calls", type=int, default=8)
    agent.add_argument("--max-model-turns", type=int, default=10)
    agent.add_argument("--rag", action="store_true", help="로컬 사업보고서 검색 Tool 활성화")
    agent.add_argument("--db-path", default="data/chroma")
    index = subparsers.add_parser("rag-index", help="대상 2024년 사업보고서 수집·색인")
    index.add_argument("companies", nargs="*", default=["005930", "000660"])
    search = subparsers.add_parser("rag-search", help="조건을 제한한 사업보고서 근거 검색")
    search.add_argument("company")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=3)
    for command in (index, search):
        command.add_argument("--year", type=int, choices=[2024], default=2024)
        command.add_argument("--db-path", default="data/chroma")
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
            if args.command in {"rag-index", "rag-search"}:
                from finsight.config import AgentSettings
                from finsight.rag.embeddings import OpenAIEmbedder
                from finsight.rag.index import DisclosureIndex
                from finsight.rag.service import DisclosureService

                embedder = OpenAIEmbedder(AgentSettings.from_env(args.env_file))
                try:
                    disclosures = DisclosureService(
                        service, DisclosureIndex(args.db_path, embedder)
                    )
                    if args.command == "rag-index":
                        result = [
                            disclosures.ingest(company, args.year) for company in args.companies
                        ]
                    else:
                        result = disclosures.search(
                            args.company, args.year, "11011", args.query, top_k=args.top_k
                        )
                finally:
                    embedder.close()
            elif args.command == "agent":
                # Lazy imports keep Day 1 commands independent of agent initialization.
                from finsight.agent.audit import AuditLog
                from finsight.agent.graph import FinancialAgent
                from finsight.agent.model import OpenAIToolModel
                from finsight.config import AgentSettings

                agent_settings = AgentSettings.from_env(args.env_file)
                audit = AuditLog(
                    args.log_file,
                    secrets=(
                        settings.opendart_api_key,
                        agent_settings.openai_api_key,
                    ),
                )
                model = OpenAIToolModel(agent_settings)
                embedder = None
                try:
                    options = {}
                    if args.rag:
                        from finsight.rag.embeddings import OpenAIEmbedder
                        from finsight.rag.index import DisclosureIndex
                        from finsight.rag.service import DisclosureService

                        embedder = OpenAIEmbedder(agent_settings)
                        options["disclosures"] = DisclosureService(
                            service, DisclosureIndex(args.db_path, embedder)
                        )
                    result = FinancialAgent(
                        service,
                        model,
                        audit=audit,
                        max_tool_calls=args.max_tool_calls,
                        max_model_turns=args.max_model_turns,
                        **options,
                    ).run(args.question)
                finally:
                    model.close()
                    if embedder is not None:
                        embedder.close()
            elif args.command == "lookup":
                result = asdict(service.directory.find(args.company))
            else:
                report = ReportType(args.report)
                basis = StatementBasis(args.basis) if args.basis else None
                if args.command == "financials":
                    result = service.get_financials(args.company, args.year, report, basis=basis)
                else:
                    result = service.compare(args.left, args.right, args.year, report, basis=basis)
        print(json.dumps(to_json_value(result), ensure_ascii=False, indent=2, allow_nan=False))
        if args.command == "agent" and result.status in {"error", "partial", "limit_reached"}:
            return 1
        if args.command == "rag-search" and result.issues:
            return 1
        return 0
    except (FinSightError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
