"""Download and inspect actual target filings; no synthetic documents or API keys in output."""

import json
from collections import Counter
from io import BytesIO
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import ZipFile

from defusedxml import ElementTree

from finsight.config import Settings
from finsight.dart.client import DartClient
from finsight.services import FinancialService


def main() -> None:
    destination = Path("data/disclosures/raw")
    destination.mkdir(parents=True, exist_ok=True)
    with DartClient(Settings.from_env().opendart_api_key) as client:
        service = FinancialService(client)
        for stock in ("005930", "000660"):
            summary = service.get_financials(stock, 2024)
            receipts = {item.receipt_no for item in summary.revenue.evidence}
            if len(receipts) != 1:
                raise ValueError("원문 접수번호를 하나로 확인할 수 없습니다.")
            receipt = receipts.pop()
            content = client.download_disclosure(receipt)
            path = destination / f"{receipt}.zip"
            path.write_bytes(content)
            members = []
            with ZipFile(BytesIO(content)) as archive:
                for item in archive.infolist():
                    record = {"file": item.filename, "bytes": item.file_size}
                    if item.filename.endswith(".xml") and item.file_size < 100_000_000:
                        xml = archive.read(item)
                        record["header"] = xml[:180].decode("utf-8", errors="replace")
                        try:
                            root = ElementTree.fromstring(xml)
                            record["root"] = root.tag
                            record["tags"] = Counter(node.tag for node in root.iter()).most_common(
                                18
                            )
                            record["titles"] = [
                                "".join(node.itertext())[:120] for node in root.iter("TITLE")
                            ][:14]
                        except ParseError as exc:
                            record["parse_error"] = str(exc)
                    members.append(record)
            print(
                json.dumps(
                    {
                        "company": summary.company.corp_name,
                        "year": 2024,
                        "receipt": receipt,
                        "zip_bytes": len(content),
                        "members": members,
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
