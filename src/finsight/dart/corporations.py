"""Exact, unambiguous lookup from the official corpCode.xml archive."""

import re
import unicodedata
from collections import defaultdict
from xml.etree.ElementTree import ParseError

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

from finsight.dart.client import DartClient
from finsight.errors import AmbiguousCompanyError, CompanyNotFoundError, InvalidResponseError
from finsight.models import Corporation


def normalize_name(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).split()).casefold()


class CorporationDirectory:
    def __init__(self, corporations: list[Corporation]) -> None:
        self._by_name: dict[str, list[Corporation]] = defaultdict(list)
        self._by_stock: dict[str, list[Corporation]] = defaultdict(list)
        for company in corporations:
            self._by_name[normalize_name(company.corp_name)].append(company)
            if company.stock_code:
                self._by_stock[company.stock_code].append(company)

    @classmethod
    def from_client(cls, client: DartClient) -> "CorporationDirectory":
        return cls.from_xml(client.download_corporations_xml())

    @classmethod
    def from_xml(cls, content: bytes) -> "CorporationDirectory":
        try:
            root = ElementTree.fromstring(content)
        except (ParseError, DefusedXmlException):
            raise InvalidResponseError("CORPCODE.xml이 유효하고 안전한 XML이 아닙니다.") from None
        if root.tag != "result":
            raise InvalidResponseError("CORPCODE.xml의 루트는 result여야 합니다.")
        corporations: list[Corporation] = []
        codes: set[str] = set()
        for item in root.findall("list"):
            code = (item.findtext("corp_code") or "").strip()
            name = (item.findtext("corp_name") or "").strip()
            stock = (item.findtext("stock_code") or "").strip() or None
            modified = (item.findtext("modify_date") or "").strip()
            if (
                not re.fullmatch(r"[0-9]{8}", code)
                or not name
                or (stock is not None and not re.fullmatch(r"[0-9A-Z]{6}", stock))
                or not re.fullmatch(r"[0-9]{8}", modified)
                or code in codes
            ):
                raise InvalidResponseError("CORPCODE.xml에 잘못되거나 중복된 기업 정보가 있습니다.")
            codes.add(code)
            corporations.append(Corporation(code, name, stock, modified))
        if not corporations:
            raise InvalidResponseError("CORPCODE.xml 기업 목록이 비어 있습니다.")
        return cls(corporations)

    def find(self, name_or_stock_code: str) -> Corporation:
        query = name_or_stock_code.strip()
        if not query:
            raise CompanyNotFoundError("기업명 또는 6자리 영숫자 종목코드를 입력하세요.")
        matches = self._by_stock.get(query.upper()) or self._by_name.get(normalize_name(query), [])
        if not matches:
            raise CompanyNotFoundError(f"기업을 찾을 수 없습니다: {query}")
        if len(matches) != 1:
            raise AmbiguousCompanyError(f"동일한 기업명 또는 종목코드가 여러 개입니다: {query}")
        return matches[0]
