"""Small synchronous client for the two official Day 1 endpoints."""

import re
from io import BytesIO
from types import TracebackType
from typing import Any, Self
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile, ZipFile, is_zipfile

import httpx
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException

from finsight.errors import (
    ConfigurationError,
    DartAPIError,
    DartTransportError,
    InvalidResponseError,
    NoDataError,
)
from finsight.models import ReportType

BASE_URL = "https://opendart.fss.or.kr/api/"
ACCOUNTS_URL = BASE_URL + "fnlttSinglAcnt.json"
# Directory archives are parsed in memory; never extract paths to disk.
MAX_DIRECTORY_XML_BYTES = 100 * 1024 * 1024


class DartClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if len(api_key) != 40 or not api_key.isascii() or not api_key.isalnum():
            raise ConfigurationError("OpenDART 인증키는 발급받은 40자리 문자열이어야 합니다.")
        self._api_key = api_key
        self._http = httpx.Client(timeout=timeout, transport=transport, follow_redirects=False)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _get(self, endpoint: str, **params: str) -> httpx.Response:
        # httpx exceptions can contain the URL and crtfc_key. Do not propagate them.
        failure: str | None = None
        try:
            response = self._http.get(
                BASE_URL + endpoint, params={"crtfc_key": self._api_key, **params}
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            failure = f"OpenDART HTTP 오류 ({exc.response.status_code})."
        except httpx.RequestError:
            failure = "OpenDART 네트워크 요청 실패 또는 시간 초과."
        if failure:
            # Raise outside the except block to avoid retaining the sensitive exception context.
            raise DartTransportError(failure)
        return response

    def _check_status(self, payload: Any) -> None:
        if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
            raise InvalidResponseError("OpenDART 응답에 유효한 status가 없습니다.")
        status = payload["status"]
        if status == "000":
            return
        message = str(payload.get("message", "설명 없는 API 오류"))
        message = message.replace(self._api_key, "[REDACTED]")
        error_type = NoDataError if status == "013" else DartAPIError
        raise error_type(status, message)

    def download_corporations_xml(self) -> bytes:
        response = self._get("corpCode.xml")
        content = response.content
        if not is_zipfile(BytesIO(content)):
            # Error responses may be XML (or JSON), despite the ZIP success format.
            if content.lstrip().startswith(b"{"):
                try:
                    payload = response.json()
                except ValueError:
                    raise InvalidResponseError(
                        "기업 고유번호 오류 응답의 JSON이 손상되었습니다."
                    ) from None
                self._check_status(payload)
            else:
                try:
                    root = ElementTree.fromstring(content)
                except (ParseError, DefusedXmlException):
                    raise InvalidResponseError(
                        "기업 고유번호 응답이 ZIP 또는 안전한 XML이 아닙니다."
                    ) from None
                self._check_status(
                    {"status": root.findtext("status"), "message": root.findtext("message")}
                )
            raise InvalidResponseError("기업 고유번호 성공 응답에 ZIP 파일이 없습니다.")
        try:
            with ZipFile(BytesIO(content)) as archive:
                members = [
                    item for item in archive.infolist() if item.filename.upper() == "CORPCODE.XML"
                ]
                if len(members) != 1 or members[0].file_size > MAX_DIRECTORY_XML_BYTES:
                    raise InvalidResponseError(
                        "ZIP에 유일하고 크기가 적절한 CORPCODE.xml이 없습니다."
                    )
                return archive.read(members[0])
        except (BadZipFile, RuntimeError, NotImplementedError):
            raise InvalidResponseError("기업 고유번호 ZIP 파일을 읽을 수 없습니다.") from None

    def get_major_accounts(
        self, corp_code: str, business_year: int, report_type: ReportType = ReportType.ANNUAL
    ) -> list[dict[str, Any]]:
        if not re.fullmatch(r"[0-9]{8}", corp_code):
            raise ValueError("corp_code는 8자리 숫자 문자열이어야 합니다.")
        if type(business_year) is not int or not 2015 <= business_year <= 9999:
            raise ValueError("business_year는 2015년 이후 4자리 정수여야 합니다.")
        report_type = ReportType(report_type)
        response = self._get(
            "fnlttSinglAcnt.json",
            corp_code=corp_code,
            bsns_year=str(business_year),
            reprt_code=report_type.value,
        )
        try:
            payload = response.json()
        except ValueError:
            raise InvalidResponseError("주요계정 응답이 유효한 JSON이 아닙니다.") from None
        self._check_status(payload)
        rows = payload.get("list")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise InvalidResponseError("주요계정 응답에 유효한 list가 없습니다.")
        if not rows:
            raise NoDataError("013", "주요계정 목록이 비어 있습니다.")
        return rows
