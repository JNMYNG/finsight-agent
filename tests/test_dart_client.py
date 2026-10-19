import traceback

import httpx
import pytest

from finsight.dart.client import DartClient
from finsight.errors import (
    ConfigurationError,
    DartAPIError,
    DartTransportError,
    InvalidResponseError,
    NoDataError,
)
from finsight.models import ReportType

from .conftest import DIRECTORY_XML, TEST_API_KEY, zip_xml


def test_official_parameters_and_endpoints(accounts_factory):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("corpCode.xml"):
            assert dict(request.url.params) == {"crtfc_key": TEST_API_KEY}
            return httpx.Response(200, content=zip_xml())
        assert request.url.path == "/api/fnlttSinglAcnt.json"
        assert dict(request.url.params) == {
            "crtfc_key": TEST_API_KEY,
            "corp_code": "00126380",
            "bsns_year": "2024",
            "reprt_code": "11011",
        }  # No fs_div request parameter on the major-accounts endpoint.
        return httpx.Response(200, json={"status": "000", "list": accounts_factory()})

    with DartClient(TEST_API_KEY, transport=httpx.MockTransport(handler)) as client:
        assert client.download_corporations_xml() == DIRECTORY_XML
        assert len(client.get_major_accounts("00126380", 2024)) == 3
    assert len(requests) == 2


@pytest.mark.parametrize("status", ["010", "011", "012", "013", "014", "020", "100", "800", "900"])
def test_json_api_errors(status):
    response = {"status": status, "message": "API 오류"}
    with DartClient(
        TEST_API_KEY, transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as client:
        with pytest.raises(NoDataError if status == "013" else DartAPIError) as exc:
            client.get_major_accounts("00126380", 2024)
        assert exc.value.status == status


@pytest.mark.parametrize(
    "content",
    [
        b"<result><status>010</status><message>invalid key</message></result>",
        b'{"status":"010","message":"invalid key"}',
    ],
)
def test_directory_error_is_not_misreported_as_bad_zip(content):
    with DartClient(
        TEST_API_KEY, transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))
    ) as client:
        with pytest.raises(DartAPIError) as exc:
            client.download_corporations_xml()
        assert exc.value.status == "010"


@pytest.mark.parametrize(
    "payload",
    [[], {}, {"status": "000"}, {"status": "000", "list": {}}, {"status": "000", "list": [1]}],
)
def test_bad_json_schema(payload):
    with DartClient(
        TEST_API_KEY, transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as client:
        with pytest.raises(InvalidResponseError):
            client.get_major_accounts("00126380", 2024)


def test_empty_success_list_is_no_data():
    with DartClient(
        TEST_API_KEY,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"status": "000", "list": []})
        ),
    ) as client:
        with pytest.raises(NoDataError):
            client.get_major_accounts("00126380", 2024)


def test_non_json_accounts_response():
    with DartClient(
        TEST_API_KEY,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, text="<html>maintenance</html>")
        ),
    ) as client:
        with pytest.raises(InvalidResponseError):
            client.get_major_accounts("00126380", 2024)


@pytest.mark.parametrize(
    "content",
    [
        b"not a zip",
        b"<result/>",
        zip_xml(name="other.xml"),
        zip_xml(name="../../CORPCODE.xml"),
        b'{"status":"000"}',
        b'{"status":',
    ],
)
def test_bad_directory_response(content):
    with DartClient(
        TEST_API_KEY, transport=httpx.MockTransport(lambda _: httpx.Response(200, content=content))
    ) as client:
        with pytest.raises(InvalidResponseError):
            client.download_corporations_xml()


@pytest.mark.parametrize("status", [302, 429, 500])
def test_http_errors_redact_key_and_do_not_follow_redirects(status):
    with DartClient(
        TEST_API_KEY,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(status, headers={"location": "https://example.org/"})
        ),
    ) as client:
        with pytest.raises(DartTransportError) as exc:
            client.get_major_accounts("00126380", 2024)
        assert TEST_API_KEY not in "".join(traceback.format_exception(exc.value))
        assert exc.value.__context__ is None


@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ConnectError])
def test_network_errors_redact_key(error_type):
    def handler(request):
        raise error_type(str(request.url), request=request)

    with DartClient(TEST_API_KEY, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DartTransportError) as exc:
            client.get_major_accounts("00126380", 2024)
        assert TEST_API_KEY not in "".join(traceback.format_exception(exc.value))


def test_api_message_redacts_key():
    with DartClient(
        TEST_API_KEY,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"status": "010", "message": TEST_API_KEY})
        ),
    ) as client:
        with pytest.raises(DartAPIError) as exc:
            client.get_major_accounts("00126380", 2024)
        assert TEST_API_KEY not in str(exc.value)


@pytest.mark.parametrize(
    "code,year,report",
    [
        ("123", 2024, "11011"),
        ("１２３４５６７８", 2024, "11011"),
        ("00126380", 2014, "11011"),
        ("00126380", True, "11011"),
        ("00126380", 2024, "bad"),
    ],
)
def test_invalid_input_never_requests_network(code, year, report):
    def unexpected_request(_):
        pytest.fail("Invalid input must not cause an API call")

    with DartClient(TEST_API_KEY, transport=httpx.MockTransport(unexpected_request)) as client:
        with pytest.raises(ValueError):
            client.get_major_accounts(code, year, report)


@pytest.mark.parametrize("key", ["", "short", "!" * 40, "한" * 40])
def test_invalid_api_key(key):
    with pytest.raises(ConfigurationError):
        DartClient(key)


@pytest.mark.parametrize("report", list(ReportType))
def test_all_report_codes_are_sent(report, accounts_factory):
    def handler(request):
        assert request.url.params["reprt_code"] == report.value
        return httpx.Response(200, json={"status": "000", "list": accounts_factory(report=report)})

    with DartClient(TEST_API_KEY, transport=httpx.MockTransport(handler)) as client:
        assert client.get_major_accounts("00126380", 2024, report)
