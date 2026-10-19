import pytest

from finsight.dart.corporations import CorporationDirectory
from finsight.errors import AmbiguousCompanyError, CompanyNotFoundError, InvalidResponseError

from .conftest import DIRECTORY_XML


def test_lookup_preserves_leading_zeroes_and_handles_case_and_spaces():
    directory = CorporationDirectory.from_xml(DIRECTORY_XML)
    samsung = directory.find("삼성전자")
    assert samsung.corp_code == "00126380"
    assert samsung.stock_code == "005930"
    assert directory.find("005930") == samsung
    assert directory.find(" sk 하이닉스 ").corp_code == "00164779"
    assert directory.find("테스트비상장").stock_code is None


def test_alphanumeric_stock_code_does_not_break_directory():
    # Regression: live corpCode.xml includes six-character codes containing letters.
    content = DIRECTORY_XML.replace(b"000660", b"0015S0")
    directory = CorporationDirectory.from_xml(content)
    assert directory.find("0015S0").corp_code == "00164779"
    assert directory.find("0015s0").stock_code == "0015S0"
    assert directory.find("삼성전자").corp_code == "00126380"


@pytest.mark.parametrize("query", ["삼성", "하이닉스", "", "missing", "00126380"])
def test_no_fuzzy_or_arbitrary_lookup(query):
    with pytest.raises(CompanyNotFoundError):
        CorporationDirectory.from_xml(DIRECTORY_XML).find(query)


def test_duplicate_company_name_is_ambiguous():
    content = DIRECTORY_XML.replace("SK하이닉스".encode(), "삼성전자".encode())
    directory = CorporationDirectory.from_xml(content)
    with pytest.raises(AmbiguousCompanyError):
        directory.find("삼성전자")
    assert directory.find("005930").corp_code == "00126380"


@pytest.mark.parametrize(
    "content",
    [
        b"broken",
        b"<result/>",
        b"<unexpected/>",
        DIRECTORY_XML.replace(b"00164779", b"00126380"),
        DIRECTORY_XML.replace(b"005930", b"123"),
        DIRECTORY_XML.replace(b"20250101", b"bad"),
        b'<!DOCTYPE x [<!ENTITY x "expanded">]><result>&x;</result>',
    ],
)
def test_invalid_or_unsafe_xml(content):
    with pytest.raises(InvalidResponseError):
        CorporationDirectory.from_xml(content)
