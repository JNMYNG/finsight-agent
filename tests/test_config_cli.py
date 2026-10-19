import json

import httpx
import pytest

from finsight import cli
from finsight.config import Settings
from finsight.dart.client import DartClient
from finsight.errors import ConfigurationError

from .conftest import TEST_API_KEY, zip_xml


def test_settings_loads_explicit_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENDART_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text(f"OPENDART_API_KEY={TEST_API_KEY}\n", encoding="utf-8")
    settings = Settings.from_env(path)
    assert settings.opendart_api_key == TEST_API_KEY
    assert TEST_API_KEY not in repr(settings)


def test_environment_wins_over_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDART_API_KEY", TEST_API_KEY)
    path = tmp_path / ".env"
    path.write_text("OPENDART_API_KEY=wrong\n", encoding="utf-8")
    assert Settings.from_env(path).opendart_api_key == TEST_API_KEY


def test_existing_dart_key_name_is_supported(tmp_path, monkeypatch):
    monkeypatch.setenv("DART_API_KEY", TEST_API_KEY)
    assert Settings.from_env(tmp_path / "missing.env").opendart_api_key == TEST_API_KEY


def test_canonical_key_name_wins_when_both_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("DART_API_KEY", "d" * 40)
    monkeypatch.setenv("OPENDART_API_KEY", TEST_API_KEY)
    assert Settings.from_env(tmp_path / "missing.env").opendart_api_key == TEST_API_KEY


def test_missing_key_is_explicit_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENDART_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        Settings.from_env(tmp_path / "missing.env")


@pytest.fixture
def mocked_cli(monkeypatch, accounts_factory):
    monkeypatch.setenv("OPENDART_API_KEY", TEST_API_KEY)

    def handler(request):
        if request.url.path.endswith("corpCode.xml"):
            return httpx.Response(200, content=zip_xml())
        stock = "005930" if request.url.params["corp_code"] == "00126380" else "000660"
        return httpx.Response(200, json={"status": "000", "list": accounts_factory(stock=stock)})

    monkeypatch.setattr(
        cli, "DartClient", lambda key: DartClient(key, transport=httpx.MockTransport(handler))
    )


def test_cli_compare_json_with_decimal_strings(mocked_cli, capsys):
    assert cli.main(["compare", "--year", "2024"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["left"]["revenue"]["value"] == "1000"
    assert output["left"]["report_name"] == "사업보고서"
    assert output["left"]["basis"] == "CFS"
    assert output["operating_margin_pct"]["difference"] == "0.00"
    assert TEST_API_KEY not in json.dumps(output)


def test_cli_lookup(mocked_cli, capsys):
    assert cli.main(["lookup", "SK하이닉스"]) == 0
    assert json.loads(capsys.readouterr().out)["corp_code"] == "00164779"


def test_cli_financials(mocked_cli, capsys):
    assert cli.main(["financials", "삼성전자", "--year", "2024", "--basis", "CFS"]) == 0
    assert json.loads(capsys.readouterr().out)["operating_margin_pct"] == "10.00"


def test_cli_failure_is_nonzero_and_json(mocked_cli, capsys):
    assert cli.main(["lookup", "없는기업"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error" in json.loads(captured.err)


def test_cli_missing_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENDART_API_KEY", raising=False)
    assert cli.main(["--env-file", str(tmp_path / "missing.env"), "lookup", "삼성전자"]) == 1
    assert "OPENDART_API_KEY" in json.loads(capsys.readouterr().err)["error"]


def test_cli_invalid_year_rejected_before_network(monkeypatch):
    monkeypatch.setattr(cli, "DartClient", lambda key: pytest.fail("unexpected network setup"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["compare", "--year", "2014"])
    assert exc.value.code == 2
