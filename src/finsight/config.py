"""Load credentials explicitly, never as a side effect of importing a module."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from finsight.errors import ConfigurationError


@dataclass(frozen=True)
class Settings:
    opendart_api_key: str = field(repr=False)

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "Settings":
        load_dotenv(env_file, override=False)
        key = (os.environ.get("OPENDART_API_KEY") or os.environ.get("DART_API_KEY", "")).strip()
        if len(key) != 40 or not key.isascii() or not key.isalnum():
            raise ConfigurationError("OPENDART_API_KEY에 발급받은 40자리 인증키를 설정하세요.")
        return cls(opendart_api_key=key)
