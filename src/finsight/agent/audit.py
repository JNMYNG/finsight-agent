"""Ordered JSON audit events, with credentials removed before every output sink."""

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finsight.serialization import to_json_value


class AuditLog:
    def __init__(self, path: str | Path | None = None, *, secrets: tuple[str, ...] = ()) -> None:
        self.path = Path(path) if path is not None else None
        self.secrets = tuple(value for value in secrets if value)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def redact(self, value: Any) -> Any:
        value = to_json_value(value)
        if isinstance(value, dict):
            return {
                key: "[REDACTED]"
                if key.lower()
                in {"api_key", "crtfc_key", "authorization", "openai_api_key", "opendart_api_key"}
                else self.redact(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
            value = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", value)
            return re.sub(r"(?i)(crtfc_key|api_key)=([^&\s]+)", r"\1=[REDACTED]", value)
        return value

    def emit(self, run_id: str, event: str, **data: Any) -> None:
        record = self.redact(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": run_id,
                "event": event,
                **data,
            }
        )
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        logging.getLogger("finsight.agent").info(line)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
