"""JSON representation shared by Day 1 CLI and agent tool observations."""

from dataclasses import is_dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from finsight.models import FinancialSummary


def to_json_value(value: Any) -> Any:
    if isinstance(value, FinancialSummary):
        result = {name: to_json_value(item) for name, item in vars(value).items()}
        result["report_name"] = value.report_type.label
        return result
    if is_dataclass(value) and not isinstance(value, type):
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
