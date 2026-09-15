from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


PLACEHOLDER_VALUES = {"", "undefined", "unknown", "null", "none"}


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def is_meaningful(value: Any) -> bool:
    return clean_text(value).lower() not in PLACEHOLDER_VALUES


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = clean_text(value).lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def atomic_json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default

