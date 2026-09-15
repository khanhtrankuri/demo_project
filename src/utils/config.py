from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "default.yaml"
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config.get("_project_root", PROJECT_ROOT)) / path).resolve()


def with_overrides(config: dict[str, Any], **sections: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(config)
    for section, values in sections.items():
        result.setdefault(section, {}).update(values)
    return result

