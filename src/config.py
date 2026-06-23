from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "sources.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load project YAML configuration."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def config_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a path from the paths section of the project config."""
    try:
        raw_path = config["paths"][key]
    except KeyError as exc:
        raise KeyError(f"Missing configured path: paths.{key}") from exc
    return resolve_project_path(raw_path)


def ensure_configured_directories(config: dict[str, Any] | None = None) -> None:
    """Create configured data directories if they do not already exist."""
    cfg = config or load_config()
    for key in ("raw_dir", "processed_dir", "features_dir", "external_dir"):
        config_path(cfg, key).mkdir(parents=True, exist_ok=True)
