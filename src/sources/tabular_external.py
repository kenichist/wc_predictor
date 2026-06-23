from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_external_table(path: str | Path, *, allow_html: bool = False) -> pd.DataFrame:
    """Read a manually supplied external data table from common tabular formats."""
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(resolved)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(resolved)
    if suffix == ".json":
        return _read_json_table(resolved)
    if suffix in {".html", ".htm"} and allow_html:
        tables = pd.read_html(resolved)
        if not tables:
            raise ValueError(f"No HTML tables found in {resolved}")
        return tables[0]
    if suffix in {".html", ".htm"}:
        raise ValueError("HTML input is disabled unless explicitly configured")
    raise ValueError(f"Unsupported external data format: {resolved.suffix}")


def _read_json_table(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        for key in ("data", "rows", "rankings", "ratings"):
            value = payload.get(key)
            if isinstance(value, list):
                return pd.DataFrame(value)
        if all(isinstance(value, list) for value in payload.values()):
            return pd.DataFrame(payload)
        return pd.DataFrame([payload])
    raise ValueError(f"JSON input must contain a list or dictionary: {path}")
