from __future__ import annotations

import re
from io import StringIO
from typing import Iterable

import pandas as pd


def extract_regex_value(text: str, patterns: Iterable[str], default: str = "unknown") -> str:
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    return default


def extract_regex_float(text: str, patterns: Iterable[str], default: float | None = None) -> float | None:
    value = extract_regex_value(text, patterns, default="")
    if value == "":
        return default
    try:
        return float(value.replace("%", ""))
    except ValueError:
        return default


def extract_regex_bool(text: str, patterns: Iterable[str], default: bool | None = None) -> bool | None:
    value = extract_regex_value(text, patterns, default="")
    if value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    return default


def markdown_table_after_heading(markdown: str, heading_text: str) -> pd.DataFrame:
    lines = (markdown or "").splitlines()
    start = None
    heading_lower = heading_text.lower()
    for index, line in enumerate(lines):
        if heading_lower in line.lower():
            start = index + 1
            break
    if start is None:
        return pd.DataFrame()
    return first_markdown_table("\n".join(lines[start:]))


def first_markdown_table(markdown: str) -> pd.DataFrame:
    table_lines: list[str] = []
    collecting = False
    for line in (markdown or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            collecting = True
            table_lines.append(stripped)
        elif collecting:
            break
    return markdown_table_to_dataframe(table_lines)


def markdown_table_to_dataframe(table_lines: list[str]) -> pd.DataFrame:
    if len(table_lines) < 2:
        return pd.DataFrame()
    rows = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    if len(rows) < 2:
        return pd.DataFrame()
    width = len(rows[0])
    rows = [row for row in rows if len(row) == width]
    if len(rows) < 2:
        return pd.DataFrame()
    csv_text = "\n".join(",".join(_csv_escape(cell) for cell in row) for row in rows)
    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception:
        return pd.DataFrame()
    return coerce_numeric_columns(df)


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in output.columns:
        cleaned = output[column].astype(str).str.strip().str.rstrip("%")
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().sum() > 0:
            output[column] = numeric
    return output


def _csv_escape(value: str) -> str:
    escaped = str(value).replace('"', '""')
    return f'"{escaped}"'
