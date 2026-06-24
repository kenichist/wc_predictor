from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.normalize import normalize_team_name


MARKET_ODDS_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_odds",
    "draw_odds",
    "away_odds",
    "bookmaker",
    "source",
    "updated_at",
]


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def read_markdown_safe(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def file_presence(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for label, path in paths.items():
        rows.append(
            {
                "file": label,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return pd.DataFrame(rows)


def validate_market_odds(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "missing_columns": MARKET_ODDS_COLUMNS,
            "duplicate_rows": 0,
            "bad_dates": 0,
            "bad_odds": 0,
            "missing_source": 0,
            "missing_updated_at": 0,
            "normalized_team_failures": 0,
        }
    missing = [column for column in MARKET_ODDS_COLUMNS if column not in df.columns]
    output = df.copy()
    for column in MARKET_ODDS_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    dates = pd.to_datetime(output["date"], errors="coerce", format="mixed")
    odds = output[["home_odds", "draw_odds", "away_odds"]].apply(pd.to_numeric, errors="coerce")
    home_norm = output["home_team"].map(normalize_team_name)
    away_norm = output["away_team"].map(normalize_team_name)
    key_frame = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "home_team": home_norm,
            "away_team": away_norm,
        }
    )
    return {
        "rows": int(len(output)),
        "missing_columns": missing,
        "duplicate_rows": int(key_frame.duplicated(["date", "home_team", "away_team"], keep=False).sum()),
        "bad_dates": int(dates.isna().sum()),
        "bad_odds": int((odds.isna() | odds.le(1)).any(axis=1).sum()),
        "missing_source": int(output["source"].fillna("").astype(str).str.strip().eq("").sum()),
        "missing_updated_at": int(output["updated_at"].fillna("").astype(str).str.strip().eq("").sum()),
        "normalized_team_failures": int(home_norm.isna().sum() + away_norm.isna().sum()),
    }


def odds_rows_by_year(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["year", "rows"])
    dates = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    return (
        pd.DataFrame({"year": dates.dt.year})
        .dropna()
        .assign(year=lambda frame: frame["year"].astype(int))
        .value_counts("year")
        .rename("rows")
        .reset_index()
        .sort_values("year", kind="stable")
    )
