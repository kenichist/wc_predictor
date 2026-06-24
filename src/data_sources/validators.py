from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.data_sources.injury_transform import INJURY_STAGING_SCHEMA
from src.data_sources.odds_transform import MARKET_ODDS_STAGING_SCHEMA
from src.data_sources.team_normalization import normalize_api_team_name


@dataclass(frozen=True)
class ValidationSummary:
    total_rows: int
    missing_columns: list[str]
    bad_dates: int
    bad_odds: int
    duplicate_rows: int
    unmatched_teams: list[str]
    safe_to_merge: bool


def validate_market_odds_schema(df: pd.DataFrame) -> list[str]:
    return [column for column in MARKET_ODDS_STAGING_SCHEMA if column not in df.columns]


def validate_market_odds_values(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {"bad_dates": 0, "bad_odds": 0}
    dates = pd.to_datetime(df.get("date"), errors="coerce", format="mixed")
    odds = df[[column for column in ("home_odds", "draw_odds", "away_odds") if column in df.columns]].apply(pd.to_numeric, errors="coerce")
    bad_odds = int((odds.isna() | odds.le(1)).any(axis=1).sum()) if not odds.empty else len(df)
    return {"bad_dates": int(dates.isna().sum()), "bad_odds": bad_odds}


def validate_no_duplicate_match_rows(df: pd.DataFrame) -> int:
    required = {"date", "home_team", "away_team"}
    if df.empty or not required.issubset(df.columns):
        return 0
    keys = _normalized_key_frame(df)
    return int(keys.duplicated(["date", "home_team", "away_team"], keep=False).sum())


def validate_injury_schema(df: pd.DataFrame) -> list[str]:
    return [column for column in INJURY_STAGING_SCHEMA if column not in df.columns]


def validate_team_normalization(df: pd.DataFrame, known_teams: set[str] | None = None) -> list[str]:
    if df.empty:
        return []
    raw_values: list[Any] = []
    for column in ("home_team", "away_team", "team"):
        if column in df.columns:
            raw_values.extend(df[column].dropna().tolist())
    unmatched: set[str] = set()
    for value in raw_values:
        normalized = normalize_api_team_name(value)
        if normalized is None:
            unmatched.add(str(value))
        elif known_teams is not None and normalized not in known_teams:
            unmatched.add(str(value))
    return sorted(unmatched)


def validate_coverage_against_active_fixtures(staged_odds: pd.DataFrame, active_fixtures: pd.DataFrame) -> dict[str, Any]:
    if active_fixtures.empty:
        return {"total_fixtures": 0, "matched_fixtures": 0, "coverage": 0.0, "missing_fixtures": pd.DataFrame()}
    required = {"date", "home_team", "away_team"}
    if not required.issubset(staged_odds.columns) or not required.issubset(active_fixtures.columns):
        missing = active_fixtures.copy()
        return {"total_fixtures": len(active_fixtures), "matched_fixtures": 0, "coverage": 0.0, "missing_fixtures": missing}
    odds_keys = _normalized_key_frame(staged_odds).drop_duplicates()
    fixture_keys = _normalized_key_frame(active_fixtures).drop_duplicates()
    merged = fixture_keys.merge(odds_keys.assign(_matched=True), on=["date", "home_team", "away_team"], how="left")
    matched_mask = merged["_matched"].fillna(False).astype(bool)
    matched = int(matched_mask.sum())
    missing_keys = merged[~matched_mask][["date", "home_team", "away_team"]]
    missing = active_fixtures.assign(
        date=pd.to_datetime(active_fixtures["date"], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d"),
        home_team=active_fixtures["home_team"].map(normalize_api_team_name),
        away_team=active_fixtures["away_team"].map(normalize_api_team_name),
    ).merge(missing_keys, on=["date", "home_team", "away_team"], how="inner")
    total = len(fixture_keys)
    return {
        "total_fixtures": int(total),
        "matched_fixtures": matched,
        "coverage": float(matched / total) if total else 0.0,
        "missing_fixtures": missing.reset_index(drop=True),
    }


def summarize_market_odds_validation(df: pd.DataFrame, *, known_teams: set[str] | None = None) -> ValidationSummary:
    missing_columns = validate_market_odds_schema(df)
    value_summary = validate_market_odds_values(df)
    duplicates = validate_no_duplicate_match_rows(df)
    unmatched = validate_team_normalization(df, known_teams=known_teams)
    safe_to_merge = (
        len(df) > 0
        and not missing_columns
        and value_summary["bad_dates"] == 0
        and value_summary["bad_odds"] == 0
        and duplicates == 0
    )
    return ValidationSummary(
        total_rows=int(len(df)),
        missing_columns=missing_columns,
        bad_dates=value_summary["bad_dates"],
        bad_odds=value_summary["bad_odds"],
        duplicate_rows=duplicates,
        unmatched_teams=unmatched,
        safe_to_merge=safe_to_merge,
    )


def rows_by_year(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["year", "rows"])
    years = pd.to_datetime(df["date"], errors="coerce", format="mixed").dt.year
    return (
        pd.DataFrame({"year": years})
        .dropna()
        .assign(year=lambda frame: frame["year"].astype(int))
        .value_counts("year")
        .rename("rows")
        .reset_index()
        .sort_values("year", kind="stable")
    )


def _normalized_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d"),
            "home_team": df["home_team"].map(normalize_api_team_name),
            "away_team": df["away_team"].map(normalize_api_team_name),
        }
    )
