from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_sources.config import (
    MARKET_ODDS_API_MERGE_REPORT,
    MARKET_ODDS_BACKUP_PATH,
    MARKET_ODDS_EXTERNAL_PATH,
    ensure_api_directories,
    write_markdown_report,
)
from src.data_sources.odds_transform import MARKET_ODDS_STAGING_SCHEMA, normalize_market_odds_rows
from src.data_sources.team_normalization import normalize_api_team_name
from src.data_sources.validators import summarize_market_odds_validation, validate_market_odds_values, validate_no_duplicate_match_rows


API_PROVIDER_NAMES = {"api_football", "the_odds_api", "sportmonks"}


def merge_staged_market_odds(
    input_path: str | Path,
    *,
    market_odds_path: str | Path = MARKET_ODDS_EXTERNAL_PATH,
    backup_path: str | Path = MARKET_ODDS_BACKUP_PATH,
    prefer_api: bool = False,
) -> dict[str, Any]:
    ensure_api_directories()
    input_path = Path(input_path)
    market_odds_path = Path(market_odds_path)
    backup_path = Path(backup_path)
    staged = _read_csv(input_path)
    staged = normalize_market_odds_rows(staged.to_dict("records")) if not staged.empty else pd.DataFrame(columns=MARKET_ODDS_STAGING_SCHEMA)
    validation = summarize_market_odds_validation(staged)
    existing = _read_csv(market_odds_path)
    rows_before = len(existing)
    staged_rows = len(staged)
    if not validation.safe_to_merge:
        summary = _summary(
            rows_before=rows_before,
            staged_rows=staged_rows,
            rows_added=0,
            rows_replaced=0,
            rows_after=rows_before,
            duplicates=validation.duplicate_rows,
            bad_odds=validation.bad_odds,
            validation_passed=False,
        )
        _write_merge_report(input_path, summary, ["Validation failed; production file was not modified."])
        return summary

    if market_odds_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(market_odds_path, backup_path)

    existing = _ensure_columns(existing, MARKET_ODDS_STAGING_SCHEMA)
    staged = _ensure_columns(staged, MARKET_ODDS_STAGING_SCHEMA)
    existing["_merge_key"] = _merge_keys(existing)
    staged["_merge_key"] = _merge_keys(staged)
    existing = existing.drop_duplicates("_merge_key", keep="last")
    staged = staged.drop_duplicates("_merge_key", keep="last")

    existing_by_key = {row["_merge_key"]: row.drop(labels=["_merge_key"]).to_dict() for _, row in existing.iterrows()}
    rows_added = 0
    rows_replaced = 0
    notes: list[str] = []
    for _, row in staged.iterrows():
        key = row["_merge_key"]
        staged_record = row.drop(labels=["_merge_key"]).to_dict()
        current = existing_by_key.get(key)
        if current is None:
            existing_by_key[key] = staged_record
            rows_added += 1
            continue
        if _should_replace(current, prefer_api=prefer_api):
            existing_by_key[key] = staged_record
            rows_replaced += 1
        else:
            notes.append(f"Kept curated existing row for {key}; staged API row was not merged.")

    merged = pd.DataFrame(list(existing_by_key.values()))
    for column in ("date",):
        if column in merged.columns:
            merged[column] = pd.to_datetime(merged[column], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    preferred_columns = [column for column in MARKET_ODDS_STAGING_SCHEMA if column in merged.columns]
    remaining_columns = [column for column in merged.columns if column not in preferred_columns]
    merged = merged[preferred_columns + remaining_columns]
    merged = merged.sort_values(["date", "home_team", "away_team"], kind="stable")
    market_odds_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(market_odds_path, index=False)

    value_summary = validate_market_odds_values(merged)
    duplicates = validate_no_duplicate_match_rows(merged)
    summary = _summary(
        rows_before=rows_before,
        staged_rows=staged_rows,
        rows_added=rows_added,
        rows_replaced=rows_replaced,
        rows_after=len(merged),
        duplicates=duplicates,
        bad_odds=value_summary["bad_odds"],
        validation_passed=True,
    )
    _write_merge_report(input_path, summary, notes)
    return summary


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    return output


def _merge_keys(df: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(df["date"], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    home = df["home_team"].map(normalize_api_team_name)
    away = df["away_team"].map(normalize_api_team_name)
    return dates.fillna("") + "|" + home.fillna("") + "|" + away.fillna("")


def _should_replace(existing_record: dict[str, Any], *, prefer_api: bool) -> bool:
    if prefer_api:
        return True
    if not _record_has_valid_odds(existing_record):
        return True
    return _is_api_record(existing_record)


def _record_has_valid_odds(record: dict[str, Any]) -> bool:
    try:
        return all(float(record[column]) > 1 for column in ("home_odds", "draw_odds", "away_odds"))
    except (TypeError, ValueError, KeyError):
        return False


def _is_api_record(record: dict[str, Any]) -> bool:
    provider = str(record.get("api_provider") or "").strip().lower()
    source = str(record.get("source") or "").strip().lower()
    return provider in API_PROVIDER_NAMES or any(name in source for name in API_PROVIDER_NAMES)


def _summary(
    *,
    rows_before: int,
    staged_rows: int,
    rows_added: int,
    rows_replaced: int,
    rows_after: int,
    duplicates: int,
    bad_odds: int,
    validation_passed: bool,
) -> dict[str, Any]:
    return {
        "MARKET_ODDS_ROWS_BEFORE": rows_before,
        "STAGED_ROWS": staged_rows,
        "ROWS_ADDED": rows_added,
        "ROWS_REPLACED": rows_replaced,
        "ROWS_AFTER": rows_after,
        "DUPLICATES": duplicates,
        "BAD_ODDS": bad_odds,
        "VALIDATION_PASSED": validation_passed,
    }


def _write_merge_report(input_path: Path, summary: dict[str, Any], notes: list[str]) -> Path:
    lines = [
        "# Market Odds API Merge Report",
        "",
        f"- Staged input: `{input_path}`",
        f"- Backup path: `{MARKET_ODDS_BACKUP_PATH}`",
        "",
        "## Summary",
        "",
    ]
    lines.extend(f"- {key}={value}" for key, value in summary.items())
    if notes:
        lines.extend(["", "## Notes", "", *[f"- {note}" for note in notes[:100]]])
    return write_markdown_report(MARKET_ODDS_API_MERGE_REPORT, lines)

