from __future__ import annotations

import logging

import pandas as pd

from src.normalize import normalize_key


logger = logging.getLogger(__name__)

SOURCE_PRIORITY = {
    "football-data": 1,
    "football_data": 1,
    "worldcup_json": 2,
    "openfootball_worldcup_json": 2,
    "historical_results": 3,
}


def source_priority(source: object) -> int:
    if source is None or pd.isna(source):
        return 99
    return SOURCE_PRIORITY.get(str(source), 99)


def _score_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def build_duplicate_group_keys(df: pd.DataFrame) -> pd.Series:
    return df.apply(_duplicate_group_key, axis=1)


def _duplicate_group_key(row: pd.Series) -> str:
    date = pd.to_datetime(row.get("date"), errors="coerce")
    date_key = "" if pd.isna(date) else date.strftime("%Y-%m-%d")
    raw_competition = _safe_text(row.get("competition_type"))
    competition_key = normalize_key(raw_competition)
    canonical_competition = raw_competition.strip().lower()
    tournament_key = normalize_key(_safe_text(row.get("tournament")))
    event_key = competition_key or tournament_key
    is_world_cup = canonical_competition == "world_cup" or competition_key == "world cup"
    if is_world_cup:
        event_key = "world_cup"

    home = normalize_key(_safe_text(row.get("home_team")))
    away = normalize_key(_safe_text(row.get("away_team")))
    home_score = _score_key(row.get("home_score"))
    away_score = _score_key(row.get("away_score"))
    neutral = _is_true(row.get("neutral")) or is_world_cup

    team_score_pairs = [(home, home_score), (away, away_score)]
    if neutral:
        team_score_pairs = sorted(team_score_pairs, key=lambda item: item[0])
    teams_key = "|".join(f"{team}:{score}" for team, score in team_score_pairs)
    return f"{date_key}|{teams_key}|{event_key}"


def _is_true(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def deduplicate_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate source rows using configured source priority."""
    if df.empty:
        output = df.copy()
        output["source_priority"] = pd.Series(dtype="int64")
        output["duplicate_group_id"] = pd.Series(dtype="object")
        return output

    output = df.copy()
    output["source_priority"] = output.get("source", pd.Series(index=output.index)).map(source_priority)
    keys = build_duplicate_group_keys(output)
    codes = pd.factorize(keys, sort=False)[0] + 1
    output["duplicate_group_id"] = [f"dup_{code:08d}" for code in codes]

    sort_columns = ["source_priority", "date"]
    if "last_updated" in output.columns:
        output["_last_updated_sort"] = pd.to_datetime(output["last_updated"], errors="coerce")
        sort_columns = ["source_priority", "_last_updated_sort", "date"]
    output["_original_order"] = range(len(output))
    output = output.sort_values(sort_columns + ["_original_order"], ascending=[True] * len(sort_columns) + [True])
    before = len(output)
    output = output.drop_duplicates(subset=["duplicate_group_id"], keep="first")
    removed = before - len(output)
    output = output.sort_values(["date", "_original_order"], kind="stable").drop(
        columns=["_original_order", "_last_updated_sort"], errors="ignore"
    )
    logger.info("Removed %s duplicate rows from %s input matches", removed, before)
    return output.reset_index(drop=True)
