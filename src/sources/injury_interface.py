from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.normalize import normalize_team_name


logger = logging.getLogger(__name__)

INJURY_FEATURE_COLUMNS = [
    "home_missing_starters_count",
    "away_missing_starters_count",
    "home_missing_key_players_count",
    "away_missing_key_players_count",
    "home_injury_impact_score",
    "away_injury_impact_score",
    "home_suspension_impact_score",
    "away_suspension_impact_score",
    "home_goalkeeper_missing",
    "away_goalkeeper_missing",
    "home_captain_missing",
    "away_captain_missing",
    "home_minutes_lost_from_expected_xi",
    "away_minutes_lost_from_expected_xi",
    "missing_starters_count_diff",
    "missing_key_players_count_diff",
    "injury_impact_score_diff",
    "suspension_impact_score_diff",
    "minutes_lost_from_expected_xi_diff",
]

AGGREGATE_INJURY_COLUMNS = [
    "missing_starters_count",
    "missing_key_players_count",
    "injury_impact_score",
    "suspension_impact_score",
    "goalkeeper_missing",
    "captain_missing",
    "minutes_lost_from_expected_xi",
]

REQUIRED_AGGREGATE_INJURY_COLUMNS = [
    "missing_starters_count",
    "missing_key_players_count",
    "injury_impact_score",
    "suspension_impact_score",
    "goalkeeper_missing",
    "captain_missing",
]


def load_injuries_suspensions(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        logger.warning("Injury/suspension file not found at %s; availability features will be null", csv_path)
        return pd.DataFrame()
    df = pd.read_csv(csv_path, low_memory=False)
    aggregate_required = {"date", "team", *REQUIRED_AGGREGATE_INJURY_COLUMNS}
    if aggregate_required.issubset(df.columns):
        return _normalize_aggregate_injuries(df)
    player_required = {"date", "team", "player", "status", "expected_start_probability", "player_rating", "position", "is_key_player"}
    missing = player_required - set(df.columns)
    if missing:
        logger.warning(
            "Injury/suspension file missing aggregate columns %s and player-level columns %s; availability features will be null",
            sorted(aggregate_required - set(df.columns)),
            sorted(missing),
        )
        return pd.DataFrame()
    return _normalize_player_level_injuries(df)


def _normalize_aggregate_injuries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["team"] = df["team"].map(normalize_team_name)
    for column in AGGREGATE_INJURY_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["_injury_schema"] = "aggregate"
    return df.dropna(subset=["date", "team"]).sort_values(["team", "date"], kind="stable").reset_index(drop=True)


def _normalize_player_level_injuries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["team"] = df["team"].map(normalize_team_name)
    df["status"] = df["status"].astype(str).str.lower()
    df["expected_start_probability"] = pd.to_numeric(df["expected_start_probability"], errors="coerce").fillna(0)
    df["player_rating"] = pd.to_numeric(df["player_rating"], errors="coerce").fillna(0)
    df["is_key_player"] = df["is_key_player"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    if "is_captain" not in df.columns:
        df["is_captain"] = False
    df["is_captain"] = df["is_captain"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    if "minutes_lost_from_expected_xi" not in df.columns:
        df["minutes_lost_from_expected_xi"] = 90 * df["expected_start_probability"]
    df["minutes_lost_from_expected_xi"] = pd.to_numeric(df["minutes_lost_from_expected_xi"], errors="coerce").fillna(0)
    df["_injury_schema"] = "player_level"
    return df.dropna(subset=["date", "team", "player"]).sort_values(["team", "player", "date"], kind="stable").reset_index(drop=True)


def add_injury_features(matches: pd.DataFrame, injury_path: str | Path) -> pd.DataFrame:
    output = matches.copy()
    injuries = load_injuries_suspensions(injury_path)
    if injuries.empty:
        for column in INJURY_FEATURE_COLUMNS:
            output[column] = np.nan
        return output
    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        prediction_flags = _prediction_flags(output)
        aggregates = [
            _aggregate_for_team(injuries, team, match_date, use_latest_current=bool(is_prediction))
            for team, match_date, is_prediction in zip(output[team_col], pd.to_datetime(output["date"], errors="coerce"), prediction_flags)
        ]
        agg_df = pd.DataFrame(aggregates, index=output.index).add_prefix(f"{side}_")
        output = pd.concat([output, agg_df], axis=1)
    for column in [
        "missing_starters_count",
        "missing_key_players_count",
        "injury_impact_score",
        "suspension_impact_score",
        "minutes_lost_from_expected_xi",
    ]:
        output[f"{column}_diff"] = output[f"home_{column}"] - output[f"away_{column}"]
    return output


def _aggregate_for_team(injuries: pd.DataFrame, team: object, match_date: pd.Timestamp, *, use_latest_current: bool = False) -> dict[str, float]:
    team = normalize_team_name(team)
    if team is None or pd.isna(team):
        return _empty_aggregate()
    if "_injury_schema" in injuries.columns and injuries["_injury_schema"].eq("aggregate").all():
        return _aggregate_snapshot_for_team(injuries, team, match_date, use_latest_current=use_latest_current)
    if pd.isna(match_date):
        return _empty_aggregate()
    subset = injuries[(injuries["team"] == team) & (injuries["date"] < match_date)]
    if subset.empty:
        return _empty_aggregate()
    latest = subset.sort_values(["player", "date"], kind="stable").groupby("player", as_index=False).tail(1)
    unavailable = latest[latest["status"].isin(["injured", "suspended", "doubtful"])]
    injury = unavailable[unavailable["status"].isin(["injured", "doubtful"])]
    suspension = unavailable[unavailable["status"].eq("suspended")]
    impact = unavailable["player_rating"] * unavailable["expected_start_probability"]
    return {
        "missing_starters_count": float((unavailable["expected_start_probability"] >= 0.5).sum()),
        "missing_key_players_count": float(unavailable["is_key_player"].sum()),
        "injury_impact_score": float((injury["player_rating"] * injury["expected_start_probability"]).sum()),
        "suspension_impact_score": float((suspension["player_rating"] * suspension["expected_start_probability"]).sum()),
        "goalkeeper_missing": float(unavailable["position"].astype(str).str.lower().isin(["gk", "goalkeeper"]).any()),
        "captain_missing": float(unavailable["is_captain"].any()),
        "minutes_lost_from_expected_xi": float(unavailable["minutes_lost_from_expected_xi"].sum() if not impact.empty else 0),
    }


def _aggregate_snapshot_for_team(injuries: pd.DataFrame, team: str, match_date: pd.Timestamp, *, use_latest_current: bool) -> dict[str, float]:
    team_rows = injuries[injuries["team"].eq(team)].sort_values("date", kind="stable").reset_index(drop=True)
    if team_rows.empty:
        return _empty_aggregate()
    if use_latest_current:
        row = team_rows.iloc[-1]
    else:
        if pd.isna(match_date):
            return _empty_aggregate()
        prior = team_rows[team_rows["date"] < match_date].reset_index(drop=True)
        if prior.empty:
            return _empty_aggregate()
        row = prior.iloc[-1]
    return {
        "missing_starters_count": _row_number(row, "missing_starters_count"),
        "missing_key_players_count": _row_number(row, "missing_key_players_count"),
        "injury_impact_score": _row_number(row, "injury_impact_score"),
        "suspension_impact_score": _row_number(row, "suspension_impact_score"),
        "goalkeeper_missing": _row_number(row, "goalkeeper_missing"),
        "captain_missing": _row_number(row, "captain_missing"),
        "minutes_lost_from_expected_xi": _row_number(row, "minutes_lost_from_expected_xi"),
    }


def _prediction_flags(matches: pd.DataFrame) -> pd.Series:
    if "_advanced_split" in matches.columns:
        return matches["_advanced_split"].eq("prediction")
    if "target_result_class" in matches.columns:
        return matches["target_result_class"].isna()
    scored = matches.get("home_score", pd.Series(np.nan, index=matches.index)).notna() & matches.get("away_score", pd.Series(np.nan, index=matches.index)).notna()
    return ~scored


def _row_number(row: pd.Series, column: str) -> float:
    value = row.get(column, np.nan)
    return float(value) if pd.notna(value) else np.nan


def _empty_aggregate() -> dict[str, float]:
    return {
        "missing_starters_count": np.nan,
        "missing_key_players_count": np.nan,
        "injury_impact_score": np.nan,
        "suspension_impact_score": np.nan,
        "goalkeeper_missing": np.nan,
        "captain_missing": np.nan,
        "minutes_lost_from_expected_xi": np.nan,
    }
