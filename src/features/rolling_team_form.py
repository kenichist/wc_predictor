from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


ROLLING_FEATURE_COLUMNS = [
    "home_matches_last_5",
    "away_matches_last_5",
    "home_matches_last_10",
    "away_matches_last_10",
    "home_win_rate_last_5",
    "away_win_rate_last_5",
    "home_win_rate_last_10",
    "away_win_rate_last_10",
    "home_goals_for_avg_last_5",
    "away_goals_for_avg_last_5",
    "home_goals_against_avg_last_5",
    "away_goals_against_avg_last_5",
    "home_goal_diff_avg_last_5",
    "away_goal_diff_avg_last_5",
    "home_goals_for_avg_last_10",
    "away_goals_for_avg_last_10",
    "home_goals_against_avg_last_10",
    "away_goals_against_avg_last_10",
    "home_goal_diff_avg_last_10",
    "away_goal_diff_avg_last_10",
    "home_days_since_last_match",
    "away_days_since_last_match",
    "rest_days_diff",
]


def compute_rolling_team_form(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling form using only matches before each match date."""
    if matches.empty:
        return pd.DataFrame(columns=["match_id", *ROLLING_FEATURE_COLUMNS], index=matches.index)

    df = matches.copy()
    df["row_id"] = np.arange(len(df))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "row_id"], kind="stable")
    histories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    feature_rows: list[dict[str, Any]] = []

    for match_date, group in df.groupby("date", sort=True, dropna=False):
        pending_updates: list[tuple[str, dict[str, Any]]] = []
        for row in group.itertuples(index=False):
            row_dict = row._asdict()
            home_team = row_dict.get("home_team")
            away_team = row_dict.get("away_team")
            home_features = _team_features(histories.get(home_team, []), match_date, "home")
            away_features = _team_features(histories.get(away_team, []), match_date, "away")
            rest_diff = np.nan
            if pd.notna(home_features["home_days_since_last_match"]) and pd.notna(away_features["away_days_since_last_match"]):
                rest_diff = home_features["home_days_since_last_match"] - away_features["away_days_since_last_match"]
            feature_rows.append(
                {
                    "row_id": row_dict["row_id"],
                    "match_id": row_dict.get("match_id"),
                    **home_features,
                    **away_features,
                    "rest_days_diff": rest_diff,
                }
            )

            home_score = row_dict.get("home_score")
            away_score = row_dict.get("away_score")
            if _has_score(home_score, away_score) and home_team and away_team:
                home_score_float = float(home_score)
                away_score_float = float(away_score)
                pending_updates.append(
                    (
                        home_team,
                        _perspective_record(match_date, away_team, home_score_float, away_score_float, row_dict),
                    )
                )
                pending_updates.append(
                    (
                        away_team,
                        _perspective_record(match_date, home_team, away_score_float, home_score_float, row_dict),
                    )
                )
        for team, record in pending_updates:
            histories[team].append(record)

    features = pd.DataFrame(feature_rows).sort_values("row_id", kind="stable").drop(columns=["row_id"])
    features.index = matches.index
    return features


def _team_features(history: list[dict[str, Any]], match_date: pd.Timestamp, prefix: str) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for window in (5, 10):
        recent = history[-window:]
        features[f"{prefix}_matches_last_{window}"] = len(recent)
        features[f"{prefix}_win_rate_last_{window}"] = _avg(recent, "win")
        features[f"{prefix}_goals_for_avg_last_{window}"] = _avg(recent, "goals_for")
        features[f"{prefix}_goals_against_avg_last_{window}"] = _avg(recent, "goals_against")
        features[f"{prefix}_goal_diff_avg_last_{window}"] = _avg(recent, "goal_diff")
    if history and pd.notna(match_date):
        features[f"{prefix}_days_since_last_match"] = int((match_date - history[-1]["date"]).days)
    else:
        features[f"{prefix}_days_since_last_match"] = np.nan
    return features


def _avg(history: list[dict[str, Any]], key: str) -> float:
    if not history:
        return np.nan
    return float(np.mean([record[key] for record in history]))


def _perspective_record(
    match_date: pd.Timestamp,
    opponent: str,
    goals_for: float,
    goals_against: float,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "date": match_date,
        "opponent": opponent,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_diff": goals_for - goals_against,
        "win": float(goals_for > goals_against),
        "draw": float(goals_for == goals_against),
        "loss": float(goals_for < goals_against),
        "neutral": row.get("neutral"),
        "tournament": row.get("tournament"),
    }


def _has_score(home_score: Any, away_score: Any) -> bool:
    return pd.notna(home_score) and pd.notna(away_score)
