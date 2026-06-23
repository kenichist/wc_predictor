from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.normalize import normalize_team_name


logger = logging.getLogger(__name__)

PLAYER_BASE_COLUMNS = [
    "squad_market_value",
    "starting_xi_market_value",
    "top_5_player_rating_avg",
    "top_11_player_rating_avg",
    "goalkeeper_strength",
    "defense_strength",
    "midfield_strength",
    "attack_strength",
    "bench_strength",
    "avg_age",
    "total_caps",
    "international_goals",
    "club_minutes_last_season",
    "club_minutes_recent_90_days",
]

PLAYER_FEATURE_COLUMNS = [f"{side}_{column}" for side in ("home", "away") for column in PLAYER_BASE_COLUMNS] + [
    "squad_market_value_diff",
    "starting_xi_market_value_diff",
    "top_11_player_rating_diff",
    "goalkeeper_strength_diff",
    "defense_strength_diff",
    "midfield_strength_diff",
    "attack_strength_diff",
    "bench_strength_diff",
    "avg_age_diff",
    "total_caps_diff",
    "club_minutes_recent_90_days_diff",
]


def load_squad_player_features(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        logger.warning("Squad/player feature file not found at %s; squad features will be null", csv_path)
        return pd.DataFrame()
    df = pd.read_csv(csv_path, low_memory=False)
    if "team" not in df.columns:
        logger.warning("Squad/player feature file missing required column 'team'; squad features will be null")
        return pd.DataFrame()
    available_features = [column for column in PLAYER_BASE_COLUMNS if column in df.columns]
    if not available_features:
        logger.warning("Squad/player feature file has no recognized numeric squad feature columns; squad features will be null")
        return pd.DataFrame()
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT
    df["team"] = df["team"].map(normalize_team_name)
    for column in PLAYER_BASE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["team"]).sort_values(["team", "date"], na_position="last", kind="stable").reset_index(drop=True)


def add_squad_player_features(matches: pd.DataFrame, features_path: str | Path) -> pd.DataFrame:
    output = matches.copy()
    features = load_squad_player_features(features_path)
    if features.empty:
        for column in PLAYER_FEATURE_COLUMNS:
            output[column] = np.nan
        return output
    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        for column in PLAYER_BASE_COLUMNS:
            output[f"{side}_{column}"] = _latest_team_value(output, features, team_col, column)
    for column in [
        "squad_market_value",
        "starting_xi_market_value",
        "top_11_player_rating",
        "goalkeeper_strength",
        "defense_strength",
        "midfield_strength",
        "attack_strength",
        "bench_strength",
        "avg_age",
        "total_caps",
        "club_minutes_recent_90_days",
    ]:
        left = "top_11_player_rating_avg" if column == "top_11_player_rating" else column
        output[f"{column}_diff"] = output[f"home_{left}"] - output[f"away_{left}"]
    return output


def _latest_team_value(matches: pd.DataFrame, history: pd.DataFrame, team_column: str, value_column: str) -> pd.Series:
    lookup = {team: group.reset_index(drop=True) for team, group in history.groupby("team")}
    dates = pd.to_datetime(matches["date"], errors="coerce")
    values = []
    for team, match_date in zip(matches[team_column], dates):
        team = normalize_team_name(team)
        team_history = lookup.get(team)
        if team_history is None:
            values.append(np.nan)
            continue
        dated = team_history[team_history["date"].notna()].reset_index(drop=True)
        if dated.empty or pd.isna(match_date):
            values.append(team_history.iloc[-1][value_column])
            continue
        idx = dated["date"].searchsorted(match_date, side="right") - 1
        values.append(dated.iloc[idx][value_column] if idx >= 0 else team_history.iloc[-1][value_column])
    return pd.Series(values, index=matches.index, dtype="float64")
