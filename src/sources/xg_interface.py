from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.normalize import normalize_team_name


logger = logging.getLogger(__name__)

XG_FEATURE_COLUMNS = [
    "home_xg_for_avg_last_5",
    "away_xg_for_avg_last_5",
    "home_xg_against_avg_last_5",
    "away_xg_against_avg_last_5",
    "home_xg_diff_last_5",
    "away_xg_diff_last_5",
    "home_xg_for_avg_last_10",
    "away_xg_for_avg_last_10",
    "home_xg_against_avg_last_10",
    "away_xg_against_avg_last_10",
    "home_xg_diff_last_10",
    "away_xg_diff_last_10",
    "home_shots_for_avg_last_5",
    "away_shots_for_avg_last_5",
    "home_shots_against_avg_last_5",
    "away_shots_against_avg_last_5",
]


def load_team_xg_stats(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        logger.warning("Team xG stats file not found at %s; xG features will be null", csv_path)
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    required = {"date", "team", "opponent", "xg_for", "xg_against"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("Team xG stats file missing columns %s; xG features will be null", sorted(missing))
        return pd.DataFrame()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["team"] = df["team"].map(normalize_team_name)
    df["opponent"] = df["opponent"].map(normalize_team_name)
    for column in [
        "xg_for",
        "xg_against",
        "shots_for",
        "shots_against",
        "big_chances_for",
        "big_chances_against",
        "set_piece_xg_for",
        "set_piece_xg_against",
    ]:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["date", "team"]).sort_values(["date", "team"], kind="stable").reset_index(drop=True)


def add_xg_rolling_features(matches: pd.DataFrame, xg_path: str | Path) -> pd.DataFrame:
    output = matches.copy()
    xg = load_team_xg_stats(xg_path)
    if xg.empty:
        for column in XG_FEATURE_COLUMNS:
            output[column] = np.nan
        return output

    xg = xg.sort_values(["date", "team"], kind="stable")
    by_team_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in xg.itertuples(index=False):
        by_team_rows[row.team].append(row._asdict())
    output = output.sort_values(["date", "match_id"], kind="stable").copy()
    dates = pd.to_datetime(output["date"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for idx, match in output.iterrows():
        match_date = dates.loc[idx]
        row = {"_idx": idx}
        row.update(_features_for_team(by_team_rows.get(match["home_team"], []), match_date, "home"))
        row.update(_features_for_team(by_team_rows.get(match["away_team"], []), match_date, "away"))
        rows.append(row)
    features = pd.DataFrame(rows).set_index("_idx")
    output = output.join(features, how="left")
    return output.sort_index()


def _features_for_team(history: list[dict[str, Any]], match_date: pd.Timestamp, prefix: str) -> dict[str, float]:
    prior = [row for row in history if pd.notna(match_date) and row["date"] < match_date]
    features: dict[str, float] = {}
    for window in (5, 10):
        recent = prior[-window:]
        features[f"{prefix}_xg_for_avg_last_{window}"] = _mean(recent, "xg_for")
        features[f"{prefix}_xg_against_avg_last_{window}"] = _mean(recent, "xg_against")
        features[f"{prefix}_xg_diff_last_{window}"] = _mean_diff(recent, "xg_for", "xg_against")
    features[f"{prefix}_shots_for_avg_last_5"] = _mean(prior[-5:], "shots_for")
    features[f"{prefix}_shots_against_avg_last_5"] = _mean(prior[-5:], "shots_against")
    return features


def _mean(rows: list[dict[str, Any]], column: str) -> float:
    values = [row[column] for row in rows if pd.notna(row[column])]
    return float(np.mean(values)) if values else np.nan


def _mean_diff(rows: list[dict[str, Any]], left: str, right: str) -> float:
    values = [row[left] - row[right] for row in rows if pd.notna(row[left]) and pd.notna(row[right])]
    return float(np.mean(values)) if values else np.nan
