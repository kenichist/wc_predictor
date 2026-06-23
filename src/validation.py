from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


CORE_COLUMNS = [
    "match_id",
    "source",
    "date",
    "home_team",
    "away_team",
    "neutral",
    "tournament",
    "competition_type",
    "stage",
    "group",
    "country",
    "city",
    "venue",
    "home_score",
    "away_score",
    "result",
    "home_win",
    "draw",
    "away_win",
    "goal_diff",
    "total_goals",
]

FEATURE_COLUMNS = [
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
    "home_elo_pre_match",
    "away_elo_pre_match",
    "elo_diff",
    "home_fifa_rank",
    "away_fifa_rank",
    "fifa_rank_diff",
    "is_world_cup",
    "is_qualifier",
    "is_continental_tournament",
    "is_friendly",
    "match_weight",
]

TARGET_COLUMNS = [
    "target_result_class",
    "target_home_goals",
    "target_away_goals",
]

REQUIRED_TRAINING_COLUMNS = CORE_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS


def add_missing_columns(df: pd.DataFrame, columns: Iterable[str], default: object = pd.NA) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = default
    return output


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    dataset_name: str = "dataset",
    raise_on_missing: bool = True,
) -> list[str]:
    missing = [column for column in required_columns if column not in df.columns]
    if missing and raise_on_missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")
    return missing


def validate_training_schema(df: pd.DataFrame, *, raise_on_missing: bool = True) -> list[str]:
    return validate_required_columns(
        df,
        REQUIRED_TRAINING_COLUMNS,
        dataset_name="match_training_dataset",
        raise_on_missing=raise_on_missing,
    )
