from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.normalize import classify_competition_type


PI_RATING_COLUMNS = [
    "home_overall_rating_pre",
    "away_overall_rating_pre",
    "overall_rating_diff",
    "home_attack_rating_pre",
    "away_attack_rating_pre",
    "attack_rating_diff",
    "home_defense_rating_pre",
    "away_defense_rating_pre",
    "defense_rating_diff",
    "home_attack_vs_away_defense",
    "away_attack_vs_home_defense",
]


@dataclass
class TeamRatings:
    overall: float = 0.0
    attack: float = 0.0
    defense: float = 0.0


def compute_pi_ratings(
    matches: pd.DataFrame,
    *,
    learning_rate: float = 0.08,
    result_learning_rate: float = 6.0,
    base_goals: float = 1.35,
) -> pd.DataFrame:
    """Compute simplified pi-style attack/defense ratings as pre-match features."""
    if matches.empty:
        return pd.DataFrame(columns=["match_id", *PI_RATING_COLUMNS], index=matches.index)
    df = matches.copy()
    df["row_id"] = np.arange(len(df))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "row_id"], kind="stable")
    ratings: defaultdict[str, TeamRatings] = defaultdict(TeamRatings)
    rows: list[dict[str, Any]] = []

    for _, day_matches in df.groupby("date", sort=True, dropna=False):
        pending: defaultdict[str, TeamRatings] = defaultdict(TeamRatings)
        for row in day_matches.itertuples(index=False):
            rec = row._asdict()
            home = rec.get("home_team")
            away = rec.get("away_team")
            h = ratings[home] if home else TeamRatings()
            a = ratings[away] if away else TeamRatings()
            rows.append(
                {
                    "row_id": rec["row_id"],
                    "match_id": rec.get("match_id"),
                    "home_overall_rating_pre": h.overall,
                    "away_overall_rating_pre": a.overall,
                    "overall_rating_diff": h.overall - a.overall,
                    "home_attack_rating_pre": h.attack,
                    "away_attack_rating_pre": a.attack,
                    "attack_rating_diff": h.attack - a.attack,
                    "home_defense_rating_pre": h.defense,
                    "away_defense_rating_pre": a.defense,
                    "defense_rating_diff": h.defense - a.defense,
                    "home_attack_vs_away_defense": h.attack - a.defense,
                    "away_attack_vs_home_defense": a.attack - h.defense,
                }
            )

            hs = rec.get("home_score")
            away_score = rec.get("away_score")
            if not home or not away or pd.isna(hs) or pd.isna(away_score):
                continue
            comp = rec.get("competition_type") or classify_competition_type(rec.get("tournament"))
            importance = _importance_multiplier(str(comp))
            actual_h = float(hs)
            actual_a = float(away_score)
            xg_h = _first_number(rec, ["home_xg", "home_xg_for", "xg_home"], fallback=actual_h)
            xg_a = _first_number(rec, ["away_xg", "away_xg_for", "xg_away"], fallback=actual_a)
            expected_h = max(0.2, base_goals + h.attack - a.defense)
            expected_a = max(0.2, base_goals + a.attack - h.defense)
            home_goal_error = xg_h - expected_h
            away_goal_error = xg_a - expected_a
            result_error = _actual_score(actual_h, actual_a) - _expected_result(h.overall, a.overall)
            gd_error = (actual_h - actual_a) - (expected_h - expected_a)

            pending[home].attack += learning_rate * importance * home_goal_error
            pending[home].defense += learning_rate * importance * (-away_goal_error)
            pending[home].overall += result_learning_rate * importance * result_error + learning_rate * importance * gd_error
            pending[away].attack += learning_rate * importance * away_goal_error
            pending[away].defense += learning_rate * importance * (-home_goal_error)
            pending[away].overall -= result_learning_rate * importance * result_error + learning_rate * importance * gd_error
        for team, delta in pending.items():
            ratings[team].overall += delta.overall
            ratings[team].attack += delta.attack
            ratings[team].defense += delta.defense
    result = pd.DataFrame(rows).sort_values("row_id", kind="stable").drop(columns=["row_id"])
    result.index = matches.index
    return result


def _first_number(row: dict[str, Any], columns: list[str], fallback: float) -> float:
    for column in columns:
        value = row.get(column)
        if pd.notna(value):
            return float(value)
    return fallback


def _actual_score(home_score: float, away_score: float) -> float:
    if home_score > away_score:
        return 1.0
    if home_score == away_score:
        return 0.5
    return 0.0


def _expected_result(home_rating: float, away_rating: float) -> float:
    return 1.0 / (1.0 + np.exp(-(home_rating - away_rating) / 10.0))


def _importance_multiplier(competition_type: str) -> float:
    return {
        "world_cup": 1.7,
        "continental_championship": 1.35,
        "world_cup_qualifier": 1.2,
        "continental_qualifier": 1.12,
        "friendly": 0.7,
    }.get(competition_type, 1.0)
