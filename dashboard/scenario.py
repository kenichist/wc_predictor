from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.normalize import normalize_team_name


@dataclass(frozen=True)
class ProbabilityColumns:
    home: str
    draw: str
    away: str


PROBABILITY_COLUMN_CANDIDATES = [
    ProbabilityColumns("home_win_prob", "draw_prob", "away_win_prob"),
    ProbabilityColumns("home_prob", "draw_prob", "away_prob"),
    ProbabilityColumns("prob_home_win", "prob_draw", "prob_away_win"),
    ProbabilityColumns("p_home", "p_draw", "p_away"),
    ProbabilityColumns("p_home_win", "p_draw", "p_home_loss"),
]


def infer_probability_columns(df: pd.DataFrame) -> ProbabilityColumns | None:
    columns = set(df.columns)
    for candidate in PROBABILITY_COLUMN_CANDIDATES:
        if {candidate.home, candidate.draw, candidate.away}.issubset(columns):
            return candidate
    return None


def normalize_probabilities(values: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    probs = np.asarray(values, dtype=float)
    probs = np.where(np.isfinite(probs), probs, np.nan)
    probs = np.clip(probs, 1e-9, None)
    total = np.nansum(probs)
    if total <= 0 or not np.isfinite(total):
        return np.full(len(probs), 1 / len(probs))
    return probs / total


def decimal_odds_to_implied(home_odds: Any, draw_odds: Any, away_odds: Any) -> tuple[float, float, float] | None:
    try:
        odds = np.array([float(home_odds), float(draw_odds), float(away_odds)], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(odds).all() or np.any(odds <= 1):
        return None
    probs = normalize_probabilities(1.0 / odds)
    return float(probs[0]), float(probs[1]), float(probs[2])


def advancement_probabilities(team_a_win_prob: float, team_b_win_prob: float) -> tuple[float, float]:
    decisive = float(team_a_win_prob) + float(team_b_win_prob)
    if decisive <= 0:
        return 0.5, 0.5
    return float(team_a_win_prob / decisive), float(team_b_win_prob / decisive)


def apply_scenario_adjustments(
    probabilities: tuple[float, float, float],
    *,
    team_a_adjustment: float = 0.0,
    team_b_adjustment: float = 0.0,
) -> tuple[float, float, float]:
    base = normalize_probabilities(probabilities)
    logits = np.log(np.clip(base, 1e-9, None))
    logits[0] += float(team_a_adjustment)
    logits[2] += float(team_b_adjustment)
    shifted = logits - np.max(logits)
    adjusted = np.exp(shifted)
    adjusted = adjusted / adjusted.sum()
    return float(adjusted[0]), float(adjusted[1]), float(adjusted[2])


def find_fixture_prediction(df: pd.DataFrame, team_a: str, team_b: str) -> dict[str, Any] | None:
    if df.empty:
        return None
    prob_cols = infer_probability_columns(df)
    if prob_cols is None or not {"home_team", "away_team"}.issubset(df.columns):
        return None
    output = df.copy()
    output["_home_norm"] = output["home_team"].map(normalize_team_name)
    output["_away_norm"] = output["away_team"].map(normalize_team_name)
    team_a_norm = normalize_team_name(team_a)
    team_b_norm = normalize_team_name(team_b)
    direct = output[output["_home_norm"].eq(team_a_norm) & output["_away_norm"].eq(team_b_norm)]
    if not direct.empty:
        row = direct.iloc[0]
        return {
            "orientation": "direct",
            "row": row.to_dict(),
            "probabilities": (
                float(row[prob_cols.home]),
                float(row[prob_cols.draw]),
                float(row[prob_cols.away]),
            ),
        }
    reverse = output[output["_home_norm"].eq(team_b_norm) & output["_away_norm"].eq(team_a_norm)]
    if not reverse.empty:
        row = reverse.iloc[0]
        return {
            "orientation": "reverse",
            "row": row.to_dict(),
            "probabilities": (
                float(row[prob_cols.away]),
                float(row[prob_cols.draw]),
                float(row[prob_cols.home]),
            ),
        }
    return None


def team_options(*frames: pd.DataFrame) -> list[str]:
    teams: set[str] = set()
    for df in frames:
        if df.empty:
            continue
        for column in ("team", "home_team", "away_team", "team_a", "team_b"):
            if column in df.columns:
                teams.update(normalize_team_name(value) for value in df[column].dropna())
    return sorted(team for team in teams if team and str(team).lower() != "nan")
