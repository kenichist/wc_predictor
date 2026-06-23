from __future__ import annotations

import json
import logging
from math import exp, factorial
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import config_path, load_config
from src.io_utils import read_dataframe, write_dataframe


logger = logging.getLogger(__name__)


def train_poisson_model(config: dict[str, Any] | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = config or load_config()
    training_path = config_path(cfg, "match_training_dataset_advanced_parquet")
    if not training_path.exists():
        training_path = config_path(cfg, "match_training_dataset_parquet")
    prediction_path = config_path(cfg, "worldcup_2026_prediction_input_advanced")
    if not prediction_path.exists():
        prediction_path = config_path(cfg, "worldcup_2026_prediction_input")
    df = read_dataframe(training_path).dropna(subset=["target_home_goals", "target_away_goals"])
    model = {
        "global_home_goals": float(df["target_home_goals"].mean()),
        "global_away_goals": float(df["target_away_goals"].mean()),
    }
    model_path = config_path(cfg, "poisson_model")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    predictions = predict_scorelines(model, read_dataframe(prediction_path))
    write_dataframe(predictions, config_path(cfg, "poisson_predictions"))
    return model, predictions


def predict_scorelines(model: dict[str, Any], matches: pd.DataFrame, *, max_goals: int = 7) -> pd.DataFrame:
    rows = []
    for match in matches.itertuples(index=False):
        rec = match._asdict()
        expected_home = _expected_home_goals(rec, model)
        expected_away = _expected_away_goals(rec, model)
        matrix = _score_matrix(expected_home, expected_away, max_goals=max_goals)
        p_home_loss = float(np.tril(matrix, -1).sum())
        p_draw = float(np.trace(matrix))
        p_home_win = float(np.triu(matrix, 1).sum())
        total = p_home_loss + p_draw + p_home_win
        rows.append(
            {
                "match_id": rec.get("match_id"),
                "date": rec.get("date"),
                "home_team": rec.get("home_team"),
                "away_team": rec.get("away_team"),
                "p_home_loss": p_home_loss / total,
                "p_draw": p_draw / total,
                "p_home_win": p_home_win / total,
                "expected_home_goals": expected_home,
                "expected_away_goals": expected_away,
                "scoreline_probs_json": json.dumps(_matrix_to_dict(matrix)),
                "model_name": "poisson",
            }
        )
    return pd.DataFrame(rows)


def _expected_home_goals(match: dict[str, Any], model: dict[str, Any]) -> float:
    gf = match.get("home_goals_for_avg_last_5")
    opp_ga = match.get("away_goals_against_avg_last_5")
    xg = match.get("home_xg_for_avg_last_5")
    values = [model["global_home_goals"]]
    for value in (gf, opp_ga, xg):
        if pd.notna(value):
            values.append(float(value))
    return float(np.clip(np.mean(values), 0.15, 5.0))


def _expected_away_goals(match: dict[str, Any], model: dict[str, Any]) -> float:
    gf = match.get("away_goals_for_avg_last_5")
    opp_ga = match.get("home_goals_against_avg_last_5")
    xg = match.get("away_xg_for_avg_last_5")
    values = [model["global_away_goals"]]
    for value in (gf, opp_ga, xg):
        if pd.notna(value):
            values.append(float(value))
    return float(np.clip(np.mean(values), 0.15, 5.0))


def _score_matrix(home_lambda: float, away_lambda: float, *, max_goals: int) -> np.ndarray:
    home_probs = np.array([_poisson_pmf(i, home_lambda) for i in range(max_goals + 1)])
    away_probs = np.array([_poisson_pmf(i, away_lambda) for i in range(max_goals + 1)])
    matrix = np.outer(home_probs, away_probs)
    return matrix / matrix.sum()


def _poisson_pmf(k: int, lam: float) -> float:
    return exp(-lam) * lam**k / factorial(k)


def _matrix_to_dict(matrix: np.ndarray) -> dict[str, float]:
    return {f"{i}-{j}": float(matrix[i, j]) for i in range(matrix.shape[0]) for j in range(matrix.shape[1])}
