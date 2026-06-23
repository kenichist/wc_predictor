from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import config_path, load_config
from src.io_utils import read_dataframe, read_json, write_dataframe


logger = logging.getLogger(__name__)


def build_ensemble_predictions(config: dict[str, Any] | None = None, weights: dict[str, float] | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    weights = weights or {"model": 0.5, "poisson": 0.3, "rating": 0.2}
    model_path = config_path(cfg, "calibrated_predictions")
    if not model_path.exists():
        model_path = config_path(cfg, "match_predictions")
    if not model_path.exists():
        raise FileNotFoundError("No match model predictions found. Run train-model first.")
    model_preds = read_dataframe(model_path)
    ensemble = model_preds.copy()
    probs = weights["model"] * model_preds[["p_home_loss", "p_draw", "p_home_win"]].to_numpy(dtype=float)

    poisson_path = config_path(cfg, "poisson_predictions")
    if poisson_path.exists():
        poisson = read_dataframe(poisson_path)[["match_id", "p_home_loss", "p_draw", "p_home_win"]]
        merged = model_preds[["match_id"]].merge(poisson, on="match_id", how="left")
        probs += weights["poisson"] * merged[["p_home_loss", "p_draw", "p_home_win"]].fillna(1 / 3).to_numpy(dtype=float)
    else:
        probs += weights["poisson"] * (1 / 3)

    rating_probs = _rating_baseline_probs(model_preds)
    probs += weights["rating"] * rating_probs
    probs = probs / probs.sum(axis=1, keepdims=True)
    ensemble[["p_home_loss", "p_draw", "p_home_win"]] = probs
    ensemble["predicted_class"] = probs.argmax(axis=1)
    ensemble["model_name"] = "football_only_ensemble"
    ensemble = _apply_market_blend(ensemble, cfg)
    write_dataframe(ensemble, config_path(cfg, "ensemble_predictions"))
    return ensemble


def _rating_baseline_probs(predictions: pd.DataFrame) -> np.ndarray:
    # Neutral fallback when rating features are unavailable in the predictions file.
    return np.tile(np.array([0.30, 0.25, 0.45]), (len(predictions), 1))


def _apply_market_blend(predictions: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    market = _prediction_market_probabilities(cfg)
    if market.empty:
        return predictions
    output = predictions.merge(market, on="match_id", how="left")
    market_columns = ["market_away_prob", "market_draw_prob", "market_home_prob"]
    covered = output[market_columns].notna().all(axis=1)
    if not covered.any():
        return output.drop(columns=market_columns, errors="ignore")
    coverage = float(covered.mean()) if len(covered) else 0.0
    min_coverage = float(cfg.get("modeling", {}).get("market_blend_min_coverage", 0.999))
    if coverage < min_coverage:
        logger.warning(
            "Skipping production market blend because market coverage is %.3f, below required %.3f",
            coverage,
            min_coverage,
        )
        output["market_blend_available"] = covered
        output["market_blend_coverage"] = coverage
        return output.drop(columns=market_columns, errors="ignore")
    alpha = _market_blend_alpha(cfg)
    model_prob = output.loc[covered, ["p_home_loss", "p_draw", "p_home_win"]].to_numpy(dtype=float)
    market_prob = output.loc[covered, market_columns].to_numpy(dtype=float)
    blended = (alpha * model_prob) + ((1.0 - alpha) * market_prob)
    blended = blended / blended.sum(axis=1, keepdims=True)
    output.loc[covered, ["p_home_loss", "p_draw", "p_home_win"]] = blended
    output.loc[covered, "predicted_class"] = blended.argmax(axis=1)
    output.loc[covered, "model_name"] = "market_blend"
    output.loc[covered, "market_blend_alpha"] = alpha
    return output.drop(columns=market_columns, errors="ignore")


def _prediction_market_probabilities(cfg: dict[str, Any]) -> pd.DataFrame:
    for key in ("worldcup_2026_prediction_input_advanced", "worldcup_2026_prediction_input"):
        path = config_path(cfg, key)
        if not path.exists():
            continue
        prediction_input = read_dataframe(path)
        columns = ["match_id", "market_away_prob", "market_draw_prob", "market_home_prob"]
        if set(columns).issubset(prediction_input.columns):
            return prediction_input[columns].copy()
    return pd.DataFrame(columns=["match_id", "market_away_prob", "market_draw_prob", "market_home_prob"])


def _market_blend_alpha(cfg: dict[str, Any]) -> float:
    try:
        path = config_path(cfg, "market_blend_params")
    except KeyError:
        return 0.5
    if not path.exists():
        return 0.5
    try:
        params = read_json(path)
        alpha = float(params.get("alpha", 0.5))
    except Exception as exc:
        logger.warning("Could not read market blend alpha from %s: %s", path, exc)
        return 0.5
    return min(1.0, max(0.0, alpha))
