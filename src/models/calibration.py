from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import config_path, load_config
from src.io_utils import read_dataframe, write_dataframe
from src.models.evaluate import evaluate_probabilities, time_aware_split
from src.models.train_match_model import train_match_model


logger = logging.getLogger(__name__)


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    scaled = np.exp(logits / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, float], dict[str, float]]:
    before = evaluate_probabilities(y_true, probabilities)
    candidates = np.linspace(0.6, 2.5, 39)
    best_t = 1.0
    best_metrics = before
    for temperature in candidates:
        metrics = evaluate_probabilities(y_true, temperature_scale(probabilities, float(temperature)))
        if metrics["log_loss"] <= best_metrics["log_loss"]:
            best_t = float(temperature)
            best_metrics = metrics
    return best_t, before, best_metrics


def calibrate_model(
    config: dict[str, Any] | None = None,
    *,
    model_name: str = "catboost",
    feature_set: str = "full_football_only",
) -> dict[str, Any]:
    cfg = config or load_config()
    model_bundle, _, _ = train_match_model(model_name=model_name, feature_set=feature_set, config=cfg)
    df_path = config_path(cfg, "match_training_dataset_advanced_parquet")
    if not df_path.exists():
        df_path = config_path(cfg, "match_training_dataset_parquet")
    df = read_dataframe(df_path).dropna(subset=["target_result_class"])
    split_cfg = cfg.get("modeling", {})
    _, validation, _ = time_aware_split(
        df,
        train_end=split_cfg.get("train_end", "2022-01-01"),
        validation_start=split_cfg.get("validation_start", "2022-01-01"),
        validation_end=split_cfg.get("validation_end", "2025-01-01"),
        test_start=split_cfg.get("test_start", "2025-01-01"),
    )
    if validation.empty:
        validation = df.sort_values("date", kind="stable").tail(max(10, int(len(df) * 0.2)))
    from src.models.train_match_model import _feature_matrix, _predict_proba_3

    raw_prob = _predict_proba_3(model_bundle["model"], _feature_matrix(validation, model_bundle["feature_columns"]))
    temperature, before, after = fit_temperature(validation["target_result_class"].astype(int).to_numpy(), raw_prob)
    params = {
        "method": "temperature_scaling",
        "temperature": temperature,
        "before": before,
        "after": after,
        "model_name": model_bundle.get("model_name", model_name),
        "feature_set": model_bundle.get("feature_set", feature_set),
    }
    output_path = config_path(cfg, "calibration_params")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

    predictions_path = config_path(cfg, "match_predictions")
    if predictions_path.exists():
        predictions = read_dataframe(predictions_path)
        probs = predictions[["p_home_loss", "p_draw", "p_home_win"]].to_numpy(dtype=float)
        calibrated = temperature_scale(probs, temperature)
        predictions[["p_home_loss", "p_draw", "p_home_win"]] = calibrated
        write_dataframe(predictions, config_path(cfg, "calibrated_predictions"))
    logger.info("Calibration before=%s after=%s", before, after)
    return params


def fit_platt_scaler(y_true: np.ndarray, probabilities: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, multi_class="auto")
    model.fit(np.log(np.clip(probabilities, 1e-12, 1.0)), y_true.astype(int))
    return model
