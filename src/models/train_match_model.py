from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import config_path, load_config
from src.features.feature_sets import columns_for_feature_set
from src.io_utils import read_dataframe, write_dataframe
from src.models.evaluate import evaluate_probabilities, time_aware_split


logger = logging.getLogger(__name__)


def train_match_model(
    *,
    model_name: str = "catboost",
    feature_set: str = "full_football_only",
    config: dict[str, Any] | None = None,
) -> tuple[Any, pd.DataFrame, dict[str, float]]:
    cfg = config or load_config()
    training_path = _best_existing_path(config_path(cfg, "match_training_dataset_advanced_parquet"), config_path(cfg, "match_training_dataset_parquet"))
    prediction_path = _best_existing_path(config_path(cfg, "worldcup_2026_prediction_input_advanced"), config_path(cfg, "worldcup_2026_prediction_input"))
    df = read_dataframe(training_path)
    prediction_input = read_dataframe(prediction_path)
    df = df.dropna(subset=["target_result_class"]).copy()
    feature_columns = columns_for_feature_set(feature_set, df.columns)
    feature_columns = [column for column in feature_columns if df[column].notna().any()]
    if not feature_columns:
        raise ValueError(f"No usable columns found for feature set {feature_set}")

    split_cfg = cfg.get("modeling", {})
    train_df, val_df, test_df = time_aware_split(
        df,
        train_end=split_cfg.get("train_end", "2022-01-01"),
        validation_start=split_cfg.get("validation_start", "2022-01-01"),
        validation_end=split_cfg.get("validation_end", "2025-01-01"),
        test_start=split_cfg.get("test_start", "2025-01-01"),
    )
    if train_df.empty or val_df.empty:
        logger.warning("Time-aware split produced an empty train/validation set; falling back to chronological 80/20 split")
        ordered = df.sort_values("date", kind="stable")
        split_idx = max(1, int(len(ordered) * 0.8))
        train_df = ordered.iloc[:split_idx].copy()
        val_df = ordered.iloc[split_idx:].copy()

    estimator, resolved_model_name = _make_estimator(
        model_name,
        random_seed=int(split_cfg.get("random_seed", 42)),
        modeling_config=split_cfg,
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("estimator", estimator),
        ]
    )
    model.fit(_feature_matrix(train_df, feature_columns), train_df["target_result_class"].astype(int))
    val_prob = _predict_proba_3(model, _feature_matrix(val_df, feature_columns))
    metrics = evaluate_probabilities(val_df["target_result_class"], val_prob)
    logger.info("Validation metrics for %s/%s: %s", resolved_model_name, feature_set, metrics)

    if not test_df.empty:
        test_prob = _predict_proba_3(model, _feature_matrix(test_df, feature_columns))
        test_metrics = evaluate_probabilities(test_df["target_result_class"], test_prob)
        metrics.update({f"test_{key}": value for key, value in test_metrics.items()})

    model_bundle = {"model": model, "feature_columns": feature_columns, "model_name": resolved_model_name, "feature_set": feature_set}
    model_path = config_path(cfg, "match_model")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)

    predictions = predict_matches(model_bundle, prediction_input)
    write_dataframe(predictions, config_path(cfg, "match_predictions"))
    return model_bundle, predictions, metrics


def predict_matches(model_bundle: dict[str, Any], matches: pd.DataFrame) -> pd.DataFrame:
    feature_columns = model_bundle["feature_columns"]
    probabilities = _predict_proba_3(model_bundle["model"], _feature_matrix(matches, feature_columns))
    return pd.DataFrame(
        {
            "match_id": matches["match_id"].values,
            "date": matches["date"].values,
            "home_team": matches["home_team"].values,
            "away_team": matches["away_team"].values,
            "p_home_loss": probabilities[:, 0],
            "p_draw": probabilities[:, 1],
            "p_home_win": probabilities[:, 2],
            "predicted_class": probabilities.argmax(axis=1),
            "actual_class": matches.get("target_result_class", pd.Series(np.nan, index=matches.index)).values,
            "model_name": model_bundle["model_name"],
            "feature_set": model_bundle["feature_set"],
        }
    )


def _make_estimator(model_name: str, *, random_seed: int, modeling_config: dict[str, Any] | None = None) -> tuple[Any, str]:
    requested = model_name.lower()
    modeling_config = modeling_config or {}
    if requested == "catboost":
        try:
            from catboost import CatBoostClassifier

            params: dict[str, Any] = {
                "loss_function": "MultiClass",
                "verbose": False,
                "random_seed": random_seed,
                "iterations": int(modeling_config.get("catboost_iterations", 300)),
                "depth": int(modeling_config.get("catboost_depth", 6)),
                "learning_rate": float(modeling_config.get("catboost_learning_rate", 0.05)),
                "allow_writing_files": False,
            }
            if _as_bool(modeling_config.get("catboost_use_gpu", False)):
                params.update(
                    {
                        "task_type": "GPU",
                        "devices": str(modeling_config.get("catboost_gpu_devices", "0")),
                    }
                )
            return CatBoostClassifier(**params), "catboost"
        except ImportError:
            logger.warning("CatBoost is not installed; falling back to HistGradientBoostingClassifier")
    elif requested == "xgboost":
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", random_state=random_seed), "xgboost"
        except ImportError:
            logger.warning("XGBoost is not installed; falling back to HistGradientBoostingClassifier")
    elif requested == "logistic":
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, multi_class="auto"))]), "logistic"
    return HistGradientBoostingClassifier(random_state=random_seed, loss="log_loss"), "hist_gradient_boosting"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    matrix = df.reindex(columns=feature_columns).copy()
    for column in matrix.columns:
        if matrix[column].dtype == "bool":
            matrix[column] = matrix[column].astype(float)
        else:
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    return matrix


def _predict_proba_3(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(matrix)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        estimator = model.named_steps.get("estimator") or model.named_steps.get("clf")
        classes = getattr(estimator, "classes_", np.array([0, 1, 2]))
    output = np.zeros((len(matrix), 3), dtype=float)
    for i, cls in enumerate(classes):
        output[:, int(cls)] = probabilities[:, i]
    row_sums = output.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return output / row_sums


def _best_existing_path(preferred: Path, fallback: Path) -> Path:
    return preferred if preferred.exists() else fallback
