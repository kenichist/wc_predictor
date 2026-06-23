from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss


def brier_score_multiclass(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int = 3) -> float:
    one_hot = np.eye(n_classes)[y_true.astype(int)]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def ranked_probability_score(y_true: np.ndarray, probabilities: np.ndarray, n_classes: int = 3) -> float:
    one_hot_cdf = np.cumsum(np.eye(n_classes)[y_true.astype(int)], axis=1)
    prob_cdf = np.cumsum(probabilities, axis=1)
    return float(np.mean(np.sum((prob_cdf - one_hot_cdf) ** 2, axis=1) / (n_classes - 1)))


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_true.astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (confidences > left) & (confidences <= right)
        if not mask.any():
            continue
        ece += float(mask.mean() * abs(correct[mask].mean() - confidences[mask].mean()))
    return ece


def evaluate_probabilities(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return {
        "accuracy": float(accuracy_score(y, clipped.argmax(axis=1))),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1, 2])),
        "brier_score": brier_score_multiclass(y, clipped),
        "ranked_probability_score": ranked_probability_score(y, clipped),
        "calibration_error": expected_calibration_error(y, clipped),
    }


def time_aware_split(
    df: pd.DataFrame,
    *,
    train_end: str = "2022-01-01",
    validation_start: str = "2022-01-01",
    validation_end: str = "2025-01-01",
    test_start: str = "2025-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(df["date"], errors="coerce")
    train = df[dates < pd.Timestamp(train_end)].copy()
    validation = df[(dates >= pd.Timestamp(validation_start)) & (dates < pd.Timestamp(validation_end))].copy()
    test = df[dates >= pd.Timestamp(test_start)].copy()
    return train, validation, test
