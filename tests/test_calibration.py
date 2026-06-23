import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import src.models.calibration as calibration
from src.models.calibration import calibrate_model, fit_temperature, temperature_scale


def test_temperature_calibration_preserves_or_improves_toy_log_loss() -> None:
    y = np.array([0, 1, 2, 2, 1, 0])
    probs = np.array(
        [
            [0.55, 0.25, 0.20],
            [0.25, 0.50, 0.25],
            [0.25, 0.20, 0.55],
            [0.20, 0.25, 0.55],
            [0.20, 0.60, 0.20],
            [0.60, 0.20, 0.20],
        ]
    )

    temperature, before, after = fit_temperature(y, probs)

    assert log_loss(y, temperature_scale(probs, temperature), labels=[0, 1, 2]) <= log_loss(y, probs, labels=[0, 1, 2]) + 1e-12
    assert after["log_loss"] <= before["log_loss"] + 1e-12


def test_calibrate_model_uses_requested_model_and_feature_set(tmp_path, monkeypatch) -> None:
    train_path = tmp_path / "train.parquet"
    pred_path = tmp_path / "pred.parquet"
    predictions_path = tmp_path / "match_predictions.parquet"
    calibrated_path = tmp_path / "calibrated.parquet"
    params_path = tmp_path / "calibration_params.json"
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=12, freq="YS"),
            "target_result_class": [0, 1, 2] * 4,
            "feature": range(12),
        }
    )
    df.to_parquet(train_path, index=False)
    df.to_parquet(pred_path, index=False)
    pd.DataFrame(
        {
            "p_home_loss": [0.4, 0.3],
            "p_draw": [0.3, 0.4],
            "p_home_win": [0.3, 0.3],
        }
    ).to_parquet(predictions_path, index=False)
    calls = {}

    class DummyModel:
        classes_ = np.array([0, 1, 2])

        def predict_proba(self, matrix):
            return np.tile(np.array([[0.4, 0.3, 0.3]]), (len(matrix), 1))

    def fake_train_match_model(*, model_name, feature_set, config):
        calls["model_name"] = model_name
        calls["feature_set"] = feature_set
        return {"model": DummyModel(), "feature_columns": ["feature"], "model_name": model_name, "feature_set": feature_set}, pd.DataFrame(), {}

    monkeypatch.setattr(calibration, "train_match_model", fake_train_match_model)
    cfg = {
        "paths": {
            "match_training_dataset_advanced_parquet": str(train_path),
            "match_training_dataset_parquet": str(train_path),
            "worldcup_2026_prediction_input_advanced": str(pred_path),
            "worldcup_2026_prediction_input": str(pred_path),
            "match_predictions": str(predictions_path),
            "calibrated_predictions": str(calibrated_path),
            "calibration_params": str(params_path),
        },
        "modeling": {"train_end": "2024-01-01", "validation_start": "2024-01-01", "validation_end": "2030-01-01"},
    }

    params = calibrate_model(config=cfg, model_name="catboost", feature_set="full_football_only")

    assert calls == {"model_name": "catboost", "feature_set": "full_football_only"}
    assert params["model_name"] == "catboost"
    assert params["feature_set"] == "full_football_only"
