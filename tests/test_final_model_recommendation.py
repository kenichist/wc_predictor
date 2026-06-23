from types import SimpleNamespace

import pandas as pd

import src.cli as cli
from src.reports.final_model_recommendation import write_final_model_recommendation


def _cfg(tmp_path):
    return {
        "paths": {
            "ablation_results_csv": str(tmp_path / "ablation_results.csv"),
            "ablation_results_md": str(tmp_path / "ablation_results.md"),
            "feature_null_rate_report_csv": str(tmp_path / "feature_null_rate_report.csv"),
            "external_rating_activation_report_csv": str(tmp_path / "external_rating_activation_report.csv"),
            "external_rating_activation_report_md": str(tmp_path / "external_rating_activation_report.md"),
            "current_external_rating_acquisition_report_md": str(tmp_path / "current_external_rating_acquisition_report.md"),
            "calibration_params": str(tmp_path / "calibration_params.json"),
            "final_model_recommendation_md": str(tmp_path / "final_model_recommendation.md"),
        }
    }


def _write_minimal_ablation(path):
    pd.DataFrame(
        [
            {"feature_set": "v1_baseline", "accuracy": 0.58, "log_loss": 0.904, "brier_score": 0.53, "ranked_probability_score": 0.17, "calibration_error": 0.03},
            {"feature_set": "core_football_only", "accuracy": 0.59, "log_loss": 0.890, "brier_score": 0.52, "ranked_probability_score": 0.16, "calibration_error": 0.02},
            {"feature_set": "v2_baseline_plus_fifa_rankings", "accuracy": 0.57, "log_loss": 0.910, "brier_score": 0.54, "ranked_probability_score": 0.18, "calibration_error": 0.04},
            {"feature_set": "v3_plus_external_elo", "accuracy": 0.57, "log_loss": 0.910, "brier_score": 0.54, "ranked_probability_score": 0.18, "calibration_error": 0.04},
        ]
    ).to_csv(path, index=False)


def _write_minimal_null_report(path):
    pd.DataFrame(
        [
            {"feature_group": "fifa_rankings", "historical_training_coverage": 0.43},
            {"feature_group": "external_elo", "historical_training_coverage": 0.0},
        ]
    ).to_csv(path, index=False)


def test_final_model_recommendation_report_is_created(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    _write_minimal_ablation(tmp_path / "ablation_results.csv")
    _write_minimal_null_report(tmp_path / "feature_null_rate_report.csv")

    path = write_final_model_recommendation(config=cfg, selected_model="catboost", selected_feature_set="core_football_only")

    text = path.read_text(encoding="utf-8")
    assert "Best feature set" in text
    assert "`core_football_only`" in text
    assert "FIFA helped validation: `False`" in text
    assert "Current World Cup simulation used: `safe` features" in text


def test_full_pipeline_runs_with_core_football_only_and_writes_recommendation(tmp_path, monkeypatch) -> None:
    cfg = _cfg(tmp_path)
    calls = {}

    def fake_run_ablation(config):
        _write_minimal_ablation(tmp_path / "ablation_results.csv")
        return pd.read_csv(tmp_path / "ablation_results.csv")

    def fake_feature_null_report(config):
        _write_minimal_null_report(tmp_path / "feature_null_rate_report.csv")
        return pd.DataFrame(), pd.DataFrame()

    def fake_train_match_model(*, model_name, feature_set, config):
        calls["train"] = (model_name, feature_set)
        return {}, pd.DataFrame(), {}

    def fake_calibrate_model(config, *, model_name, feature_set):
        calls["calibrate"] = (model_name, feature_set)
        (tmp_path / "calibration_params.json").write_text(
            '{"model_name":"catboost","feature_set":"core_football_only","after":{"log_loss":0.89,"accuracy":0.59}}',
            encoding="utf-8",
        )
        return {}

    monkeypatch.setattr(cli, "build_advanced_features", lambda config: None)
    monkeypatch.setattr(cli, "run_ablation", fake_run_ablation)
    monkeypatch.setattr(cli, "train_match_model", fake_train_match_model)
    monkeypatch.setattr(cli, "train_poisson_model", lambda config: None)
    monkeypatch.setattr(cli, "calibrate_model", fake_calibrate_model)
    monkeypatch.setattr(cli, "build_ensemble_predictions", lambda config: None)
    monkeypatch.setattr(cli, "split_prediction_outputs", lambda config: None)
    monkeypatch.setattr(cli, "generate_feature_null_rate_report", fake_feature_null_report)
    monkeypatch.setattr(cli, "generate_external_rating_activation_report", lambda config: pd.DataFrame())
    monkeypatch.setattr(cli, "write_current_external_rating_acquisition_report", lambda config: tmp_path / "current.md")
    monkeypatch.setattr(cli, "simulate_worldcup", lambda n_sims, config: None)

    cli.run_full_sota_pipeline(SimpleNamespace(model="catboost", feature_set="core_football_only", n_sims=10), cfg)

    assert calls["train"] == ("catboost", "core_football_only")
    assert calls["calibrate"] == ("catboost", "core_football_only")
    assert (tmp_path / "final_model_recommendation.md").exists()
