from __future__ import annotations

import pandas as pd

import src.cli as cli
from src.backtesting.stage5_failure_diagnosis import run_stage5_failure_diagnosis


def test_stage5_failure_diagnosis_creates_outputs(tmp_path) -> None:
    _write_stage5_fixture(tmp_path)

    result = run_stage5_failure_diagnosis(
        output_dir=tmp_path,
        report_path=tmp_path / "stage5_failure_diagnosis_report.md",
    )

    assert result.report_path.exists()
    assert result.by_year_path.exists()
    assert result.by_stage_path.exists()
    assert result.by_outcome_path.exists()
    assert result.worst_matches_path.exists()
    assert result.ablation_reconciliation_path.exists()
    assert not result.by_year.empty
    assert result.worst_matches.iloc[0]["match_id"] == "m2"
    assert "Why Market Blend Equals Bookmaker Odds" in result.report_path.read_text(encoding="utf-8")


def test_stage5_diagnose_failures_cli_parser() -> None:
    args = cli.build_parser().parse_args(["stage5-diagnose-failures", "--output-dir", "data/backtests"])

    assert args.command == "stage5-diagnose-failures"
    assert args.output_dir == "data/backtests"


def _write_stage5_fixture(path) -> None:
    predictions = pd.DataFrame(
        [
            _prediction("m1", "official_model", "home_win", 0.70, 0.20, 0.10, 0.357, 0.140, 0.070),
            _prediction("m2", "official_model", "away_win", 0.60, 0.25, 0.15, 1.897, 1.085, 0.505),
            _prediction("m1", "bookmaker_odds_only", "home_win", 0.55, 0.25, 0.20, 0.598, 0.305, 0.130),
            _prediction("m2", "bookmaker_odds_only", "away_win", 0.20, 0.25, 0.55, 0.598, 0.305, 0.130),
            _prediction("m1", "market_blend", "home_win", 0.55, 0.25, 0.20, 0.598, 0.305, 0.130),
            _prediction("m2", "market_blend", "away_win", 0.20, 0.25, 0.55, 0.598, 0.305, 0.130),
        ]
    )
    predictions.to_csv(path / "stage5_match_predictions.csv", index=False)

    metrics = pd.DataFrame(
        [
            _metric("official_model", 1.127, 0.612, 0.288),
            _metric("bookmaker_odds_only", 0.598, 0.305, 0.130),
            _metric("market_blend", 0.598, 0.305, 0.130),
        ]
    )
    metrics.to_csv(path / "stage5_benchmark_metrics.csv", index=False)

    significance = pd.DataFrame(
        [
            {
                "candidate_model": "official_model",
                "baseline_model": "bookmaker_odds_only",
                "metric": "log_loss",
                "mean_delta": 0.529,
                "ci_lower": -0.1,
                "ci_upper": 1.0,
                "statistically_meaningful": False,
                "n_matches": 2,
            }
        ]
    )
    significance.to_csv(path / "stage5_significance_tests.csv", index=False)

    calibration = pd.DataFrame(
        [
            {
                "model_name": "official_model",
                "test_year": 2022,
                "bucket": "(0.6,0.7]",
                "mean_confidence": 0.65,
                "empirical_accuracy": 0.5,
                "count": 2,
                "calibration_gap": -0.15,
            }
        ]
    )
    calibration.to_csv(path / "stage5_calibration_table.csv", index=False)


def _prediction(
    match_id: str,
    model_name: str,
    predicted_result: str,
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    log_loss: float,
    brier: float,
    rps: float,
) -> dict[str, object]:
    actual_class = 2 if match_id == "m1" else 0
    actual_result = "home_win" if match_id == "m1" else "away_win"
    return {
        "match_id": match_id,
        "date": "2022-11-20",
        "tournament_year": 2022,
        "stage": "group_stage",
        "home_team": "A",
        "away_team": "B",
        "actual_result": actual_result,
        "actual_class": actual_class,
        "model_name": model_name,
        "feature_set": "fixture",
        "baseline_type": "fixture",
        "home_prob": home_prob,
        "draw_prob": draw_prob,
        "away_prob": away_prob,
        "predicted_result": predicted_result,
        "confidence": max(home_prob, draw_prob, away_prob),
        "log_loss": log_loss,
        "brier": brier,
        "rps": rps,
        "data_source_notes": "fixture",
    }


def _metric(model_name: str, log_loss: float, brier: float, rps: float) -> dict[str, object]:
    return {
        "model_name": model_name,
        "feature_set": "fixture",
        "baseline_type": "fixture",
        "test_year": 2022,
        "n_matches": 2,
        "n_skipped": 0,
        "accuracy": 0.5,
        "log_loss": log_loss,
        "brier": brier,
        "rps": rps,
        "ece": 0.1,
        "mean_confidence": 0.6,
    }
