import numpy as np
import pandas as pd
import pytest

from src.backtesting.stage5_benchmark import (
    _add_prediction_scores,
    _calibration_table,
    _stage5_metrics,
    _stage5_significance,
    normalize_probability_rows,
    paired_bootstrap_ci,
    per_row_brier,
    per_row_log_loss,
    per_row_rps,
    write_stage5_benchmark_report,
    write_stage5_decision_report,
)
from src.sources.market_odds import decimal_odds_to_probabilities


def test_stage5_probability_rows_are_valid() -> None:
    probabilities = normalize_probability_rows(np.array([[2.0, 1.0, 1.0], [np.nan, np.nan, np.nan]]))

    assert np.all(probabilities >= 0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_stage5_row_metrics_are_finite() -> None:
    y = np.array([0, 1, 2])
    probabilities = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])

    assert np.isfinite(per_row_log_loss(y, probabilities)).all()
    assert np.isfinite(per_row_brier(y, probabilities)).all()
    assert np.isfinite(per_row_rps(y, probabilities)).all()
    assert per_row_log_loss(y, probabilities).mean() < 1.0


def test_stage5_odds_conversion_normalizes_probabilities() -> None:
    converted = decimal_odds_to_probabilities(pd.DataFrame([{"home_odds": 2.0, "draw_odds": 3.5, "away_odds": 4.0}]))

    total = converted.loc[0, ["market_home_prob", "market_draw_prob", "market_away_prob"]].sum()
    assert total == pytest.approx(1.0)


def test_stage5_bootstrap_ci_contains_mean() -> None:
    mean_delta, lower, upper = paired_bootstrap_ci(np.array([-0.1, -0.2, 0.05, -0.05]), n_bootstrap=200, seed=1)

    assert lower <= mean_delta <= upper


def test_stage5_metrics_calibration_and_significance_are_created() -> None:
    predictions = _synthetic_stage5_predictions()
    predictions = _add_prediction_scores(predictions)
    metrics = _stage5_metrics(predictions, {2022: 2})
    calibration = _calibration_table(predictions)
    significance = _stage5_significance(predictions, n_bootstrap=100)

    assert {"log_loss", "brier", "rps", "ece", "mean_confidence"}.issubset(metrics.columns)
    assert not calibration.empty
    assert not significance.empty
    assert significance["n_matches"].max() > 0


def test_stage5_report_generation_creates_expected_files(tmp_path) -> None:
    predictions = _add_prediction_scores(_synthetic_stage5_predictions())
    metrics = _stage5_metrics(predictions, {2022: 2})
    calibration = _calibration_table(predictions)
    significance = _stage5_significance(predictions, n_bootstrap=50)
    skipped = pd.DataFrame([{"model_name": "dixon_coles", "test_year": 2022, "reason": "skipped", "n_skipped": 2}])

    report = write_stage5_benchmark_report(
        metrics=metrics,
        predictions=predictions,
        significance=significance,
        calibration=calibration,
        stage_calibration=pd.DataFrame(),
        skipped_notes=skipped,
        years=[2022],
        stage5_achieved=False,
        path=tmp_path / "stage5_benchmark_report.md",
    )
    decision = write_stage5_decision_report(
        metrics=metrics,
        significance=significance,
        skipped_notes=skipped,
        stage5_achieved=False,
        path=tmp_path / "stage5_research_decision.md",
    )

    assert report.exists()
    assert decision.exists()
    assert "Stage 5" in report.read_text(encoding="utf-8")
    assert "Stage 5 not yet achieved" in decision.read_text(encoding="utf-8")


def _synthetic_stage5_predictions() -> pd.DataFrame:
    rows = []
    for model, probs in [
        ("official_model", [(0.65, 0.2, 0.15), (0.2, 0.25, 0.55)]),
        ("bookmaker_odds_only", [(0.50, 0.25, 0.25), (0.35, 0.30, 0.35)]),
        ("elo_only", [(0.45, 0.30, 0.25), (0.3, 0.3, 0.4)]),
        ("poisson_goal_model", [(0.40, 0.30, 0.30), (0.25, 0.3, 0.45)]),
        ("market_blend", [(0.60, 0.22, 0.18), (0.25, 0.27, 0.48)]),
    ]:
        for idx, (home, draw, away) in enumerate(probs):
            actual_class = 2 if idx == 0 else 0
            rows.append(
                {
                    "match_id": f"m{idx}",
                    "date": "2022-11-20",
                    "tournament_year": 2022,
                    "stage": "group",
                    "home_team": f"H{idx}",
                    "away_team": f"A{idx}",
                    "actual_result": "home_win" if actual_class == 2 else "away_win",
                    "actual_class": actual_class,
                    "model_name": model,
                    "feature_set": "test",
                    "baseline_type": "test",
                    "home_prob": home,
                    "draw_prob": draw,
                    "away_prob": away,
                    "predicted_result": "home_win",
                    "confidence": max(home, draw, away),
                    "data_source_notes": "synthetic",
                }
            )
    return pd.DataFrame(rows)
