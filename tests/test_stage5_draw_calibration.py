from __future__ import annotations

import numpy as np
import pandas as pd

import src.cli as cli
from src.backtesting.stage5_draw_calibration import (
    apply_confidence_calibration,
    apply_draw_multiplier,
    paired_official_bookmaker,
    run_stage5_draw_calibration,
    tune_draw_multiplier,
)


def test_draw_calibration_probabilities_sum_to_one() -> None:
    probabilities = np.array([[0.25, 0.20, 0.55], [0.40, 0.30, 0.30]])

    adjusted = apply_draw_multiplier(probabilities, 1.4)

    assert np.all(adjusted >= 0)
    assert np.allclose(adjusted.sum(axis=1), 1.0)


def test_confidence_calibration_has_no_negative_probabilities() -> None:
    probabilities = np.array([[0.25, 0.20, 0.55], [0.40, 0.30, 0.30]])
    buckets = pd.Series(["0.50-0.60", "0.40-0.50"])

    adjusted = apply_confidence_calibration(probabilities, buckets, {"0.50-0.60": 0.7, "0.40-0.50": 1.2})

    assert np.all(adjusted >= 0)
    assert np.allclose(adjusted.sum(axis=1), 1.0)


def test_draw_multiplier_tuning_uses_prior_years_only() -> None:
    predictions = _synthetic_predictions([2014, 2018, 2022], rows_per_year=24)
    paired = paired_official_bookmaker(predictions, [2014, 2018, 2022])
    prior_2022 = paired[paired["tournament_year"].lt(2022)]

    _, tuning = tune_draw_multiplier(prior_2022, 2022, grid=[0.8, 1.0, 1.2])
    selected = pd.DataFrame(tuning)

    assert set(selected["tuning_source_years"].unique()) == {"2014,2018"}
    assert not selected["tuning_source_years"].astype(str).str.contains("2022").any()


def test_stage5_draw_calibration_creates_outputs(tmp_path) -> None:
    predictions = _synthetic_predictions([2014, 2018, 2022], rows_per_year=24)
    predictions.to_csv(tmp_path / "stage5_match_predictions.csv", index=False)

    result = run_stage5_draw_calibration(
        years=[2014, 2018, 2022],
        n_bootstrap=25,
        output_dir=tmp_path,
        report_path=tmp_path / "stage5_draw_calibration_report.md",
        decision_path=tmp_path / "stage5_draw_calibration_decision.md",
    )

    assert (tmp_path / "stage5_draw_calibration_metrics.csv").exists()
    assert (tmp_path / "stage5_draw_calibration_predictions.csv").exists()
    assert (tmp_path / "stage5_draw_calibration_significance.csv").exists()
    assert (tmp_path / "stage5_draw_calibration_tuning.csv").exists()
    assert result.report_path.exists()
    assert result.decision_path.exists()
    assert result.stage5_achieved is False

    selected = result.tuning[result.tuning["selected"].astype(bool)]
    selected_2022 = selected[
        selected["test_year"].eq(2022)
        & selected["parameter"].astype(str).eq("draw_multiplier")
    ]
    assert not selected_2022.empty
    assert set(selected_2022["tuning_source_years"].unique()) == {"2014,2018"}
    assert not result.predictions[["home_prob", "draw_prob", "away_prob"]].lt(0).any().any()


def test_stage5_draw_calibration_cli_parser() -> None:
    args = cli.build_parser().parse_args(["stage5-draw-calibration", "--years", "2014,2018,2022", "--n-bootstrap", "500"])

    assert args.command == "stage5-draw-calibration"
    assert args.years == "2014,2018,2022"
    assert args.n_bootstrap == 500


def _synthetic_predictions(years: list[int], *, rows_per_year: int) -> pd.DataFrame:
    rows = []
    outcomes = [2, 1, 0]
    for year in years:
        for idx in range(rows_per_year):
            actual_class = outcomes[idx % 3]
            match_id = f"{year}_{idx}"
            official = np.array([0.24, 0.18, 0.58], dtype=float)
            bookmaker = np.array([0.25, 0.30, 0.45], dtype=float)
            if actual_class == 1:
                bookmaker = np.array([0.30, 0.40, 0.30], dtype=float)
            elif actual_class == 0:
                official = np.array([0.48, 0.20, 0.32], dtype=float)
                bookmaker = np.array([0.55, 0.25, 0.20], dtype=float)
            rows.append(_prediction_row(match_id, year, actual_class, "official_model", official))
            rows.append(_prediction_row(match_id, year, actual_class, "bookmaker_odds_only", bookmaker))
    return pd.DataFrame(rows)


def _prediction_row(match_id: str, year: int, actual_class: int, model_name: str, probs_away_draw_home: np.ndarray) -> dict[str, object]:
    predicted_class = int(probs_away_draw_home.argmax())
    return {
        "match_id": match_id,
        "date": f"{year}-06-01",
        "tournament_year": year,
        "stage": "group_stage",
        "home_team": f"Home {match_id}",
        "away_team": f"Away {match_id}",
        "actual_result": {0: "away_win", 1: "draw", 2: "home_win"}[actual_class],
        "actual_class": actual_class,
        "model_name": model_name,
        "feature_set": "fixture",
        "baseline_type": "fixture",
        "home_prob": probs_away_draw_home[2],
        "draw_prob": probs_away_draw_home[1],
        "away_prob": probs_away_draw_home[0],
        "predicted_result": {0: "away_win", 1: "draw", 2: "home_win"}[predicted_class],
        "confidence": float(probs_away_draw_home.max()),
        "log_loss": 0.0,
        "brier": 0.0,
        "rps": 0.0,
        "data_source_notes": "fixture",
    }
