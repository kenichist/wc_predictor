from __future__ import annotations

import numpy as np
import pandas as pd

import src.cli as cli
from src.backtesting.stage5_calibrated_market import (
    blend_probabilities,
    paired_official_bookmaker,
    run_stage5_calibrated_market,
    select_alpha_for_year,
)


def test_alpha_blend_probabilities_sum_to_one() -> None:
    model = np.array([[0.2, 0.3, 0.5], [0.5, 0.2, 0.3]])
    market = np.array([[0.4, 0.3, 0.3], [0.2, 0.3, 0.5]])

    blended = blend_probabilities(model, market, 0.35)

    assert np.all(blended >= 0)
    assert np.allclose(blended.sum(axis=1), 1.0)


def test_selected_alpha_uses_only_prior_years() -> None:
    predictions = _synthetic_stage5_predictions([2014, 2018, 2022], rows_per_year=12)
    paired = paired_official_bookmaker(predictions, [2014, 2018, 2022])

    selected_alpha, tuning = select_alpha_for_year(paired, 2022, [0.0, 0.5, 1.0])

    assert selected_alpha == 0.0
    assert set(tuning["tuning_years"].unique()) == {"2014,2018"}
    assert not tuning["tuning_years"].astype(str).str.contains("2022").any()


def test_stacker_skips_when_insufficient_training_data(tmp_path) -> None:
    predictions = _synthetic_stage5_predictions([2014], rows_per_year=12)
    predictions.to_csv(tmp_path / "stage5_match_predictions.csv", index=False)

    result = run_stage5_calibrated_market(
        years=[2014],
        n_bootstrap=25,
        include_stacker=True,
        output_dir=tmp_path,
        report_path=tmp_path / "stage5_calibrated_market_report.md",
        decision_path=tmp_path / "stage5_calibrated_market_decision.md",
    )

    stacker_metrics = result.metrics[result.metrics["model_name"].eq("stacked_model_market")]
    assert not stacker_metrics.empty
    assert int(stacker_metrics.iloc[0]["n_matches"]) == 0
    assert "insufficient prior" in stacker_metrics.iloc[0]["notes"]


def test_market_value_significance_delta_direction_and_outputs(tmp_path) -> None:
    predictions = _synthetic_stage5_predictions([2014, 2018, 2022], rows_per_year=18)
    predictions.to_csv(tmp_path / "stage5_match_predictions.csv", index=False)

    result = run_stage5_calibrated_market(
        years=[2014, 2018, 2022],
        n_bootstrap=50,
        alpha_grid_step=0.5,
        include_stacker=True,
        output_dir=tmp_path,
        report_path=tmp_path / "stage5_calibrated_market_report.md",
        decision_path=tmp_path / "stage5_calibrated_market_decision.md",
    )

    assert (tmp_path / "stage5_calibrated_market_metrics.csv").exists()
    assert (tmp_path / "stage5_calibrated_market_predictions.csv").exists()
    assert (tmp_path / "stage5_market_value_significance.csv").exists()
    assert (tmp_path / "stage5_alpha_tuning.csv").exists()
    assert (tmp_path / "stage5_stacker_coefficients.csv").exists()
    assert result.report_path.exists()
    assert result.decision_path.exists()

    alpha_2022 = result.alpha_tuning[result.alpha_tuning["test_year"].eq(2022)]
    assert not alpha_2022.empty
    assert not alpha_2022["tuning_years"].astype(str).str.contains("2022").any()

    row = result.significance[
        result.significance["candidate_model"].eq("market_blend_tuned")
        & result.significance["baseline_model"].eq("bookmaker_odds_only")
        & result.significance["metric"].eq("log_loss")
    ].iloc[0]
    assert row["mean_delta"] == 0.0
    assert row["statistically_meaningful"] in {False, np.False_}


def test_stage5_calibrated_market_cli_parser() -> None:
    args = cli.build_parser().parse_args(
        [
            "stage5-calibrated-market",
            "--years",
            "2014,2018,2022",
            "--n-bootstrap",
            "1000",
            "--alpha-grid-step",
            "0.05",
            "--include-stacker",
        ]
    )

    assert args.command == "stage5-calibrated-market"
    assert args.years == "2014,2018,2022"
    assert args.n_bootstrap == 1000
    assert args.alpha_grid_step == 0.05
    assert args.include_stacker is True


def _synthetic_stage5_predictions(years: list[int], *, rows_per_year: int) -> pd.DataFrame:
    rows = []
    outcomes = [2, 1, 0]
    for year in years:
        for idx in range(rows_per_year):
            actual = outcomes[idx % len(outcomes)]
            match_id = f"{year}_{idx}"
            official = np.array([0.50, 0.25, 0.25], dtype=float)
            bookmaker = np.array([0.20, 0.25, 0.55], dtype=float)
            if actual == 1:
                bookmaker = np.array([0.25, 0.50, 0.25], dtype=float)
            elif actual == 0:
                bookmaker = np.array([0.55, 0.25, 0.20], dtype=float)
            rows.append(_prediction_row(match_id, year, actual, "official_model", official))
            rows.append(_prediction_row(match_id, year, actual, "bookmaker_odds_only", bookmaker))
    return pd.DataFrame(rows)


def _prediction_row(match_id: str, year: int, actual: int, model_name: str, probs_away_draw_home: np.ndarray) -> dict[str, object]:
    return {
        "match_id": match_id,
        "date": f"{year}-06-01",
        "tournament_year": year,
        "stage": "group_stage",
        "home_team": f"Home {match_id}",
        "away_team": f"Away {match_id}",
        "actual_result": {0: "away_win", 1: "draw", 2: "home_win"}[actual],
        "actual_class": actual,
        "model_name": model_name,
        "feature_set": "fixture",
        "baseline_type": "fixture",
        "home_prob": probs_away_draw_home[2],
        "draw_prob": probs_away_draw_home[1],
        "away_prob": probs_away_draw_home[0],
        "predicted_result": "home_win",
        "confidence": float(probs_away_draw_home.max()),
        "log_loss": 0.0,
        "brier": 0.0,
        "rps": 0.0,
        "data_source_notes": "fixture",
    }
