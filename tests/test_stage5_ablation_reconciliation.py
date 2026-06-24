from __future__ import annotations

import pandas as pd

import src.cli as cli
from src.backtesting.stage5_ablation_reconciliation import run_stage5_ablation_reconciliation


def test_stage5_ablation_reconciliation_creates_report_and_csv(tmp_path) -> None:
    final_report = tmp_path / "final_model_recommendation.md"
    stage5_report = tmp_path / "stage5_benchmark_report.md"
    stage5_decision = tmp_path / "stage5_research_decision.md"
    _write_final_recommendation_fixture(final_report)
    stage5_report.write_text("# Stage 5 Benchmark Report\n", encoding="utf-8")
    stage5_decision.write_text("Decision: **Stage 5 not yet achieved**\n", encoding="utf-8")
    _write_stage5_fixtures(tmp_path)

    result = run_stage5_ablation_reconciliation(
        output_dir=tmp_path,
        final_model_recommendation_path=final_report,
        stage5_benchmark_report_path=stage5_report,
        stage5_decision_path=stage5_decision,
        report_path=tmp_path / "stage5_ablation_reconciliation_report.md",
    )

    assert result.csv_path.exists()
    assert result.report_path.exists()
    assert result.stage5_achieved is False
    assert {"metric_source", "evaluation_protocol", "comparable_to_stage5"}.issubset(result.reconciliation.columns)
    ablation = result.reconciliation[result.reconciliation["model_label"].eq("ablation_core_football_only")].iloc[0]
    stage5 = result.reconciliation[result.reconciliation["model_label"].eq("official_model_weighted_average")].iloc[0]
    assert ablation["log_loss"] == 0.894662
    assert stage5["row_count"] == 128
    report_text = result.report_path.read_text(encoding="utf-8")
    assert "Ablation is feature-selection validation" in report_text
    assert "Stage 5 is a held-out World Cup benchmark" in report_text
    assert "should not be used to claim Stage 5" in report_text


def test_stage5_reconcile_ablation_cli_parser() -> None:
    args = cli.build_parser().parse_args(["stage5-reconcile-ablation", "--output-dir", "data/backtests"])

    assert args.command == "stage5-reconcile-ablation"
    assert args.output_dir == "data/backtests"


def _write_final_recommendation_fixture(path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Final Model Recommendation",
                "",
                "## Ablation Summary",
                "",
                "| feature_set | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | warning |",
                "| --- | --- | --- | --- | --- | --- | --- |",
                "| core_football_only | 0.584946 | 0.894662 | 0.525878 | 0.175830 | 0.035662 | unknown |",
                "| v1_baseline | 0.589247 | 0.904491 | 0.531969 | 0.177840 | 0.028365 | unknown |",
            ]
        ),
        encoding="utf-8",
    )


def _write_stage5_fixtures(path) -> None:
    metrics = pd.DataFrame(
        [
            {
                "model_name": "official_model",
                "feature_set": "football_only_ensemble/core_football_only",
                "baseline_type": "official_model",
                "test_year": 2018,
                "n_matches": 64,
                "n_skipped": 0,
                "accuracy": 0.50,
                "log_loss": 0.98,
                "brier": 0.59,
                "rps": 0.21,
                "ece": 0.07,
                "mean_confidence": 0.52,
            },
            {
                "model_name": "official_model",
                "feature_set": "football_only_ensemble/core_football_only",
                "baseline_type": "official_model",
                "test_year": 2022,
                "n_matches": 64,
                "n_skipped": 0,
                "accuracy": 0.48,
                "log_loss": 1.08,
                "brier": 0.64,
                "rps": 0.23,
                "ece": 0.12,
                "mean_confidence": 0.55,
            },
        ]
    )
    metrics.to_csv(path / "stage5_benchmark_metrics.csv", index=False)

    predictions = pd.DataFrame(
        [
            {"model_name": "official_model", "match_id": "m1"},
            {"model_name": "official_model", "match_id": "m2"},
        ]
    )
    predictions.to_csv(path / "stage5_match_predictions.csv", index=False)

    significance = pd.DataFrame(
        [
            {
                "candidate_model": "official_model",
                "baseline_model": "bookmaker_odds_only",
                "metric": "log_loss",
                "mean_delta": 0.04,
                "ci_lower": -0.01,
                "ci_upper": 0.08,
                "statistically_meaningful": False,
                "n_matches": 126,
            }
        ]
    )
    significance.to_csv(path / "stage5_significance_tests.csv", index=False)
