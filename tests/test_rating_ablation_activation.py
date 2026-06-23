import pandas as pd

from src.experiments.ablation_runner import run_ablation
from src.features.null_rate_report import generate_feature_null_rate_report


def test_feature_null_report_detects_activated_rating_groups(tmp_path) -> None:
    train_path = tmp_path / "train.parquet"
    pred_path = tmp_path / "pred.parquet"
    cfg = {
        "paths": {
            "match_training_dataset_advanced_parquet": str(train_path),
            "match_training_dataset_parquet": str(tmp_path / "fallback.parquet"),
            "worldcup_2026_prediction_input_advanced": str(pred_path),
            "feature_null_rate_report_csv": str(tmp_path / "nulls.csv"),
            "feature_null_rate_report_md": str(tmp_path / "nulls.md"),
        }
    }
    pd.DataFrame(
        [
            {"match_id": "m1", "date": "2024-01-01", "home_fifa_rank": 1, "away_fifa_rank": 2, "home_external_elo": 2100, "away_external_elo": 2000},
            {"match_id": "m2", "date": "2024-01-02", "home_fifa_rank": 3, "away_fifa_rank": 4, "home_external_elo": 1900, "away_external_elo": 1800},
        ]
    ).to_parquet(train_path, index=False)
    pd.DataFrame(
        [{"match_id": "p1", "date": "2026-06-11", "home_fifa_rank": 5, "away_fifa_rank": 6, "home_external_elo": 1880, "away_external_elo": 1770}]
    ).to_parquet(pred_path, index=False)

    _, summary = generate_feature_null_rate_report(cfg)

    fifa = summary[summary["feature_group"].eq("fifa_rankings")].iloc[0]
    elo = summary[summary["feature_group"].eq("external_elo")].iloc[0]
    assert fifa["average_null_rate"] < 1.0
    assert fifa["usable_columns"] > 0
    assert elo["average_null_rate"] < 1.0
    assert elo["usable_columns"] > 0


def test_ablation_runner_counts_usable_fifa_and_external_elo_columns(tmp_path) -> None:
    train_path = tmp_path / "train.parquet"
    rows = []
    for i in range(60):
        year = 2020 if i < 36 else 2023
        rows.append(
            {
                "date": f"{year}-01-{(i % 28) + 1:02d}",
                "target_result_class": i % 3,
                "elo_diff": float((i % 9) - 4),
                "match_weight": 1.0,
                "home_fifa_rank": float((i % 20) + 1),
                "away_fifa_rank": float((i % 18) + 2),
                "fifa_rank_diff": float((i % 20) - (i % 18)),
                "home_fifa_points": 1600.0 + i,
                "away_fifa_points": 1550.0 + i,
                "fifa_points_diff": 50.0,
                "home_external_elo": 1800.0 + i,
                "away_external_elo": 1750.0 + i,
                "external_elo_diff": 50.0,
            }
        )
    pd.DataFrame(rows).to_parquet(train_path, index=False)
    cfg = {
        "paths": {
            "match_training_dataset_advanced_parquet": str(train_path),
            "match_training_dataset_parquet": str(tmp_path / "fallback.parquet"),
            "ablation_results_csv": str(tmp_path / "ablation_results.csv"),
            "ablation_results_md": str(tmp_path / "ablation_results.md"),
            "final_model_recommendation_md": str(tmp_path / "final_model_recommendation.md"),
        },
        "modeling": {
            "train_end": "2022-01-01",
            "validation_start": "2022-01-01",
            "validation_end": "2025-01-01",
            "test_start": "2025-01-01",
            "random_seed": 42,
        },
    }

    results = run_ablation(cfg)

    assert list(results.columns) == [
        "feature_set",
        "num_features",
        "new_columns_added",
        "usable_new_columns",
        "null_new_columns",
        "constant_new_columns",
        "identical_to_previous",
        "accuracy",
        "log_loss",
        "brier_score",
        "ranked_probability_score",
        "calibration_error",
        "warning",
    ]
    v2 = results[results["feature_set"].eq("v2_baseline_plus_fifa_rankings")].iloc[0]
    v3 = results[results["feature_set"].eq("v3_plus_external_elo")].iloc[0]
    assert v2["usable_new_columns"] > 0
    assert v3["usable_new_columns"] > 0
    assert v3["num_features"] > v2["num_features"]
