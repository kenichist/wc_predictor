import pandas as pd

from src.features.null_rate_report import generate_feature_null_rate_report


def test_feature_null_rate_report_flags_fully_null_groups(tmp_path) -> None:
    cfg = {
        "paths": {
            "match_training_dataset_advanced_parquet": str(tmp_path / "train.parquet"),
            "match_training_dataset_parquet": str(tmp_path / "fallback.parquet"),
            "worldcup_2026_prediction_input_advanced": str(tmp_path / "pred.parquet"),
            "feature_null_rate_report_csv": str(tmp_path / "nulls.csv"),
            "feature_null_rate_report_md": str(tmp_path / "nulls.md"),
        }
    }
    pd.DataFrame(
        [
            {"match_id": "m1", "date": "2024-01-01", "home_team": "A", "away_team": "B", "home_fifa_rank": None, "elo_diff": 10, "target_result_class": 2},
            {"match_id": "m2", "date": "2024-01-02", "home_team": "C", "away_team": "D", "home_fifa_rank": None, "elo_diff": 20, "target_result_class": 0},
        ]
    ).to_parquet(cfg["paths"]["match_training_dataset_advanced_parquet"], index=False)
    pd.DataFrame([{"match_id": "p1", "date": "2026-01-01", "home_team": "A", "away_team": "C", "home_fifa_rank": None, "elo_diff": 5}]).to_parquet(
        cfg["paths"]["worldcup_2026_prediction_input_advanced"], index=False
    )

    detail, summary = generate_feature_null_rate_report(cfg)

    fifa = detail[detail["feature_name"].eq("home_fifa_rank")].iloc[0]
    assert fifa["null_rate"] == 1.0
    assert "fifa_rankings" in set(summary["feature_group"])
    assert (tmp_path / "nulls.csv").exists()
