import pandas as pd

from src.sources.external_rating_acquisition import generate_external_rating_activation_report


def _rating_file(path, *, dataset: str) -> None:
    rows = []
    for idx in range(32):
        team = ["United States", "France", "Spain", "Argentina"][idx] if idx < 4 else f"Team {idx:02d}"
        if dataset == "fifa_rankings":
            rows.append({"date": "2024-12-19", "team": team, "rank": idx + 1, "points": 1900 - idx, "source": "test", "retrieved_at": "2026-01-01"})
        else:
            rows.append({"date": "2024-12-31", "team": team, "elo": 2100 - idx, "source": "test", "retrieved_at": "2026-01-01"})
    pd.DataFrame(rows).to_csv(path, index=False)


def _cfg(tmp_path):
    return {
        "external_sources": {
            "fifa_rankings": {"output_path": str(tmp_path / "fifa_rankings.csv")},
            "world_football_elo": {"output_path": str(tmp_path / "world_football_elo.csv")},
        },
        "paths": {
            "feature_null_rate_report_csv": str(tmp_path / "feature_null_rate_report.csv"),
            "ablation_results_csv": str(tmp_path / "ablation_results.csv"),
            "external_rating_activation_report_csv": str(tmp_path / "activation.csv"),
            "external_rating_activation_report_md": str(tmp_path / "activation.md"),
        },
    }


def test_activation_report_shows_inactive_when_files_missing(tmp_path) -> None:
    cfg = _cfg(tmp_path)

    report = generate_external_rating_activation_report(config=cfg)

    assert set(report["active"]) == {False}
    assert (tmp_path / "activation.csv").exists()
    assert "feature group is inactive" in (tmp_path / "activation.md").read_text(encoding="utf-8")


def test_activation_report_shows_active_when_mock_valid_files_exist(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    _rating_file(tmp_path / "fifa_rankings.csv", dataset="fifa_rankings")
    _rating_file(tmp_path / "world_football_elo.csv", dataset="world_football_elo")
    pd.DataFrame(
        [
            {"feature_name": "home_fifa_rank", "feature_group": "fifa_rankings", "null_rate": 0.2, "non_null_count": 8},
            {"feature_name": "away_fifa_rank", "feature_group": "fifa_rankings", "null_rate": 0.2, "non_null_count": 8},
            {"feature_name": "home_external_elo", "feature_group": "external_elo", "null_rate": 0.1, "non_null_count": 9},
            {"feature_name": "away_external_elo", "feature_group": "external_elo", "null_rate": 0.1, "non_null_count": 9},
        ]
    ).to_csv(tmp_path / "feature_null_rate_report.csv", index=False)
    pd.DataFrame(
        [
            {"feature_set": "v1_baseline", "accuracy": 0.50, "log_loss": 1.00, "identical_to_previous": False},
            {"feature_set": "v2_baseline_plus_fifa_rankings", "accuracy": 0.54, "log_loss": 0.95, "identical_to_previous": False},
            {"feature_set": "v3_plus_external_elo", "accuracy": 0.56, "log_loss": 0.93, "identical_to_previous": False},
        ]
    ).to_csv(tmp_path / "ablation_results.csv", index=False)

    report = generate_external_rating_activation_report(config=cfg, before_null_rates={"fifa_rankings": 1.0, "external_elo": 1.0})

    assert bool(report.set_index("dataset").loc["fifa_rankings", "active"]) is True
    assert bool(report.set_index("dataset").loc["world_football_elo", "active"]) is True
    assert report.set_index("dataset").loc["fifa_rankings", "null_rate_before"] == 1.0
