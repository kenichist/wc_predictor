import pandas as pd

from src.sources.external_data_validation import validate_external_data


def test_external_data_validation_report_outputs_sections_and_metrics(tmp_path) -> None:
    fifa_path = tmp_path / "fifa_rankings.csv"
    elo_path = tmp_path / "world_football_elo.csv"
    train_path = tmp_path / "match_training_dataset_advanced.parquet"
    wc_path = tmp_path / "worldcup_2026_prediction_input_advanced.parquet"
    report_csv = tmp_path / "external_data_validation_report.csv"
    report_md = tmp_path / "external_data_validation_report.md"

    teams = [
        {"home_team": "United States", "away_team": "France"},
        {"home_team": "South Korea", "away_team": "Iran"},
    ]
    pd.DataFrame(teams).to_parquet(train_path, index=False)
    pd.DataFrame(teams).to_parquet(wc_path, index=False)
    pd.DataFrame(
        [
            {"date": "2024-12-19", "team": "USA", "rank": 18, "points": 1630},
            {"date": "2024-12-19", "team": "United States", "rank": 18, "points": 1630},
            {"date": "2024-12-19", "team": "France", "rank": 2, "points": 1850},
            {"date": "2024-12-19", "team": "Korea Republic", "rank": 23, "points": 1580},
            {"date": "2024-12-19", "team": "IR Iran", "rank": 20, "points": 1600},
        ]
    ).to_csv(fifa_path, index=False)
    pd.DataFrame(
        [
            {"date": "2024-12-31", "team": "United States", "elo": 1780},
            {"date": "2024-12-31", "team": "France", "elo": 2098},
            {"date": "2024-12-31", "team": "South Korea", "elo": 1820},
            {"date": "2024-12-31", "team": "Iran", "elo": 1810},
        ]
    ).to_csv(elo_path, index=False)
    cfg = {
        "external_sources": {
            "fifa_rankings": {"output_path": str(fifa_path)},
            "world_football_elo": {"output_path": str(elo_path)},
        },
        "paths": {
            "match_training_dataset_advanced_parquet": str(train_path),
            "worldcup_2026_prediction_input_advanced": str(wc_path),
            "external_data_validation_report_csv": str(report_csv),
            "external_data_validation_report_md": str(report_md),
        },
    }

    report = validate_external_data(cfg)

    fifa = report[report["dataset"].eq("fifa_rankings")].iloc[0]
    elo = report[report["dataset"].eq("world_football_elo")].iloc[0]
    assert fifa["duplicate_date_team_rows"] == 1
    assert fifa["worldcup_2026_team_coverage"] == 1.0
    assert elo["historical_team_coverage"] == 1.0
    assert report_csv.exists()
    markdown = report_md.read_text(encoding="utf-8")
    assert "## File Existence" in markdown
    assert "## Coverage" in markdown
    assert "## Warnings And Next Actions" in markdown
