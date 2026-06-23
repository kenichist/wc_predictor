import pandas as pd

from src.sources.world_football_elo import WORLD_FOOTBALL_ELO_COLUMNS, add_world_football_elo_features


def test_external_elo_join_latest_prior_row_and_changes(tmp_path) -> None:
    path = tmp_path / "world_football_elo.csv"
    pd.DataFrame(
        [
            {"date": "2024-01-01", "team": "Korea Republic", "elo": 1800},
            {"date": "2024-12-31", "team": "South Korea", "elo": 1825},
            {"date": "2024-01-01", "team": "IR Iran", "elo": 1750},
        ]
    ).to_csv(path, index=False)
    matches = pd.DataFrame(
        [
            {"date": "2024-12-31", "home_team": "South Korea", "away_team": "Iran"},
            {"date": "2025-01-01", "home_team": "South Korea", "away_team": "Iran"},
        ]
    )

    result = add_world_football_elo_features(matches, path)

    assert result.loc[0, "home_external_elo"] == 1800
    assert result.loc[1, "home_external_elo"] == 1825
    assert result.loc[1, "home_external_elo_change"] == 25


def test_external_elo_missing_file_returns_null_columns(tmp_path) -> None:
    matches = pd.DataFrame([{"date": "2024-12-20", "home_team": "Spain", "away_team": "France"}])

    result = add_world_football_elo_features(matches, tmp_path / "missing.csv")

    assert set(WORLD_FOOTBALL_ELO_COLUMNS).issubset(result.columns)
    assert result[WORLD_FOOTBALL_ELO_COLUMNS].isna().all().all()
