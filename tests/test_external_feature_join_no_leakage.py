import pandas as pd

from src.sources.fifa_rankings import FIFA_RANKING_COLUMNS, add_fifa_ranking_features
from src.sources.world_football_elo import WORLD_FOOTBALL_ELO_COLUMNS, add_world_football_elo_features


def test_external_feature_joins_prevent_same_day_leakage(tmp_path) -> None:
    fifa_path = tmp_path / "fifa_rankings.csv"
    elo_path = tmp_path / "world_football_elo.csv"
    pd.DataFrame(
        [
            {"date": "2024-01-01", "team": "USA", "rank": 20, "points": 1500},
            {"date": "2024-02-01", "team": "United States", "rank": 1, "points": 1900},
            {"date": "2024-01-01", "team": "Mexico", "rank": 15, "points": 1510},
        ]
    ).to_csv(fifa_path, index=False)
    pd.DataFrame(
        [
            {"date": "2024-01-01", "team": "USA", "elo": 1750},
            {"date": "2024-02-01", "team": "United States", "elo": 2200},
            {"date": "2024-01-01", "team": "Mexico", "elo": 1780},
        ]
    ).to_csv(elo_path, index=False)
    matches = pd.DataFrame([{"date": "2024-02-01", "home_team": "United States", "away_team": "Mexico"}])

    fifa = add_fifa_ranking_features(matches, fifa_path)
    elo = add_world_football_elo_features(matches, elo_path)

    assert fifa.loc[0, "home_fifa_rank"] == 20
    assert fifa.loc[0, "home_fifa_points"] == 1500
    assert elo.loc[0, "home_external_elo"] == 1750


def test_missing_external_files_create_null_columns(tmp_path) -> None:
    matches = pd.DataFrame([{"date": "2024-02-01", "home_team": "United States", "away_team": "Mexico"}])

    fifa = add_fifa_ranking_features(matches, tmp_path / "missing_fifa.csv")
    elo = add_world_football_elo_features(matches, tmp_path / "missing_elo.csv")

    assert fifa[FIFA_RANKING_COLUMNS].isna().all().all()
    assert elo[WORLD_FOOTBALL_ELO_COLUMNS].isna().all().all()
