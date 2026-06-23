import pandas as pd

from src.sources.fifa_rankings import FIFA_RANKING_COLUMNS, add_fifa_ranking_features


def test_fifa_rankings_join_latest_prior_row_and_changes(tmp_path) -> None:
    path = tmp_path / "fifa_rankings.csv"
    pd.DataFrame(
        [
            {"date": "2024-10-01", "team": "USA", "rank": 20, "points": 1500},
            {"date": "2024-12-19", "team": "USA", "rank": 10, "points": 1600},
            {"date": "2024-10-01", "team": "Congo DR", "rank": 60, "points": 1300},
            {"date": "2024-12-19", "team": "DR Congo", "rank": 55, "points": 1320},
        ]
    ).to_csv(path, index=False)
    matches = pd.DataFrame(
        [
            {"date": "2024-12-18", "home_team": "United States", "away_team": "DR Congo"},
            {"date": "2024-12-20", "home_team": "United States", "away_team": "DR Congo"},
        ]
    )

    result = add_fifa_ranking_features(matches, path)

    assert result.loc[0, "home_fifa_rank"] == 20
    assert result.loc[1, "home_fifa_rank"] == 10
    assert result.loc[1, "home_fifa_rank_change"] == -10
    assert result.loc[1, "away_fifa_points_change"] == 20


def test_fifa_rankings_missing_file_returns_null_columns(tmp_path) -> None:
    matches = pd.DataFrame([{"date": "2024-12-20", "home_team": "USA", "away_team": "IR Iran"}])

    result = add_fifa_ranking_features(matches, tmp_path / "missing.csv")

    assert set(FIFA_RANKING_COLUMNS).issubset(result.columns)
    assert result[FIFA_RANKING_COLUMNS].isna().all().all()
