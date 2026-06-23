import pandas as pd

from src.sources.fifa_rankings import add_fifa_ranking_features


def test_fifa_rankings_use_only_publications_before_match_date(tmp_path) -> None:
    path = tmp_path / "fifa_rankings.csv"
    pd.DataFrame(
        [
            {"date": "2020-01-01", "team": "USA", "rank": 20, "points": 1500},
            {"date": "2020-02-01", "team": "USA", "rank": 1, "points": 1900},
            {"date": "2020-01-01", "team": "Mexico", "rank": 15, "points": 1550},
        ]
    ).to_csv(path, index=False)
    matches = pd.DataFrame([{"match_id": "m1", "date": "2020-02-01", "home_team": "United States", "away_team": "Mexico"}])

    result = add_fifa_ranking_features(matches, path)

    assert result.loc[0, "home_fifa_rank"] == 20
    assert result.loc[0, "home_fifa_points"] == 1500
