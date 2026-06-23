import pandas as pd

from src.sources.xg_interface import add_xg_rolling_features


def test_xg_rolling_uses_only_prior_team_rows(tmp_path) -> None:
    path = tmp_path / "team_xg_match_stats.csv"
    pd.DataFrame(
        [
            {"date": "2020-01-01", "team": "A", "opponent": "B", "xg_for": 2.0, "xg_against": 0.5, "shots_for": 10, "shots_against": 3},
            {"date": "2020-02-01", "team": "A", "opponent": "C", "xg_for": 9.0, "xg_against": 9.0, "shots_for": 99, "shots_against": 99},
        ]
    ).to_csv(path, index=False)
    matches = pd.DataFrame([{"match_id": "m2", "date": "2020-02-01", "home_team": "A", "away_team": "C"}])

    result = add_xg_rolling_features(matches, path)

    assert result.loc[0, "home_xg_for_avg_last_5"] == 2.0
    assert result.loc[0, "home_shots_for_avg_last_5"] == 10
