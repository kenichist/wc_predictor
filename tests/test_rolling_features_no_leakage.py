import pandas as pd

from src.features.rolling_team_form import compute_rolling_team_form


def test_rolling_features_use_only_prior_matches() -> None:
    matches = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2020-01-01",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "neutral": True,
                "tournament": "Friendly",
            },
            {
                "match_id": "m2",
                "date": "2020-02-01",
                "home_team": "A",
                "away_team": "C",
                "home_score": 2,
                "away_score": 2,
                "neutral": True,
                "tournament": "Friendly",
            },
            {
                "match_id": "m3",
                "date": "2020-03-01",
                "home_team": "C",
                "away_team": "A",
                "home_score": 0,
                "away_score": 3,
                "neutral": True,
                "tournament": "Friendly",
            },
        ]
    )

    features = compute_rolling_team_form(matches)

    first = features[features["match_id"] == "m1"].iloc[0]
    second = features[features["match_id"] == "m2"].iloc[0]

    assert first["home_matches_last_5"] == 0
    assert pd.isna(first["home_win_rate_last_5"])
    assert second["home_matches_last_5"] == 1
    assert second["home_win_rate_last_5"] == 1.0
    assert second["home_goals_for_avg_last_5"] == 1.0
    assert second["away_matches_last_5"] == 0
