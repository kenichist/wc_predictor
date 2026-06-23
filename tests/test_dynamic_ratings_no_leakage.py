import pandas as pd

from src.features.dynamic_ratings import compute_dynamic_ratings


def test_dynamic_ratings_are_pre_match_values() -> None:
    matches = pd.DataFrame(
        [
            {"match_id": "m1", "date": "2020-01-01", "home_team": "A", "away_team": "B", "home_score": 3, "away_score": 0, "neutral": True, "competition_type": "friendly"},
            {"match_id": "m2", "date": "2020-02-01", "home_team": "A", "away_team": "C", "home_score": 0, "away_score": 0, "neutral": True, "competition_type": "friendly"},
        ]
    )

    features = compute_dynamic_ratings(matches)

    first = features[features["match_id"].eq("m1")].iloc[0]
    second = features[features["match_id"].eq("m2")].iloc[0]
    assert first["home_dynamic_rating_pre"] == 1500
    assert second["home_dynamic_rating_pre"] > 1500
