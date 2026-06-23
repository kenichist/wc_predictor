import pandas as pd

from src.features.pi_ratings import compute_pi_ratings


def test_pi_ratings_emit_pre_match_values() -> None:
    matches = pd.DataFrame(
        [
            {"match_id": "m1", "date": "2020-01-01", "home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0, "competition_type": "friendly"},
            {"match_id": "m2", "date": "2020-02-01", "home_team": "A", "away_team": "C", "home_score": 1, "away_score": 1, "competition_type": "friendly"},
        ]
    )

    features = compute_pi_ratings(matches)

    assert features.loc[0, "home_overall_rating_pre"] == 0
    assert features.loc[1, "home_attack_rating_pre"] != 0
    assert "home_attack_vs_away_defense" in features.columns
