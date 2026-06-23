import pandas as pd

from src.features.match_weights import compute_match_weights


def test_recent_world_cup_weight_exceeds_old_friendly() -> None:
    df = pd.DataFrame(
        [
            {"date": "2023-12-01", "competition_type": "world_cup", "tournament": "FIFA World Cup"},
            {"date": "2010-01-01", "competition_type": "friendly", "tournament": "Friendly"},
        ]
    )

    weights = compute_match_weights(df, as_of_date="2024-01-01")

    assert weights.iloc[0] > weights.iloc[1]
