import random

import pandas as pd

from src.simulation.tiebreakers import rank_group


def test_group_tiebreakers_sort_points_goal_diff_goals_for() -> None:
    standings = pd.DataFrame(
        [
            {"team": "A", "points": 6, "goal_diff": 2, "goals_for": 3, "wins": 2, "goals_against": 1},
            {"team": "B", "points": 6, "goal_diff": 3, "goals_for": 4, "wins": 2, "goals_against": 1},
            {"team": "C", "points": 3, "goal_diff": 0, "goals_for": 2, "wins": 1, "goals_against": 2},
        ]
    )

    ranked = rank_group(standings, rng=random.Random(1))

    assert ranked.iloc[0]["team"] == "B"
