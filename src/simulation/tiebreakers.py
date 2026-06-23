from __future__ import annotations

import random

import pandas as pd


STANDING_COLUMNS = ["team", "points", "goal_diff", "goals_for", "wins", "goals_against"]


def rank_group(standings: pd.DataFrame, *, rng: random.Random | None = None) -> pd.DataFrame:
    """Rank group standings with FIFA-style primary tiebreakers and random fallback."""
    rng = rng or random.Random()
    ranked = standings.copy()
    ranked["_draw"] = [rng.random() for _ in range(len(ranked))]
    ranked = ranked.sort_values(
        ["points", "goal_diff", "goals_for", "wins", "_draw"],
        ascending=[False, False, False, False, True],
        kind="stable",
    ).drop(columns=["_draw"])
    return ranked.reset_index(drop=True)
