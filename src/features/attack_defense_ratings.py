from __future__ import annotations

import pandas as pd

from src.features.pi_ratings import PI_RATING_COLUMNS, compute_pi_ratings


ATTACK_DEFENSE_RATING_COLUMNS = PI_RATING_COLUMNS


def compute_attack_defense_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper around the simplified pi-style attack/defense ratings."""
    return compute_pi_ratings(matches)
