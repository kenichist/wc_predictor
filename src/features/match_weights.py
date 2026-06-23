from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.normalize import classify_competition_type


DEFAULT_MULTIPLIERS = {
    "world_cup": 1.50,
    "continental_championship": 1.25,
    "world_cup_qualifier": 1.15,
    "continental_qualifier": 1.10,
    "nations_league": 1.00,
    "friendly": 0.70,
    "other": 1.00,
}


def compute_match_weights(
    df: pd.DataFrame,
    *,
    as_of_date: str | pd.Timestamp | None = None,
    config: dict[str, Any] | None = None,
) -> pd.Series:
    cfg = config or load_config()
    weight_cfg = cfg.get("match_weights", {})
    decay_days = float(weight_cfg.get("recency_decay_days", 730))
    multipliers = DEFAULT_MULTIPLIERS | dict(weight_cfg.get("multipliers", {}))

    dates = pd.to_datetime(df["date"], errors="coerce")
    reference = pd.Timestamp(as_of_date) if as_of_date is not None else pd.Timestamp.today().normalize()
    if reference.tzinfo is not None:
        reference = reference.tz_convert(None)
    days_old = (reference - dates).dt.days.clip(lower=0)
    recency = np.exp(-days_old / decay_days)

    if "competition_type" in df.columns:
        competition_type = df["competition_type"].fillna("other")
    else:
        competition_type = df.get("tournament", pd.Series("other", index=df.index)).map(classify_competition_type)
    importance = competition_type.map(lambda value: multipliers.get(str(value), multipliers["other"]))
    return recency * importance
