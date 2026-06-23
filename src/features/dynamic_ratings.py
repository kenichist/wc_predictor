from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.normalize import classify_competition_type


DYNAMIC_RATING_COLUMNS = ["home_dynamic_rating_pre", "away_dynamic_rating_pre", "dynamic_rating_diff"]


def compute_dynamic_ratings(matches: pd.DataFrame, *, config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    rating_cfg = cfg.get("dynamic_ratings", {})
    base = float(rating_cfg.get("base_rating", 1500))
    k_factor = float(rating_cfg.get("k_factor", 24))
    home_advantage = float(rating_cfg.get("home_advantage", 55))
    goal_diff_cap = float(rating_cfg.get("goal_diff_cap", 4))
    multipliers = dict(rating_cfg.get("k_multipliers", {}))

    df = matches.copy()
    if df.empty:
        return pd.DataFrame(columns=["match_id", *DYNAMIC_RATING_COLUMNS], index=matches.index)
    df["row_id"] = np.arange(len(df))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "row_id"], kind="stable")
    ratings: defaultdict[str, float] = defaultdict(lambda: base)
    rows: list[dict[str, Any]] = []

    for _, day_matches in df.groupby("date", sort=True, dropna=False):
        pending: defaultdict[str, float] = defaultdict(float)
        for row in day_matches.itertuples(index=False):
            rec = row._asdict()
            home = rec.get("home_team")
            away = rec.get("away_team")
            home_pre = float(ratings[home]) if home else base
            away_pre = float(ratings[away]) if away else base
            rows.append(
                {
                    "row_id": rec["row_id"],
                    "match_id": rec.get("match_id"),
                    "home_dynamic_rating_pre": home_pre,
                    "away_dynamic_rating_pre": away_pre,
                    "dynamic_rating_diff": home_pre - away_pre,
                }
            )
            hs = rec.get("home_score")
            away_score = rec.get("away_score")
            if not home or not away or pd.isna(hs) or pd.isna(away_score):
                continue
            advantage = 0.0 if bool(rec.get("neutral")) else home_advantage
            expected = 1.0 / (1.0 + 10 ** (((away_pre) - (home_pre + advantage)) / 400.0))
            actual = _actual_score(float(hs), float(away_score))
            goal_diff = min(abs(float(hs) - float(away_score)), goal_diff_cap)
            gd_multiplier = 1.0 if goal_diff <= 1 else np.log(goal_diff + 1.0)
            comp = rec.get("competition_type") or classify_competition_type(rec.get("tournament"))
            k = k_factor * float(multipliers.get(str(comp), multipliers.get("other", 1.0))) * gd_multiplier
            delta = k * (actual - expected)
            pending[home] += delta
            pending[away] -= delta
        for team, delta in pending.items():
            ratings[team] += delta
    result = pd.DataFrame(rows).sort_values("row_id", kind="stable").drop(columns=["row_id"])
    result.index = matches.index
    return result


def _actual_score(home_score: float, away_score: float) -> float:
    if home_score > away_score:
        return 1.0
    if home_score == away_score:
        return 0.5
    return 0.0
