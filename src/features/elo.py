from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config
from src.normalize import classify_competition_type


ELO_COLUMNS = ["home_elo_pre_match", "away_elo_pre_match", "elo_diff"]


def compute_elo_features(
    matches: pd.DataFrame,
    *,
    default_elo: float | None = None,
    k_factor: float | None = None,
    home_advantage: float | None = None,
    k_multipliers: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compute simple pre-match Elo ratings without future leakage."""
    cfg = config or load_config()
    elo_cfg = cfg.get("elo", {})
    base_elo = float(default_elo if default_elo is not None else elo_cfg.get("default_elo", 1500))
    base_k = float(k_factor if k_factor is not None else elo_cfg.get("k_factor", 20))
    advantage = float(home_advantage if home_advantage is not None else elo_cfg.get("home_advantage", 50))
    multipliers = dict(elo_cfg.get("k_multipliers", {}))
    if k_multipliers:
        multipliers.update(k_multipliers)

    if matches.empty:
        return pd.DataFrame(columns=["match_id", *ELO_COLUMNS], index=matches.index)

    df = matches.copy()
    df["row_id"] = np.arange(len(df))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "row_id"], kind="stable")
    ratings: defaultdict[str, float] = defaultdict(lambda: base_elo)
    rows: list[dict[str, Any]] = []

    for _, group in df.groupby("date", sort=True, dropna=False):
        pending_deltas: defaultdict[str, float] = defaultdict(float)
        for row in group.itertuples(index=False):
            row_dict = row._asdict()
            home_team = row_dict.get("home_team")
            away_team = row_dict.get("away_team")
            home_pre = float(ratings[home_team]) if home_team else base_elo
            away_pre = float(ratings[away_team]) if away_team else base_elo
            rows.append(
                {
                    "row_id": row_dict["row_id"],
                    "match_id": row_dict.get("match_id"),
                    "home_elo_pre_match": home_pre,
                    "away_elo_pre_match": away_pre,
                    "elo_diff": home_pre - away_pre,
                }
            )

            home_score = row_dict.get("home_score")
            away_score = row_dict.get("away_score")
            if not home_team or not away_team or pd.isna(home_score) or pd.isna(away_score):
                continue
            expected_home = _expected_home_score(home_pre, away_pre, 0 if bool(row_dict.get("neutral")) else advantage)
            actual_home = _actual_home_score(float(home_score), float(away_score))
            competition_type = row_dict.get("competition_type") or classify_competition_type(row_dict.get("tournament"))
            k = base_k * float(multipliers.get(str(competition_type), multipliers.get("other", 1.0)))
            delta = k * (actual_home - expected_home)
            pending_deltas[home_team] += delta
            pending_deltas[away_team] -= delta
        for team, delta in pending_deltas.items():
            ratings[team] += delta

    features = pd.DataFrame(rows).sort_values("row_id", kind="stable").drop(columns=["row_id"])
    features.index = matches.index
    return features


def _expected_home_score(home_elo: float, away_elo: float, home_advantage: float) -> float:
    adjusted_home = home_elo + home_advantage
    return 1.0 / (1.0 + 10 ** ((away_elo - adjusted_home) / 400.0))


def _actual_home_score(home_goals: float, away_goals: float) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals == away_goals:
        return 0.5
    return 0.0
