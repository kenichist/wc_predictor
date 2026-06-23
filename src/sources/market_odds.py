from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.normalize import normalize_team_name


logger = logging.getLogger(__name__)

MARKET_ODDS_SCHEMA = ["date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds", "bookmaker", "source", "updated_at"]
MARKET_FEATURE_COLUMNS = ["home_odds", "draw_odds", "away_odds", "market_home_prob", "market_draw_prob", "market_away_prob"]
MARKET_JOIN_COLUMNS = ["_market_date", "_market_home_team", "_market_away_team"]
MARKET_PROBABILITY_COLUMNS = ["market_home_prob", "market_draw_prob", "market_away_prob"]
ODDS_CONVERSION_METHODS = ["multiplicative", "additive", "power", "shin", "favorite_longshot"]
MARKET_JOIN_DATE_TOLERANCE_DAYS = 1


def decimal_odds_to_probabilities(odds: pd.DataFrame, *, method: str = "multiplicative") -> pd.DataFrame:
    """Convert decimal 1X2 odds into normalized implied probabilities."""
    required = ["home_odds", "draw_odds", "away_odds"]
    missing = [column for column in required if column not in odds.columns]
    if missing:
        raise ValueError(f"Missing odds columns: {missing}")
    method = method.strip().lower()
    if method not in ODDS_CONVERSION_METHODS:
        raise ValueError(f"Unsupported odds conversion method: {method}")
    output = odds.copy()
    for column in required:
        output[column] = pd.to_numeric(output[column], errors="coerce")
        output.loc[output[column] <= 0, column] = np.nan
    raw = np.column_stack(
        [
            1.0 / output["home_odds"].to_numpy(dtype=float),
            1.0 / output["draw_odds"].to_numpy(dtype=float),
            1.0 / output["away_odds"].to_numpy(dtype=float),
        ]
    )
    probabilities = convert_decimal_odds_array(raw, method=method)
    output["market_home_prob"] = probabilities[:, 0]
    output["market_draw_prob"] = probabilities[:, 1]
    output["market_away_prob"] = probabilities[:, 2]
    return output


def convert_decimal_odds_array(raw_implied: np.ndarray, *, method: str) -> np.ndarray:
    raw = np.asarray(raw_implied, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError("Expected a two-dimensional raw implied probability array with 3 columns")
    method = method.strip().lower()
    if method == "multiplicative":
        return _normalize_probabilities(raw)
    if method == "additive":
        return _additive_normalization(raw)
    if method == "power":
        return np.vstack([_power_normalization_row(row) for row in raw])
    if method == "shin":
        return np.vstack([_shin_normalization_row(row) for row in raw])
    if method == "favorite_longshot":
        return _normalize_probabilities(np.power(np.clip(raw, 1e-12, None), 1.05))
    raise ValueError(f"Unsupported odds conversion method: {method}")


def load_market_odds(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        logger.warning("Market odds file not found at %s; market features will be null", csv_path)
        return _empty_market_odds()
    try:
        raw = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        logger.warning("Could not read market odds from %s: %s", csv_path, exc)
        return _empty_market_odds()
    if raw.empty:
        return _empty_market_odds(columns=list(raw.columns) or MARKET_ODDS_SCHEMA)
    missing = [column for column in ["date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds"] if column not in raw.columns]
    if missing:
        logger.warning("Market odds file missing columns %s; market features will be null", missing)
        return _empty_market_odds(columns=list(raw.columns))
    return normalize_market_odds(raw)


def normalize_market_odds(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in MARKET_ODDS_SCHEMA:
        if column not in output.columns:
            output[column] = pd.NA
    output["date"] = pd.to_datetime(output["date"], errors="coerce", format="mixed").dt.normalize()
    output["home_team"] = output["home_team"].map(normalize_team_name)
    output["away_team"] = output["away_team"].map(normalize_team_name)
    output = decimal_odds_to_probabilities(output)
    output = output.dropna(subset=["date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds"])
    output = output[MARKET_ODDS_SCHEMA + ["market_home_prob", "market_draw_prob", "market_away_prob"]]
    output = output.drop_duplicates(["date", "home_team", "away_team"], keep="last")
    return output.sort_values(["date", "home_team", "away_team"], kind="stable").reset_index(drop=True)


def add_market_odds_features(matches: pd.DataFrame, odds_path: str | Path) -> pd.DataFrame:
    output = matches.copy()
    output = _clear_market_columns(output)
    odds = load_market_odds(odds_path)
    for column in MARKET_FEATURE_COLUMNS:
        output[column] = np.nan
    if odds.empty:
        return output

    left = _market_join_keys(output)
    right = _market_join_keys(odds)
    right = pd.concat([right, odds[MARKET_FEATURE_COLUMNS].reset_index(drop=True)], axis=1)
    right = right.dropna(subset=MARKET_JOIN_COLUMNS).drop_duplicates(MARKET_JOIN_COLUMNS, keep="last")
    merged = left.merge(right, on=MARKET_JOIN_COLUMNS, how="left")
    for column in MARKET_FEATURE_COLUMNS:
        output[column] = pd.to_numeric(merged[column], errors="coerce").values
    output = _fill_near_date_market_matches(output, left, right)
    return output


def _market_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    index = df.index
    return pd.DataFrame(
        {
            "_market_date": pd.to_datetime(df.get("date", pd.Series(pd.NaT, index=index)), errors="coerce", format="mixed").dt.normalize(),
            "_market_home_team": df.get("home_team", pd.Series(pd.NA, index=index)).map(normalize_team_name),
            "_market_away_team": df.get("away_team", pd.Series(pd.NA, index=index)).map(normalize_team_name),
        },
        index=index,
    )


def _clear_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[column for column in MARKET_FEATURE_COLUMNS if column in df.columns], errors="ignore").copy()


def _empty_market_odds(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=columns or [*MARKET_ODDS_SCHEMA, "market_home_prob", "market_draw_prob", "market_away_prob"])


def _fill_near_date_market_matches(output: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    missing = output[MARKET_PROBABILITY_COLUMNS].isna().all(axis=1)
    if not bool(missing.any()):
        return output
    keyed_right = right.dropna(subset=MARKET_JOIN_COLUMNS).copy()
    if keyed_right.empty:
        return output
    keyed_right["_market_date"] = pd.to_datetime(keyed_right["_market_date"], errors="coerce", format="mixed").dt.normalize()
    for idx in output.index[missing]:
        key = left.loc[idx]
        match_date = pd.Timestamp(key["_market_date"]) if pd.notna(key["_market_date"]) else pd.NaT
        if pd.isna(match_date):
            continue
        candidates = keyed_right[
            keyed_right["_market_home_team"].eq(key["_market_home_team"])
            & keyed_right["_market_away_team"].eq(key["_market_away_team"])
        ].copy()
        if candidates.empty:
            continue
        candidates["_date_delta"] = (candidates["_market_date"] - match_date).abs().dt.days
        candidates = candidates[candidates["_date_delta"].le(MARKET_JOIN_DATE_TOLERANCE_DAYS)]
        if candidates.empty:
            continue
        best_delta = candidates["_date_delta"].min()
        best = candidates[candidates["_date_delta"].eq(best_delta)]
        if len(best) != 1:
            continue
        for column in MARKET_FEATURE_COLUMNS:
            output.loc[idx, column] = pd.to_numeric(best.iloc[0][column], errors="coerce")
    return output


def _additive_normalization(raw: np.ndarray) -> np.ndarray:
    margin = raw.sum(axis=1, keepdims=True) - 1.0
    adjusted = raw - (margin / raw.shape[1])
    return _normalize_probabilities(adjusted)


def _power_normalization_row(raw: np.ndarray) -> np.ndarray:
    row = np.clip(np.asarray(raw, dtype=float), 1e-12, None)
    if not np.isfinite(row).all():
        return np.full(3, np.nan)
    low, high = 0.01, 10.0
    for _ in range(80):
        mid = (low + high) / 2.0
        value = float(np.power(row, mid).sum())
        if value > 1.0:
            low = mid
        else:
            high = mid
    return _normalize_probabilities(np.power(row, (low + high) / 2.0).reshape(1, -1))[0]


def _shin_normalization_row(raw: np.ndarray) -> np.ndarray:
    row = np.clip(np.asarray(raw, dtype=float), 1e-12, None)
    if not np.isfinite(row).all():
        return np.full(3, np.nan)
    if row.sum() <= 1.0:
        return _normalize_probabilities(row.reshape(1, -1))[0]

    def probabilities(z: float) -> np.ndarray:
        z = min(max(float(z), 1e-9), 0.999999)
        total = row.sum()
        values = (np.sqrt(z * z + (4.0 * (1.0 - z) * row * row / total)) - z) / (2.0 * (1.0 - z))
        return np.clip(values, 1e-12, None)

    low, high = 0.0, 0.999999
    low_sum = probabilities(low).sum()
    high_sum = probabilities(high).sum()
    if not (low_sum >= 1.0 >= high_sum):
        return _normalize_probabilities(probabilities(0.0).reshape(1, -1))[0]
    for _ in range(80):
        mid = (low + high) / 2.0
        if probabilities(mid).sum() > 1.0:
            low = mid
        else:
            high = mid
    return _normalize_probabilities(probabilities((low + high) / 2.0).reshape(1, -1))[0]


def _normalize_probabilities(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float)
    output = np.where(np.isfinite(output), output, np.nan)
    output = np.clip(output, 1e-12, None)
    row_sums = np.nansum(output, axis=1, keepdims=True)
    row_sums[row_sums <= 0] = np.nan
    return output / row_sums
