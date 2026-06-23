from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.io_utils import write_dataframe
from src.normalize import normalize_team_name
from src.sources.tabular_external import read_external_table


logger = logging.getLogger(__name__)

WORLD_FOOTBALL_ELO_SCHEMA = ["date", "team", "elo", "source", "retrieved_at"]

WORLD_FOOTBALL_ELO_COLUMNS = [
    "home_external_elo",
    "away_external_elo",
    "external_elo_diff",
    "home_external_elo_change",
    "away_external_elo_change",
    "home_external_elo_date",
    "away_external_elo_date",
]

WORLD_FOOTBALL_ELO_MODEL_COLUMNS = [
    "home_external_elo",
    "away_external_elo",
    "external_elo_diff",
    "home_external_elo_change",
    "away_external_elo_change",
]


def load_world_football_elo(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        logger.warning("World Football Elo file not found at %s; external Elo features will be null", csv_path)
        return _empty_elo()
    try:
        raw = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Could not read World Football Elo from %s: %s", csv_path, exc)
        return _empty_elo()
    errors = validate_world_football_elo(raw)
    if errors:
        logger.warning("World Football Elo file %s is invalid: %s", csv_path, "; ".join(errors))
        return _empty_elo()
    return normalize_world_football_elo(raw)


def validate_world_football_elo(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = {"date", "team", "elo"}
    missing = sorted(required - set(df.columns))
    if missing:
        return [f"missing required columns: {missing}"]

    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        errors.append(f"date parse failures: {int(dates.isna().sum())}")

    teams = df["team"].astype("string").str.strip()
    if teams.isna().any() or teams.eq("").any():
        errors.append(f"empty team values: {int(teams.isna().sum() + teams.eq('').sum())}")

    elo = pd.to_numeric(df["elo"], errors="coerce")
    if elo.isna().any():
        errors.append(f"elo numeric parse failures: {int(elo.isna().sum())}")

    normalized_teams = df["team"].map(normalize_team_name)
    duplicate_count = int(pd.DataFrame({"date": dates, "team": normalized_teams}).duplicated(["date", "team"]).sum())
    if duplicate_count:
        errors.append(f"duplicate date/team rows after normalization: {duplicate_count}")
    return errors


def normalize_world_football_elo(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if "source" not in output.columns:
        output["source"] = "manual"
    if "retrieved_at" not in output.columns:
        output["retrieved_at"] = _utc_now()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output["team"] = output["team"].map(normalize_team_name)
    output["elo"] = pd.to_numeric(output["elo"], errors="coerce")
    output["source"] = output["source"].fillna("manual").astype(str)
    output["retrieved_at"] = output["retrieved_at"].fillna(_utc_now()).astype(str)
    output = output.dropna(subset=["date", "team", "elo"])
    output = output[WORLD_FOOTBALL_ELO_SCHEMA].drop_duplicates(["date", "team"], keep="last")
    return output.sort_values(["team", "date"], kind="stable").reset_index(drop=True)


def prepare_world_football_elo(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    raw = read_external_table(input_path)
    errors = validate_world_football_elo(raw)
    if errors:
        raise ValueError("Invalid World Football Elo input: " + "; ".join(errors))
    prepared = normalize_world_football_elo(raw)
    write_dataframe(prepared, output_path)
    return prepared


def join_world_football_elo_asof(matches: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    output = matches.copy()
    prepared = _ensure_prepared_elo(elo)
    if prepared.empty:
        return _add_null_elo_columns(output)

    prepared = prepared.sort_values(["team", "date"], kind="stable").reset_index(drop=True)
    prepared["elo_change"] = prepared.groupby("team")["elo"].diff()
    lookup = {team: group.reset_index(drop=True) for team, group in prepared.groupby("team")}

    dates = pd.to_datetime(output["date"], errors="coerce")
    home = output["home_team"].map(normalize_team_name)
    away = output["away_team"].map(normalize_team_name)
    home_rows = [_latest_team_row(lookup, team, match_date) for team, match_date in zip(home, dates)]
    away_rows = [_latest_team_row(lookup, team, match_date) for team, match_date in zip(away, dates)]

    output["home_external_elo"] = _row_value(home_rows, "elo", output.index)
    output["away_external_elo"] = _row_value(away_rows, "elo", output.index)
    output["external_elo_diff"] = output["home_external_elo"] - output["away_external_elo"]
    output["home_external_elo_change"] = _row_value(home_rows, "elo_change", output.index)
    output["away_external_elo_change"] = _row_value(away_rows, "elo_change", output.index)
    output["home_external_elo_date"] = _row_date(home_rows, output.index)
    output["away_external_elo_date"] = _row_date(away_rows, output.index)
    return output


def add_world_football_elo_features(matches: pd.DataFrame, elo_path: str | Path) -> pd.DataFrame:
    elo = load_world_football_elo(elo_path)
    return join_world_football_elo_asof(matches, elo)


def _empty_elo() -> pd.DataFrame:
    return pd.DataFrame(columns=WORLD_FOOTBALL_ELO_SCHEMA)


def _ensure_prepared_elo(elo: pd.DataFrame) -> pd.DataFrame:
    if elo.empty:
        return _empty_elo()
    required = {"date", "team", "elo"}
    if not required.issubset(elo.columns):
        logger.warning("World Football Elo missing columns %s; external Elo features will be null", sorted(required - set(elo.columns)))
        return _empty_elo()
    return normalize_world_football_elo(elo)


def _add_null_elo_columns(matches: pd.DataFrame) -> pd.DataFrame:
    output = matches.copy()
    for column in WORLD_FOOTBALL_ELO_MODEL_COLUMNS:
        output[column] = np.nan
    output["home_external_elo_date"] = pd.NaT
    output["away_external_elo_date"] = pd.NaT
    return output


def _latest_team_row(
    lookup: dict[str, pd.DataFrame],
    team: str | None,
    match_date: pd.Timestamp,
) -> pd.Series | None:
    if team is None or pd.isna(match_date):
        return None
    history = lookup.get(team)
    if history is None or history.empty:
        return None
    idx = history["date"].searchsorted(match_date, side="left") - 1
    if idx < 0:
        return None
    return history.iloc[idx]


def _row_value(rows: list[pd.Series | None], column: str, index: pd.Index) -> pd.Series:
    values = [np.nan if row is None or pd.isna(row[column]) else row[column] for row in rows]
    return pd.Series(values, index=index, dtype="float64")


def _row_date(rows: list[pd.Series | None], index: pd.Index) -> pd.Series:
    values = [pd.NaT if row is None else row["date"] for row in rows]
    return pd.to_datetime(pd.Series(values, index=index), errors="coerce")


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()
