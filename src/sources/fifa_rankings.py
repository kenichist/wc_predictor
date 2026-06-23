from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.io_utils import write_dataframe
from src.normalize import normalize_team_name
from src.sources.tabular_external import read_external_table


logger = logging.getLogger(__name__)

FIFA_RANKINGS_SCHEMA = ["date", "team", "rank", "points", "source", "retrieved_at"]

FIFA_RANKING_COLUMNS = [
    "home_fifa_rank",
    "away_fifa_rank",
    "fifa_rank_diff",
    "home_fifa_points",
    "away_fifa_points",
    "fifa_points_diff",
    "home_fifa_rank_change",
    "away_fifa_rank_change",
    "home_fifa_points_change",
    "away_fifa_points_change",
    "home_fifa_ranking_date",
    "away_fifa_ranking_date",
]

FIFA_RANKING_MODEL_COLUMNS = [
    "home_fifa_rank",
    "away_fifa_rank",
    "fifa_rank_diff",
    "home_fifa_points",
    "away_fifa_points",
    "fifa_points_diff",
    "home_fifa_rank_change",
    "away_fifa_rank_change",
    "home_fifa_points_change",
    "away_fifa_points_change",
]


def load_fifa_rankings(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        logger.warning("FIFA rankings file not found at %s; ranking features will be null", csv_path)
        return _empty_rankings()
    try:
        raw = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Could not read FIFA rankings from %s: %s", csv_path, exc)
        return _empty_rankings()
    errors = validate_fifa_rankings(raw)
    if errors:
        logger.warning("FIFA rankings file %s is invalid: %s", csv_path, "; ".join(errors))
        return _empty_rankings()
    return normalize_fifa_rankings(raw)


def validate_fifa_rankings(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = {"date", "team", "rank"}
    missing = sorted(required - set(df.columns))
    if missing:
        return [f"missing required columns: {missing}"]

    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any():
        errors.append(f"date parse failures: {int(dates.isna().sum())}")

    teams = df["team"].astype("string").str.strip()
    if teams.isna().any() or teams.eq("").any():
        errors.append(f"empty team values: {int(teams.isna().sum() + teams.eq('').sum())}")

    ranks = pd.to_numeric(df["rank"], errors="coerce")
    if ranks.isna().any():
        errors.append(f"rank numeric parse failures: {int(ranks.isna().sum())}")
    integer_like = ranks.dropna().map(lambda value: float(value).is_integer())
    if not integer_like.all():
        errors.append("rank values must be integer-like")

    if "points" in df.columns:
        point_values = df["points"]
        present_points = ~_optional_numeric_missing_mask(point_values)
        points = pd.to_numeric(_clean_optional_numeric(point_values), errors="coerce")
        parse_failures = int((present_points & points.isna()).sum())
        if parse_failures:
            errors.append(f"points numeric parse failures: {parse_failures}")

    normalized_teams = df["team"].map(normalize_team_name)
    duplicate_count = int(pd.DataFrame({"date": dates, "team": normalized_teams}).duplicated(["date", "team"]).sum())
    if duplicate_count:
        errors.append(f"duplicate date/team rows after normalization: {duplicate_count}")
    return errors


def normalize_fifa_rankings(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if "points" not in output.columns:
        output["points"] = np.nan
    if "source" not in output.columns:
        output["source"] = "manual"
    if "retrieved_at" not in output.columns:
        output["retrieved_at"] = _utc_now()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output["team"] = output["team"].map(normalize_team_name)
    output["rank"] = pd.to_numeric(output["rank"], errors="coerce").round().astype("Int64")
    output["points"] = pd.to_numeric(_clean_optional_numeric(output["points"]), errors="coerce")
    output["source"] = output["source"].fillna("manual").astype(str)
    output["retrieved_at"] = output["retrieved_at"].fillna(_utc_now()).astype(str)
    output = output.dropna(subset=["date", "team", "rank"])
    output = output[FIFA_RANKINGS_SCHEMA].drop_duplicates(["date", "team"], keep="last")
    return output.sort_values(["team", "date"], kind="stable").reset_index(drop=True)


def prepare_fifa_rankings(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    raw = read_external_table(input_path)
    errors = validate_fifa_rankings(raw)
    if errors:
        raise ValueError("Invalid FIFA rankings input: " + "; ".join(errors))
    prepared = normalize_fifa_rankings(raw)
    write_dataframe(prepared, output_path)
    return prepared


def join_fifa_rankings_asof(matches: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    output = matches.copy()
    prepared = _ensure_prepared_rankings(rankings)
    if prepared.empty:
        return _add_null_fifa_columns(output)

    prepared = prepared.sort_values(["team", "date"], kind="stable").reset_index(drop=True)
    prepared["rank_change"] = prepared.groupby("team")["rank"].diff()
    prepared["points_change"] = prepared.groupby("team")["points"].diff()
    lookup = {team: group.reset_index(drop=True) for team, group in prepared.groupby("team")}

    dates = pd.to_datetime(output["date"], errors="coerce")
    home = output["home_team"].map(normalize_team_name)
    away = output["away_team"].map(normalize_team_name)

    home_rows = [_latest_team_row(lookup, team, match_date) for team, match_date in zip(home, dates)]
    away_rows = [_latest_team_row(lookup, team, match_date) for team, match_date in zip(away, dates)]

    output["home_fifa_rank"] = _row_value(home_rows, "rank", output.index)
    output["away_fifa_rank"] = _row_value(away_rows, "rank", output.index)
    output["fifa_rank_diff"] = output["home_fifa_rank"] - output["away_fifa_rank"]
    output["home_fifa_points"] = _row_value(home_rows, "points", output.index)
    output["away_fifa_points"] = _row_value(away_rows, "points", output.index)
    output["fifa_points_diff"] = output["home_fifa_points"] - output["away_fifa_points"]
    output["home_fifa_rank_change"] = _row_value(home_rows, "rank_change", output.index)
    output["away_fifa_rank_change"] = _row_value(away_rows, "rank_change", output.index)
    output["home_fifa_points_change"] = _row_value(home_rows, "points_change", output.index)
    output["away_fifa_points_change"] = _row_value(away_rows, "points_change", output.index)
    output["home_fifa_ranking_date"] = _row_date(home_rows, output.index)
    output["away_fifa_ranking_date"] = _row_date(away_rows, output.index)
    return output


def add_fifa_ranking_features(matches: pd.DataFrame, rankings_path: str | Path) -> pd.DataFrame:
    rankings = load_fifa_rankings(rankings_path)
    return join_fifa_rankings_asof(matches, rankings)


def _empty_rankings() -> pd.DataFrame:
    return pd.DataFrame(columns=FIFA_RANKINGS_SCHEMA)


def _ensure_prepared_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    if rankings.empty:
        return _empty_rankings()
    required = {"date", "team", "rank"}
    if not required.issubset(rankings.columns):
        logger.warning("FIFA rankings missing columns %s; ranking features will be null", sorted(required - set(rankings.columns)))
        return _empty_rankings()
    return normalize_fifa_rankings(rankings)


def _clean_optional_numeric(values: pd.Series) -> pd.Series:
    return values.astype("string").str.replace(",", "", regex=False).str.strip().replace(
        {
            "": pd.NA,
            "-": pd.NA,
            "NA": pd.NA,
            "N/A": pd.NA,
            "na": pd.NA,
            "n/a": pd.NA,
            "null": pd.NA,
            "None": pd.NA,
            "nan": pd.NA,
        }
    )


def _optional_numeric_missing_mask(values: pd.Series) -> pd.Series:
    cleaned = _clean_optional_numeric(values)
    return cleaned.isna()


def _add_null_fifa_columns(matches: pd.DataFrame) -> pd.DataFrame:
    output = matches.copy()
    for column in FIFA_RANKING_MODEL_COLUMNS:
        output[column] = np.nan
    output["home_fifa_ranking_date"] = pd.NaT
    output["away_fifa_ranking_date"] = pd.NaT
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
