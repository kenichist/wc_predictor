from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config_path, load_config
from src.io_utils import read_dataframe, write_dataframe


logger = logging.getLogger(__name__)


def is_placeholder_team_name(name: object) -> bool:
    if name is None or pd.isna(name):
        return True
    text = str(name).strip()
    if not text:
        return True
    upper = text.upper()
    if upper == "TBD":
        return True
    return bool(
        re.fullmatch(r"[123][A-L]", upper)
        or re.fullmatch(r"3[A-L](?:/[A-L])+", upper)
        or re.fullmatch(r"[WL]\d+", upper)
        or "/" in text
    )


def split_prediction_outputs(config: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    cfg = config or load_config()
    predictions_path = _latest_prediction_path(cfg)
    predictions = read_dataframe(predictions_path)
    prediction_input = _load_prediction_input(cfg)

    if "match_id" in prediction_input.columns:
        metadata_columns = [column for column in ["match_id", "stage", "group", "source"] if column in prediction_input.columns]
        predictions = predictions.merge(
            prediction_input[metadata_columns].drop_duplicates("match_id"),
            on="match_id",
            how="left",
        )

    home_placeholder = predictions["home_team"].map(is_placeholder_team_name)
    away_placeholder = predictions["away_team"].map(is_placeholder_team_name)
    placeholder_mask = home_placeholder | away_placeholder
    actual_mask = ~placeholder_mask
    group_mask = actual_mask & predictions.get("group", pd.Series(pd.NA, index=predictions.index)).notna()

    group_stage = predictions[group_mask].copy().reset_index(drop=True)
    actual = predictions[actual_mask].copy().reset_index(drop=True)
    placeholder = predictions[placeholder_mask].copy().reset_index(drop=True)
    dynamic = _load_dynamic_predictions(cfg)

    _write_pair(group_stage, config_path(cfg, "group_stage_predictions_parquet"), config_path(cfg, "group_stage_predictions_csv"))
    _write_pair(actual, config_path(cfg, "actual_team_match_predictions_parquet"), config_path(cfg, "actual_team_match_predictions_csv"))
    _write_pair(placeholder, config_path(cfg, "placeholder_fixture_predictions_parquet"), config_path(cfg, "placeholder_fixture_predictions_csv"))
    _write_pair(dynamic, config_path(cfg, "simulation_dynamic_predictions_parquet"), config_path(cfg, "simulation_dynamic_predictions_csv"))
    logger.info(
        "Split predictions from %s into %s group, %s actual, %s placeholder, %s dynamic rows",
        predictions_path,
        len(group_stage),
        len(actual),
        len(placeholder),
        len(dynamic),
    )
    return {
        "group_stage_predictions": group_stage,
        "actual_team_match_predictions": actual,
        "placeholder_fixture_predictions": placeholder,
        "simulation_dynamic_predictions": dynamic,
    }


def _latest_prediction_path(cfg: dict[str, Any]) -> Path:
    for key in ("ensemble_predictions", "calibrated_predictions", "match_predictions", "poisson_predictions"):
        path = config_path(cfg, key)
        if path.exists():
            return path
    raise FileNotFoundError("No prediction file found. Run train-model/build-ensemble first.")


def _load_prediction_input(cfg: dict[str, Any]) -> pd.DataFrame:
    for key in ("worldcup_2026_prediction_input_advanced", "worldcup_2026_prediction_input"):
        path = config_path(cfg, key)
        if path.exists():
            return read_dataframe(path)
    return pd.DataFrame(columns=["match_id", "stage", "group", "source"])


def _load_dynamic_predictions(cfg: dict[str, Any]) -> pd.DataFrame:
    path = config_path(cfg, "simulation_dynamic_predictions_parquet")
    if path.exists():
        return read_dataframe(path)
    columns = [
        "home_team",
        "away_team",
        "p_home_loss",
        "p_draw",
        "p_home_win",
        "source",
    ]
    return pd.DataFrame(columns=columns)


def _write_pair(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    write_dataframe(df, parquet_path)
    write_dataframe(df, csv_path)
