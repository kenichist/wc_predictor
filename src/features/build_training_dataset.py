from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import config_path, ensure_configured_directories, load_config
from src.features.elo import ELO_COLUMNS, compute_elo_features
from src.features.match_weights import compute_match_weights
from src.features.rolling_team_form import ROLLING_FEATURE_COLUMNS, compute_rolling_team_form
from src.io_utils import read_dataframe, write_dataframe
from src.normalize import add_match_outcome_columns, canonicalize_match_teams, classify_competition_type
from src.sources.fifa_rankings import add_fifa_ranking_features
from src.validation import CORE_COLUMNS, FEATURE_COLUMNS, REQUIRED_TRAINING_COLUMNS, TARGET_COLUMNS, add_missing_columns, validate_training_schema


logger = logging.getLogger(__name__)


def build_training_dataset(
    *,
    all_matches_path: str | Path | None = None,
    start_date: str = "2010-01-01",
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the scored training dataset and unscored World Cup prediction input."""
    cfg = config or load_config()
    ensure_configured_directories(cfg)
    input_path = Path(all_matches_path) if all_matches_path else config_path(cfg, "all_matches_clean")
    matches = read_dataframe(input_path)
    if matches.empty:
        raise ValueError(f"No matches found in {input_path}")

    matches = _prepare_base_matches(matches, cfg)
    form_features = compute_rolling_team_form(matches)
    elo_features = compute_elo_features(matches, config=cfg)

    form_output = config_path(cfg, "team_form_features")
    write_dataframe(form_features, form_output)

    enriched = _merge_feature_frame(matches, form_features, ROLLING_FEATURE_COLUMNS)
    enriched = _merge_feature_frame(enriched, elo_features, ELO_COLUMNS)
    enriched = add_fifa_ranking_features(enriched, config_path(cfg, "fifa_rankings"))
    enriched["match_weight"] = compute_match_weights(enriched, config=cfg)
    enriched = _add_flags_and_targets(enriched)
    enriched = add_missing_columns(enriched, REQUIRED_TRAINING_COLUMNS)
    enriched = enriched[REQUIRED_TRAINING_COLUMNS + [column for column in enriched.columns if column not in REQUIRED_TRAINING_COLUMNS]]

    scored = enriched["home_score"].notna() & enriched["away_score"].notna()
    date_filter = pd.to_datetime(enriched["date"], errors="coerce") >= pd.Timestamp(start_date)
    training = enriched[scored & date_filter].copy().reset_index(drop=True)
    validate_training_schema(training)

    prediction_input = enriched[enriched["is_world_cup"].fillna(False) & ~scored].copy().reset_index(drop=True)
    for target in TARGET_COLUMNS:
        prediction_input[target] = pd.NA
    prediction_input = add_missing_columns(prediction_input, REQUIRED_TRAINING_COLUMNS)

    write_dataframe(training, config_path(cfg, "match_training_dataset_parquet"))
    write_dataframe(training, config_path(cfg, "match_training_dataset_csv"))
    write_dataframe(prediction_input, config_path(cfg, "worldcup_2026_prediction_input"))
    write_dataframe(prediction_input, config_path(cfg, "worldcup_2026_prediction_input").with_suffix(".csv"))
    _write_data_dictionary(config_path(cfg, "data_dictionary"))

    logger.info("Final training dataset contains %s rows", len(training))
    logger.info("World Cup prediction input contains %s rows", len(prediction_input))
    return training, prediction_input

def _prepare_base_matches(matches: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    output = matches.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output = output.dropna(subset=["date", "home_team", "away_team"])
    output = canonicalize_match_teams(output, mapping_path=config_path(cfg, "team_name_mapping"))
    output["tournament"] = output.get("tournament", pd.Series("Other", index=output.index)).fillna("Other")
    output["competition_type"] = output.get("competition_type", pd.Series(index=output.index)).fillna(
        output["tournament"].map(classify_competition_type)
    )
    output = add_match_outcome_columns(output)
    output = add_missing_columns(output, CORE_COLUMNS)
    output = output.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
    return output


def _merge_feature_frame(base: pd.DataFrame, features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    output = base.drop(columns=[column for column in feature_columns if column in base.columns], errors="ignore").copy()
    merge_cols = ["match_id", *feature_columns]
    return output.merge(features[merge_cols], on="match_id", how="left")


def _add_flags_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["is_world_cup"] = output["competition_type"].eq("world_cup")
    output["is_qualifier"] = output["competition_type"].isin(["world_cup_qualifier", "continental_qualifier"])
    output["is_continental_tournament"] = output["competition_type"].eq("continental_championship")
    output["is_friendly"] = output["competition_type"].eq("friendly")

    known = output["home_score"].notna() & output["away_score"].notna()
    output["target_result_class"] = np.select(
        [
            known & (output["home_score"] < output["away_score"]),
            known & (output["home_score"] == output["away_score"]),
            known & (output["home_score"] > output["away_score"]),
        ],
        [0, 1, 2],
        default=np.nan,
    )
    output["target_home_goals"] = np.where(known, output["home_score"], np.nan)
    output["target_away_goals"] = np.where(known, output["away_score"], np.nan)
    return output


def _write_data_dictionary(path: Path) -> None:
    descriptions = {
        "match_id": "Stable source-derived match identifier.",
        "source": "Source selected after deduplication.",
        "date": "Match date or kickoff timestamp when available.",
        "result": "home_win, draw, or away_win for scored matches.",
        "match_weight": "Recency decay multiplied by tournament importance.",
        "target_result_class": "0=home loss, 1=draw, 2=home win.",
    }
    rows = ["# Data Dictionary", "", "All rolling and Elo features are generated using only matches with dates before the row's match date.", ""]
    for column in REQUIRED_TRAINING_COLUMNS:
        rows.append(f"- `{column}`: {descriptions.get(column, 'Pipeline-generated match, feature, or target column.')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
