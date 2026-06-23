from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import config_path, ensure_configured_directories, load_config
from src.features.dynamic_ratings import DYNAMIC_RATING_COLUMNS, compute_dynamic_ratings
from src.features.internal_historical_elo import (
    INTERNAL_HISTORICAL_ELO_COLUMNS,
    compute_internal_historical_elo,
    load_internal_elo_match_base,
    write_internal_elo_reconstruction_report,
    write_internal_historical_elo_history,
)
from src.features.pi_ratings import PI_RATING_COLUMNS, compute_pi_ratings
from src.io_utils import read_dataframe, write_dataframe
from src.reports.external_feature_activation_debug import write_external_feature_activation_debug_report
from src.reports.market_odds_join_debug import write_market_odds_join_debug_report
from src.sources.fifa_rankings import add_fifa_ranking_features
from src.sources.injury_interface import INJURY_FEATURE_COLUMNS, add_injury_features
from src.sources.player_stats_interface import add_squad_player_features
from src.sources.market_odds import add_market_odds_features
from src.sources.world_football_elo import add_world_football_elo_features
from src.sources.xg_interface import add_xg_rolling_features


logger = logging.getLogger(__name__)


def build_advanced_features(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or load_config()
    ensure_configured_directories(cfg)
    training_path = config_path(cfg, "match_training_dataset_parquet")
    prediction_path = config_path(cfg, "worldcup_2026_prediction_input")
    if not training_path.exists() or not prediction_path.exists():
        raise FileNotFoundError("Baseline feature files are missing. Run `python -m src.cli build-features` first.")

    training = read_dataframe(training_path)
    prediction = read_dataframe(prediction_path)
    combined = pd.concat(
        [training.assign(_advanced_split="training"), prediction.assign(_advanced_split="prediction")],
        ignore_index=True,
        sort=False,
    )
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce", format="mixed")

    try:
        rating_base = load_internal_elo_match_base(cfg)
    except FileNotFoundError:
        rating_base = combined
    rating_base["date"] = pd.to_datetime(rating_base["date"], errors="coerce", format="mixed")
    dynamic = compute_dynamic_ratings(rating_base, config=cfg)
    pi = compute_pi_ratings(rating_base)
    internal_elo = compute_internal_historical_elo(rating_base, config=cfg)
    write_internal_historical_elo_history(internal_elo.history, cfg)
    rating_features = (
        rating_base[["match_id"]]
        .merge(dynamic, on="match_id", how="left")
        .merge(pi, on="match_id", how="left")
        .merge(internal_elo.features[["match_id", *INTERNAL_HISTORICAL_ELO_COLUMNS]], on="match_id", how="left")
    )

    combined = combined.drop(columns=[*DYNAMIC_RATING_COLUMNS, *PI_RATING_COLUMNS, *INTERNAL_HISTORICAL_ELO_COLUMNS], errors="ignore")
    combined = combined.merge(rating_features, on="match_id", how="left")
    combined = add_fifa_ranking_features(combined, config_path(cfg, "fifa_rankings"))
    combined = add_world_football_elo_features(combined, config_path(cfg, "world_football_elo"))
    combined = add_xg_rolling_features(combined, config_path(cfg, "team_xg_match_stats"))
    combined = add_squad_player_features(combined, config_path(cfg, "squad_player_features"))
    combined = add_injury_features(combined, config_path(cfg, "injuries_suspensions"))
    combined = add_market_features(combined, config_path(cfg, "betting_odds"))

    advanced_training = combined[combined["_advanced_split"].eq("training")].drop(columns=["_advanced_split"]).reset_index(drop=True)
    advanced_prediction = combined[combined["_advanced_split"].eq("prediction")].drop(columns=["_advanced_split"]).reset_index(drop=True)
    advanced_prediction = _refresh_prediction_injury_features(advanced_prediction, cfg)
    write_dataframe(advanced_training, config_path(cfg, "match_training_dataset_advanced_parquet"))
    write_dataframe(advanced_training, config_path(cfg, "match_training_dataset_advanced_csv"))
    write_dataframe(advanced_prediction, config_path(cfg, "worldcup_2026_prediction_input_advanced"))
    write_dataframe(advanced_prediction, config_path(cfg, "worldcup_2026_prediction_input_advanced").with_suffix(".csv"))
    write_internal_elo_reconstruction_report(config=cfg, result=internal_elo)
    write_external_feature_activation_debug_report(
        config=cfg,
        advanced_training=advanced_training,
        advanced_prediction=advanced_prediction,
    )
    write_market_odds_join_debug_report(config=cfg)
    logger.info("Advanced training dataset contains %s rows", len(advanced_training))
    logger.info("Advanced World Cup prediction input contains %s rows", len(advanced_prediction))
    logger.info("Run `python -m src.cli feature-null-report` to see which advanced feature groups are active")
    return advanced_training, advanced_prediction


def _refresh_prediction_injury_features(prediction: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    base = prediction.drop(columns=[column for column in INJURY_FEATURE_COLUMNS if column in prediction.columns], errors="ignore")
    repaired = add_injury_features(base.assign(_advanced_split="prediction"), config_path(cfg, "injuries_suspensions"))
    return repaired.drop(columns=["_advanced_split"], errors="ignore")


def add_market_features(matches: pd.DataFrame, odds_path: str | Path) -> pd.DataFrame:
    return add_market_odds_features(matches, odds_path)
