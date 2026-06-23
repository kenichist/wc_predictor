from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config_path, ensure_configured_directories, load_config
from src.io_utils import download_file, stable_id, write_dataframe
from src.normalize import add_match_outcome_columns, canonicalize_match_teams, classify_competition_type


logger = logging.getLogger(__name__)


def download_historical_results(config: dict[str, Any] | None = None, *, force: bool = False) -> dict[str, Path]:
    """Download raw historical international match files."""
    cfg = config or load_config()
    ensure_configured_directories(cfg)
    source_cfg = cfg["sources"]["historical"]
    path_pairs = {
        "results": ("results_url", "historical_results_raw"),
        "shootouts": ("shootouts_url", "historical_shootouts_raw"),
        "goalscorers": ("goalscorers_url", "historical_goalscorers_raw"),
    }
    output: dict[str, Path] = {}
    retry_count = int(cfg.get("sources", {}).get("football_data", {}).get("retry_count", 3))
    timeout = int(cfg.get("sources", {}).get("football_data", {}).get("timeout_seconds", 30))
    for name, (url_key, path_key) in path_pairs.items():
        path = config_path(cfg, path_key)
        if path.exists() and not force:
            logger.info("Using existing raw historical %s file at %s", name, path)
            output[name] = path
            continue
        output[name] = download_file(
            source_cfg[url_key],
            path,
            timeout_seconds=timeout,
            retry_count=retry_count,
            sleep_seconds=1.0,
        )
    return output


def load_historical_results(path: str | Path | None = None, config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    raw_path = Path(path) if path else config_path(cfg, "historical_results_raw")
    return pd.read_csv(raw_path)


def clean_historical_results(
    df: pd.DataFrame | None = None,
    *,
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Clean the martj42 international results dataset into the common match schema."""
    cfg = config or load_config()
    if df is None:
        df = load_historical_results(input_path, cfg)

    output = df.copy()
    output.columns = [column.strip().lower() for column in output.columns]
    output = output.rename(columns={"home_team": "home_team", "away_team": "away_team"})
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output["home_score"] = pd.to_numeric(output["home_score"], errors="coerce")
    output["away_score"] = pd.to_numeric(output["away_score"], errors="coerce")
    output = output.dropna(subset=["date", "home_team", "away_team"])
    output = output[output["home_team"].astype(str).str.strip().ne("")]
    output = output[output["away_team"].astype(str).str.strip().ne("")]

    if "neutral" in output.columns:
        output["neutral"] = output["neutral"].map(_parse_bool).fillna(False)
    else:
        output["neutral"] = False

    mapping = mapping_path if mapping_path is not None else config_path(cfg, "team_name_mapping")
    output = canonicalize_match_teams(output, mapping_path=mapping)
    output["source"] = "historical_results"
    output["tournament"] = output.get("tournament", pd.Series("Other", index=output.index)).fillna("Other")
    output["competition_type"] = output["tournament"].map(classify_competition_type)
    output["stage"] = pd.NA
    output["group"] = pd.NA
    output["venue"] = pd.NA
    output["status"] = "FINISHED"
    output["last_updated"] = pd.NA
    output["match_id"] = [
        stable_id("historical", row.date.strftime("%Y-%m-%d"), row.home_team, row.away_team, row.home_score, row.away_score, row.tournament)
        for row in output.itertuples(index=False)
    ]
    output = add_match_outcome_columns(output)

    columns = [
        "match_id",
        "source",
        "date",
        "home_team",
        "away_team",
        "neutral",
        "tournament",
        "competition_type",
        "stage",
        "group",
        "country",
        "city",
        "venue",
        "home_score",
        "away_score",
        "result",
        "home_win",
        "draw",
        "away_win",
        "goal_diff",
        "total_goals",
        "status",
        "last_updated",
    ]
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    output = output[columns].sort_values(["date", "home_team", "away_team"], kind="stable").reset_index(drop=True)

    target = Path(output_path) if output_path else config_path(cfg, "historical_matches_clean")
    write_dataframe(output, target)
    logger.info("Cleaned %s historical match rows", len(output))
    return output


def _parse_bool(value: object) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}
