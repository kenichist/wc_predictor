from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config_path, load_config, resolve_project_path
from src.features.feature_sets import FEATURE_GROUPS
from src.io_utils import read_dataframe
from src.models.prediction_outputs import is_placeholder_team_name
from src.normalize import normalize_team_name
from src.sources.injury_interface import REQUIRED_AGGREGATE_INJURY_COLUMNS
from src.sources.player_stats_interface import PLAYER_BASE_COLUMNS, PLAYER_FEATURE_COLUMNS


def write_external_feature_activation_debug_report(
    *,
    config: dict[str, Any] | None = None,
    advanced_training: pd.DataFrame | None = None,
    advanced_prediction: pd.DataFrame | None = None,
) -> Path:
    cfg = config or load_config()
    squad_path = config_path(cfg, "squad_player_features")
    injury_path = config_path(cfg, "injuries_suspensions")
    market_path = config_path(cfg, "betting_odds")

    squad_df = _read_csv_if_exists(squad_path)
    injury_df = _read_csv_if_exists(injury_path)
    market_df = _read_csv_if_exists(market_path)
    prediction_df = advanced_prediction if advanced_prediction is not None else _read_prediction_input(cfg)
    training_df = advanced_training if advanced_training is not None else _read_training_input(cfg)
    generated = pd.concat([training_df, prediction_df], ignore_index=True, sort=False)

    prediction_teams = _match_teams(prediction_df)
    squad_teams = _external_teams(squad_df)
    injury_teams = _external_teams(injury_df)

    lines = [
        "# External Feature Activation Debug",
        "",
        "## Files",
        "",
        _file_line("squad_player_features", squad_path, squad_df),
        _file_line("injuries_suspensions", injury_path, injury_df),
        _file_line("market_odds", market_path, market_df),
        "",
        "## Squad Player Features",
        "",
        *_source_lines(
            source=squad_df,
            external_teams=squad_teams,
            prediction_teams=prediction_teams,
            required_missing=_squad_required_missing(squad_df),
            generated=generated,
            generated_columns=PLAYER_FEATURE_COLUMNS,
        ),
        "",
        "## Injuries And Suspensions",
        "",
        *_source_lines(
            source=injury_df,
            external_teams=injury_teams,
            prediction_teams=prediction_teams,
            required_missing=_injury_required_missing(injury_df),
            generated=generated,
            generated_columns=FEATURE_GROUPS["injuries"],
        ),
        "",
        "## Market Odds",
        "",
        f"- Configured path: `{market_path}`",
        f"- Rows: `{len(market_df)}`",
        f"- Detected columns: `{', '.join(market_df.columns.astype(str)) if not market_df.empty or len(market_df.columns) else 'none'}`",
        f"- Status: `{('inactive: file has zero rows' if market_df.empty else 'available')}`",
        "",
    ]
    path = _optional_config_path(cfg, "external_feature_activation_debug_md", "data/reports/external_feature_activation_debug.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _source_lines(
    *,
    source: pd.DataFrame,
    external_teams: set[str],
    prediction_teams: set[str],
    required_missing: list[str],
    generated: pd.DataFrame,
    generated_columns: list[str],
) -> list[str]:
    matched = sorted(external_teams & prediction_teams)
    unmatched_external = sorted(external_teams - prediction_teams)
    unmatched_prediction = sorted(prediction_teams - external_teams)
    non_null = _non_null_counts(generated, generated_columns)
    total_non_null = sum(non_null.values())
    reason = _fully_null_reason(source, required_missing, matched, total_non_null)
    return [
        f"- Detected columns: `{', '.join(source.columns.astype(str)) if len(source.columns) else 'none'}`",
        f"- Required columns missing: `{', '.join(required_missing) if required_missing else 'none'}`",
        f"- External teams: `{len(external_teams)}`",
        f"- World Cup prediction teams: `{len(prediction_teams)}`",
        f"- Matched teams: `{len(matched)}`",
        f"- Matched team names: `{', '.join(matched) if matched else 'none'}`",
        f"- Unmatched external teams: `{', '.join(unmatched_external) if unmatched_external else 'none'}`",
        f"- Unmatched prediction teams: `{', '.join(unmatched_prediction) if unmatched_prediction else 'none'}`",
        "- Generated non-null counts:",
        *_non_null_lines(non_null),
        f"- Fully null reason: `{reason}`",
    ]


def _file_line(label: str, path: Path, df: pd.DataFrame) -> str:
    return f"- `{label}` path: `{path}`; exists: `{path.exists()}`; rows: `{len(df)}`"


def _squad_required_missing(df: pd.DataFrame) -> list[str]:
    missing: list[str] = []
    if "team" not in df.columns:
        missing.append("team")
    if not any(column in df.columns for column in PLAYER_BASE_COLUMNS):
        missing.append("at least one squad numeric feature")
    return missing


def _injury_required_missing(df: pd.DataFrame) -> list[str]:
    required = ["date", "team", *REQUIRED_AGGREGATE_INJURY_COLUMNS]
    return [column for column in required if column not in df.columns]


def _fully_null_reason(source: pd.DataFrame, required_missing: list[str], matched: list[str], total_non_null: int) -> str:
    if total_non_null > 0:
        return "active"
    if source.empty:
        return "source file has zero rows or could not be read"
    if required_missing:
        return "required columns are missing"
    if not matched:
        return "no normalized teams matched prediction rows"
    return "generated columns are still fully null after join"


def _non_null_counts(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    return {column: int(df[column].notna().sum()) if column in df.columns else 0 for column in columns}


def _non_null_lines(counts: dict[str, int]) -> list[str]:
    if not counts:
        return ["  - none: `0`"]
    return [f"  - `{column}`: `{count}`" for column, count in counts.items()]


def _external_teams(df: pd.DataFrame) -> set[str]:
    if df.empty or "team" not in df.columns:
        return set()
    return {team for team in df["team"].map(normalize_team_name).dropna().tolist() if team}


def _match_teams(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    teams: set[str] = set()
    for column in ("home_team", "away_team"):
        if column not in df.columns:
            continue
        for team in df[column].map(normalize_team_name).dropna().tolist():
            if team and not is_placeholder_team_name(team):
                teams.add(team)
    return teams


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _read_prediction_input(cfg: dict[str, Any]) -> pd.DataFrame:
    for key in ("worldcup_2026_prediction_input_advanced", "worldcup_2026_prediction_input"):
        path = _optional_config_path(cfg, key, "")
        if path.exists():
            return read_dataframe(path)
    return pd.DataFrame()


def _read_training_input(cfg: dict[str, Any]) -> pd.DataFrame:
    for key in ("match_training_dataset_advanced_parquet", "match_training_dataset_parquet"):
        path = _optional_config_path(cfg, key, "")
        if path.exists():
            return read_dataframe(path)
    return pd.DataFrame()


def _optional_config_path(cfg: dict[str, Any], key: str, default: str) -> Path:
    try:
        return config_path(cfg, key)
    except KeyError:
        if not default:
            return resolve_project_path("__missing__")
        return resolve_project_path(default)
