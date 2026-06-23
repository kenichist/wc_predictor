from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import config_path, load_config, resolve_project_path
from src.io_utils import read_dataframe, write_dataframe
from src.normalize import normalize_team_name


EXTERNAL_SPECS = {
    "fifa_rankings": {
        "path_key": "fifa_rankings",
        "source_key": "fifa_rankings",
        "default_output_path": "data/external/fifa_rankings.csv",
        "required": ["date", "team", "rank", "points"],
        "numeric": ["rank", "points"],
        "template_name": "fifa_rankings_template.csv",
        "template_rows": [
            {"date": "2024-12-19", "team": "Argentina", "rank": 1, "points": 1867.25, "source": "manual_template"},
            {"date": "2024-12-19", "team": "France", "rank": 2, "points": 1859.78, "source": "manual_template"},
            {"date": "2024-12-19", "team": "Spain", "rank": 3, "points": 1853.27, "source": "manual_template"},
        ],
    },
    "world_football_elo": {
        "path_key": "world_football_elo",
        "source_key": "world_football_elo",
        "default_output_path": "data/external/world_football_elo.csv",
        "required": ["date", "team", "elo"],
        "numeric": ["elo"],
        "template_name": "world_football_elo_template.csv",
        "template_rows": [
            {"date": "2024-12-31", "team": "Argentina", "elo": 2145, "source": "manual_template"},
            {"date": "2024-12-31", "team": "France", "elo": 2098, "source": "manual_template"},
            {"date": "2024-12-31", "team": "Spain", "elo": 2075, "source": "manual_template"},
        ],
    },
}


def prepare_external_templates(config: dict[str, Any] | None = None) -> dict[str, Path]:
    cfg = config or load_config()
    external_dir = _optional_config_path(cfg, "external_dir", "data/external")
    external_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for dataset, spec in EXTERNAL_SPECS.items():
        path = external_dir / str(spec["template_name"])
        pd.DataFrame(spec["template_rows"]).to_csv(path, index=False)
        paths[f"{dataset}_template"] = path
    return paths


def validate_external_data(config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    historical_teams = _dataset_teams(
        cfg,
        preferred_key="match_training_dataset_advanced_parquet",
        fallback_key="match_training_dataset_parquet",
    )
    worldcup_teams = _worldcup_teams(cfg)
    rows = []
    for dataset, spec in EXTERNAL_SPECS.items():
        path = _external_source_path(cfg, spec)
        rows.append(_validate_one(dataset, path, spec, historical_teams, worldcup_teams))
    report = pd.DataFrame(rows)
    write_dataframe(report, _optional_config_path(cfg, "external_data_validation_report_csv", "data/reports/external_data_validation_report.csv"))
    _write_markdown(report, _optional_config_path(cfg, "external_data_validation_report_md", "data/reports/external_data_validation_report.md"))
    return report


def _validate_one(
    dataset: str,
    path: Path,
    spec: dict[str, Any],
    historical_teams: set[str],
    worldcup_teams: set[str],
) -> dict[str, Any]:
    required_columns = list(spec["required"])
    numeric_columns = list(spec["numeric"])
    row: dict[str, Any] = {
        "dataset": dataset,
        "path": str(path),
        "exists": path.exists(),
        "required_columns": ",".join(required_columns),
        "required_columns_present": False,
        "missing_required_columns": "",
        "row_count": 0,
        "date_min": "",
        "date_max": "",
        "date_coverage": "",
        "unique_teams": 0,
        "missing_values": "",
        "duplicate_date_team_rows": 0,
        "numeric_parse_errors": 0,
        "unmapped_team_names": "",
        "unmapped_team_count": 0,
        "historical_team_coverage": 0.0,
        "worldcup_2026_team_coverage": 0.0,
        "warnings": "",
        "next_actions": "",
        "status": "missing",
    }
    warnings: list[str] = []
    next_actions: list[str] = []
    if not path.exists():
        warnings.append("file does not exist")
        next_actions.append(f"provide {path} or run the prepare command with --input")
        row["warnings"] = "; ".join(warnings)
        row["next_actions"] = "; ".join(next_actions)
        return row

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        row["status"] = "unreadable"
        row["warnings"] = f"could not read CSV: {exc}"
        row["next_actions"] = "replace the file with a readable CSV or rerun the prepare command"
        return row

    row["row_count"] = len(df)
    missing = [column for column in required_columns if column not in df.columns]
    row["missing_required_columns"] = ",".join(missing)
    row["required_columns_present"] = not missing
    if missing:
        row["status"] = "invalid_schema"
        row["warnings"] = f"missing required columns: {missing}"
        row["next_actions"] = f"add columns: {missing}"
        return row

    dates = pd.to_datetime(df["date"], errors="coerce")
    normalized_teams = df["team"].map(normalize_team_name)
    row["unique_teams"] = int(normalized_teams.dropna().nunique())
    if dates.notna().any():
        date_min = dates.min()
        date_max = dates.max()
        row["date_min"] = str(date_min.date())
        row["date_max"] = str(date_max.date())
        row["date_coverage"] = f"{row['date_min']} to {row['date_max']}"
    missing_values = {column: int(df[column].isna().sum()) for column in required_columns}
    row["missing_values"] = ";".join(f"{column}:{count}" for column, count in missing_values.items())
    row["duplicate_date_team_rows"] = int(pd.DataFrame({"date": dates, "team": normalized_teams}).duplicated(["date", "team"]).sum())

    numeric_errors = 0
    for column in numeric_columns:
        parsed = pd.to_numeric(df[column], errors="coerce")
        numeric_errors += int(parsed.isna().sum() - df[column].isna().sum())
    row["numeric_parse_errors"] = numeric_errors

    known_teams = historical_teams | worldcup_teams
    unmapped = sorted(
        {
            str(team)
            for team in normalized_teams.dropna().unique()
            if known_teams and str(team) not in known_teams
        }
    )
    invalid_team_count = int(normalized_teams.isna().sum())
    row["unmapped_team_count"] = len(unmapped) + invalid_team_count
    row["unmapped_team_names"] = ",".join(unmapped[:25])
    external_teams = {str(team) for team in normalized_teams.dropna().unique()}
    row["historical_team_coverage"] = _coverage(historical_teams, external_teams)
    row["worldcup_2026_team_coverage"] = _coverage(worldcup_teams, external_teams)

    if dates.isna().any():
        warnings.append(f"{int(dates.isna().sum())} date values failed to parse")
    if numeric_errors:
        warnings.append(f"{numeric_errors} numeric values failed to parse")
    if row["duplicate_date_team_rows"]:
        warnings.append(f"{row['duplicate_date_team_rows']} duplicate date/team rows")
    if any(count > 0 for count in missing_values.values()):
        warnings.append("required columns contain missing values")
    if row["historical_team_coverage"] < 0.5:
        warnings.append("historical team coverage is below 50%")
        next_actions.append("add more historical teams to this external file")
    if row["worldcup_2026_team_coverage"] < 0.9:
        warnings.append("World Cup 2026 team coverage is below 90%")
        next_actions.append("add rows for all 2026 World Cup teams")
    if row["unmapped_team_count"]:
        warnings.append("some team names do not match known training or World Cup teams")
        next_actions.append("check team names and update normalization if needed")

    row["warnings"] = "; ".join(warnings)
    row["next_actions"] = "; ".join(dict.fromkeys(next_actions))
    row["status"] = "ok" if not warnings else "warning"
    return row


def _external_source_path(cfg: dict[str, Any], spec: dict[str, Any]) -> Path:
    source_cfg = cfg.get("external_sources", {}).get(str(spec["source_key"]), {})
    if source_cfg.get("output_path"):
        return resolve_project_path(source_cfg["output_path"])
    return _optional_config_path(cfg, str(spec["path_key"]), str(spec["default_output_path"]))


def _dataset_teams(cfg: dict[str, Any], *, preferred_key: str, fallback_key: str) -> set[str]:
    frame = _read_first_available_dataframe(cfg, preferred_key, fallback_key)
    if frame.empty:
        return set()
    return _teams_from_match_frame(frame)


def _worldcup_teams(cfg: dict[str, Any]) -> set[str]:
    frame = _read_first_available_dataframe(
        cfg,
        "worldcup_2026_prediction_input_advanced",
        "worldcup_2026_prediction_input",
        "worldcup_2026_fixtures_clean",
    )
    if frame.empty:
        return set()
    return {team for team in _teams_from_match_frame(frame) if not _is_placeholder_team(team)}


def _teams_from_match_frame(df: pd.DataFrame) -> set[str]:
    teams: set[str] = set()
    for column in ("home_team", "away_team"):
        if column in df.columns:
            teams.update(str(team) for team in df[column].map(normalize_team_name).dropna())
    return teams


def _first_existing_path(cfg: dict[str, Any], *keys: str) -> Path | None:
    for key in keys:
        try:
            path = config_path(cfg, key)
        except KeyError:
            continue
        if path.exists():
            return path
    return None


def _read_first_available_dataframe(cfg: dict[str, Any], *keys: str) -> pd.DataFrame:
    for key in keys:
        try:
            path = config_path(cfg, key)
        except KeyError:
            continue
        if not path.exists():
            continue
        try:
            return read_dataframe(path)
        except Exception:
            continue
    return pd.DataFrame()


def _optional_config_path(cfg: dict[str, Any], key: str, default: str) -> Path:
    try:
        return config_path(cfg, key)
    except KeyError:
        return resolve_project_path(default)


def _coverage(required_teams: set[str], external_teams: set[str]) -> float:
    if not required_teams:
        return 0.0
    return float(len(required_teams & external_teams) / len(required_teams))


def _is_placeholder_team(team: object) -> bool:
    text = str(team).strip()
    if "/" in text:
        return True
    return bool(re.match(r"^([123][A-L]|W\d+|L\d+)$", text))


def _write_markdown(report: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# External Data Validation Report",
        "",
        "## File Existence",
        "",
        "| dataset | exists | path | status |",
        "| --- | --- | --- | --- |",
    ]
    for row in report.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.exists} | `{row.path}` | {row.status} |")

    lines.extend(
        [
            "",
            "## Required Columns And Values",
            "",
            "| dataset | required_columns_present | missing_required_columns | row_count | missing_values | numeric_parse_errors | duplicate_date_team_rows |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.required_columns_present} | {row.missing_required_columns} | "
            f"{row.row_count} | {row.missing_values} | {row.numeric_parse_errors} | {row.duplicate_date_team_rows} |"
        )

    lines.extend(
        [
            "",
            "## Date Range And Teams",
            "",
            "| dataset | date_range | unique_teams | unmapped_team_count | unmapped_team_names |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.date_coverage} | {row.unique_teams} | "
            f"{row.unmapped_team_count} | {row.unmapped_team_names} |"
        )

    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "| dataset | historical_team_coverage | worldcup_2026_team_coverage |",
            "| --- | --- | --- |",
        ]
    )
    for row in report.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.historical_team_coverage:.3f} | {row.worldcup_2026_team_coverage:.3f} |")

    lines.extend(
        [
            "",
            "## Warnings And Next Actions",
            "",
            "| dataset | warnings | next_actions |",
            "| --- | --- | --- |",
        ]
    )
    for row in report.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.warnings} | {row.next_actions} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
