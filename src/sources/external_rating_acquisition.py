from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from src.config import config_path, load_config, resolve_project_path
from src.io_utils import download_file, read_dataframe, write_dataframe
from src.normalize import normalize_key, normalize_team_name
from src.sources.fifa_rankings import FIFA_RANKINGS_SCHEMA, validate_fifa_rankings
from src.sources.tabular_external import read_external_table
from src.sources.world_football_elo import WORLD_FOOTBALL_ELO_SCHEMA, validate_world_football_elo


logger = logging.getLogger(__name__)

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "fifa_rankings": {
        "source_key": "fifa_rankings",
        "output_path": "data/external/fifa_rankings.csv",
        "raw_dir": "data/raw/fifa_rankings",
        "schema": FIFA_RANKINGS_SCHEMA,
        "numeric": ["rank", "points"],
        "quality_numeric": ["rank", "points"],
        "value_column": "rank",
        "parse": "ranking",
        "required_manual_schema": "date,team,rank,points,source,retrieved_at",
    },
    "world_football_elo": {
        "source_key": "world_football_elo",
        "output_path": "data/external/world_football_elo.csv",
        "raw_dir": "data/raw/world_football_elo",
        "schema": WORLD_FOOTBALL_ELO_SCHEMA,
        "numeric": ["elo"],
        "quality_numeric": ["elo"],
        "value_column": "elo",
        "parse": "elo",
        "required_manual_schema": "date,team,elo,source,retrieved_at",
    },
}

FIFA_COLUMN_CANDIDATES = {
    "date": ["date", "ranking_date", "release_date", "rank_date", "published", "publication_date"],
    "team": ["team", "country", "nation", "association", "country_full", "name"],
    "rank": ["rank", "position", "ranking", "rank_position", "pos"],
    "points": ["points", "total_points", "rating", "score", "pts"],
}

ELO_COLUMN_CANDIDATES = {
    "date": ["date", "rating_date", "elo_date", "rank_date", "published", "publication_date"],
    "team": ["team", "country", "nation", "association", "country_full", "name"],
    "elo": ["elo", "rating", "points", "elo_rating", "score"],
}


@dataclass
class QualityGateResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AcquisitionResult:
    dataset: str
    output_path: Path
    status: str
    rows: int = 0
    source: str = ""
    raw_path: Path | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    preserved_existing: bool = False
    quality: QualityGateResult | None = None


def parse_ranking_table(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    selected = _select_columns(df, FIFA_COLUMN_CANDIDATES)
    missing = [column for column in ("team", "rank", "points") if column not in selected]
    if missing:
        raise ValueError(f"FIFA ranking table missing recognizable columns: {missing}")
    output = pd.DataFrame()
    output["team"] = df[selected["team"]].map(normalize_team_name)
    output["rank"] = pd.to_numeric(df[selected["rank"]], errors="coerce")
    output["points"] = pd.to_numeric(df[selected["points"]], errors="coerce")
    output["date"] = _date_series(df[selected["date"]]) if "date" in selected else pd.Timestamp.now(tz="UTC").normalize()
    output["source"] = source_name
    output["retrieved_at"] = _utc_now()
    return _finalize_parsed_table(output, FIFA_RANKINGS_SCHEMA)


def parse_elo_table(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    selected = _select_columns(df, ELO_COLUMN_CANDIDATES)
    missing = [column for column in ("team", "elo") if column not in selected]
    if missing:
        raise ValueError(f"World Football Elo table missing recognizable columns: {missing}")
    output = pd.DataFrame()
    output["team"] = df[selected["team"]].map(normalize_team_name)
    output["elo"] = pd.to_numeric(df[selected["elo"]], errors="coerce")
    output["date"] = _date_series(df[selected["date"]]) if "date" in selected else pd.Timestamp.now(tz="UTC").normalize()
    output["source"] = source_name
    output["retrieved_at"] = _utc_now()
    return _finalize_parsed_table(output, WORLD_FOOTBALL_ELO_SCHEMA)


def acquire_fifa_rankings(
    *,
    input_path: str | Path | None = None,
    url: str | None = None,
    force: bool = False,
    config: dict[str, Any] | None = None,
    run_post_checks: bool = False,
) -> AcquisitionResult:
    return acquire_rating_dataset(
        "fifa_rankings",
        input_path=input_path,
        url=url,
        force=force,
        config=config,
        run_post_checks=run_post_checks,
    )


def acquire_world_football_elo(
    *,
    input_path: str | Path | None = None,
    url: str | None = None,
    force: bool = False,
    config: dict[str, Any] | None = None,
    run_post_checks: bool = False,
) -> AcquisitionResult:
    return acquire_rating_dataset(
        "world_football_elo",
        input_path=input_path,
        url=url,
        force=force,
        config=config,
        run_post_checks=run_post_checks,
    )


def acquire_external_ratings(
    *,
    force: bool = False,
    config: dict[str, Any] | None = None,
    run_post_checks: bool = True,
) -> list[AcquisitionResult]:
    cfg = config or load_config()
    before_null_rates = _current_null_rates(cfg)
    results = [
        acquire_fifa_rankings(force=force, config=cfg, run_post_checks=False),
        acquire_world_football_elo(force=force, config=cfg, run_post_checks=False),
    ]
    _write_acquisition_report(results, cfg)
    if run_post_checks and any(result.status == "saved" for result in results):
        run_external_rating_post_acquisition_checks(cfg, before_null_rates=before_null_rates)
    else:
        generate_external_rating_activation_report(config=cfg, before_null_rates=before_null_rates)
    write_current_external_rating_acquisition_report(config=cfg, results=results, before_null_rates=before_null_rates)
    return results


def acquire_rating_dataset(
    dataset: str,
    *,
    input_path: str | Path | None = None,
    url: str | None = None,
    force: bool = False,
    config: dict[str, Any] | None = None,
    run_post_checks: bool = False,
) -> AcquisitionResult:
    cfg = config or load_config()
    spec = DATASET_SPECS[dataset]
    source_cfg = cfg.get("external_sources", {}).get(spec["source_key"], {})
    output_path = _configured_source_path(source_cfg, "output_path", spec["output_path"])
    raw_dir = _configured_source_path(source_cfg, "raw_dir", spec["raw_dir"])
    source_url = url or source_cfg.get("source_url")
    source_type = str(source_cfg.get("source_type", "auto") or "auto")
    before_null_rates = _current_null_rates(cfg)

    result = AcquisitionResult(dataset=dataset, output_path=output_path, status="not_run")
    try:
        raw_path, source_name = _resolve_raw_source(
            dataset=dataset,
            raw_dir=raw_dir,
            input_path=input_path,
            source_url=source_url,
            source_type=source_type,
        )
        if raw_path is None:
            result.status = "manual_required"
            result.warnings.append(_manual_instruction(dataset, output_path, spec))
            logger.warning("%s", result.warnings[-1])
            _write_acquisition_report([result], cfg)
            generate_external_rating_activation_report(config=cfg, before_null_rates=before_null_rates)
            return result
        result.raw_path = raw_path
        result.source = source_name
        parsed = _parse_rating_source(raw_path, dataset=dataset, source_name=source_name, source_type=source_type, raw_dir=raw_dir)
        result.rows = len(parsed)
        quality = validate_rating_quality(parsed, dataset=dataset, config=cfg)
        result.quality = quality
        result.warnings.extend(quality.warnings)
        if not quality.ok:
            result.status = "failed_quality_gate"
            result.errors.extend(quality.errors)
            result.preserved_existing = output_path.exists()
            _write_acquisition_report([result], cfg)
            _write_failure_report(result, cfg)
            logger.warning("External rating acquisition failed for %s: %s", dataset, "; ".join(quality.errors))
            if run_post_checks:
                generate_external_rating_activation_report(config=cfg, before_null_rates=before_null_rates)
            return result

        _save_valid_external_file(parsed, output_path, force=force)
        result.status = "saved"
        _write_acquisition_report([result], cfg)
        if run_post_checks:
            run_external_rating_post_acquisition_checks(cfg, before_null_rates=before_null_rates)
        return result
    except Exception as exc:
        result.status = "failed"
        result.errors.append(str(exc))
        result.preserved_existing = output_path.exists()
        _write_acquisition_report([result], cfg)
        _write_failure_report(result, cfg)
        logger.warning("External rating acquisition failed for %s; keeping existing file if present: %s", dataset, exc)
        if run_post_checks:
            generate_external_rating_activation_report(config=cfg, before_null_rates=before_null_rates)
        return result


def validate_rating_quality(df: pd.DataFrame, *, dataset: str, config: dict[str, Any] | None = None) -> QualityGateResult:
    cfg = config or load_config()
    errors: list[str] = []
    warnings: list[str] = []
    spec = DATASET_SPECS[dataset]
    schema = list(spec["schema"])
    missing = [column for column in schema if column not in df.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")
        return QualityGateResult(ok=False, errors=errors)

    validation_errors = validate_fifa_rankings(df) if dataset == "fifa_rankings" else validate_world_football_elo(df)
    errors.extend(validation_errors)
    unique_teams = int(df["team"].dropna().nunique())
    unique_dates = int(pd.to_datetime(df["date"], errors="coerce").dropna().nunique())
    if unique_teams < 30:
        errors.append(f"quality gate failed: at least 30 teams required, found {unique_teams}")
    if unique_dates < 1:
        errors.append("quality gate failed: at least one valid date is required")
    elif unique_dates == 1:
        warnings.append("only one rating date found; historical training coverage will be limited")

    worldcup_teams = _worldcup_teams(cfg)
    historical_teams = _historical_teams(cfg)
    external_teams = set(df["team"].dropna().astype(str))
    worldcup_coverage = _coverage(worldcup_teams, external_teams)
    historical_coverage = _coverage(historical_teams, external_teams)
    if worldcup_teams and worldcup_coverage < 0.8:
        errors.append(f"quality gate failed: World Cup 2026 team coverage {worldcup_coverage:.1%} is below 80%")
    if historical_teams and historical_coverage < 0.5:
        warnings.append(f"historical match team coverage {historical_coverage:.1%} is below 50%")
    duplicate_count = int(df.attrs.get("duplicate_date_team_rows", 0))
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate date/team rows were detected and removed")

    dates = pd.to_datetime(df["date"], errors="coerce")
    metrics = {
        "rows": len(df),
        "unique_teams": unique_teams,
        "unique_dates": unique_dates,
        "date_min": str(dates.min().date()) if dates.notna().any() else "",
        "date_max": str(dates.max().date()) if dates.notna().any() else "",
        "worldcup_2026_team_coverage": worldcup_coverage,
        "historical_team_coverage": historical_coverage,
        "duplicate_date_team_rows": duplicate_count,
        "missing_values": int(df[schema].isna().sum().sum()),
    }
    return QualityGateResult(ok=not errors, errors=errors, warnings=warnings, metrics=metrics)


def run_external_rating_post_acquisition_checks(
    config: dict[str, Any] | None = None,
    *,
    before_null_rates: dict[str, float] | None = None,
) -> pd.DataFrame:
    cfg = config or load_config()
    from src.experiments.ablation_runner import run_ablation
    from src.features.external_feature_joiner import build_advanced_features
    from src.features.null_rate_report import generate_feature_null_rate_report
    from src.sources.external_data_validation import validate_external_data

    validate_external_data(config=cfg)
    build_advanced_features(config=cfg)
    _, null_summary = generate_feature_null_rate_report(config=cfg)
    ablation = run_ablation(config=cfg)
    report = generate_external_rating_activation_report(config=cfg, before_null_rates=before_null_rates)
    _log_activation_checks(null_summary, ablation)
    return report


def generate_external_rating_activation_report(
    config: dict[str, Any] | None = None,
    *,
    before_null_rates: dict[str, float] | None = None,
) -> pd.DataFrame:
    cfg = config or load_config()
    rows = [
        _activation_row("fifa_rankings", "v2_baseline_plus_fifa_rankings", "v1_baseline", cfg, before_null_rates or {}),
        _activation_row("world_football_elo", "v3_plus_external_elo", "v2_baseline_plus_fifa_rankings", cfg, before_null_rates or {}),
    ]
    report = pd.DataFrame(rows)
    csv_path = _optional_config_path(cfg, "external_rating_activation_report_csv", "data/reports/external_rating_activation_report.csv")
    md_path = _optional_config_path(cfg, "external_rating_activation_report_md", "data/reports/external_rating_activation_report.md")
    write_dataframe(report, csv_path)
    _write_activation_markdown(report, md_path)
    return report


def write_current_external_rating_acquisition_report(
    *,
    config: dict[str, Any] | None = None,
    results: list[AcquisitionResult] | None = None,
    before_null_rates: dict[str, float] | None = None,
) -> Path:
    cfg = config or load_config()
    activation = generate_external_rating_activation_report(config=cfg, before_null_rates=before_null_rates)
    acquisition = _acquisition_status_frame(results, cfg)
    path = _optional_config_path(
        cfg,
        "current_external_rating_acquisition_report_md",
        "data/reports/current_external_rating_acquisition_report.md",
    )
    fifa = activation[activation["dataset"].eq("fifa_rankings")].iloc[0]
    elo = activation[activation["dataset"].eq("world_football_elo")].iloc[0]
    lines = [
        "# Current External Rating Acquisition Report",
        "",
        "## Acquisition Status",
        "",
        f"- FIFA rankings acquired automatically: {_auto_status(acquisition, 'fifa_rankings')}",
        f"- World Football Elo acquired automatically: {_auto_status(acquisition, 'world_football_elo')}",
        "",
        "## File Coverage",
        "",
        "| dataset | file_exists | rows | date_min | date_max | unique_teams | worldcup_2026_team_coverage | historical_training_coverage | active |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in activation.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.file_exists} | {row.rows} | {row.date_min} | {row.date_max} | "
            f"{row.unique_teams} | {row.worldcup_2026_team_coverage:.3f} | {row.historical_training_coverage:.3f} | {row.active} |"
        )
    lines.extend(
        [
            "",
            "## Feature Activation",
            "",
            "| feature_group | null_rate_before | null_rate_after | usable_columns | ablation_feature_set | ablation_log_loss_delta | ablation_accuracy_delta | identical_to_previous |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in activation.itertuples(index=False):
        lines.append(
            f"| {row.feature_group} | {_format_report_value(row.null_rate_before)} | {_format_report_value(row.null_rate_after)} | "
            f"{row.usable_columns} | {row.ablation_feature_set} | {_format_report_value(row.ablation_log_loss_delta)} | "
            f"{_format_report_value(row.ablation_accuracy_delta)} | {_format_report_value(row.ablation_identical_to_previous)} |"
        )
    manual_steps = _manual_steps(acquisition, activation)
    lines.extend(["", "## Manual Steps Still Needed", ""])
    if manual_steps:
        lines.extend(f"- {step}" for step in manual_steps)
    else:
        lines.append("No manual steps required by the latest acquisition report.")
    lines.extend(
        [
            "",
            "## Snapshot Warning",
            "",
            "Current snapshots help the 2026 prediction input, but they do not provide strong historical training coverage. For stronger validation and SOTA-style modeling, historical ranking releases from 2010-2026 should be added later.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _parse_rating_source(raw_path: Path, *, dataset: str, source_name: str, source_type: str, raw_dir: Path) -> pd.DataFrame:
    suffix = raw_path.suffix.lower()
    if source_type == "html" or suffix in {".html", ".htm"}:
        tables = _read_html_tables(raw_path)
        selected = _select_best_html_table(tables, dataset=dataset)
        if selected is None:
            _save_debug_tables(tables, raw_dir / "debug", dataset)
            raise ValueError(f"No HTML table matched the expected {dataset} columns; raw tables saved under {raw_dir / 'debug'}")
        logger.info("Selected HTML table %s for %s from %s", selected[0], dataset, raw_path)
        table = selected[1]
    else:
        table = read_external_table(raw_path, allow_html=True)
    if dataset == "fifa_rankings":
        return parse_ranking_table(table, source_name)
    return parse_elo_table(table, source_name)


def _resolve_raw_source(
    *,
    dataset: str,
    raw_dir: Path,
    input_path: str | Path | None,
    source_url: str | None,
    source_type: str = "auto",
) -> tuple[Path | None, str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d%H%M%S")
    if input_path:
        source_path = Path(input_path)
        suffix = source_path.suffix or ".csv"
        raw_path = raw_dir / f"{dataset}_local_{timestamp}{suffix}"
        shutil.copy2(source_path, raw_path)
        return raw_path, f"local:{source_path}"
    if source_url:
        suffix = Path(urlparse(source_url).path).suffix or (".html" if source_type == "html" else ".csv")
        raw_path = raw_dir / f"{dataset}_download_{timestamp}{suffix}"
        return download_file(source_url, raw_path, timeout_seconds=20, retry_count=1), source_url
    return None, ""


def _save_valid_external_file(df: pd.DataFrame, output_path: Path, *, force: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    if output_path.exists() and force:
        output_path.unlink()
    temp_path.replace(output_path)
    logger.info("Saved validated external rating file to %s", output_path)


def _finalize_parsed_table(df: pd.DataFrame, schema: list[str]) -> pd.DataFrame:
    output = df.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output["team"] = output["team"].map(normalize_team_name)
    output["source"] = output["source"].fillna("unknown").astype(str)
    output["retrieved_at"] = output["retrieved_at"].fillna(_utc_now()).astype(str)
    output = output.dropna(subset=["date", "team"])
    duplicate_count = int(output.duplicated(["date", "team"]).sum())
    output = output.drop_duplicates(["date", "team"], keep="last")
    output = output[schema].sort_values(["team", "date"], kind="stable").reset_index(drop=True)
    output.attrs["duplicate_date_team_rows"] = duplicate_count
    return output


def _select_columns(df: pd.DataFrame, candidates: dict[str, list[str]]) -> dict[str, str]:
    normalized = {_column_key(column): column for column in df.columns}
    selected: dict[str, str] = {}
    for target, names in candidates.items():
        for name in names:
            key = _column_key(name)
            if key in normalized:
                selected[target] = normalized[key]
                break
        if target not in selected:
            for key, original in normalized.items():
                if any(_column_key(name) in key for name in names):
                    selected[target] = original
                    break
    return selected


def _select_best_html_table(tables: list[pd.DataFrame], *, dataset: str) -> tuple[int, pd.DataFrame] | None:
    candidates = FIFA_COLUMN_CANDIDATES if dataset == "fifa_rankings" else ELO_COLUMN_CANDIDATES
    required = {"team", "rank", "points"} if dataset == "fifa_rankings" else {"team", "elo"}
    best: tuple[int, pd.DataFrame] | None = None
    best_score = -1
    for idx, table in enumerate(tables):
        selected = _select_columns(table, candidates)
        score = len(set(selected) & required) + (1 if "date" in selected else 0)
        if score > best_score:
            best_score = score
            best = (idx, table)
    if best and required.issubset(set(_select_columns(best[1], candidates))):
        return best
    return None


def _save_debug_tables(tables: list[pd.DataFrame], debug_dir: Path, dataset: str) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for idx, table in enumerate(tables):
        table.to_csv(debug_dir / f"{dataset}_html_table_{idx}.csv", index=False)


def _read_html_tables(path: Path) -> list[pd.DataFrame]:
    try:
        return pd.read_html(path)
    except ImportError as exc:
        parser = _SimpleTableParser()
        parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
        tables = parser.to_dataframes()
        if tables:
            return tables
        raise ValueError(f"No parseable HTML tables found in {path}; pandas optional HTML parser error was: {exc}") from exc


class _SimpleTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "table":
            self._current_table = []
        elif tag == "tr" and self._current_table is not None:
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            self._current_row.append(" ".join(part.strip() for part in self._current_cell if part.strip()))
            self._current_cell = None
        elif tag == "tr" and self._current_table is not None and self._current_row is not None:
            if self._current_row:
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._current_table is not None:
            if self._current_table:
                self.tables.append(self._current_table)
            self._current_table = None

    def to_dataframes(self) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        for table in self.tables:
            if not table:
                continue
            header = table[0]
            rows = [row for row in table[1:] if len(row) == len(header)]
            if rows:
                frames.append(pd.DataFrame(rows, columns=header))
        return frames


def _activation_row(
    dataset: str,
    feature_set: str,
    baseline_feature_set: str,
    cfg: dict[str, Any],
    before_null_rates: dict[str, float],
) -> dict[str, Any]:
    group = "fifa_rankings" if dataset == "fifa_rankings" else "external_elo"
    path_key = "fifa_rankings" if dataset == "fifa_rankings" else "world_football_elo"
    path = _external_output_path(cfg, dataset)
    stats = _external_file_stats(path, cfg)
    null_stats = _feature_group_null_stats(cfg, group)
    ablation_stats = _ablation_delta(cfg, feature_set, baseline_feature_set)
    warnings = []
    if stats["unique_dates"] == 1:
        warnings.append("only current snapshot data exists; historical coverage is limited")
    if stats["historical_training_coverage"] < 0.5:
        warnings.append("historical coverage is below 50%")
    if stats["worldcup_2026_team_coverage"] < 0.8:
        warnings.append("World Cup 2026 coverage is below 80%")
    if null_stats["average_null_rate"] >= 1.0 or null_stats["usable_columns"] <= 0:
        warnings.append("feature group is inactive; add real external data and rebuild advanced features")
    return {
        "dataset": dataset,
        "feature_group": group,
        "file_exists": path.exists(),
        "path": str(path),
        "rows": stats["rows"],
        "date_min": stats["date_min"],
        "date_max": stats["date_max"],
        "unique_teams": stats["unique_teams"],
        "unique_dates": stats["unique_dates"],
        "worldcup_2026_team_coverage": stats["worldcup_2026_team_coverage"],
        "historical_training_coverage": stats["historical_training_coverage"],
        "null_rate_before": before_null_rates.get(group, pd.NA),
        "null_rate_after": null_stats["average_null_rate"],
        "usable_columns": null_stats["usable_columns"],
        "ablation_feature_set": feature_set,
        "ablation_baseline_feature_set": baseline_feature_set,
        "ablation_log_loss_delta": ablation_stats["log_loss_delta"],
        "ablation_accuracy_delta": ablation_stats["accuracy_delta"],
        "ablation_identical_to_previous": ablation_stats["identical_to_previous"],
        "active": bool(path.exists() and null_stats["average_null_rate"] < 1.0 and null_stats["usable_columns"] > 0),
        "warning": "; ".join(warnings),
    }


def _external_file_stats(path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {
            "rows": 0,
            "date_min": "",
            "date_max": "",
            "unique_teams": 0,
            "unique_dates": 0,
            "worldcup_2026_team_coverage": 0.0,
            "historical_training_coverage": 0.0,
        }
    df = pd.read_csv(path)
    dates = pd.to_datetime(df.get("date"), errors="coerce")
    teams = set(df.get("team", pd.Series(dtype=object)).map(normalize_team_name).dropna().astype(str))
    return {
        "rows": len(df),
        "date_min": str(dates.min().date()) if dates.notna().any() else "",
        "date_max": str(dates.max().date()) if dates.notna().any() else "",
        "unique_teams": len(teams),
        "unique_dates": int(dates.dropna().nunique()),
        "worldcup_2026_team_coverage": _coverage(_worldcup_teams(cfg), teams),
        "historical_training_coverage": _coverage(_historical_teams(cfg), teams),
    }


def _feature_group_null_stats(cfg: dict[str, Any], group: str) -> dict[str, Any]:
    detail_path = _optional_config_path(cfg, "feature_null_rate_report_csv", "data/reports/feature_null_rate_report.csv")
    if not detail_path.exists():
        return {"average_null_rate": 1.0, "usable_columns": 0}
    detail = pd.read_csv(detail_path)
    group_detail = detail[detail["feature_group"].eq(group)]
    if group_detail.empty:
        return {"average_null_rate": 1.0, "usable_columns": 0}
    return {
        "average_null_rate": float(group_detail["null_rate"].mean()),
        "usable_columns": int((group_detail["non_null_count"] > 0).sum()),
    }


def _ablation_delta(cfg: dict[str, Any], feature_set: str, baseline_feature_set: str) -> dict[str, Any]:
    path = _optional_config_path(cfg, "ablation_results_csv", "data/experiments/ablation_results.csv")
    if not path.exists():
        return {"log_loss_delta": pd.NA, "accuracy_delta": pd.NA, "identical_to_previous": pd.NA}
    ablation = pd.read_csv(path)
    current = ablation[ablation["feature_set"].eq(feature_set)]
    baseline = ablation[ablation["feature_set"].eq(baseline_feature_set)]
    if current.empty or baseline.empty:
        return {"log_loss_delta": pd.NA, "accuracy_delta": pd.NA, "identical_to_previous": pd.NA}
    current_row = current.iloc[0]
    baseline_row = baseline.iloc[0]
    return {
        "log_loss_delta": float(current_row["log_loss"] - baseline_row["log_loss"]),
        "accuracy_delta": float(current_row["accuracy"] - baseline_row["accuracy"]),
        "identical_to_previous": bool(current_row.get("identical_to_previous", False)),
    }


def _current_null_rates(cfg: dict[str, Any]) -> dict[str, float]:
    path = _optional_config_path(cfg, "feature_null_rate_report_csv", "data/reports/feature_null_rate_report.csv")
    if not path.exists():
        return {}
    detail = pd.read_csv(path)
    return {
        group: float(frame["null_rate"].mean())
        for group, frame in detail.groupby("feature_group")
        if group in {"fifa_rankings", "external_elo"}
    }


def _log_activation_checks(null_summary: pd.DataFrame, ablation: pd.DataFrame) -> None:
    for group in ("fifa_rankings", "external_elo"):
        row = null_summary[null_summary["feature_group"].eq(group)]
        if row.empty:
            logger.warning("%s did not appear in the feature null-rate report", group)
            continue
        values = row.iloc[0]
        if values["average_null_rate"] >= 1.0:
            logger.warning("%s is inactive because average_null_rate is %.3f", group, values["average_null_rate"])
        if values["usable_columns"] <= 0:
            logger.warning("%s is inactive because usable_columns is 0", group)
    checks = [
        ("v2_baseline_plus_fifa_rankings", "v1_baseline"),
        ("v3_plus_external_elo", "v2_baseline_plus_fifa_rankings"),
    ]
    for feature_set, previous in checks:
        row = ablation[ablation["feature_set"].eq(feature_set)]
        if not row.empty and bool(row.iloc[0].get("identical_to_previous", False)):
            logger.warning("%s is identical to %s: %s", feature_set, previous, row.iloc[0].get("warning", ""))


def _write_activation_markdown(report: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "file_exists",
        "rows",
        "date_min",
        "date_max",
        "worldcup_2026_team_coverage",
        "historical_training_coverage",
        "null_rate_before",
        "null_rate_after",
        "usable_columns",
        "ablation_log_loss_delta",
        "active",
        "warning",
    ]
    lines = [
        "# External Rating Activation Report",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in report[columns].itertuples(index=False):
        lines.append("| " + " | ".join(_format_report_value(value) for value in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_acquisition_report(results: list[AcquisitionResult], cfg: dict[str, Any]) -> None:
    csv_path = _optional_config_path(cfg, "external_rating_acquisition_report_csv", "data/reports/external_rating_acquisition_report.csv")
    md_path = _optional_config_path(cfg, "external_rating_acquisition_report_md", "data/reports/external_rating_acquisition_report.md")
    rows = []
    for result in results:
        rows.append(
            {
                "dataset": result.dataset,
                "status": result.status,
                "output_path": str(result.output_path),
                "raw_path": "" if result.raw_path is None else str(result.raw_path),
                "rows": result.rows,
                "preserved_existing": result.preserved_existing,
                "errors": "; ".join(result.errors),
                "warnings": "; ".join(result.warnings),
            }
        )
    report = pd.DataFrame(rows)
    write_dataframe(report, csv_path)
    lines = ["# External Rating Acquisition Report", ""]
    for row in report.itertuples(index=False):
        lines.append(f"- `{row.dataset}`: {row.status}. {row.errors or row.warnings}")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _acquisition_status_frame(results: list[AcquisitionResult] | None, cfg: dict[str, Any]) -> pd.DataFrame:
    if results is not None:
        return pd.DataFrame(
            [
                {
                    "dataset": result.dataset,
                    "status": result.status,
                    "source": result.source,
                    "rows": result.rows,
                    "errors": "; ".join(result.errors),
                    "warnings": "; ".join(result.warnings),
                }
                for result in results
            ]
        )
    path = _optional_config_path(cfg, "external_rating_acquisition_report_csv", "data/reports/external_rating_acquisition_report.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=["dataset", "status", "source", "rows", "errors", "warnings"])


def _auto_status(acquisition: pd.DataFrame, dataset: str) -> str:
    row = acquisition[acquisition["dataset"].eq(dataset)]
    if row.empty:
        return "unknown"
    status = str(row.iloc[0].get("status", "unknown"))
    return "yes" if status == "saved" else f"no ({status})"


def _manual_steps(acquisition: pd.DataFrame, activation: pd.DataFrame) -> list[str]:
    steps: list[str] = []
    commands = {
        "fifa_rankings": r'python -m src.cli acquire-fifa-rankings --input "C:\Users\kenic\Downloads\fifa_rankings.csv"',
        "world_football_elo": r'python -m src.cli acquire-world-football-elo --input "C:\Users\kenic\Downloads\world_football_elo.csv"',
    }
    for dataset, command in commands.items():
        active = activation[activation["dataset"].eq(dataset)]
        status = acquisition[acquisition["dataset"].eq(dataset)]
        if active.empty or not bool(active.iloc[0].get("active", False)):
            reason = "not active"
            if not status.empty:
                reason = str(status.iloc[0].get("status", reason))
            steps.append(f"{dataset}: {reason}. Import a real current file with `{command}` after replacing the example path with your actual file path.")
    if steps:
        steps.extend(
            [
                "Run `python -m src.cli validate-external-data`.",
                "Run `python -m src.cli build-advanced-features`.",
                "Run `python -m src.cli feature-null-report`.",
                "Run `python -m src.cli run-ablation`.",
            ]
        )
    return steps


def _write_failure_report(result: AcquisitionResult, cfg: dict[str, Any]) -> None:
    reports_dir = _optional_config_path(cfg, "reports_dir", "data/reports")
    path = reports_dir / f"{result.dataset}_acquisition_failure.md"
    lines = [
        f"# {result.dataset} Acquisition Failure",
        "",
        f"- status: {result.status}",
        f"- output_path: `{result.output_path}`",
        f"- raw_path: `{result.raw_path}`",
        f"- preserved_existing: {result.preserved_existing}",
        f"- errors: {'; '.join(result.errors)}",
        f"- warnings: {'; '.join(result.warnings)}",
        "",
        "## Manual Import Instructions",
        "",
        _manual_instruction(result.dataset, result.output_path, DATASET_SPECS[result.dataset]),
        "",
        "- Validate after import: `python -m src.cli validate-external-data`",
        "- Rebuild features: `python -m src.cli build-advanced-features`",
        "- Check activation: `python -m src.cli feature-null-report`",
        "- Rerun ablation: `python -m src.cli run-ablation`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _manual_instruction(dataset: str, output_path: Path, spec: dict[str, Any]) -> str:
    command = "acquire-fifa-rankings" if dataset == "fifa_rankings" else "acquire-world-football-elo"
    example = (
        r"C:\Users\kenic\Downloads\fifa_rankings.csv"
        if dataset == "fifa_rankings"
        else r"C:\Users\kenic\Downloads\world_football_elo.csv"
    )
    return (
        f"Provide a real CSV/Excel/JSON/HTML file for {dataset} with schema "
        f"{spec['required_manual_schema']} at {output_path}, or run "
        f"`python -m src.cli {command} --input \"{example}\"` after replacing the example path with your actual downloaded file path."
    )


def _date_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().any():
        return parsed
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    return parsed


def _column_key(column: object) -> str:
    return normalize_key(str(column))


def _configured_source_path(source_cfg: dict[str, Any], key: str, default: str) -> Path:
    value = source_cfg.get(key) or default
    return resolve_project_path(value)


def _external_output_path(cfg: dict[str, Any], dataset: str) -> Path:
    spec = DATASET_SPECS[dataset]
    source_cfg = cfg.get("external_sources", {}).get(spec["source_key"], {})
    if source_cfg.get("output_path"):
        return resolve_project_path(source_cfg["output_path"])
    path_key = "fifa_rankings" if dataset == "fifa_rankings" else "world_football_elo"
    try:
        return config_path(cfg, path_key)
    except KeyError:
        return resolve_project_path(spec["output_path"])


def _optional_config_path(cfg: dict[str, Any], key: str, default: str) -> Path:
    try:
        return config_path(cfg, key)
    except KeyError:
        return resolve_project_path(default)


def _historical_teams(cfg: dict[str, Any]) -> set[str]:
    path = _first_existing_path(cfg, "match_training_dataset_advanced_parquet", "match_training_dataset_parquet", "all_matches_clean")
    if path is None:
        return set()
    return _teams_from_match_frame(read_dataframe(path))


def _worldcup_teams(cfg: dict[str, Any]) -> set[str]:
    path = _first_existing_path(cfg, "worldcup_2026_prediction_input_advanced", "worldcup_2026_prediction_input", "worldcup_2026_fixtures_clean")
    if path is None:
        return set()
    return {team for team in _teams_from_match_frame(read_dataframe(path)) if not _is_placeholder_team(team)}


def _first_existing_path(cfg: dict[str, Any], *keys: str) -> Path | None:
    for key in keys:
        try:
            path = config_path(cfg, key)
        except KeyError:
            continue
        if path.exists():
            return path
    return None


def _teams_from_match_frame(df: pd.DataFrame) -> set[str]:
    teams: set[str] = set()
    for column in ("home_team", "away_team"):
        if column in df.columns:
            teams.update(str(team) for team in df[column].map(normalize_team_name).dropna())
    return teams


def _coverage(required_teams: set[str], external_teams: set[str]) -> float:
    if not required_teams:
        return 0.0
    return float(len(required_teams & external_teams) / len(required_teams))


def _is_placeholder_team(team: object) -> bool:
    text = str(team).strip()
    if "/" in text:
        return True
    return bool(re.match(r"^([123][A-L]|W\d+|L\d+)$", text))


def _format_report_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()
