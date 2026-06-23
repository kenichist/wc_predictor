from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config_path, load_config, resolve_project_path
from src.experiments.ablation_runner import run_ablation
from src.features.external_feature_joiner import build_advanced_features
from src.features.null_rate_report import generate_feature_null_rate_report
from src.io_utils import read_dataframe, write_dataframe
from src.normalize import normalize_key, normalize_team_name
from src.sources.external_data_validation import validate_external_data
from src.sources.fifa_rankings import (
    FIFA_RANKINGS_SCHEMA,
    join_fifa_rankings_asof,
    normalize_fifa_rankings,
    validate_fifa_rankings,
)
from src.sources.world_football_elo import (
    WORLD_FOOTBALL_ELO_SCHEMA,
    join_world_football_elo_asof,
    normalize_world_football_elo,
    validate_world_football_elo,
)


logger = logging.getLogger(__name__)

SNAPSHOT_SUFFIXES = {".csv", ".txt", ".md", ".markdown", ".html", ".htm", ".xlsx", ".xls"}

FIFA_ALIASES = {
    "date": ["date", "ranking_date", "release_date", "rank_date", "published", "publication_date"],
    "rank": ["rank", "ranking", "position", "pos"],
    "team": ["team", "country", "nation", "association"],
    "points": ["points", "pts", "rating", "total_points", "score"],
}

ELO_ALIASES = {
    "date": ["date", "rating_date", "elo_date", "rank_date", "published", "publication_date"],
    "rank": ["rank", "ranking", "position", "pos"],
    "team": ["team", "country", "nation"],
    "elo": ["elo", "rating", "points", "pts", "score"],
}

FIFA_FILENAME_PATTERNS = [
    r"^fifa[_-](?P<date>\d{4}[-_]\d{2}[-_]\d{2})",
    r"^fifa-ranking-(?P<date>\d{4}-\d{2}-\d{2})",
    r"^fifa_rankings_(?P<date>\d{4}-\d{2}-\d{2})",
]

ELO_FILENAME_PATTERNS = [
    r"^elo[_-](?P<date>\d{4}[-_]\d{2}[-_]\d{2})",
    r"^world-football-elo-(?P<date>\d{4}-\d{2}-\d{2})",
    r"^elo_ratings_(?P<date>\d{4}-\d{2}-\d{2})",
]


@dataclass
class SnapshotFileResult:
    dataset: str
    path: Path
    status: str
    rows: int = 0
    date_min: str = ""
    date_max: str = ""
    latest_snapshot_team_count: int = 0
    reason: str = ""
    warning: str = ""


@dataclass
class DatasetImportResult:
    dataset: str
    input_dir: Path
    output_path: Path
    files_found: int
    snapshots_imported: int
    snapshots_imported_with_missing_points: int
    snapshots_failed: int
    snapshot_files: str
    snapshot_statuses: str
    total_rows: int
    unique_teams: int
    date_min: str
    date_max: str
    latest_snapshot_date: str
    latest_snapshot_team_count: int
    historical_training_coverage: float
    worldcup_2026_coverage: float
    null_rate_by_year: str
    ablation_can_meaningfully_test: bool
    failed_files: str
    failed_reasons: str
    output_written: bool
    preserved_existing: bool
    combined_errors: str

    def as_row(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "input_dir": str(self.input_dir),
            "output_path": str(self.output_path),
            "files_found": self.files_found,
            "snapshots_imported": self.snapshots_imported,
            "snapshots_imported_with_missing_points": self.snapshots_imported_with_missing_points,
            "snapshots_failed": self.snapshots_failed,
            "snapshot_files": self.snapshot_files,
            "snapshot_statuses": self.snapshot_statuses,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "total_rows": self.total_rows,
            "unique_teams": self.unique_teams,
            "latest_snapshot_date": self.latest_snapshot_date,
            "latest_snapshot_team_count": self.latest_snapshot_team_count,
            "historical_training_coverage": self.historical_training_coverage,
            "worldcup_2026_coverage": self.worldcup_2026_coverage,
            "null_rate_by_year": self.null_rate_by_year,
            "ablation_can_meaningfully_test": self.ablation_can_meaningfully_test,
            "failed_files": self.failed_files,
            "failed_reasons": self.failed_reasons,
            "output_written": self.output_written,
            "preserved_existing": self.preserved_existing,
            "combined_errors": self.combined_errors,
        }


def import_fifa_ranking_snapshots(
    input_dir: Path,
    output_path: Path,
    min_date: str | None = None,
    max_date: str | None = None,
    force: bool = False,
    *,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Import dated FIFA ranking snapshots into the canonical external file."""
    cfg = config or load_config()
    result, frame = _import_dataset(
        dataset="fifa_rankings",
        input_dir=Path(input_dir),
        output_path=Path(output_path),
        min_date=min_date,
        max_date=max_date,
        force=force,
        config=cfg,
    )
    _write_coverage_report([result], cfg)
    return frame


def import_world_football_elo_snapshots(
    input_dir: Path,
    output_path: Path,
    min_date: str | None = None,
    max_date: str | None = None,
    force: bool = False,
    *,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Import dated World Football Elo snapshots into the canonical external file."""
    cfg = config or load_config()
    result, frame = _import_dataset(
        dataset="world_football_elo",
        input_dir=Path(input_dir),
        output_path=Path(output_path),
        min_date=min_date,
        max_date=max_date,
        force=force,
        config=cfg,
    )
    _write_coverage_report([result], cfg)
    return frame


def import_historical_rating_snapshots(
    fifa_input_dir: Path | None = None,
    elo_input_dir: Path | None = None,
    force: bool = False,
    rebuild: bool = False,
    *,
    min_date: str | None = None,
    max_date: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Import both historical external rating snapshot folders and optionally rebuild features."""
    cfg = config or load_config()
    fifa_dir = Path(fifa_input_dir) if fifa_input_dir is not None else _optional_config_path(
        cfg, "fifa_rankings_snapshots_dir", "data/external/fifa_rankings_snapshots"
    )
    elo_dir = Path(elo_input_dir) if elo_input_dir is not None else _optional_config_path(
        cfg, "world_football_elo_snapshots_dir", "data/external/world_football_elo_snapshots"
    )
    fifa_output = _optional_config_path(cfg, "fifa_rankings", "data/external/fifa_rankings.csv")
    elo_output = _optional_config_path(cfg, "world_football_elo", "data/external/world_football_elo.csv")

    fifa_result, fifa_frame = _import_dataset(
        dataset="fifa_rankings",
        input_dir=fifa_dir,
        output_path=fifa_output,
        min_date=min_date,
        max_date=max_date,
        force=force,
        config=cfg,
    )
    elo_result, elo_frame = _import_dataset(
        dataset="world_football_elo",
        input_dir=elo_dir,
        output_path=elo_output,
        min_date=min_date,
        max_date=max_date,
        force=force,
        config=cfg,
    )
    report_paths = _write_coverage_report([fifa_result, elo_result], cfg)

    rebuild_status = "not_requested"
    if rebuild:
        _run_rebuild_steps(cfg)
        rebuild_status = "completed"

    return {
        "fifa_rankings": fifa_frame,
        "world_football_elo": elo_frame,
        "results": {
            "fifa_rankings": fifa_result.as_row(),
            "world_football_elo": elo_result.as_row(),
        },
        "coverage_report_csv": report_paths["csv"],
        "coverage_report_md": report_paths["md"],
        "rebuild_status": rebuild_status,
    }


def _import_dataset(
    *,
    dataset: str,
    input_dir: Path,
    output_path: Path,
    min_date: str | None,
    max_date: str | None,
    force: bool,
    config: dict[str, Any],
) -> tuple[DatasetImportResult, pd.DataFrame]:
    input_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SNAPSHOT_SUFFIXES)
    valid_frames: list[pd.DataFrame] = []
    file_results: list[SnapshotFileResult] = []

    for path in files:
        file_result, frame = _parse_and_validate_snapshot(path, dataset=dataset, min_date=min_date, max_date=max_date)
        file_results.append(file_result)
        if file_result.status in {"imported", "imported_with_missing_points"}:
            valid_frames.append(frame)

    combined = _combine_frames(valid_frames, dataset)
    combined_errors = _validate_combined(combined, dataset)
    output_written = False
    preserved_existing = False

    if combined_errors:
        if not combined.empty:
            _save_debug_candidate(combined, dataset)
        preserved_existing = output_path.exists()
        if force and not combined.empty:
            write_dataframe(combined, output_path)
            output_written = True
            preserved_existing = False
            logger.warning("Force-wrote %s despite combined validation errors: %s", output_path, "; ".join(combined_errors))
        else:
            logger.warning(
                "Historical %s import did not overwrite %s: %s",
                dataset,
                output_path,
                "; ".join(combined_errors),
            )
    else:
        write_dataframe(combined, output_path)
        output_written = True

    frame_to_report = combined if not combined.empty else _read_existing_output(output_path)
    result = _dataset_result(
        dataset=dataset,
        input_dir=input_dir,
        output_path=output_path,
        file_results=file_results,
        frame=frame_to_report,
        imported_frame=combined,
        config=config,
        output_written=output_written,
        preserved_existing=preserved_existing,
        combined_errors=combined_errors,
    )
    return result, frame_to_report


def _parse_and_validate_snapshot(
    path: Path,
    *,
    dataset: str,
    min_date: str | None,
    max_date: str | None,
) -> tuple[SnapshotFileResult, pd.DataFrame]:
    try:
        parsed = _parse_snapshot_file(path, dataset)
        parsed = _filter_date_range(parsed, min_date=min_date, max_date=max_date)
        if parsed.empty:
            raise ValueError("no rows remain after applying date filters")
        errors = _validate_snapshot(parsed, dataset)
        if errors:
            raise ValueError("; ".join(errors))
        normalized = normalize_fifa_rankings(parsed) if dataset == "fifa_rankings" else normalize_world_football_elo(parsed)
        dates = pd.to_datetime(normalized["date"], errors="coerce")
        latest_date = dates.max()
        latest_count = int(normalized.loc[dates.eq(latest_date), "team"].nunique()) if pd.notna(latest_date) else 0
        status = "imported"
        warning = ""
        if dataset == "fifa_rankings" and normalized["points"].isna().all():
            status = "imported_with_missing_points"
            warning = "points missing for this snapshot; rank features only"
            logger.warning("%s: %s", path.name, warning)
        return (
            SnapshotFileResult(
                dataset=dataset,
                path=path,
                status=status,
                rows=len(normalized),
                date_min=str(dates.min().date()),
                date_max=str(dates.max().date()),
                latest_snapshot_team_count=latest_count,
                warning=warning,
            ),
            normalized,
        )
    except Exception as exc:
        logger.warning("Skipping %s snapshot %s: %s", dataset, path, exc)
        return SnapshotFileResult(dataset=dataset, path=path, status="failed", reason=str(exc)), _empty_frame(dataset)


def _parse_snapshot_file(path: Path, dataset: str) -> pd.DataFrame:
    source = f"snapshot:{path.name}"
    errors: list[str] = []
    for table in _read_snapshot_tables(path, dataset):
        try:
            return _parse_snapshot_table(table, dataset=dataset, source=source, path=path)
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("; ".join(errors) if errors else f"no parseable table found in {path}")


def _read_snapshot_tables(path: Path, dataset: str) -> list[pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return [pd.read_csv(path)]
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        return list(sheets.values())
    if suffix in {".html", ".htm"}:
        try:
            return pd.read_html(path)
        except Exception as exc:
            parser = _SimpleTableParser()
            parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
            tables = parser.to_dataframes()
            if tables:
                return tables
            raise ValueError(f"No parseable HTML tables found in {path}: {exc}") from exc
    if suffix in {".txt", ".md", ".markdown"}:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        tables = []
        try:
            tables.append(pd.read_csv(path, sep=None, engine="python"))
        except Exception:
            pass
        markdown = _parse_markdown_table(text)
        if markdown is not None:
            tables.append(markdown)
        text_table = _parse_whitespace_table(text, dataset)
        if text_table is not None:
            tables.append(text_table)
        return tables
    raise ValueError(f"unsupported snapshot format: {path.suffix}")


def _parse_snapshot_table(table: pd.DataFrame, *, dataset: str, source: str, path: Path) -> pd.DataFrame:
    aliases = FIFA_ALIASES if dataset == "fifa_rankings" else ELO_ALIASES
    selected = _select_columns(table, aliases)
    required = {"team", "rank"} if dataset == "fifa_rankings" else {"team", "elo"}
    missing = sorted(required - set(selected))
    if missing:
        raise ValueError(f"missing recognizable columns: {missing}")

    output = pd.DataFrame(index=table.index)
    if "date" in selected:
        output["date"] = _date_series(table[selected["date"]])
    else:
        inferred = _infer_snapshot_date(path.name, dataset)
        if inferred is None:
            raise ValueError("no date column and no supported date pattern in filename")
        output["date"] = inferred
    output["team"] = table[selected["team"]].map(normalize_team_name)
    if dataset == "fifa_rankings":
        output["rank"] = _numeric_series(table[selected["rank"]])
        if "points" in selected:
            raw_points = table[selected["points"]]
            output["points"] = _numeric_series(raw_points)
            output.attrs["points_parse_failures"] = _optional_numeric_parse_failures(raw_points, output["points"])
        else:
            output["points"] = pd.NA
            output.attrs["points_parse_failures"] = 0
        schema = FIFA_RANKINGS_SCHEMA
    else:
        output["elo"] = _numeric_series(table[selected["elo"]])
        schema = WORLD_FOOTBALL_ELO_SCHEMA
    output["source"] = source
    output["retrieved_at"] = _utc_now()
    output = output.dropna(how="all")
    return output[schema]


def _parse_markdown_table(text: str) -> pd.DataFrame | None:
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or "|" not in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells or all(re.fullmatch(r":?-{2,}:?", cell or "") for cell in cells):
            continue
        rows.append(cells)
    if len(rows) < 2:
        return None
    header = rows[0]
    data = [row for row in rows[1:] if len(row) == len(header)]
    if not data:
        return None
    return pd.DataFrame(data, columns=header)


def _parse_whitespace_table(text: str, dataset: str) -> pd.DataFrame | None:
    rows = []
    value_column = "points" if dataset == "fifa_rankings" else "elo"
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "|" in stripped:
            continue
        if _looks_like_header(stripped):
            continue
        pattern = (
            r"^(?P<rank>\d+)\s+(?P<team>.+?)(?:\s+(?P<value>-?\d+(?:[,.]\d+)?))?$"
            if dataset == "fifa_rankings"
            else r"^(?:(?P<rank>\d+)\s+)?(?P<team>.+?)\s+(?P<value>-?\d+(?:[,.]\d+)?)$"
        )
        match = re.match(pattern, stripped)
        if match is None:
            continue
        row = {"team": match.group("team"), value_column: match.group("value")}
        if dataset == "fifa_rankings" or match.groupdict().get("rank"):
            row["rank"] = match.groupdict().get("rank")
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows)


def _looks_like_header(line: str) -> bool:
    key = normalize_key(line)
    header_terms = {"rank", "ranking", "position", "pos", "team", "country", "nation", "points", "pts", "elo", "rating"}
    return len(set(key.split()) & header_terms) >= 2


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


def _select_columns(df: pd.DataFrame, aliases: dict[str, list[str]]) -> dict[str, str]:
    normalized = {normalize_key(str(column)): str(column) for column in df.columns}
    selected: dict[str, str] = {}
    for target, candidates in aliases.items():
        for candidate in candidates:
            key = normalize_key(candidate)
            if key in normalized:
                selected[target] = normalized[key]
                break
        if target in selected:
            continue
        for normalized_name, original in normalized.items():
            if any(normalize_key(candidate) in normalized_name for candidate in candidates):
                selected[target] = original
                break
    return selected


def _infer_snapshot_date(filename: str, dataset: str) -> pd.Timestamp | None:
    stem = Path(filename).stem.lower()
    patterns = FIFA_FILENAME_PATTERNS if dataset == "fifa_rankings" else ELO_FILENAME_PATTERNS
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            text = match.group("date").replace("_", "-")
            parsed = pd.to_datetime(text, errors="coerce")
            return None if pd.isna(parsed) else parsed
    return None


def _validate_snapshot(df: pd.DataFrame, dataset: str) -> list[str]:
    errors = validate_fifa_rankings(df) if dataset == "fifa_rankings" else validate_world_football_elo(df)
    if dataset == "fifa_rankings":
        point_failures = int(df.attrs.get("points_parse_failures", 0))
        if point_failures:
            errors.append(f"points numeric parse failures: {point_failures}")
    teams = df["team"].map(normalize_team_name).dropna().nunique() if "team" in df.columns else 0
    if teams < 30:
        errors.append(f"at least 30 parsed teams required, found {teams}")
    return errors


def _combine_frames(frames: list[pd.DataFrame], dataset: str) -> pd.DataFrame:
    if not frames:
        return _empty_frame(dataset)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined["team"] = combined["team"].map(normalize_team_name)
    combined["source"] = combined["source"].fillna("snapshot").astype(str)
    combined["retrieved_at"] = combined["retrieved_at"].fillna(_utc_now()).astype(str)
    if dataset == "fifa_rankings":
        combined["rank"] = pd.to_numeric(combined["rank"], errors="coerce").round().astype("Int64")
        combined["points"] = pd.to_numeric(combined["points"], errors="coerce")
        schema = FIFA_RANKINGS_SCHEMA
    else:
        combined["elo"] = pd.to_numeric(combined["elo"], errors="coerce")
        schema = WORLD_FOOTBALL_ELO_SCHEMA
    return combined[schema].sort_values(["team", "date"], kind="stable").reset_index(drop=True)


def _validate_combined(df: pd.DataFrame, dataset: str) -> list[str]:
    errors: list[str] = []
    if df.empty:
        return ["at least 1 valid snapshot is required"]
    errors.extend(validate_fifa_rankings(df) if dataset == "fifa_rankings" else validate_world_football_elo(df))
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.notna().sum() == 0:
        errors.append("combined output has no parseable dates")
    if df.duplicated(["date", "team"]).any():
        errors.append("combined output contains duplicate date/team rows")
    latest_date = dates.max()
    latest_count = int(df.loc[dates.eq(latest_date), "team"].nunique()) if pd.notna(latest_date) else 0
    if latest_count < 30:
        errors.append(f"latest snapshot must contain at least 30 teams, found {latest_count}")
    return errors


def _dataset_result(
    *,
    dataset: str,
    input_dir: Path,
    output_path: Path,
    file_results: list[SnapshotFileResult],
    frame: pd.DataFrame,
    imported_frame: pd.DataFrame,
    config: dict[str, Any],
    output_written: bool,
    preserved_existing: bool,
    combined_errors: list[str],
) -> DatasetImportResult:
    imported = [result for result in file_results if result.status in {"imported", "imported_with_missing_points"}]
    imported_with_missing_points = [result for result in file_results if result.status == "imported_with_missing_points"]
    imported_statuses = {"imported", "imported_with_missing_points"}
    failed = [result for result in file_results if result.status not in imported_statuses]
    dates = pd.to_datetime(frame.get("date"), errors="coerce") if not frame.empty and "date" in frame.columns else pd.Series(dtype="datetime64[ns]")
    unique_teams = int(frame.get("team", pd.Series(dtype=object)).map(normalize_team_name).dropna().nunique()) if not frame.empty else 0
    latest_date = dates.max() if dates.notna().any() else pd.NaT
    latest_count = int(frame.loc[dates.eq(latest_date), "team"].map(normalize_team_name).nunique()) if pd.notna(latest_date) else 0
    coverage = _rating_coverage(frame, dataset, config)
    can_test = (
        imported_frame["date"].nunique() >= 2
        and coverage["historical_training_coverage"] >= 0.5
        and latest_count >= 30
        and not combined_errors
    ) if not imported_frame.empty else False
    return DatasetImportResult(
        dataset=dataset,
        input_dir=input_dir,
        output_path=output_path,
        files_found=len(file_results),
        snapshots_imported=len(imported),
        snapshots_imported_with_missing_points=len(imported_with_missing_points),
        snapshots_failed=len(failed),
        snapshot_files="; ".join(result.path.name for result in file_results),
        snapshot_statuses="; ".join(f"{result.path.name}:{result.status}" for result in file_results),
        total_rows=len(frame),
        unique_teams=unique_teams,
        date_min=str(dates.min().date()) if dates.notna().any() else "",
        date_max=str(dates.max().date()) if dates.notna().any() else "",
        latest_snapshot_date=str(latest_date.date()) if pd.notna(latest_date) else "",
        latest_snapshot_team_count=latest_count,
        historical_training_coverage=coverage["historical_training_coverage"],
        worldcup_2026_coverage=coverage["worldcup_2026_coverage"],
        null_rate_by_year=coverage["null_rate_by_year"],
        ablation_can_meaningfully_test=can_test,
        failed_files="; ".join(result.path.name for result in failed),
        failed_reasons="; ".join(f"{result.path.name}: {result.reason}" for result in failed),
        output_written=output_written,
        preserved_existing=preserved_existing,
        combined_errors="; ".join(combined_errors),
    )


def _rating_coverage(frame: pd.DataFrame, dataset: str, config: dict[str, Any]) -> dict[str, Any]:
    if frame.empty:
        return {"historical_training_coverage": 0.0, "worldcup_2026_coverage": 0.0, "null_rate_by_year": ""}
    train = _read_first_existing(
        config,
        "match_training_dataset_advanced_parquet",
        "match_training_dataset_advanced_csv",
        "match_training_dataset_parquet",
        "match_training_dataset_csv",
    )
    prediction = _read_first_existing(config, "worldcup_2026_prediction_input_advanced", "worldcup_2026_prediction_input")
    train_coverage, by_year = _coverage_for_matches(train, frame, dataset)
    prediction_coverage, _ = _coverage_for_matches(prediction, frame, dataset)
    return {
        "historical_training_coverage": train_coverage,
        "worldcup_2026_coverage": prediction_coverage,
        "null_rate_by_year": by_year,
    }


def _coverage_for_matches(matches: pd.DataFrame, ratings: pd.DataFrame, dataset: str) -> tuple[float, str]:
    if matches.empty or ratings.empty:
        return 0.0, ""
    working = matches.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    if dataset == "fifa_rankings":
        joined = join_fifa_rankings_asof(working, ratings)
        columns = ["home_fifa_rank", "away_fifa_rank"]
    else:
        joined = join_world_football_elo_asof(working, ratings)
        columns = ["home_external_elo", "away_external_elo"]
    available = joined[columns].notna().all(axis=1)
    coverage = float(available.mean()) if len(available) else 0.0
    by_year_rows = []
    for year, values in available.groupby(working["date"].dt.year):
        if pd.isna(year):
            continue
        by_year_rows.append(f"{int(year)}:{1.0 - float(values.mean()):.3f}")
    return coverage, ";".join(by_year_rows)


def _write_coverage_report(results: list[DatasetImportResult], config: dict[str, Any]) -> dict[str, Path]:
    report = pd.DataFrame([result.as_row() for result in results])
    csv_path = _optional_config_path(config, "historical_rating_coverage_report_csv", "data/reports/historical_rating_coverage_report.csv")
    md_path = _optional_config_path(config, "historical_rating_coverage_report_md", "data/reports/historical_rating_coverage_report.md")
    write_dataframe(report, csv_path)
    _write_coverage_markdown(report, md_path)
    return {"csv": csv_path, "md": md_path}


def _write_coverage_markdown(report: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Historical Rating Coverage Report",
        "",
        "## Summary",
        "",
        "| dataset | files_found | snapshots_imported | snapshots_imported_with_missing_points | snapshots_failed | date_min | date_max | total_rows | unique_teams | latest_snapshot_date | latest_snapshot_team_count | historical_training_coverage | worldcup_2026_coverage | ablation_can_meaningfully_test | output_written | preserved_existing |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.files_found} | {row.snapshots_imported} | {row.snapshots_imported_with_missing_points} | {row.snapshots_failed} | "
            f"{row.date_min} | {row.date_max} | {row.total_rows} | {row.unique_teams} | {row.latest_snapshot_date} | "
            f"{row.latest_snapshot_team_count} | {row.historical_training_coverage:.3f} | {row.worldcup_2026_coverage:.3f} | "
            f"{row.ablation_can_meaningfully_test} | {row.output_written} | {row.preserved_existing} |"
        )
    lines.extend(["", "## Null Rate By Year", ""])
    for row in report.itertuples(index=False):
        lines.append(f"- `{row.dataset}`: {row.null_rate_by_year or 'not available'}")
    lines.extend(["", "## Failed Files", ""])
    failures = report[report["failed_files"].fillna("").astype(str).ne("")]
    if failures.empty:
        lines.append("No failed snapshot files.")
    else:
        for row in failures.itertuples(index=False):
            lines.append(f"- `{row.dataset}`: {row.failed_reasons}")
    lines.extend(["", "## Snapshot Files", ""])
    for row in report.itertuples(index=False):
        lines.append(f"- `{row.dataset}`: {row.snapshot_statuses or 'no snapshot files found'}")
    lines.extend(["", "## Ablation Readiness", ""])
    for row in report.itertuples(index=False):
        if row.ablation_can_meaningfully_test:
            lines.append(f"- `{row.dataset}` has enough historical coverage for FIFA/Elo ablation to become meaningful.")
        else:
            lines.append(
                f"- `{row.dataset}` is not yet ready for meaningful ablation. Add more dated historical snapshots before validation matches."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_rebuild_steps(config: dict[str, Any]) -> None:
    logger.info("Running post-import external data validation")
    validate_external_data(config=config)
    logger.info("Rebuilding advanced features")
    build_advanced_features(config=config)
    logger.info("Writing feature null-rate report")
    generate_feature_null_rate_report(config=config)
    logger.info("Running ablation")
    run_ablation(config=config)


def _filter_date_range(df: pd.DataFrame, *, min_date: str | None, max_date: str | None) -> pd.DataFrame:
    if min_date is None and max_date is None:
        return df
    output = df.copy()
    dates = pd.to_datetime(output["date"], errors="coerce")
    if min_date is not None:
        dates_min = pd.to_datetime(min_date, errors="raise")
        output = output.loc[dates >= dates_min].copy()
        dates = dates.loc[output.index]
    if max_date is not None:
        dates_max = pd.to_datetime(max_date, errors="raise")
        output = output.loc[dates <= dates_max].copy()
    return output


def _numeric_series(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def _optional_numeric_parse_failures(raw_values: pd.Series, numeric_values: pd.Series) -> int:
    cleaned = raw_values.astype("string").str.strip()
    missing = cleaned.isna() | cleaned.isin(["", "-", "NA", "N/A", "na", "n/a", "null", "None", "nan"])
    return int((~missing & numeric_values.isna()).sum())


def _date_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().any():
        return parsed
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    return parsed


def _empty_frame(dataset: str) -> pd.DataFrame:
    schema = FIFA_RANKINGS_SCHEMA if dataset == "fifa_rankings" else WORLD_FOOTBALL_ELO_SCHEMA
    return pd.DataFrame(columns=schema)


def _read_existing_output(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(output_path)
    except Exception:
        return pd.DataFrame()


def _read_first_existing(config: dict[str, Any], *keys: str) -> pd.DataFrame:
    for key in keys:
        try:
            path = config_path(config, key)
        except KeyError:
            continue
        if path.exists():
            try:
                return read_dataframe(path)
            except Exception as exc:
                logger.warning("Could not read %s while building historical rating coverage report: %s", path, exc)
                continue
    return pd.DataFrame()


def _save_debug_candidate(df: pd.DataFrame, dataset: str) -> None:
    debug_dir = resolve_project_path("data/raw/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d%H%M%S")
    df.to_csv(debug_dir / f"{dataset}_historical_snapshot_candidate_failed_{timestamp}.csv", index=False)


def _optional_config_path(config: dict[str, Any], key: str, default: str) -> Path:
    try:
        return config_path(config, key)
    except KeyError:
        return resolve_project_path(default)


def _utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()
