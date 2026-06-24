from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from dashboard.live_api import generate_live_predictions
from src.config import resolve_project_path
from src.data_sources.api_football_client import ApiFootballClient
from src.data_sources.config import (
    API_FOOTBALL_ODDS_2026_PATH,
    MARKET_ODDS_EXTERNAL_PATH,
    RAW_API_FOOTBALL_DIR,
    RAW_SPORTMONKS_DIR,
    RAW_THE_ODDS_API_DIR,
    SPORTMONKS_ODDS_2026_PATH,
    THE_ODDS_API_ODDS_2026_PATH,
    ensure_api_directories,
    get_credentials,
    safe_key_label,
    write_markdown_report,
)
from src.data_sources.coverage import load_active_2026_fixtures
from src.data_sources.merge_market_odds import FINAL_MARKET_ODDS_SCHEMA, merge_staged_market_odds
from src.data_sources.odds_transform import (
    MARKET_ODDS_STAGING_SCHEMA,
    api_football_odds_to_market_rows,
    normalize_market_odds_rows,
    save_raw_payload,
    the_odds_api_to_market_rows,
    write_market_staging,
)
from src.data_sources.sportmonks_client import SportmonksClient
from src.data_sources.team_normalization import normalize_api_team_name
from src.data_sources.the_odds_api_client import TheOddsApiClient
from src.data_sources.validators import summarize_market_odds_validation, validate_coverage_against_active_fixtures
from src.features.null_rate_report import generate_feature_null_rate_report


PROVIDER_DISCOVERY_RAW_DIR = resolve_project_path("data/raw/provider_discovery")
ODDS_PROVIDER_DISCOVERY_REPORT = resolve_project_path("data/reports/odds_provider_competition_discovery.md")
REFRESH_MARKET_ODDS_REPORT = resolve_project_path("data/reports/refresh_market_odds_report.md")
MISSING_ODDS_TEMPLATE_PATH = resolve_project_path("data/staging/missing_market_odds_template_2026.csv")
MISSING_ODDS_TEMPLATE_REPORT = resolve_project_path("data/reports/missing_market_odds_template_report.md")
MANUAL_MARKET_ODDS_VALIDATED_PATH = resolve_project_path("data/staging/manual_market_odds_validated_2026.csv")
MANUAL_MARKET_ODDS_VALIDATION_ERRORS = resolve_project_path("data/reports/manual_market_odds_validation_errors.md")
COMBINED_MARKET_ODDS_2026_PATH = resolve_project_path("data/staging/combined_market_odds_2026.csv")
REFRESH_MARKET_ODDS_BACKUP_PATH = resolve_project_path("data/external/market_odds_backup_before_refresh_market_odds.csv")
ODDS_PROVIDER_CONFIG_PATH = resolve_project_path("config/odds_provider_keys.json")


@dataclass(frozen=True)
class ProviderFetchResult:
    provider: str
    path: Path
    rows: int
    valid_rows: int
    invalid_rows: int
    safe_to_merge: bool
    error: str = ""


def ensure_odds_provider_config(path: Path = ODDS_PROVIDER_CONFIG_PATH) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    default = {
        "the_odds_api": {
            "enabled": True,
            "competition_keys": [],
            "excluded_keys": ["soccer_fifa_club_world_cup"],
        },
        "api_football": {
            "enabled": True,
            "league_ids": [],
        },
        "sportmonks": {
            "enabled": True,
            "league_ids": [],
        },
    }
    path.write_text(json.dumps(default, indent=2), encoding="utf-8")
    return path


def load_odds_provider_config(path: Path = ODDS_PROVIDER_CONFIG_PATH) -> dict[str, Any]:
    ensure_odds_provider_config(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def discover_odds_competitions() -> Path:
    ensure_api_directories()
    ensure_odds_provider_config()
    PROVIDER_DISCOVERY_RAW_DIR.mkdir(parents=True, exist_ok=True)
    credentials = get_credentials()
    rows: list[dict[str, Any]] = []
    notes: list[str] = []

    if credentials.the_odds_api_available:
        client = TheOddsApiClient(raw_dir=RAW_THE_ODDS_API_DIR)
        try:
            payload = client.get_sports(all=True)
            raw_path = _save_discovery_payload(payload, "the_odds_api_sports")
            for item in payload if isinstance(payload, list) else payload.get("data", []) if isinstance(payload, dict) else []:
                if not isinstance(item, dict):
                    continue
                key = str(item.get("key") or "")
                title = str(item.get("title") or item.get("description") or "")
                text = f"{key} {title}"
                if "soccer" not in text.lower() and "football" not in text.lower():
                    continue
                rows.append(_competition_row("the_odds_api", True, "/sports", key, title, item.get("group"), None, raw_path))
        except Exception as exc:
            notes.append(f"The Odds API discovery failed: {exc}")
    else:
        rows.append(_missing_provider_row("the_odds_api", "/sports"))

    if credentials.api_football_available:
        client = ApiFootballClient(raw_dir=RAW_API_FOOTBALL_DIR)
        try:
            payload = client.get_leagues(search="World Cup")
            raw_path = _save_discovery_payload(payload, "api_football_leagues_world_cup")
            for item in payload.get("response", []) or []:
                league = item.get("league", {}) if isinstance(item, dict) else {}
                country = item.get("country", {}) if isinstance(item, dict) else {}
                seasons = item.get("seasons", []) if isinstance(item, dict) else []
                season_text = ",".join(str(season.get("year")) for season in seasons if isinstance(season, dict) and season.get("year"))
                rows.append(
                    _competition_row(
                        "api_football",
                        True,
                        "/leagues?search=World Cup",
                        league.get("id"),
                        league.get("name"),
                        country.get("name"),
                        season_text,
                        raw_path,
                    )
                )
        except Exception as exc:
            notes.append(f"API-Football discovery failed: {exc}")
    else:
        rows.append(_missing_provider_row("api_football", "/leagues?search=World Cup"))

    if credentials.sportmonks_available:
        client = SportmonksClient(raw_dir=RAW_SPORTMONKS_DIR)
        try:
            payload = client.search_leagues("World Cup")
            raw_path = _save_discovery_payload(payload, "sportmonks_leagues_world_cup")
            for item in _sportmonks_items(payload):
                rows.append(
                    _competition_row(
                        "sportmonks",
                        True,
                        "/leagues/search/World Cup",
                        item.get("id"),
                        item.get("name"),
                        item.get("country", {}).get("name") if isinstance(item.get("country"), dict) else item.get("country_name"),
                        item.get("season") or item.get("currentseason", {}).get("name") if isinstance(item.get("currentseason"), dict) else None,
                        raw_path,
                    )
                )
        except Exception as exc:
            notes.append(f"Sportmonks discovery failed: {exc}")
    else:
        rows.append(_missing_provider_row("sportmonks", "/leagues/search/World Cup"))

    lines = [
        "# Odds Provider Competition Discovery",
        "",
        "This report lists provider competition keys/league IDs. It does not modify production odds.",
        "",
        "## API Keys",
        "",
        f"- API_FOOTBALL_KEY: `{safe_key_label(credentials.api_football_key)}`",
        f"- THE_ODDS_API_KEY: `{safe_key_label(credentials.the_odds_api_key)}`",
        f"- SPORTMONKS_API_TOKEN: `{safe_key_label(credentials.sportmonks_api_token)}`",
        "",
        "## Competitions",
        "",
        _markdown_table(
            rows,
            [
                "provider",
                "api_key_detected",
                "endpoint",
                "raw_key_or_league_id",
                "competition_name",
                "country_or_region",
                "season_availability",
                "looks_like_national_world_cup",
                "looks_like_club_world_cup",
                "recommendation",
            ],
        ),
    ]
    if notes:
        lines.extend(["", "## Provider Errors", "", *[f"- {note}" for note in notes]])
    return write_markdown_report(ODDS_PROVIDER_DISCOVERY_REPORT, lines)


def refresh_market_odds(
    *,
    provider: str = "all",
    season: int = 2026,
    prefer_api: bool = False,
    dry_run: bool = False,
    skip_merge: bool = False,
    rerun_reports: bool = True,
    rerun_live_predictions: bool = True,
) -> dict[str, Any]:
    ensure_api_directories()
    ensure_odds_provider_config()
    selected = _selected_providers(provider)
    before_coverage = active_fixture_coverage()
    results: list[ProviderFetchResult] = []
    staged_frames: list[pd.DataFrame] = []

    for name in selected:
        result, frame = _fetch_provider_odds(name, season)
        results.append(result)
        if not frame.empty:
            staged_frames.append(_valid_row_subset(frame))

    combined = normalize_market_odds_rows(pd.concat(staged_frames, ignore_index=True, sort=False).to_dict("records")) if staged_frames else pd.DataFrame(columns=MARKET_ODDS_STAGING_SCHEMA)
    combined = combined.drop_duplicates(["date", "home_team", "away_team"], keep="last") if not combined.empty else combined
    write_market_staging(combined, COMBINED_MARKET_ODDS_2026_PATH)
    combined_validation = summarize_market_odds_validation(combined)

    merge_summary = {
        "MARKET_ODDS_ROWS_BEFORE": len(_read_market_odds()),
        "STAGED_ROWS": len(combined),
        "ROWS_ADDED": 0,
        "ROWS_REPLACED": 0,
        "ROWS_AFTER": len(_read_market_odds()),
        "DUPLICATES": combined_validation.duplicate_rows,
        "BAD_ODDS": combined_validation.bad_odds,
        "VALIDATION_PASSED": combined_validation.safe_to_merge,
    }
    if combined_validation.safe_to_merge and not dry_run and not skip_merge:
        merge_summary = merge_staged_market_odds(
            COMBINED_MARKET_ODDS_2026_PATH,
            backup_path=REFRESH_MARKET_ODDS_BACKUP_PATH,
            prefer_api=prefer_api,
        )

    template_path, template_summary = generate_missing_odds_template()
    if rerun_reports:
        generate_feature_null_rate_report()
    if rerun_live_predictions:
        generate_live_predictions()
    after_coverage = active_fixture_coverage()
    summary = {
        "PROVIDERS_CHECKED": ",".join(selected),
        "STAGED_ROWS": int(sum(result.rows for result in results)),
        "VALID_ROWS": int(sum(result.valid_rows for result in results)),
        "INVALID_ROWS": int(sum(result.invalid_rows for result in results)),
        "ROWS_ADDED": int(merge_summary.get("ROWS_ADDED", 0)),
        "ROWS_REPLACED": int(merge_summary.get("ROWS_REPLACED", 0)),
        "ROWS_AFTER": int(merge_summary.get("ROWS_AFTER", len(_read_market_odds()))),
        "ACTIVE_FIXTURE_COVERAGE_BEFORE": before_coverage["coverage"],
        "ACTIVE_FIXTURE_COVERAGE_AFTER": after_coverage["coverage"],
        "MISSING_ACTIVE_FIXTURES": int(template_summary["missing_fixtures"]),
        "SUCCESS": bool((combined_validation.safe_to_merge and (dry_run or skip_merge or merge_summary.get("VALIDATION_PASSED"))) or sum(result.rows for result in results) == 0),
    }
    _write_refresh_report(summary, results, merge_summary, template_path, dry_run=dry_run, skip_merge=skip_merge)
    return summary


def generate_missing_odds_template(
    *,
    output_path: Path = MISSING_ODDS_TEMPLATE_PATH,
    report_path: Path = MISSING_ODDS_TEMPLATE_REPORT,
) -> tuple[Path, dict[str, Any]]:
    active = load_active_live_fixtures()
    market = _read_market_odds()
    coverage = validate_coverage_against_active_fixtures(market, active)
    missing = coverage["missing_fixtures"].copy()
    today = datetime.now(timezone.utc).date().isoformat()
    if missing.empty:
        template = pd.DataFrame(columns=["date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds", "bookmaker", "source", "updated_at", "notes"])
    else:
        template = missing[["date", "home_team", "away_team"]].drop_duplicates().copy()
        template["home_odds"] = ""
        template["draw_odds"] = ""
        template["away_odds"] = ""
        template["bookmaker"] = "manual_market_average"
        template["source"] = "manual_verified"
        template["updated_at"] = today
        template["notes"] = "Fill only with verified decimal 1X2 market odds. Do not guess."
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    summary = {
        "total_active_fixtures": int(coverage["total_fixtures"]),
        "odds_covered_fixtures": int(coverage["matched_fixtures"]),
        "missing_fixtures": int(len(template)),
        "template_path": str(output_path),
    }
    lines = [
        "# Missing Market Odds Template Report",
        "",
        f"- total_active_fixtures: `{summary['total_active_fixtures']}`",
        f"- odds_covered_fixtures: `{summary['odds_covered_fixtures']}`",
        f"- missing_fixtures: `{summary['missing_fixtures']}`",
        f"- template_path: `{output_path}`",
        "",
        "Fill only with verified decimal 1X2 market odds. Do not guess.",
    ]
    write_markdown_report(report_path, lines)
    return output_path, summary


def validate_manual_market_odds(input_path: str | Path) -> dict[str, Any]:
    input_path = Path(input_path)
    df = pd.read_csv(input_path, low_memory=False) if input_path.exists() else pd.DataFrame()
    errors: list[str] = []
    required = ["date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds", "bookmaker", "source", "updated_at"]
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        errors.append(f"Missing columns: {', '.join(missing_columns)}")
        filled = pd.DataFrame(columns=required)
    else:
        odds_cols = ["home_odds", "draw_odds", "away_odds"]
        filled_mask = df[odds_cols].notna().all(axis=1) & df[odds_cols].astype(str).apply(lambda col: col.str.strip().ne("")).all(axis=1)
        filled = df[filled_mask].copy()
    active = load_active_live_fixtures()
    active_keys = set(_fixture_key(active))
    if not filled.empty:
        normalized = normalize_market_odds_rows(filled.to_dict("records"))
        odds = normalized[["home_odds", "draw_odds", "away_odds"]].apply(pd.to_numeric, errors="coerce")
        bad_odds = odds.isna().any(axis=1) | odds.le(1).any(axis=1) | odds.ge(100).any(axis=1)
        for idx in normalized[bad_odds].index:
            errors.append(f"Invalid odds on row {idx + 2}; odds must be numeric decimal odds > 1 and below 100.")
        duplicate_count = int(normalized.duplicated(["date", "home_team", "away_team"], keep=False).sum())
        if duplicate_count:
            errors.append(f"Duplicate fixture rows detected: {duplicate_count}")
        empty_bookmaker = normalized["bookmaker"].astype(str).str.strip().isin(["", "nan", "None"]).sum()
        empty_source = normalized["source"].astype(str).str.strip().isin(["", "nan", "None"]).sum()
        if empty_bookmaker:
            errors.append("Bookmaker must be non-empty for all filled rows.")
        if empty_source:
            errors.append("Source must be non-empty for all filled rows.")
        for key in _fixture_key(normalized):
            if key not in active_keys:
                errors.append(f"Filled odds row does not match an active fixture exactly: {key}")
        validated = normalized[~bad_odds].copy()
    else:
        validated = pd.DataFrame(columns=MARKET_ODDS_STAGING_SCHEMA)
    if errors:
        lines = ["# Manual Market Odds Validation Errors", "", *[f"- {error}" for error in errors]]
        write_markdown_report(MANUAL_MARKET_ODDS_VALIDATION_ERRORS, lines)
    else:
        write_market_staging(validated, MANUAL_MARKET_ODDS_VALIDATED_PATH)
    return {
        "INPUT_ROWS": int(len(df)),
        "FILLED_ROWS": int(len(filled)),
        "VALID_ROWS": int(len(validated)) if not errors else 0,
        "ERRORS": len(errors),
        "OUTPUT_PATH": str(MANUAL_MARKET_ODDS_VALIDATED_PATH if not errors else MANUAL_MARKET_ODDS_VALIDATION_ERRORS),
        "SUCCESS": not errors,
    }


def active_fixture_coverage() -> dict[str, Any]:
    active = load_active_live_fixtures()
    market = _read_market_odds()
    coverage = validate_coverage_against_active_fixtures(market, active)
    return {
        "total": int(coverage["total_fixtures"]),
        "matched": int(coverage["matched_fixtures"]),
        "coverage": float(coverage["coverage"]),
    }


def load_active_live_fixtures() -> pd.DataFrame:
    live_path = resolve_project_path("data/live/live_predictions.csv")
    if live_path.exists():
        live = pd.read_csv(live_path, low_memory=False)
        if {"date", "home_team", "away_team", "status"}.issubset(live.columns):
            status = live["status"].astype(str).str.lower()
            active = live[~status.str.contains("finished|full", na=False)].copy()
            if not active.empty:
                active["date"] = pd.to_datetime(active["date"], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
                active["home_team"] = active["home_team"].map(normalize_api_team_name)
                active["away_team"] = active["away_team"].map(normalize_api_team_name)
                return active[["date", "home_team", "away_team"]].dropna().drop_duplicates().reset_index(drop=True)
    return load_active_2026_fixtures()


def looks_like_club_world_cup(key: Any, name: Any) -> bool:
    text = f"{key or ''} {name or ''}".lower()
    return "club world cup" in text or "club_world_cup" in text


def looks_like_national_world_cup(key: Any, name: Any) -> bool:
    text = f"{key or ''} {name or ''}".lower()
    return "world cup" in text and not looks_like_club_world_cup(key, name)


def _fetch_provider_odds(provider: str, season: int) -> tuple[ProviderFetchResult, pd.DataFrame]:
    try:
        config = load_odds_provider_config()
        provider_config = config.get(provider, {}) if isinstance(config, dict) else {}
        if isinstance(provider_config, dict) and provider_config.get("enabled") is False:
            path = _provider_path(provider)
            return (
                ProviderFetchResult(
                    provider=provider,
                    path=path,
                    rows=0,
                    valid_rows=0,
                    invalid_rows=0,
                    safe_to_merge=False,
                    error="provider disabled in config/odds_provider_keys.json",
                ),
                pd.DataFrame(columns=MARKET_ODDS_STAGING_SCHEMA),
            )
        if provider == "api_football":
            client = ApiFootballClient()
            odds, raw_path, _ = _fetch_api_football_odds_with_config(client, config, season)
            odds = _enrich_api_football_odds_from_live_fixtures(odds)
            path = API_FOOTBALL_ODDS_2026_PATH
        elif provider == "the_odds_api":
            client = TheOddsApiClient()
            odds, raw_path, _ = _fetch_the_odds_api_odds_with_config(client, config)
            path = THE_ODDS_API_ODDS_2026_PATH
        elif provider == "sportmonks":
            client = SportmonksClient()
            odds, raw_path = client.fetch_worldcup_2026_odds_if_available()
            path = SPORTMONKS_ODDS_2026_PATH
        else:
            raise ValueError(f"Unknown provider: {provider}")
        write_market_staging(odds, path)
        normalized = normalize_market_odds_rows(odds.to_dict("records")) if not odds.empty else pd.DataFrame(columns=MARKET_ODDS_STAGING_SCHEMA)
        valid = _valid_row_subset(normalized)
        validation = summarize_market_odds_validation(normalized)
        return (
            ProviderFetchResult(
                provider=provider,
                path=path,
                rows=len(normalized),
                valid_rows=len(valid),
                invalid_rows=max(0, len(normalized) - len(valid)),
                safe_to_merge=validation.safe_to_merge,
            ),
            normalized,
        )
    except Exception as exc:
        path = _provider_path(provider)
        return ProviderFetchResult(provider=provider, path=path, rows=0, valid_rows=0, invalid_rows=0, safe_to_merge=False, error=str(exc)), pd.DataFrame(columns=MARKET_ODDS_STAGING_SCHEMA)


def _fetch_api_football_odds_with_config(
    client: ApiFootballClient,
    config: dict[str, Any],
    season: int,
) -> tuple[pd.DataFrame, Path, list[dict[str, Any]]]:
    provider_config = config.get("api_football", {}) if isinstance(config, dict) else {}
    league_ids = _clean_int_list(provider_config.get("league_ids") if isinstance(provider_config, dict) else None)
    if not league_ids:
        return client.fetch_worldcup_odds(season)

    frames: list[pd.DataFrame] = []
    raw_paths: list[Path] = []
    candidates: list[dict[str, Any]] = []
    for league_id in league_ids:
        payload = client.get_odds(league=league_id, season=season)
        if isinstance(payload, dict):
            payload["configured_league_id"] = league_id
            payload["configured_season"] = season
        raw_path = save_raw_payload(payload, client.raw_dir, f"worldcup_{season}_odds_league_{league_id}")
        frames.append(api_football_odds_to_market_rows(payload, raw_payload_file=str(raw_path)))
        raw_paths.append(raw_path)
        candidates.append({"league_id": league_id, "source": "config/odds_provider_keys.json"})

    odds = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=MARKET_ODDS_STAGING_SCHEMA)
    first_path = raw_paths[0] if raw_paths else client.raw_dir / f"worldcup_{season}_odds_configured_empty.json"
    return normalize_market_odds_rows(odds.to_dict("records")), first_path, candidates


def _fetch_the_odds_api_odds_with_config(
    client: TheOddsApiClient,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, Path, str | None]:
    provider_config = config.get("the_odds_api", {}) if isinstance(config, dict) else {}
    competition_keys = _clean_string_list(provider_config.get("competition_keys") if isinstance(provider_config, dict) else None)
    excluded_keys = set(_clean_string_list(provider_config.get("excluded_keys") if isinstance(provider_config, dict) else None))
    competition_keys = [key for key in competition_keys if key not in excluded_keys]
    if not competition_keys:
        return client.fetch_worldcup_2026_odds_if_available()

    frames: list[pd.DataFrame] = []
    raw_paths: list[Path] = []
    for sport_key in competition_keys:
        payload = client.get_odds(sport_key)
        raw_path = save_raw_payload(payload, client.raw_dir, f"worldcup_2026_odds_{sport_key}")
        frames.append(the_odds_api_to_market_rows(payload, raw_payload_file=str(raw_path)))
        raw_paths.append(raw_path)
    odds = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=MARKET_ODDS_STAGING_SCHEMA)
    first_path = raw_paths[0] if raw_paths else client.raw_dir / "worldcup_2026_odds_configured_empty.json"
    return normalize_market_odds_rows(odds.to_dict("records")), first_path, ",".join(competition_keys)


def _enrich_api_football_odds_from_live_fixtures(odds: pd.DataFrame) -> pd.DataFrame:
    if odds.empty or "api_fixture_id" not in odds.columns:
        return odds
    frames = []
    for path in [resolve_project_path("data/live/live_scores.csv"), resolve_project_path("data/live/live_fixtures.csv")]:
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            if {"fixture_id", "date", "home_team", "away_team"}.issubset(frame.columns):
                frames.append(frame[["fixture_id", "date", "home_team", "away_team"]])
    if not frames:
        return odds
    lookup = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates("fixture_id", keep="last")
    lookup["_fixture_key"] = lookup["fixture_id"].astype(str)
    output = odds.copy()
    output["_fixture_key"] = output["api_fixture_id"].astype(str)
    merged = output.merge(lookup.drop(columns=["fixture_id"]), on="_fixture_key", how="left", suffixes=("", "_fixture"))
    for column in ("date", "home_team", "away_team"):
        fixture_col = f"{column}_fixture"
        if fixture_col in merged.columns:
            current = merged[column] if column in merged.columns else pd.Series(pd.NA, index=merged.index)
            merged[column] = current.where(current.notna() & current.astype(str).str.strip().ne(""), merged[fixture_col])
    return merged.drop(columns=[column for column in merged.columns if column.endswith("_fixture") or column == "_fixture_key"], errors="ignore")


def _valid_row_subset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=MARKET_ODDS_STAGING_SCHEMA)
    normalized = normalize_market_odds_rows(df.to_dict("records"))
    odds = normalized[["home_odds", "draw_odds", "away_odds"]].apply(pd.to_numeric, errors="coerce")
    valid = (
        normalized["date"].notna()
        & normalized["home_team"].notna()
        & normalized["away_team"].notna()
        & ~normalized.apply(lambda row: looks_like_club_world_cup(row.get("api_fixture_id"), row.get("bookmaker")), axis=1)
        & odds.notna().all(axis=1)
        & odds.gt(1).all(axis=1)
    )
    return normalized[valid].copy()


def _selected_providers(provider: str) -> list[str]:
    valid = ["api_football", "the_odds_api", "sportmonks"]
    if provider == "all":
        return valid
    if provider not in valid:
        raise ValueError(f"Unknown provider: {provider}")
    return [provider]


def _clean_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    output: list[int] = []
    for value in values:
        try:
            output.append(int(value))
        except (TypeError, ValueError):
            continue
    return output


def _clean_string_list(values: Any) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    return [str(value).strip() for value in values if str(value).strip()]


def _provider_path(provider: str) -> Path:
    return {
        "api_football": API_FOOTBALL_ODDS_2026_PATH,
        "the_odds_api": THE_ODDS_API_ODDS_2026_PATH,
        "sportmonks": SPORTMONKS_ODDS_2026_PATH,
    }[provider]


def _read_market_odds() -> pd.DataFrame:
    if not MARKET_ODDS_EXTERNAL_PATH.exists():
        return pd.DataFrame(columns=FINAL_MARKET_ODDS_SCHEMA)
    return pd.read_csv(MARKET_ODDS_EXTERNAL_PATH, low_memory=False)


def _write_refresh_report(
    summary: dict[str, Any],
    results: list[ProviderFetchResult],
    merge_summary: dict[str, Any],
    template_path: Path,
    *,
    dry_run: bool,
    skip_merge: bool,
) -> Path:
    rows = [result.__dict__ | {"path": str(result.path)} for result in results]
    lines = [
        "# Refresh Market Odds Report",
        "",
        "This command stages provider data first, validates it, and merges only valid odds rows.",
        "",
        f"- dry_run: `{dry_run}`",
        f"- skip_merge: `{skip_merge}`",
        f"- missing_odds_template: `{template_path}`",
        "",
        "## Summary",
        "",
        *[f"- {key}={value}" for key, value in summary.items()],
        "",
        "## Provider Fetch Results",
        "",
        _markdown_table(rows, ["provider", "rows", "valid_rows", "invalid_rows", "safe_to_merge", "path", "error"]),
        "",
        "## Merge Status",
        "",
        *[f"- {key}={value}" for key, value in merge_summary.items()],
    ]
    return write_markdown_report(REFRESH_MARKET_ODDS_REPORT, lines)


def _competition_row(
    provider: str,
    key_detected: bool,
    endpoint: str,
    key_or_id: Any,
    name: Any,
    country: Any,
    seasons: Any,
    raw_path: Path,
) -> dict[str, Any]:
    club = looks_like_club_world_cup(key_or_id, name)
    national = looks_like_national_world_cup(key_or_id, name)
    if club:
        recommendation = "exclude: Club World Cup is not national-team World Cup"
    elif national:
        recommendation = "candidate: verify season and odds availability"
    else:
        recommendation = "not a clear national-team World Cup candidate"
    return {
        "provider": provider,
        "api_key_detected": key_detected,
        "endpoint": endpoint,
        "raw_key_or_league_id": key_or_id,
        "competition_name": name,
        "country_or_region": country,
        "season_availability": seasons,
        "looks_like_national_world_cup": national,
        "looks_like_club_world_cup": club,
        "recommendation": recommendation,
        "raw_response": str(raw_path),
    }


def _missing_provider_row(provider: str, endpoint: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "api_key_detected": False,
        "endpoint": endpoint,
        "raw_key_or_league_id": "",
        "competition_name": "",
        "country_or_region": "",
        "season_availability": "",
        "looks_like_national_world_cup": False,
        "looks_like_club_world_cup": False,
        "recommendation": "key missing; provider skipped",
    }


def _save_discovery_payload(payload: Any, label: str) -> Path:
    return save_raw_payload(payload, PROVIDER_DISCOVERY_RAW_DIR, label)


def _sportmonks_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    response = payload.get("response") if isinstance(payload, dict) else None
    return [item for item in response if isinstance(item, dict)] if isinstance(response, list) else []


def _fixture_key(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    dates = pd.to_datetime(df["date"], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    home = df["home_team"].map(normalize_api_team_name)
    away = df["away_team"].map(normalize_api_team_name)
    return (dates.fillna("") + "|" + home.fillna("") + "|" + away.fillna("")).tolist()


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    output = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        output.append("| " + " | ".join("" if row.get(column) is None else str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(output)
