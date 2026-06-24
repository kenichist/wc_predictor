from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import config_path, load_config
from src.data_sources.api_football_client import ApiFootballClient
from src.data_sources.config import (
    API_COVERAGE_REPORT,
    API_FOOTBALL_INJURIES_LIVE_PATH,
    API_FOOTBALL_ODDS_2010_PROBE_PATH,
    API_FOOTBALL_ODDS_2026_PATH,
    API_INJURY_INGESTION_REPORT,
    API_MARKET_ODDS_INGESTION_REPORT,
    MARKET_ODDS_2010_COVERAGE_REPORT,
    MARKET_ODDS_EXTERNAL_PATH,
    SPORTMONKS_INJURIES_LIVE_PATH,
    SPORTMONKS_ODDS_2026_PATH,
    STAGING_API_COVERAGE_REPORT,
    THE_ODDS_API_ODDS_2026_PATH,
    credential_status,
    ensure_api_directories,
    markdown_table,
    write_markdown_report,
)
from src.data_sources.injury_transform import INJURY_STAGING_SCHEMA
from src.data_sources.odds_transform import MARKET_ODDS_STAGING_SCHEMA
from src.data_sources.sportmonks_client import SportmonksClient
from src.data_sources.team_normalization import normalize_api_team_name
from src.data_sources.the_odds_api_client import TheOddsApiClient
from src.data_sources.validators import (
    rows_by_year,
    summarize_market_odds_validation,
    validate_coverage_against_active_fixtures,
    validate_injury_schema,
    validate_team_normalization,
)


logger = logging.getLogger(__name__)


REQUIRED_2010_WORLD_CUP_MATCHES = [
    ("2010-06-11", "South Africa", "Mexico"),
    ("2010-06-11", "Uruguay", "France"),
    ("2010-06-12", "England", "United States"),
    ("2010-06-12", "South Korea", "Greece"),
    ("2010-06-12", "Argentina", "Nigeria"),
    ("2010-06-13", "Serbia", "Ghana"),
    ("2010-06-13", "Algeria", "Slovenia"),
    ("2010-06-13", "Germany", "Australia"),
    ("2010-06-14", "Netherlands", "Denmark"),
    ("2010-06-14", "Italy", "Paraguay"),
    ("2010-06-14", "Japan", "Cameroon"),
    ("2010-06-15", "Ivory Coast", "Portugal"),
    ("2010-06-15", "Brazil", "North Korea"),
    ("2010-06-15", "New Zealand", "Slovakia"),
    ("2010-06-16", "Honduras", "Chile"),
    ("2010-06-16", "Spain", "Switzerland"),
    ("2010-06-16", "South Africa", "Uruguay"),
    ("2010-06-17", "France", "Mexico"),
    ("2010-06-17", "Argentina", "South Korea"),
    ("2010-06-17", "Greece", "Nigeria"),
    ("2010-06-18", "England", "Algeria"),
    ("2010-06-18", "Slovenia", "United States"),
    ("2010-06-18", "Germany", "Serbia"),
    ("2010-06-19", "Netherlands", "Japan"),
    ("2010-06-19", "Ghana", "Australia"),
    ("2010-06-19", "Cameroon", "Denmark"),
    ("2010-06-20", "Slovakia", "Paraguay"),
    ("2010-06-20", "Italy", "New Zealand"),
    ("2010-06-20", "Brazil", "Ivory Coast"),
    ("2010-06-21", "Spain", "Honduras"),
    ("2010-06-21", "Portugal", "North Korea"),
    ("2010-06-21", "Chile", "Switzerland"),
    ("2010-06-22", "South Africa", "France"),
    ("2010-06-22", "Nigeria", "South Korea"),
    ("2010-06-22", "Mexico", "Uruguay"),
    ("2010-06-22", "Greece", "Argentina"),
    ("2010-06-23", "Ghana", "Germany"),
    ("2010-06-23", "Australia", "Serbia"),
    ("2010-06-23", "Slovenia", "England"),
    ("2010-06-23", "United States", "Algeria"),
    ("2010-06-24", "Denmark", "Japan"),
    ("2010-06-24", "Paraguay", "New Zealand"),
    ("2010-06-24", "Slovakia", "Italy"),
    ("2010-06-24", "Cameroon", "Netherlands"),
    ("2010-06-25", "Switzerland", "Honduras"),
    ("2010-06-25", "Portugal", "Brazil"),
    ("2010-06-25", "North Korea", "Ivory Coast"),
    ("2010-06-25", "Chile", "Spain"),
    ("2010-06-26", "United States", "Ghana"),
    ("2010-06-26", "Uruguay", "South Korea"),
    ("2010-06-27", "Argentina", "Mexico"),
    ("2010-06-27", "Germany", "England"),
    ("2010-06-28", "Brazil", "Chile"),
    ("2010-06-28", "Netherlands", "Slovakia"),
    ("2010-06-29", "Spain", "Portugal"),
    ("2010-06-29", "Paraguay", "Japan"),
    ("2010-07-02", "Netherlands", "Brazil"),
    ("2010-07-02", "Uruguay", "Ghana"),
    ("2010-07-03", "Argentina", "Germany"),
    ("2010-07-03", "Paraguay", "Spain"),
    ("2010-07-06", "Uruguay", "Netherlands"),
    ("2010-07-07", "Germany", "Spain"),
    ("2010-07-10", "Uruguay", "Germany"),
    ("2010-07-11", "Netherlands", "Spain"),
]


def run_api_coverage_check() -> Path:
    ensure_api_directories()
    status: dict[str, Any] = credential_status()
    details: list[str] = []

    api_football_client = ApiFootballClient()
    if api_football_client.available:
        try:
            candidates = api_football_client.find_world_cup_league_ids()
            status["API_FOOTBALL_LEAGUE_CANDIDATES"] = len(candidates)
            odds_2026, _, _ = api_football_client.fetch_worldcup_2026_odds()
            status["API_FOOTBALL_2026_ODDS_AVAILABLE"] = bool(len(odds_2026))
            odds_2010, _, _ = api_football_client.probe_worldcup_2010_odds()
            status["API_FOOTBALL_2010_ODDS_AVAILABLE"] = bool(len(odds_2010))
            injuries, _, _ = api_football_client.fetch_live_injuries_for_worldcup_teams()
            status["API_FOOTBALL_INJURIES_AVAILABLE"] = bool(len(injuries))
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            status["API_FOOTBALL_2026_ODDS_AVAILABLE"] = "Unknown"
            status["API_FOOTBALL_2010_ODDS_AVAILABLE"] = "Unknown"
            status["API_FOOTBALL_INJURIES_AVAILABLE"] = "Unknown"
            details.append(f"API-Football probe failed: {exc}")
    else:
        status["API_FOOTBALL_2026_ODDS_AVAILABLE"] = "Unknown"
        status["API_FOOTBALL_2010_ODDS_AVAILABLE"] = "Unknown"
        status["API_FOOTBALL_INJURIES_AVAILABLE"] = "Unknown"

    the_odds_client = TheOddsApiClient()
    status["THE_ODDS_API_2010_ODDS_AVAILABLE"] = False
    if the_odds_client.available:
        try:
            odds_2026, _, sport_key = the_odds_client.fetch_worldcup_2026_odds_if_available()
            status["THE_ODDS_API_2026_ODDS_AVAILABLE"] = bool(len(odds_2026))
            status["THE_ODDS_API_SPORT_KEY"] = sport_key or ""
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            status["THE_ODDS_API_2026_ODDS_AVAILABLE"] = "Unknown"
            details.append(f"The Odds API probe failed: {exc}")
    else:
        status["THE_ODDS_API_2026_ODDS_AVAILABLE"] = "Unknown"

    sportmonks_client = SportmonksClient()
    if sportmonks_client.available:
        try:
            odds_2026, _ = sportmonks_client.fetch_worldcup_2026_odds_if_available()
            status["SPORTMONKS_2026_ODDS_AVAILABLE"] = bool(len(odds_2026))
            odds_2010, _ = sportmonks_client.probe_worldcup_2010_odds_if_available()
            status["SPORTMONKS_2010_ODDS_AVAILABLE"] = bool(len(odds_2010))
            injuries, _ = sportmonks_client.fetch_worldcup_injuries_if_available()
            status["SPORTMONKS_INJURIES_AVAILABLE"] = bool(len(injuries))
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            status["SPORTMONKS_2026_ODDS_AVAILABLE"] = "Unknown"
            status["SPORTMONKS_2010_ODDS_AVAILABLE"] = "Unknown"
            status["SPORTMONKS_INJURIES_AVAILABLE"] = "Unknown"
            details.append(f"Sportmonks probe failed: {exc}")
    else:
        status["SPORTMONKS_2026_ODDS_AVAILABLE"] = "Unknown"
        status["SPORTMONKS_2010_ODDS_AVAILABLE"] = "Unknown"
        status["SPORTMONKS_INJURIES_AVAILABLE"] = "Unknown"

    recommended_odds = _recommend_provider(status, ["API_FOOTBALL", "SPORTMONKS", "THE_ODDS_API"], "2026_ODDS_AVAILABLE")
    recommended_injuries = _recommend_provider(status, ["API_FOOTBALL", "SPORTMONKS"], "INJURIES_AVAILABLE")
    lines = [
        "# API Coverage Check Report",
        "",
        "This report uses official API endpoints only. It does not modify production files.",
        "",
        "## Availability",
        "",
    ]
    for key in [
        "API_FOOTBALL_AVAILABLE",
        "THE_ODDS_API_AVAILABLE",
        "SPORTMONKS_AVAILABLE",
        "API_FOOTBALL_2026_ODDS_AVAILABLE",
        "API_FOOTBALL_2010_ODDS_AVAILABLE",
        "API_FOOTBALL_INJURIES_AVAILABLE",
        "THE_ODDS_API_2026_ODDS_AVAILABLE",
        "THE_ODDS_API_2010_ODDS_AVAILABLE",
        "SPORTMONKS_2026_ODDS_AVAILABLE",
        "SPORTMONKS_2010_ODDS_AVAILABLE",
        "SPORTMONKS_INJURIES_AVAILABLE",
    ]:
        lines.append(f"- {key}={status.get(key)}")
    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            f"- Recommended provider for 2026 odds: `{recommended_odds}`",
            f"- Recommended provider for live injuries: `{recommended_injuries}`",
            "- Whether 2010 still needs manual/archive collection: `True` unless a provider returns all 64 valid 2010 1X2 odds rows.",
            "- The Odds API 2010 odds: `False`; public historical availability starts from 2020.",
        ]
    )
    if details:
        lines.extend(["", "## Probe Notes", "", *[f"- {detail}" for detail in details]])
    report_path = write_markdown_report(API_COVERAGE_REPORT, lines)
    write_markdown_report(STAGING_API_COVERAGE_REPORT, lines)
    return report_path


def validate_staged_api_data(config: dict[str, Any] | None = None) -> tuple[Path, Path]:
    cfg = config or load_config()
    ensure_api_directories()
    active_2026 = load_active_2026_fixtures(cfg)
    market_paths = [API_FOOTBALL_ODDS_2026_PATH, API_FOOTBALL_ODDS_2010_PROBE_PATH, THE_ODDS_API_ODDS_2026_PATH, SPORTMONKS_ODDS_2026_PATH]
    market_rows = []
    year_rows = []
    for path in market_paths:
        df = _read_staging_csv(path, MARKET_ODDS_STAGING_SCHEMA)
        summary = summarize_market_odds_validation(df)
        coverage = validate_coverage_against_active_fixtures(df, active_2026) if "2026" in path.name else None
        coverage_2010 = _coverage_against_required_2010(df) if "2010" in path.name or _has_year(df, 2010) else None
        for year_row in rows_by_year(df).to_dict("records"):
            year_rows.append({"file": str(path), **year_row})
        market_rows.append(
            {
                "file": str(path),
                "rows": summary.total_rows,
                "bad_dates": summary.bad_dates,
                "bad_odds": summary.bad_odds,
                "duplicates": summary.duplicate_rows,
                "unmatched_teams": ",".join(summary.unmatched_teams),
                "safe_to_merge": summary.safe_to_merge,
                "coverage_2026": "" if coverage is None else f"{coverage['matched_fixtures']}/{coverage['total_fixtures']} ({coverage['coverage']:.3f})",
                "missing_2026_fixtures": "" if coverage is None else len(coverage["missing_fixtures"]),
                "coverage_2010": "" if coverage_2010 is None else f"{coverage_2010['matched_fixtures']}/64 ({coverage_2010['coverage']:.3f})",
            }
        )
    market_lines = [
        "# API Market Odds Ingestion Report",
        "",
        "All API odds are staged first and are not production inputs until explicitly merged.",
        "",
        "## Validation Summary",
        "",
        markdown_table(
            market_rows,
            [
                "file",
                "rows",
                "bad_dates",
                "bad_odds",
                "duplicates",
                "unmatched_teams",
                "safe_to_merge",
                "coverage_2026",
                "missing_2026_fixtures",
                "coverage_2010",
            ],
        ),
        "",
        "## Rows By Year",
        "",
        markdown_table(year_rows, ["file", "year", "rows"]),
    ]
    market_report = write_markdown_report(API_MARKET_ODDS_INGESTION_REPORT, market_lines)

    injury_paths = [API_FOOTBALL_INJURIES_LIVE_PATH, SPORTMONKS_INJURIES_LIVE_PATH]
    injury_rows = []
    for path in injury_paths:
        df = _read_staging_csv(path, INJURY_STAGING_SCHEMA)
        missing_schema = validate_injury_schema(df)
        teams = sorted({team for team in df.get("team", pd.Series(dtype=str)).dropna().map(normalize_api_team_name) if team})
        scenario_only = bool(df.get("scenario_only", pd.Series(dtype=bool)).astype(str).str.lower().isin(["true", "1"]).all()) if not df.empty else True
        injury_rows.append(
            {
                "file": str(path),
                "injuries_fetched_count": len(df),
                "teams_covered": len(teams),
                "source_api": ",".join(sorted(df.get("api_provider", pd.Series(dtype=str)).dropna().astype(str).unique())),
                "scenario_only": scenario_only,
                "missing_schema": ",".join(missing_schema),
            }
        )
    injury_lines = [
        "# API Injury Ingestion Report",
        "",
        "Live injury/suspension data is scenario-only and is not used in the official model unless validated historical as-of backtests prove it improves performance.",
        "",
        markdown_table(
            injury_rows,
            ["file", "injuries_fetched_count", "teams_covered", "source_api", "scenario_only", "missing_schema"],
        ),
    ]
    injury_report = write_markdown_report(API_INJURY_INGESTION_REPORT, injury_lines)
    return market_report, injury_report


def check_2010_odds_coverage(
    odds_path: str | Path = MARKET_ODDS_EXTERNAL_PATH,
    *,
    report_path: str | Path = MARKET_ODDS_2010_COVERAGE_REPORT,
) -> tuple[Path, dict[str, Any]]:
    required = required_2010_matches_frame()
    odds = _read_csv(Path(odds_path))
    if odds.empty:
        matched = pd.DataFrame(columns=required.columns)
        missing = required.copy()
    else:
        odds_keys = _fixture_key_frame(odds).drop_duplicates()
        required_keys = _fixture_key_frame(required)
        merged = required_keys.merge(odds_keys.assign(_matched=True), on=["date", "home_team", "away_team"], how="left")
        matched_mask = merged["_matched"].fillna(False).astype(bool)
        matched_keys = merged[matched_mask][["date", "home_team", "away_team"]]
        missing_keys = merged[~matched_mask][["date", "home_team", "away_team"]]
        matched = required.merge(matched_keys, on=["date", "home_team", "away_team"], how="inner")
        missing = required.merge(missing_keys, on=["date", "home_team", "away_team"], how="inner")
    summary = {
        "rows_present": int(len(matched)),
        "rows_missing": int(len(missing)),
        "coverage": float(len(matched) / len(required)) if len(required) else 0.0,
    }
    lines = [
        "# Market Odds 2010 Coverage Report",
        "",
        f"- WORLD_CUP_2010_ODDS_COVERAGE={summary['coverage']:.3f}",
        f"- WORLD_CUP_2010_ROWS_PRESENT={summary['rows_present']}",
        f"- WORLD_CUP_2010_ROWS_MISSING={summary['rows_missing']}",
        "",
        "Do not claim 2010 is solved unless all 64 rows are present and valid.",
        "",
        "## Missing 2010 Matches",
        "",
        _df_markdown(missing),
    ]
    return write_markdown_report(report_path, lines), summary


def required_2010_matches_frame() -> pd.DataFrame:
    return pd.DataFrame(REQUIRED_2010_WORLD_CUP_MATCHES, columns=["date", "home_team", "away_team"]).assign(
        date=lambda frame: pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d"),
        home_team=lambda frame: frame["home_team"].map(normalize_api_team_name),
        away_team=lambda frame: frame["away_team"].map(normalize_api_team_name),
    )


def load_active_2026_fixtures(config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    for key in ("worldcup_2026_prediction_input_advanced", "worldcup_2026_prediction_input", "worldcup_2026_fixtures_clean"):
        try:
            path = config_path(cfg, key)
        except KeyError:
            continue
        if path.exists():
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
            break
    else:
        return pd.DataFrame(columns=["date", "home_team", "away_team"])
    output = df.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce", format="mixed")
    output = output[output["date"].dt.year.eq(2026)]
    output["home_team"] = output["home_team"].map(normalize_api_team_name)
    output["away_team"] = output["away_team"].map(normalize_api_team_name)
    placeholder = output["home_team"].astype(str).str.match(r"^[123][A-L]|^[WL]\d+", na=False) | output["away_team"].astype(str).str.match(r"^[123][A-L]|^[WL]\d+", na=False)
    return output[~placeholder][["date", "home_team", "away_team"]].drop_duplicates().reset_index(drop=True)


def known_fixture_teams(config: dict[str, Any] | None = None) -> set[str]:
    fixtures = load_active_2026_fixtures(config)
    teams = set(fixtures.get("home_team", pd.Series(dtype=str)).dropna()) | set(fixtures.get("away_team", pd.Series(dtype=str)).dropna())
    teams.update(required_2010_matches_frame()["home_team"].dropna())
    teams.update(required_2010_matches_frame()["away_team"].dropna())
    return {str(team) for team in teams}


def _coverage_against_required_2010(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"matched_fixtures": 0, "total_fixtures": 64, "coverage": 0.0}
    required = required_2010_matches_frame()
    odds_keys = _fixture_key_frame(df).drop_duplicates()
    required_keys = _fixture_key_frame(required)
    merged = required_keys.merge(odds_keys.assign(_matched=True), on=["date", "home_team", "away_team"], how="left")
    matched = int(merged["_matched"].fillna(False).astype(bool).sum())
    return {"matched_fixtures": matched, "total_fixtures": len(required), "coverage": float(matched / len(required))}


def _has_year(df: pd.DataFrame, year: int) -> bool:
    if df.empty or "date" not in df.columns:
        return False
    return bool(pd.to_datetime(df["date"], errors="coerce", format="mixed").dt.year.eq(year).any())


def _recommend_provider(status: dict[str, Any], providers: list[str], suffix: str) -> str:
    for provider in providers:
        if status.get(f"{provider}_{suffix}") is True:
            return provider.lower()
    return "none confirmed"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        schema = INJURY_STAGING_SCHEMA if "injur" in path.name else MARKET_ODDS_STAGING_SCHEMA
        return pd.DataFrame(columns=schema)
    return pd.read_csv(path, low_memory=False)


def _read_staging_csv(path: Path, schema: list[str]) -> pd.DataFrame:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=schema).to_csv(path, index=False)
    return pd.read_csv(path, low_memory=False)


def _fixture_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(df["date"], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d"),
            "home_team": df["home_team"].map(normalize_api_team_name),
            "away_team": df["away_team"].map(normalize_api_team_name),
        }
    )


def _df_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    return markdown_table(df.astype(str).to_dict("records"), columns)
