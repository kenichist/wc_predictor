from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import resolve_project_path


API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
THE_ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
SPORTMONKS_BASE_URL = "https://api.sportmonks.com/v3/football"

STAGING_DIR = resolve_project_path("data/staging")
RAW_API_FOOTBALL_DIR = resolve_project_path("data/raw/api_football")
RAW_THE_ODDS_API_DIR = resolve_project_path("data/raw/the_odds_api")
RAW_SPORTMONKS_DIR = resolve_project_path("data/raw/sportmonks")

API_FOOTBALL_ODDS_2026_PATH = STAGING_DIR / "api_football_market_odds_2026.csv"
API_FOOTBALL_ODDS_2010_PROBE_PATH = STAGING_DIR / "api_football_market_odds_2010_probe.csv"
API_FOOTBALL_INJURIES_LIVE_PATH = STAGING_DIR / "api_football_injuries_live.csv"
THE_ODDS_API_ODDS_2026_PATH = STAGING_DIR / "the_odds_api_market_odds_2026.csv"
SPORTMONKS_ODDS_2026_PATH = STAGING_DIR / "sportmonks_market_odds_2026.csv"
SPORTMONKS_INJURIES_LIVE_PATH = STAGING_DIR / "sportmonks_injuries_live.csv"
STAGING_API_COVERAGE_REPORT = STAGING_DIR / "api_coverage_check_report.md"

API_COVERAGE_REPORT = resolve_project_path("data/reports/api_coverage_check_report.md")
API_MARKET_ODDS_INGESTION_REPORT = resolve_project_path("data/reports/api_market_odds_ingestion_report.md")
API_INJURY_INGESTION_REPORT = resolve_project_path("data/reports/api_injury_ingestion_report.md")
MARKET_ODDS_API_MERGE_REPORT = resolve_project_path("data/reports/market_odds_api_merge_report.md")
MARKET_ODDS_2010_COVERAGE_REPORT = resolve_project_path("data/reports/market_odds_2010_coverage_report.md")

MARKET_ODDS_EXTERNAL_PATH = resolve_project_path("data/external/market_odds.csv")
MARKET_ODDS_BACKUP_PATH = resolve_project_path("data/external/market_odds_backup_before_api_merge.csv")


@dataclass(frozen=True)
class ApiCredentials:
    api_football_key: str | None
    the_odds_api_key: str | None
    sportmonks_api_token: str | None

    @property
    def api_football_available(self) -> bool:
        return bool(self.api_football_key)

    @property
    def the_odds_api_available(self) -> bool:
        return bool(self.the_odds_api_key)

    @property
    def sportmonks_available(self) -> bool:
        return bool(self.sportmonks_api_token)


def load_environment() -> None:
    """Load .env only when python-dotenv is already installed."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv()


def get_credentials() -> ApiCredentials:
    load_environment()
    return ApiCredentials(
        api_football_key=_clean_env("API_FOOTBALL_KEY"),
        the_odds_api_key=_clean_env("THE_ODDS_API_KEY"),
        sportmonks_api_token=_clean_env("SPORTMONKS_API_TOKEN"),
    )


def credential_status() -> dict[str, bool]:
    credentials = get_credentials()
    return {
        "API_FOOTBALL_AVAILABLE": credentials.api_football_available,
        "THE_ODDS_API_AVAILABLE": credentials.the_odds_api_available,
        "SPORTMONKS_AVAILABLE": credentials.sportmonks_available,
    }


def ensure_api_directories() -> None:
    for path in (
        STAGING_DIR,
        RAW_API_FOOTBALL_DIR,
        RAW_THE_ODDS_API_DIR,
        RAW_SPORTMONKS_DIR,
        API_COVERAGE_REPORT.parent,
        MARKET_ODDS_EXTERNAL_PATH.parent,
    ):
        Path(path).mkdir(parents=True, exist_ok=True)


def safe_key_label(value: str | None) -> str:
    if not value:
        return "missing"
    return f"present:{len(value)} chars"


def _clean_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def write_markdown_report(path: str | Path, lines: list[str]) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text("\n".join(lines), encoding="utf-8")
    return resolved


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    output = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        output.append("| " + " | ".join("" if row.get(column) is None else str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(output)
