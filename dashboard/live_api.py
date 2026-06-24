from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from dashboard.data_loader import read_csv_safe
from dashboard.prediction_logic import (
    DEFAULT_MARKET_ALPHA,
    LIVE_FIXTURES_SCHEMA,
    LIVE_INJURIES_SCHEMA,
    LIVE_LINEUPS_SCHEMA,
    LIVE_ODDS_SCHEMA,
    LIVE_PREDICTIONS_SCHEMA,
    LIVE_SCORES_SCHEMA,
    build_live_predictions,
    empty_frame,
    ensure_schema,
    normalize_dashboard_team,
    write_live_prediction_outputs,
)
from src.config import PROJECT_ROOT
from src.data_sources.config import safe_key_label
from src.data_sources.odds_transform import api_football_odds_to_market_rows


LIVE_DIR = PROJECT_ROOT / "data" / "live"
LIVE_RAW_DIR = LIVE_DIR / "raw"
LIVE_REPORTS_DIR = LIVE_DIR / "reports"

LIVE_FIXTURES_PATH = LIVE_DIR / "live_fixtures.csv"
LIVE_ODDS_PATH = LIVE_DIR / "live_odds.csv"
LIVE_INJURIES_PATH = LIVE_DIR / "live_injuries.csv"
LIVE_LINEUPS_PATH = LIVE_DIR / "live_lineups.csv"
LIVE_SCORES_PATH = LIVE_DIR / "live_scores.csv"
LIVE_PREDICTIONS_PATH = LIVE_DIR / "live_predictions.csv"
LIVE_FACTORS_PATH = LIVE_DIR / "live_prediction_factors.csv"
LIVE_API_STATUS_REPORT = LIVE_REPORTS_DIR / "live_api_status_report.md"
LIVE_PREDICTION_REPORT = LIVE_REPORTS_DIR / "live_prediction_report.md"

OFFICIAL_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "predictions" / "actual_team_match_predictions.csv"
EXTERNAL_MARKET_ODDS_PATH = PROJECT_ROOT / "data" / "external" / "market_odds.csv"

API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"


def load_dashboard_environment() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def api_available() -> bool:
    load_dashboard_environment()
    return bool(os.getenv("API_FOOTBALL_KEY", "").strip())


def fetch_api_football_status() -> dict[str, Any]:
    """Fetch API-Football quota/account status without modifying project data.

    The API response shape can differ by provider/plan/version, so this parser
    accepts both `response` and `results` payload roots and falls back to
    rate-limit headers when the JSON body omits request counters.
    """
    ensure_live_directories()
    load_dashboard_environment()
    api_key = os.getenv("API_FOOTBALL_KEY", "").strip()
    status: dict[str, Any] = {
        "API_FOOTBALL_AVAILABLE": bool(api_key),
        "API_FOOTBALL_KEY": safe_key_label(api_key),
        "checked_at": _now(),
        "endpoint": f"{API_FOOTBALL_BASE_URL}/status",
        "used_today": None,
        "daily_limit": None,
        "remaining_today": None,
        "plan": "unknown",
        "subscription_active": None,
        "reset_note": "Daily quota normally resets at 00:00 UTC.",
        "errors": [],
    }
    if not api_key:
        status["errors"].append("API_FOOTBALL_KEY is not configured")
        _write_api_football_quota_report(status)
        return status

    try:
        response = requests.get(
            f"{API_FOOTBALL_BASE_URL}/status",
            headers={"x-apisports-key": api_key},
            timeout=30,
        )
        status["http_status_code"] = response.status_code
        header_map = {str(key).lower(): value for key, value in response.headers.items()}
        status["rate_limit_remaining_header"] = _header_value(
            header_map,
            "x-ratelimit-requests-remaining",
            "x-ratelimit-remaining",
            "x-ratelimit-requests-remaining-day",
        )
        status["rate_limit_limit_header"] = _header_value(
            header_map,
            "x-ratelimit-limit",
            "x-ratelimit-requests-limit",
            "x-ratelimit-limit-day",
        )
        response.raise_for_status()
        payload = response.json()
        raw_path = save_raw_payload(_redact_status_payload(payload), "api_football_status")
        status["raw_payload_file"] = str(raw_path)

        root = _status_payload_root(payload)
        account = root.get("account", {}) if isinstance(root, dict) else {}
        subscription = root.get("subscription", {}) if isinstance(root, dict) else {}
        requests_info = _status_requests_info(payload)
        if not isinstance(requests_info, dict):
            requests_info = {}

        status["account_email_present"] = bool(account.get("email")) if isinstance(account, dict) else False
        status["plan"] = _first_present_value(subscription.get("plan"), subscription.get("name"), status["plan"]) if isinstance(subscription, dict) else status["plan"]
        status["subscription_active"] = _first_present_value(subscription.get("active"), status["subscription_active"]) if isinstance(subscription, dict) else status["subscription_active"]
        status["subscription_end"] = _first_present_value(subscription.get("end"), subscription.get("ends_at"), None) if isinstance(subscription, dict) else None

        used = _first_numeric_value(
            requests_info.get("current"),
            requests_info.get("used"),
            requests_info.get("requests_current"),
            requests_info.get("current_day"),
        )
        limit = _first_numeric_value(
            requests_info.get("limit_day"),
            requests_info.get("limit"),
            requests_info.get("daily_limit"),
            status.get("rate_limit_limit_header"),
        )
        remaining = _first_numeric_value(
            requests_info.get("remaining"),
            status.get("rate_limit_remaining_header"),
        )
        if remaining is None and used is not None and limit is not None:
            remaining = max(0, limit - used)
        if used is None and remaining is not None and limit is not None:
            used = max(0, limit - remaining)

        status["used_today"] = used
        status["daily_limit"] = limit
        status["remaining_today"] = remaining
        if limit and remaining is not None:
            status["usage_percent"] = (limit - remaining) / limit
        elif limit and used is not None:
            status["usage_percent"] = used / limit
        else:
            status["usage_percent"] = None
    except Exception as exc:
        status["errors"].append(str(exc))
    _write_api_football_quota_report(status)
    return status


def ensure_live_directories() -> None:
    for path in (LIVE_DIR, LIVE_RAW_DIR, LIVE_REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def initialize_live_files() -> None:
    ensure_live_directories()
    for path, schema in [
        (LIVE_FIXTURES_PATH, LIVE_FIXTURES_SCHEMA),
        (LIVE_ODDS_PATH, LIVE_ODDS_SCHEMA),
        (LIVE_INJURIES_PATH, LIVE_INJURIES_SCHEMA),
        (LIVE_LINEUPS_PATH, LIVE_LINEUPS_SCHEMA),
        (LIVE_SCORES_PATH, LIVE_SCORES_SCHEMA),
        (LIVE_PREDICTIONS_PATH, LIVE_PREDICTIONS_SCHEMA),
    ]:
        if not path.exists():
            empty_frame(schema).to_csv(path, index=False)
    if not LIVE_FACTORS_PATH.exists():
        pd.DataFrame(columns=["fixture_id", "factor_type", "team", "description", "impact_direction", "impact_strength", "source", "updated_at"]).to_csv(LIVE_FACTORS_PATH, index=False)


class ApiFootballLiveClient:
    def __init__(self, api_key: str | None = None, *, base_url: str = API_FOOTBALL_BASE_URL) -> None:
        load_dashboard_environment()
        self.api_key = api_key or os.getenv("API_FOOTBALL_KEY", "").strip()
        self.base_url = base_url.rstrip("/")

    def api_available(self) -> bool:
        return bool(self.api_key)

    def request(self, endpoint: str, params: dict[str, Any] | None = None, *, label: str) -> tuple[dict[str, Any], Path]:
        if not self.api_key:
            raise RuntimeError("API_FOOTBALL_KEY is not configured")
        response = requests.get(
            f"{self.base_url}/{endpoint.lstrip('/')}",
            headers={"x-apisports-key": self.api_key},
            params=params or {},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        raw_path = save_raw_payload(payload, label)
        return payload, raw_path

    def find_world_cup_league_ids(self) -> tuple[list[dict[str, Any]], Path | None]:
        payload, raw_path = self.request("/leagues", {"search": "World Cup"}, label="api_football_leagues")
        candidates = []
        for item in payload.get("response", []) or []:
            league = item.get("league", {}) if isinstance(item, dict) else {}
            country = item.get("country", {}) if isinstance(item, dict) else {}
            name = str(league.get("name") or "")
            if "world cup" in name.lower():
                candidates.append({"league_id": league.get("id"), "name": name, "country": country.get("name")})
        return candidates, raw_path

    def preferred_world_cup_league_id(self) -> tuple[int | None, list[dict[str, Any]], Path | None]:
        candidates, raw_path = self.find_world_cup_league_ids()
        for candidate in candidates:
            name = str(candidate.get("name") or "").lower()
            if name in {"fifa world cup", "world cup"}:
                return _safe_int(candidate.get("league_id")), candidates, raw_path
        if candidates:
            return _safe_int(candidates[0].get("league_id")), candidates, raw_path
        return None, candidates, raw_path

    def fetch_worldcup_fixtures_2026(self) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        league_id, candidates, _ = self.preferred_world_cup_league_id()
        if league_id is None:
            return empty_frame(LIVE_FIXTURES_SCHEMA), empty_frame(LIVE_SCORES_SCHEMA), {"error": "No World Cup league candidate found", "league_candidates": candidates}
        payload, raw_path = self.request("/fixtures", {"league": league_id, "season": 2026}, label="api_football_fixtures_2026")
        fixtures, scores = parse_api_football_fixtures(payload)
        return fixtures, scores, {"raw_payload_file": str(raw_path), "league_id": league_id, "league_candidates": candidates}

    def fetch_current_or_upcoming_worldcup_matches(self) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        fixtures, scores, meta = self.fetch_worldcup_fixtures_2026()
        if fixtures.empty:
            return fixtures, scores, meta
        status = fixtures["status"].astype(str).str.lower()
        active = fixtures[status.str.contains("not started|time to be defined|first half|second half|halftime|live|extra time|penalty|upcoming", na=False)].copy()
        if active.empty:
            active = fixtures.copy()
        return active, scores, meta

    def fetch_live_scores(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        payload, raw_path = self.request("/fixtures", {"live": "all"}, label="api_football_live_scores")
        _, scores = parse_api_football_fixtures(payload)
        return scores, {"raw_payload_file": str(raw_path)}

    def fetch_match_odds(self, *, league_id: int | None = None, season: int = 2026) -> tuple[pd.DataFrame, dict[str, Any]]:
        if league_id is None:
            league_id, candidates, _ = self.preferred_world_cup_league_id()
        else:
            candidates = []
        if league_id is None:
            return empty_frame(LIVE_ODDS_SCHEMA), {"error": "No World Cup league candidate found", "league_candidates": candidates}
        payload, raw_path = self.request("/odds", {"league": league_id, "season": season}, label=f"api_football_odds_{season}")
        return api_football_odds_to_live_odds(payload, raw_payload_file=str(raw_path)), {"raw_payload_file": str(raw_path), "league_id": league_id}

    def fetch_injuries(self, *, league_id: int | None = None, season: int = 2026) -> tuple[pd.DataFrame, dict[str, Any]]:
        if league_id is None:
            league_id, candidates, _ = self.preferred_world_cup_league_id()
        else:
            candidates = []
        if league_id is None:
            return empty_frame(LIVE_INJURIES_SCHEMA), {"error": "No World Cup league candidate found", "league_candidates": candidates}
        payload, raw_path = self.request("/injuries", {"league": league_id, "season": season}, label=f"api_football_injuries_{season}")
        return parse_api_football_injuries(payload), {"raw_payload_file": str(raw_path), "league_id": league_id}

    def fetch_lineups(self, fixture_ids: list[Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        rows = []
        errors = []
        raw_paths = []
        for fixture_id in fixture_ids[:20]:
            try:
                payload, raw_path = self.request("/fixtures/lineups", {"fixture": fixture_id}, label=f"api_football_lineups_{fixture_id}")
                raw_paths.append(str(raw_path))
                rows.extend(parse_api_football_lineups(payload).to_dict("records"))
            except Exception as exc:  # API add-ons vary; keep this non-fatal.
                errors.append(f"{fixture_id}: {exc}")
        return ensure_schema(pd.DataFrame(rows), LIVE_LINEUPS_SCHEMA), {"raw_payload_files": raw_paths, "errors": errors}

    def fetch_fixture_details(self, fixture_id: Any) -> tuple[dict[str, Any], Path]:
        return self.request("/fixtures", {"id": fixture_id}, label=f"api_football_fixture_{fixture_id}")


def refresh_live_data(*, fetch_lineups: bool = False, alpha: float = DEFAULT_MARKET_ALPHA) -> dict[str, Any]:
    ensure_live_directories()
    initialize_live_files()
    client = ApiFootballLiveClient()
    status: dict[str, Any] = {
        "API_FOOTBALL_AVAILABLE": client.api_available(),
        "API_FOOTBALL_KEY": safe_key_label(client.api_key),
        "started_at": _now(),
        "errors": [],
    }
    if not client.api_available():
        status["errors"].append("No API key configured. Using local CSV/report outputs only.")
        _write_live_api_status_report(status)
        predictions_summary = generate_live_predictions(alpha=alpha)
        return {**status, **predictions_summary, "SUCCESS": False}

    try:
        fixtures, scores_from_fixtures, fixture_meta = client.fetch_worldcup_fixtures_2026()
        status["fixtures_meta"] = fixture_meta
        league_id = fixture_meta.get("league_id")
        scores_live, scores_meta = client.fetch_live_scores()
        odds, odds_meta = client.fetch_match_odds(league_id=league_id)
        odds_fixture_lookup = pd.concat([fixtures, scores_from_fixtures], ignore_index=True, sort=False)
        odds = enrich_live_odds_from_fixtures(odds, odds_fixture_lookup)
        injuries, injury_meta = client.fetch_injuries(league_id=league_id)
        if fetch_lineups:
            lineup_fixtures = _current_or_upcoming_fixture_subset(fixtures)
            fixture_ids = lineup_fixtures["fixture_id"].dropna().astype(str).unique().tolist() if not lineup_fixtures.empty else []
            lineups, lineups_meta = client.fetch_lineups(fixture_ids)
        else:
            lineups, lineups_meta = empty_frame(LIVE_LINEUPS_SCHEMA), {"skipped": "lineups disabled by default to respect API limits"}
        scores = pd.concat([scores_from_fixtures, scores_live], ignore_index=True, sort=False).drop_duplicates(["fixture_id"], keep="last")
        ensure_schema(fixtures, LIVE_FIXTURES_SCHEMA).to_csv(LIVE_FIXTURES_PATH, index=False)
        ensure_schema(scores, LIVE_SCORES_SCHEMA).to_csv(LIVE_SCORES_PATH, index=False)
        ensure_schema(odds, LIVE_ODDS_SCHEMA).to_csv(LIVE_ODDS_PATH, index=False)
        ensure_schema(injuries, LIVE_INJURIES_SCHEMA).to_csv(LIVE_INJURIES_PATH, index=False)
        ensure_schema(lineups, LIVE_LINEUPS_SCHEMA).to_csv(LIVE_LINEUPS_PATH, index=False)
        status.update(
            {
                "fixtures_rows": len(fixtures),
                "scores_rows": len(scores),
                "odds_rows": len(odds),
                "injury_rows": len(injuries),
                "lineup_rows": len(lineups),
                "odds_meta": odds_meta,
                "injury_meta": injury_meta,
                "lineups_meta": lineups_meta,
                "scores_meta": scores_meta,
                "finished_at": _now(),
            }
        )
    except Exception as exc:
        status["errors"].append(str(exc))
        status["finished_at"] = _now()
        _write_live_api_status_report(status)
        predictions_summary = generate_live_predictions(alpha=alpha)
        return {**status, **predictions_summary, "SUCCESS": False}
    _write_live_api_status_report(status)
    predictions_summary = generate_live_predictions(alpha=alpha)
    return {**status, **predictions_summary, "SUCCESS": not bool(status["errors"])}


def generate_live_predictions(*, alpha: float = DEFAULT_MARKET_ALPHA) -> dict[str, Any]:
    ensure_live_directories()
    initialize_live_files()
    fixtures = read_csv_safe(LIVE_FIXTURES_PATH)
    official = read_csv_safe(OFFICIAL_PREDICTIONS_PATH)
    live_odds = read_csv_safe(LIVE_ODDS_PATH)
    external_odds = read_csv_safe(EXTERNAL_MARKET_ODDS_PATH)
    injuries = read_csv_safe(LIVE_INJURIES_PATH)
    lineups = read_csv_safe(LIVE_LINEUPS_PATH)
    scores = read_csv_safe(LIVE_SCORES_PATH)
    fixtures = combine_fixtures_with_scores(fixtures, scores)
    predictions, factors, summary = build_live_predictions(
        fixtures=fixtures,
        official_predictions=official,
        live_odds=live_odds,
        external_odds=external_odds,
        injuries=injuries,
        lineups=lineups,
        scores=scores,
        alpha=alpha,
    )
    write_live_prediction_outputs(predictions, factors, live_dir=LIVE_DIR)
    _write_live_prediction_report(summary)
    return summary


def combine_fixtures_with_scores(fixtures: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    base = ensure_schema(fixtures, LIVE_FIXTURES_SCHEMA) if not fixtures.empty else empty_frame(LIVE_FIXTURES_SCHEMA)
    if scores.empty:
        return base
    score_frame = ensure_schema(scores, LIVE_SCORES_SCHEMA)
    now = _now()
    score_fixtures = pd.DataFrame(
        {
            "fixture_id": score_frame["fixture_id"],
            "date": score_frame["date"],
            "kickoff_time": pd.NA,
            "status": score_frame["status"],
            "minute": score_frame["minute"],
            "home_team": score_frame["home_team"],
            "away_team": score_frame["away_team"],
            "venue": pd.NA,
            "stage": pd.NA,
            "group": pd.NA,
            "source": "live_scores",
            "updated_at": score_frame.get("updated_at", pd.Series(now, index=score_frame.index)),
        }
    )
    combined = pd.concat([score_fixtures, base], ignore_index=True, sort=False)
    if "fixture_id" in combined.columns:
        keyed = combined[combined["fixture_id"].notna()].drop_duplicates("fixture_id", keep="last")
        unkeyed = combined[combined["fixture_id"].isna()]
        combined = pd.concat([keyed, unkeyed], ignore_index=True, sort=False)
    if {"date", "home_team", "away_team"}.issubset(combined.columns):
        combined = combined.drop_duplicates(["date", "home_team", "away_team"], keep="last")
    return ensure_schema(combined, LIVE_FIXTURES_SCHEMA)


def parse_api_football_fixtures(payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixture_rows = []
    score_rows = []
    updated_at = _now()
    for item in payload.get("response", []) or []:
        fixture = item.get("fixture", {}) if isinstance(item, dict) else {}
        teams = item.get("teams", {}) if isinstance(item, dict) else {}
        league = item.get("league", {}) if isinstance(item, dict) else {}
        goals = item.get("goals", {}) if isinstance(item, dict) else {}
        status = fixture.get("status", {}) if isinstance(fixture, dict) else {}
        venue = fixture.get("venue", {}) if isinstance(fixture, dict) else {}
        fixture_id = fixture.get("id")
        date = fixture.get("date")
        home = normalize_dashboard_team((teams.get("home") or {}).get("name") if isinstance(teams, dict) else None)
        away = normalize_dashboard_team((teams.get("away") or {}).get("name") if isinstance(teams, dict) else None)
        status_text = status.get("long") or status.get("short") or ""
        fixture_rows.append(
            {
                "fixture_id": fixture_id,
                "date": _date_key(date),
                "kickoff_time": date,
                "status": status_text,
                "minute": status.get("elapsed"),
                "home_team": home,
                "away_team": away,
                "venue": venue.get("name"),
                "stage": league.get("round"),
                "group": league.get("round") if "group" in str(league.get("round", "")).lower() else "",
                "source": "api_football:/fixtures",
                "updated_at": updated_at,
            }
        )
        score_rows.append(
            {
                "fixture_id": fixture_id,
                "date": _date_key(date),
                "home_team": home,
                "away_team": away,
                "status": status_text,
                "minute": status.get("elapsed"),
                "home_score": goals.get("home"),
                "away_score": goals.get("away"),
                "home_red_cards": 0,
                "away_red_cards": 0,
                "source": "api_football:/fixtures",
                "updated_at": updated_at,
            }
        )
    return ensure_schema(pd.DataFrame(fixture_rows), LIVE_FIXTURES_SCHEMA), ensure_schema(pd.DataFrame(score_rows), LIVE_SCORES_SCHEMA)


def api_football_odds_to_live_odds(payload: dict[str, Any], *, raw_payload_file: str = "") -> pd.DataFrame:
    staged = api_football_odds_to_market_rows(payload, raw_payload_file=raw_payload_file)
    if staged.empty:
        return empty_frame(LIVE_ODDS_SCHEMA)
    return ensure_schema(
        pd.DataFrame(
            {
                "fixture_id": staged["api_fixture_id"],
                "date": staged["date"],
                "home_team": staged["home_team"],
                "away_team": staged["away_team"],
                "home_odds": staged["home_odds"],
                "draw_odds": staged["draw_odds"],
                "away_odds": staged["away_odds"],
                "bookmaker": staged["bookmaker"],
                "source": staged["source"],
                "updated_at": staged["updated_at"],
                "api_provider": staged["api_provider"],
            }
        ),
        LIVE_ODDS_SCHEMA,
    )


def enrich_live_odds_from_fixtures(odds: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    if odds.empty or fixtures.empty or "fixture_id" not in odds.columns or "fixture_id" not in fixtures.columns:
        return ensure_schema(odds, LIVE_ODDS_SCHEMA)
    output = ensure_schema(odds, LIVE_ODDS_SCHEMA).copy()
    fixture_cols = ["fixture_id", "date", "home_team", "away_team"]
    lookup = fixtures[fixture_cols].drop_duplicates("fixture_id", keep="last").copy()
    lookup["_fixture_id_key"] = lookup["fixture_id"].astype(str)
    output["_fixture_id_key"] = output["fixture_id"].astype(str)
    merged = output.merge(lookup.drop(columns=["fixture_id"]), on="_fixture_id_key", how="left", suffixes=("", "_fixture"))
    for column in ("date", "home_team", "away_team"):
        fixture_column = f"{column}_fixture"
        if fixture_column in merged.columns:
            merged[column] = merged[column].where(merged[column].notna() & merged[column].astype(str).str.strip().ne(""), merged[fixture_column])
    return ensure_schema(merged.drop(columns=[column for column in merged.columns if column.endswith("_fixture") or column == "_fixture_id_key"], errors="ignore"), LIVE_ODDS_SCHEMA)


def parse_api_football_injuries(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    updated_at = _now()
    for item in payload.get("response", []) or []:
        fixture = item.get("fixture", {}) if isinstance(item, dict) else {}
        team = item.get("team", {}) if isinstance(item, dict) else {}
        player = item.get("player", {}) if isinstance(item, dict) else {}
        rows.append(
            {
                "fixture_id": fixture.get("id"),
                "date": _date_key(fixture.get("date") or updated_at),
                "team": normalize_dashboard_team(team.get("name")),
                "player": player.get("name"),
                "status": str(item.get("type") or item.get("status") or item.get("reason") or "unknown").lower(),
                "reason": item.get("reason") or item.get("type"),
                "importance": item.get("importance"),
                "source": "api_football:/injuries",
                "updated_at": updated_at,
                "api_provider": "api_football",
                "scenario_only": True,
            }
        )
    return ensure_schema(pd.DataFrame(rows), LIVE_INJURIES_SCHEMA)


def parse_api_football_lineups(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    updated_at = _now()
    for team_row in payload.get("response", []) or []:
        team = team_row.get("team", {}) if isinstance(team_row, dict) else {}
        team_name = normalize_dashboard_team(team.get("name"))
        formation = team_row.get("formation") if isinstance(team_row, dict) else None
        for player_row in team_row.get("startXI", []) or []:
            player = player_row.get("player", {}) if isinstance(player_row, dict) else {}
            rows.append(_lineup_row(team_name, player, True, formation, updated_at))
        for player_row in team_row.get("substitutes", []) or []:
            player = player_row.get("player", {}) if isinstance(player_row, dict) else {}
            rows.append(_lineup_row(team_name, player, False, formation, updated_at))
    return ensure_schema(pd.DataFrame(rows), LIVE_LINEUPS_SCHEMA)


def save_raw_payload(payload: Any, label: str) -> Path:
    ensure_live_directories()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in label)
    path = LIVE_RAW_DIR / f"{safe_label}_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def _write_live_api_status_report(status: dict[str, Any]) -> Path:
    lines = [
        "# Live API Status Report",
        "",
        "Live API data is staged under `data/live/` and does not overwrite official model outputs.",
        "",
        "## Status",
        "",
    ]
    for key, value in status.items():
        if key == "errors":
            continue
        lines.append(f"- {key}: `{value}`")
    if status.get("errors"):
        lines.extend(["", "## Errors", "", *[f"- {error}" for error in status["errors"]]])
    return _write_report(LIVE_API_STATUS_REPORT, lines)


def _write_live_prediction_report(summary: dict[str, Any]) -> Path:
    lines = [
        "# Live Prediction Report",
        "",
        "Live-adjusted predictions are scenario-based and are not official model outputs.",
        "",
        "## Summary",
        "",
    ]
    lines.extend(f"- {key}={value}" for key, value in summary.items())
    return _write_report(LIVE_PREDICTION_REPORT, lines)



def _write_api_football_quota_report(status: dict[str, Any]) -> Path:
    path = LIVE_REPORTS_DIR / "api_football_quota_report.md"
    lines = [
        "# API-Football Quota Report",
        "",
        "This report is generated by the dashboard quota checker. It does not modify prediction or odds files.",
        "",
        "## Summary",
        "",
        f"- checked_at: `{status.get('checked_at')}`",
        f"- API_FOOTBALL_AVAILABLE: `{status.get('API_FOOTBALL_AVAILABLE')}`",
        f"- API_FOOTBALL_KEY: `{status.get('API_FOOTBALL_KEY')}`",
        f"- plan: `{status.get('plan')}`",
        f"- subscription_active: `{status.get('subscription_active')}`",
        f"- subscription_end: `{status.get('subscription_end', 'unknown')}`",
        f"- used_today: `{status.get('used_today')}`",
        f"- daily_limit: `{status.get('daily_limit')}`",
        f"- remaining_today: `{status.get('remaining_today')}`",
        f"- usage_percent: `{status.get('usage_percent')}`",
        f"- reset_note: `{status.get('reset_note')}`",
        f"- raw_payload_file: `{status.get('raw_payload_file', '')}`",
        "",
        "## Header Fallbacks",
        "",
        f"- rate_limit_remaining_header: `{status.get('rate_limit_remaining_header')}`",
        f"- rate_limit_limit_header: `{status.get('rate_limit_limit_header')}`",
    ]
    if status.get("errors"):
        lines.extend(["", "## Errors", "", *[f"- {error}" for error in status["errors"]]])
    return _write_report(path, lines)


def _status_payload_root(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    for key in ("response", "results", "result", "data"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return value[0]
    return payload


def _status_requests_info(payload: Any) -> dict[str, Any]:
    for path in (
        ("results", "requests"),
        ("response", "requests"),
        ("response", "results", "requests"),
        ("requests",),
        ("data", "requests"),
    ):
        value = _nested_dict_value(payload, path)
        if isinstance(value, dict):
            return value
    root = _status_payload_root(payload)
    value = root.get("requests") if isinstance(root, dict) else None
    return value if isinstance(value, dict) else {}


def _nested_dict_value(payload: Any, path: tuple[str, ...]) -> Any:
    value = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _redact_status_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        redacted = {}
        for key, value in payload.items():
            if str(key).lower() in {"email", "firstname", "lastname", "first_name", "last_name"}:
                redacted[key] = "REDACTED"
            else:
                redacted[key] = _redact_status_payload(value)
        return redacted
    if isinstance(payload, list):
        return [_redact_status_payload(item) for item in payload]
    return payload


def _header_value(headers: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = headers.get(key.lower())
        if value is not None:
            return value
    return None


def _first_present_value(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value).strip() != "":
            return value
    return None


def _first_numeric_value(*values: Any) -> int | None:
    for value in values:
        if value is None or str(value).strip() == "":
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return None


def _write_report(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _current_or_upcoming_fixture_subset(fixtures: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty or "status" not in fixtures.columns:
        return fixtures
    status = fixtures["status"].astype(str).str.lower()
    active = fixtures[status.str.contains("not started|time to be defined|first half|second half|halftime|live|extra time|penalty|upcoming", na=False)].copy()
    return active if not active.empty else fixtures.copy()


def _lineup_row(team: str | None, player: dict[str, Any], starter: bool, formation: Any, updated_at: str) -> dict[str, Any]:
    return {
        "fixture_id": pd.NA,
        "date": pd.NA,
        "team": team,
        "player": player.get("name"),
        "starter": starter,
        "position": player.get("pos"),
        "formation": formation,
        "status": "starter" if starter else "substitute",
        "source": "api_football:/fixtures/lineups",
        "updated_at": updated_at,
        "api_provider": "api_football",
        "scenario_only": True,
    }


def _date_key(value: Any) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce", format="mixed", utc=True)
    if pd.isna(timestamp):
        return None
    return timestamp.strftime("%Y-%m-%d")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
