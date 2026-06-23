from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import config_path, load_config
from src.io_utils import stable_id, write_dataframe, write_json
from src.normalize import add_match_outcome_columns, canonicalize_match_teams, classify_competition_type


logger = logging.getLogger(__name__)


class MissingFootballDataToken(RuntimeError):
    """Raised when FOOTBALL_DATA_TOKEN is not available."""


def fetch_matches(
    date_from: str | date,
    date_to: str | date,
    *,
    only_finished: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch matches from football-data.org, chunking ranges over the API limit."""
    start_date = _parse_api_date(date_from)
    end_date = _parse_api_date(date_to)
    if start_date > end_date:
        raise ValueError(f"date_from must be on or before date_to: {start_date} > {end_date}")

    payloads: list[dict[str, Any]] = []
    cursor = start_date
    max_span_days = 10
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=max_span_days), end_date)
        payloads.append(
            _fetch_matches_window(
                cursor,
                chunk_end,
                only_finished=only_finished,
                config=config,
            )
        )
        cursor = chunk_end + timedelta(days=1)

    if len(payloads) == 1:
        return payloads[0]

    matches: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()
    for payload in payloads:
        for match in payload.get("matches", []):
            match_id = match.get("id")
            if match_id is not None and match_id in seen_ids:
                continue
            if match_id is not None:
                seen_ids.add(match_id)
            matches.append(match)
    logger.info("Fetched %s football-data.org matches across %s date chunks", len(matches), len(payloads))
    return {
        "filters": {"dateFrom": str(start_date), "dateTo": str(end_date)},
        "resultSet": {"count": len(matches)},
        "matches": matches,
    }


def _fetch_matches_window(
    date_from: date,
    date_to: date,
    *,
    only_finished: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch a single API-allowed date window."""
    cfg = config or load_config()
    load_dotenv()
    token = os.getenv("FOOTBALL_DATA_TOKEN")
    if not token:
        raise MissingFootballDataToken("FOOTBALL_DATA_TOKEN is not set")

    api_cfg = cfg["sources"]["football_data"]
    params = {"dateFrom": str(date_from), "dateTo": str(date_to)}
    session = requests.Session()
    headers = {"X-Auth-Token": token}
    retry_count = int(api_cfg.get("retry_count", 3))
    timeout_seconds = int(api_cfg.get("timeout_seconds", 30))
    rate_limit_sleep = float(api_cfg.get("rate_limit_sleep_seconds", 60))
    last_error: Exception | None = None

    for attempt in range(1, retry_count + 1):
        try:
            response = session.get(api_cfg["base_url"], params=params, headers=headers, timeout=timeout_seconds)
        except requests.RequestException as exc:
            last_error = exc
            _sleep_before_retry(attempt, retry_count, rate_limit_sleep, "network error")
            continue

        if response.status_code == 403:
            raise PermissionError("football-data.org returned 403. Check token permissions or request limits.")
        if response.status_code == 429:
            retry_after = _retry_after_seconds(response, rate_limit_sleep)
            last_error = RuntimeError("football-data.org rate limit exceeded")
            _sleep_before_retry(attempt, retry_count, retry_after, "rate limit")
            continue
        if response.status_code >= 500:
            last_error = RuntimeError(f"football-data.org server error {response.status_code}")
            _sleep_before_retry(attempt, retry_count, rate_limit_sleep, "server error")
            continue

        try:
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as exc:
            detail = _response_error_detail(response)
            raise RuntimeError(f"football-data.org request failed with status {response.status_code}: {detail}") from exc
        except ValueError as exc:
            raise ValueError("football-data.org returned malformed JSON") from exc

        if only_finished:
            matches = payload.get("matches", [])
            payload["matches"] = [match for match in matches if match.get("status") == "FINISHED"]
        logger.info("Fetched %s football-data.org matches", len(payload.get("matches", [])))
        return payload

    raise RuntimeError("Failed to fetch football-data.org matches") from last_error


def _parse_api_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def fetch_recent_matches(
    *,
    days_back: int = 14,
    days_forward: int = 2,
    only_finished: bool = False,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    today = date.today()
    return fetch_matches(
        today - timedelta(days=days_back),
        today + timedelta(days=days_forward),
        only_finished=only_finished,
        config=config,
    )


def parse_matches_response(
    json_data: dict[str, Any],
    *,
    mapping_path: str | Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for match in json_data.get("matches", []):
        competition = match.get("competition") or {}
        home_team = match.get("homeTeam") or {}
        away_team = match.get("awayTeam") or {}
        score = match.get("score") or {}
        full_time = score.get("fullTime") or {}
        utc_date = pd.to_datetime(match.get("utcDate"), errors="coerce", utc=True)
        date_value = utc_date.tz_convert(None) if not pd.isna(utc_date) else pd.NaT
        home_score = full_time.get("home")
        away_score = full_time.get("away")
        rows.append(
            {
                "match_id": f"football_data_{match.get('id')}" if match.get("id") is not None else stable_id("football_data", match),
                "source": "football-data",
                "date": date_value,
                "home_team": _team_name(home_team),
                "away_team": _team_name(away_team),
                "neutral": False,
                "tournament": competition.get("name"),
                "competition_type": classify_competition_type(competition.get("name")),
                "stage": match.get("stage"),
                "group": match.get("group"),
                "country": pd.NA,
                "city": pd.NA,
                "venue": pd.NA,
                "home_score": home_score,
                "away_score": away_score,
                "status": match.get("status"),
                "last_updated": match.get("lastUpdated"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return _empty_recent_matches()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = canonicalize_match_teams(df, mapping_path=mapping_path)
    df = add_match_outcome_columns(df)
    return df


def save_raw_response(json_data: dict[str, Any], path: str | Path | None = None, config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config()
    output_path = Path(path) if path else config_path(cfg, "raw_dir") / f"football_data_matches_raw_{date.today():%Y%m%d}.json"
    return write_json(json_data, output_path)


def save_clean_matches(df: pd.DataFrame, path: str | Path | None = None, config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config()
    output_path = Path(path) if path else config_path(cfg, "recent_matches_clean")
    return write_dataframe(df, output_path)


def _team_name(team: dict[str, Any]) -> str | None:
    return team.get("name") or team.get("shortName") or team.get("tla")


def _retry_after_seconds(response: requests.Response, default_seconds: float) -> float:
    header = response.headers.get("Retry-After")
    if header is None:
        return default_seconds
    try:
        return float(header)
    except ValueError:
        return default_seconds


def _response_error_detail(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(data, dict):
        return str(data.get("message") or data.get("error") or data)
    return str(data)


def _sleep_before_retry(attempt: int, retry_count: int, seconds: float, reason: str) -> None:
    if attempt >= retry_count:
        return
    logger.warning("football-data.org %s on attempt %s/%s; retrying in %.1fs", reason, attempt, retry_count, seconds)
    time.sleep(seconds)


def _empty_recent_matches() -> pd.DataFrame:
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
        "status",
        "last_updated",
        "result",
        "home_win",
        "draw",
        "away_win",
        "goal_diff",
        "total_goals",
    ]
    return pd.DataFrame(columns=columns)
