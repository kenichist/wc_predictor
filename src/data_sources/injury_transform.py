from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_sources.team_normalization import normalize_api_team_name


INJURY_STAGING_SCHEMA = [
    "date",
    "team",
    "player",
    "status",
    "reason",
    "fixture_id",
    "fixture_date",
    "source",
    "updated_at",
    "api_provider",
    "raw_team",
    "raw_player",
    "importance",
    "scenario_only",
]


STATUS_ALIASES = {
    "injured": "injured",
    "injury": "injured",
    "suspended": "suspended",
    "suspension": "suspended",
    "doubtful": "doubtful",
    "questionable": "doubtful",
    "unavailable": "unavailable",
    "out": "unavailable",
    "returned": "returned",
    "available": "returned",
}


def empty_injury_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=INJURY_STAGING_SCHEMA)


def normalize_injury_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return empty_injury_frame()
    output = pd.DataFrame(rows)
    for column in INJURY_STAGING_SCHEMA:
        if column not in output.columns:
            output[column] = pd.NA
    output["date"] = pd.to_datetime(output["date"], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    output["fixture_date"] = pd.to_datetime(output["fixture_date"], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    output["team"] = output["team"].map(normalize_api_team_name)
    output["status"] = output["status"].map(normalize_injury_status)
    output["scenario_only"] = True
    return output[INJURY_STAGING_SCHEMA]


def api_football_injuries_to_rows(payload: dict[str, Any], *, raw_payload_file: str = "") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    updated_at = _utc_now()
    for item in _response_items(payload):
        fixture = item.get("fixture") or {}
        team = item.get("team") or {}
        player = item.get("player") or {}
        raw_team = team.get("name") or item.get("team_name")
        raw_player = player.get("name") or item.get("player_name")
        rows.append(
            {
                "date": fixture.get("date") or item.get("date") or updated_at,
                "team": raw_team,
                "player": raw_player,
                "status": item.get("type") or item.get("status") or item.get("reason"),
                "reason": item.get("reason") or item.get("type"),
                "fixture_id": fixture.get("id") or item.get("fixture_id"),
                "fixture_date": fixture.get("date") or item.get("fixture_date"),
                "source": "api_football:/injuries",
                "updated_at": updated_at,
                "api_provider": "api_football",
                "raw_team": raw_team,
                "raw_player": raw_player,
                "importance": item.get("importance") or pd.NA,
                "scenario_only": True,
                "raw_payload_file": raw_payload_file,
            }
        )
    return normalize_injury_rows(rows)


def sportmonks_injuries_to_rows(payload: dict[str, Any], *, raw_payload_file: str = "") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    updated_at = _utc_now()
    for item in _sportmonks_items(payload):
        if _looks_like_fixture(item):
            fixture_id = item.get("id")
            fixture_date = item.get("starting_at") or item.get("date")
            for injury in _sportmonks_injury_items(item):
                rows.append(_sportmonks_injury_record(injury, fixture_id, fixture_date, updated_at, raw_payload_file))
        else:
            rows.append(_sportmonks_injury_record(item, item.get("fixture_id"), item.get("fixture_date"), updated_at, raw_payload_file))
    return normalize_injury_rows(rows)


def write_injury_staging(df: pd.DataFrame, path: str | Path) -> Path:
    output = normalize_injury_rows(df.to_dict("records")) if not df.empty else empty_injury_frame()
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(resolved, index=False)
    return resolved


def normalize_injury_status(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "unknown"
    for key, normalized in STATUS_ALIASES.items():
        if key in text:
            return normalized
    return text.replace(" ", "_")


def _response_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        response = payload.get("response")
        if isinstance(response, list):
            return [item for item in response if isinstance(item, dict)]
        if isinstance(response, dict):
            return [response]
    return []


def _sportmonks_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return _response_items(payload)


def _looks_like_fixture(item: dict[str, Any]) -> bool:
    return any(key in item for key in ("sidelined", "lineups", "participants", "starting_at"))


def _sportmonks_injury_items(item: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("sidelined", "injuries", "suspensions"):
        value = item.get(key)
        if isinstance(value, dict):
            value = value.get("data")
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _sportmonks_injury_record(
    injury: dict[str, Any],
    fixture_id: Any,
    fixture_date: Any,
    updated_at: str,
    raw_payload_file: str,
) -> dict[str, Any]:
    participant = injury.get("participant") or injury.get("team") or {}
    player = injury.get("player") or {}
    raw_team = participant.get("name") if isinstance(participant, dict) else injury.get("team_name")
    raw_player = player.get("name") if isinstance(player, dict) else injury.get("player_name")
    raw_team = raw_team or injury.get("team_name")
    raw_player = raw_player or injury.get("name")
    return {
        "date": injury.get("date") or fixture_date or updated_at,
        "team": raw_team,
        "player": raw_player,
        "status": injury.get("status") or injury.get("type") or injury.get("category"),
        "reason": injury.get("reason") or injury.get("type") or injury.get("category"),
        "fixture_id": fixture_id or injury.get("fixture_id"),
        "fixture_date": fixture_date or injury.get("fixture_date"),
        "source": "sportmonks:/fixtures with sidelined include",
        "updated_at": updated_at,
        "api_provider": "sportmonks",
        "raw_team": raw_team,
        "raw_player": raw_player,
        "importance": injury.get("importance") or pd.NA,
        "scenario_only": True,
        "raw_payload_file": raw_payload_file,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

