from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_sources.team_normalization import normalize_api_team_name


MARKET_ODDS_STAGING_SCHEMA = [
    "date",
    "home_team",
    "away_team",
    "home_odds",
    "draw_odds",
    "away_odds",
    "bookmaker",
    "source",
    "updated_at",
    "api_provider",
    "api_fixture_id",
    "raw_home_team",
    "raw_away_team",
    "raw_payload_file",
]

MATCH_WINNER_MARKET_NAMES = {
    "match winner",
    "winner",
    "1x2",
    "home/draw/away",
    "home draw away",
    "full time result",
    "fulltime result",
    "h2h",
}

HOME_VALUES = {"home", "1", "home team"}
DRAW_VALUES = {"draw", "x", "tie"}
AWAY_VALUES = {"away", "2", "away team"}


def empty_market_odds_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=MARKET_ODDS_STAGING_SCHEMA)


def normalize_market_odds_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return empty_market_odds_frame()
    output = pd.DataFrame(rows)
    for column in MARKET_ODDS_STAGING_SCHEMA:
        if column not in output.columns:
            output[column] = pd.NA
    output["date"] = pd.to_datetime(output["date"], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    output["home_team"] = output["home_team"].map(normalize_api_team_name)
    output["away_team"] = output["away_team"].map(normalize_api_team_name)
    for column in ("home_odds", "draw_odds", "away_odds"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output[MARKET_ODDS_STAGING_SCHEMA]


def api_football_odds_to_market_rows(payload: dict[str, Any], *, raw_payload_file: str = "") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    updated_at = _utc_now()
    for item in _response_items(payload):
        fixture = item.get("fixture", {}) if isinstance(item, dict) else {}
        fixture_id = fixture.get("id") or item.get("fixture_id")
        match_date = fixture.get("date") or item.get("date")
        raw_home, raw_away = _api_football_teams(item)
        bookmaker_prices: list[tuple[str, float, float, float]] = []
        for bookmaker in item.get("bookmakers", []) or []:
            bookmaker_name = str(bookmaker.get("name") or bookmaker.get("id") or "api_football")
            for bet in bookmaker.get("bets", []) or []:
                if not _is_match_winner_market(bet.get("name") or bet.get("label") or bet.get("id")):
                    continue
                prices = _values_to_1x2_prices(bet.get("values", []), home_name=raw_home, away_name=raw_away)
                if prices is not None:
                    bookmaker_prices.append((bookmaker_name, *prices))
        if not bookmaker_prices:
            continue
        odds = np.array([[home, draw, away] for _, home, draw, away in bookmaker_prices], dtype=float)
        avg_home, avg_draw, avg_away = np.nanmean(odds, axis=0)
        if not _valid_1x2(avg_home, avg_draw, avg_away):
            continue
        rows.append(
            {
                "date": match_date,
                "home_team": raw_home,
                "away_team": raw_away,
                "home_odds": avg_home,
                "draw_odds": avg_draw,
                "away_odds": avg_away,
                "bookmaker": "api_football_average" if len(bookmaker_prices) > 1 else bookmaker_prices[0][0],
                "source": "api_football:/odds",
                "updated_at": updated_at,
                "api_provider": "api_football",
                "api_fixture_id": fixture_id,
                "raw_home_team": raw_home,
                "raw_away_team": raw_away,
                "raw_payload_file": raw_payload_file,
            }
        )
    return normalize_market_odds_rows(rows)


def the_odds_api_to_market_rows(payload: Any, *, raw_payload_file: str = "") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    updated_at = _utc_now()
    for item in payload if isinstance(payload, list) else _response_items(payload):
        if not isinstance(item, dict):
            continue
        raw_home = item.get("home_team")
        raw_away = item.get("away_team")
        match_date = item.get("commence_time") or item.get("date")
        fixture_id = item.get("id")
        bookmaker_prices: list[tuple[str, float, float, float]] = []
        for bookmaker in item.get("bookmakers", []) or []:
            bookmaker_name = str(bookmaker.get("title") or bookmaker.get("key") or "the_odds_api")
            for market in bookmaker.get("markets", []) or []:
                if str(market.get("key", "")).lower() != "h2h":
                    continue
                prices = _outcomes_to_named_1x2(market.get("outcomes", []), raw_home, raw_away)
                if prices is not None:
                    bookmaker_prices.append((bookmaker_name, *prices))
        if not bookmaker_prices:
            continue
        odds = np.array([[home, draw, away] for _, home, draw, away in bookmaker_prices], dtype=float)
        avg_home, avg_draw, avg_away = np.nanmean(odds, axis=0)
        if not _valid_1x2(avg_home, avg_draw, avg_away):
            continue
        rows.append(
            {
                "date": match_date,
                "home_team": raw_home,
                "away_team": raw_away,
                "home_odds": avg_home,
                "draw_odds": avg_draw,
                "away_odds": avg_away,
                "bookmaker": "the_odds_api_average" if len(bookmaker_prices) > 1 else bookmaker_prices[0][0],
                "source": "the_odds_api:/sports/{sport}/odds",
                "updated_at": updated_at,
                "api_provider": "the_odds_api",
                "api_fixture_id": fixture_id,
                "raw_home_team": raw_home,
                "raw_away_team": raw_away,
                "raw_payload_file": raw_payload_file,
            }
        )
    return normalize_market_odds_rows(rows)


def sportmonks_odds_to_market_rows(payload: dict[str, Any], *, raw_payload_file: str = "") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    updated_at = _utc_now()
    for item in _sportmonks_items(payload):
        fixture_id = item.get("id") or item.get("fixture_id")
        match_date = item.get("starting_at") or item.get("date")
        raw_home, raw_away = _sportmonks_teams(item)
        prices = _sportmonks_1x2_prices(item, raw_home, raw_away)
        if prices is None:
            continue
        home_odds, draw_odds, away_odds = prices
        rows.append(
            {
                "date": match_date,
                "home_team": raw_home,
                "away_team": raw_away,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "bookmaker": "sportmonks_average",
                "source": "sportmonks:/fixtures with odds include",
                "updated_at": updated_at,
                "api_provider": "sportmonks",
                "api_fixture_id": fixture_id,
                "raw_home_team": raw_home,
                "raw_away_team": raw_away,
                "raw_payload_file": raw_payload_file,
            }
        )
    return normalize_market_odds_rows(rows)


def write_market_staging(df: pd.DataFrame, path: str | Path) -> Path:
    output = normalize_market_odds_rows(df.to_dict("records")) if not df.empty else empty_market_odds_frame()
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(resolved, index=False)
    return resolved


def save_raw_payload(payload: Any, raw_dir: Path, label: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = raw_dir / f"{label}_{timestamp}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def _response_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        response = payload.get("response")
        if isinstance(response, list):
            return [item for item in response if isinstance(item, dict)]
        if isinstance(response, dict):
            return [response]
    return []


def _api_football_teams(item: dict[str, Any]) -> tuple[str | None, str | None]:
    teams = item.get("teams", {}) or {}
    home = teams.get("home", {}) if isinstance(teams, dict) else {}
    away = teams.get("away", {}) if isinstance(teams, dict) else {}
    return home.get("name") or item.get("home_team"), away.get("name") or item.get("away_team")


def _sportmonks_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return _response_items(payload)


def _sportmonks_teams(item: dict[str, Any]) -> tuple[str | None, str | None]:
    participants = item.get("participants") or item.get("participants_data") or []
    home = item.get("home_team") or item.get("localteam", {}).get("name") if isinstance(item.get("localteam"), dict) else item.get("home_team")
    away = item.get("away_team") or item.get("visitorteam", {}).get("name") if isinstance(item.get("visitorteam"), dict) else item.get("away_team")
    if isinstance(participants, list):
        for participant in participants:
            if not isinstance(participant, dict):
                continue
            meta = participant.get("meta") or {}
            location = str(meta.get("location") or participant.get("location") or "").lower()
            name = participant.get("name") or participant.get("team", {}).get("name") if isinstance(participant.get("team"), dict) else participant.get("name")
            if location == "home":
                home = name
            elif location == "away":
                away = name
    return home, away


def _sportmonks_1x2_prices(item: dict[str, Any], raw_home: str | None, raw_away: str | None) -> tuple[float, float, float] | None:
    odds_nodes = item.get("odds") or item.get("premiumOdds") or item.get("bookmakers") or []
    candidate_rows: list[tuple[float, float, float]] = []
    if isinstance(odds_nodes, dict):
        odds_nodes = odds_nodes.get("data") or odds_nodes.get("odds") or []
    for node in odds_nodes if isinstance(odds_nodes, list) else []:
        if not isinstance(node, dict):
            continue
        market_name = node.get("market_description") or node.get("market") or node.get("name") or node.get("label")
        if market_name and not _is_match_winner_market(market_name):
            continue
        values = node.get("values") or node.get("outcomes") or node.get("bookmaker") or node.get("odds") or []
        prices = _values_to_1x2_prices(values, home_name=raw_home, away_name=raw_away)
        if prices is not None:
            candidate_rows.append(prices)
    if not candidate_rows:
        return None
    return tuple(np.nanmean(np.asarray(candidate_rows, dtype=float), axis=0))  # type: ignore[return-value]


def _is_match_winner_market(name: Any) -> bool:
    text = str(name or "").strip().lower().replace("-", " ").replace("_", " ")
    return text in MATCH_WINNER_MARKET_NAMES or "match winner" in text or "full time result" in text


def _values_to_1x2_prices(values: Any, *, home_name: str | None, away_name: str | None) -> tuple[float, float, float] | None:
    if not isinstance(values, list):
        return None
    home = draw = away = np.nan
    for item in values:
        if not isinstance(item, dict):
            continue
        label = str(item.get("value") or item.get("label") or item.get("name") or item.get("team") or item.get("participant") or "").strip()
        price = _price(item.get("odd") or item.get("odds") or item.get("price") or item.get("value_decimal"))
        key = label.lower()
        if key in HOME_VALUES or _same_team(label, home_name):
            home = price
        elif key in DRAW_VALUES:
            draw = price
        elif key in AWAY_VALUES or _same_team(label, away_name):
            away = price
    if _valid_1x2(home, draw, away):
        return float(home), float(draw), float(away)
    return None


def _outcomes_to_named_1x2(outcomes: Any, raw_home: str | None, raw_away: str | None) -> tuple[float, float, float] | None:
    if not isinstance(outcomes, list):
        return None
    home = draw = away = np.nan
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        name = outcome.get("name")
        price = _price(outcome.get("price"))
        if _same_team(name, raw_home):
            home = price
        elif _same_team(name, raw_away):
            away = price
        elif str(name).strip().lower() in DRAW_VALUES:
            draw = price
    if _valid_1x2(home, draw, away):
        return float(home), float(draw), float(away)
    return None


def _same_team(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return normalize_api_team_name(left) == normalize_api_team_name(right)


def _price(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _valid_1x2(home: Any, draw: Any, away: Any) -> bool:
    values = pd.to_numeric(pd.Series([home, draw, away]), errors="coerce")
    return bool(values.notna().all() and values.gt(1).all())


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

