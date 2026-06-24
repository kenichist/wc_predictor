from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data_sources.config import API_FOOTBALL_BASE_URL, RAW_API_FOOTBALL_DIR, get_credentials
from src.data_sources.injury_transform import api_football_injuries_to_rows
from src.data_sources.odds_transform import api_football_odds_to_market_rows, save_raw_payload


logger = logging.getLogger(__name__)


class ApiFootballClient:
    def __init__(self, api_key: str | None = None, *, base_url: str = API_FOOTBALL_BASE_URL, raw_dir: Path = RAW_API_FOOTBALL_DIR) -> None:
        self.api_key = api_key or get_credentials().api_football_key
        self.base_url = base_url.rstrip("/")
        self.raw_dir = Path(raw_dir)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("API_FOOTBALL_KEY is not configured")
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = requests.get(url, headers={"x-apisports-key": self.api_key}, params=params or {}, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_leagues(self, search: str | None = None, country: str | None = None, season: int | None = None) -> dict[str, Any]:
        params = {key: value for key, value in {"search": search, "country": country, "season": season}.items() if value is not None}
        return self.request("/leagues", params=params)

    def get_fixtures(
        self,
        league: int | None = None,
        season: int | None = None,
        date: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        params = {
            key: value
            for key, value in {"league": league, "season": season, "date": date, "from": from_date, "to": to_date}.items()
            if value is not None
        }
        return self.request("/fixtures", params=params)

    def get_odds(
        self,
        league: int | None = None,
        season: int | None = None,
        fixture: int | None = None,
        bookmaker: int | None = None,
        bet: int | None = None,
    ) -> dict[str, Any]:
        params = {
            key: value
            for key, value in {"league": league, "season": season, "fixture": fixture, "bookmaker": bookmaker, "bet": bet}.items()
            if value is not None
        }
        return self.request("/odds", params=params)

    def get_injuries(
        self,
        league: int | None = None,
        season: int | None = None,
        fixture: int | None = None,
        team: int | None = None,
        player: int | None = None,
        date: str | None = None,
    ) -> dict[str, Any]:
        params = {
            key: value
            for key, value in {"league": league, "season": season, "fixture": fixture, "team": team, "player": player, "date": date}.items()
            if value is not None
        }
        return self.request("/injuries", params=params)

    def find_world_cup_league_ids(self) -> list[dict[str, Any]]:
        payload = self.get_leagues(search="World Cup")
        candidates = []
        for item in payload.get("response", []) or []:
            league = item.get("league", {}) if isinstance(item, dict) else {}
            country = item.get("country", {}) if isinstance(item, dict) else {}
            name = str(league.get("name") or "")
            if "world cup" not in name.lower():
                continue
            candidates.append({"league_id": league.get("id"), "name": name, "country": country.get("name")})
        return candidates

    def fetch_worldcup_2026_odds(self) -> tuple[pd.DataFrame, Path, list[dict[str, Any]]]:
        return self.fetch_worldcup_odds(2026)

    def probe_worldcup_2010_odds(self) -> tuple[pd.DataFrame, Path, list[dict[str, Any]]]:
        return self.fetch_worldcup_odds(2010, label="worldcup_2010_odds_probe")

    def fetch_worldcup_odds(self, season: int, *, label: str | None = None) -> tuple[pd.DataFrame, Path, list[dict[str, Any]]]:
        return self._fetch_worldcup_odds_for_season(season, label or f"worldcup_{season}_odds")

    def fetch_live_injuries_for_worldcup_teams(self, *, season: int = 2026) -> tuple[pd.DataFrame, Path, list[dict[str, Any]]]:
        candidates = self.find_world_cup_league_ids()
        league_id = _preferred_world_cup_league_id(candidates)
        if league_id is None:
            payload = {"response": [], "errors": ["No World Cup league candidate found"], "league_candidates": candidates}
        else:
            payload = self.get_injuries(league=league_id, season=season)
            payload["league_candidates"] = candidates
        raw_path = save_raw_payload(payload, self.raw_dir, f"worldcup_{season}_injuries")
        return api_football_injuries_to_rows(payload, raw_payload_file=str(raw_path)), raw_path, candidates

    def _fetch_worldcup_odds_for_season(self, season: int, label: str) -> tuple[pd.DataFrame, Path, list[dict[str, Any]]]:
        candidates = self.find_world_cup_league_ids()
        league_id = _preferred_world_cup_league_id(candidates)
        if league_id is None:
            payload = {"response": [], "errors": ["No World Cup league candidate found"], "league_candidates": candidates}
        else:
            payload = self.get_odds(league=league_id, season=season)
            payload["league_candidates"] = candidates
        raw_path = save_raw_payload(payload, self.raw_dir, label)
        return api_football_odds_to_market_rows(payload, raw_payload_file=str(raw_path)), raw_path, candidates


def _preferred_world_cup_league_id(candidates: list[dict[str, Any]]) -> int | None:
    if not candidates:
        return None
    for candidate in candidates:
        name = str(candidate.get("name") or "").lower()
        if name in {"fifa world cup", "world cup"}:
            league_id = candidate.get("league_id")
            return int(league_id) if league_id is not None else None
    league_id = candidates[0].get("league_id")
    return int(league_id) if league_id is not None else None
