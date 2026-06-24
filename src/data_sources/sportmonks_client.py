from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data_sources.config import RAW_SPORTMONKS_DIR, SPORTMONKS_BASE_URL, get_credentials
from src.data_sources.injury_transform import sportmonks_injuries_to_rows
from src.data_sources.odds_transform import save_raw_payload, sportmonks_odds_to_market_rows


class SportmonksClient:
    def __init__(self, api_token: str | None = None, *, base_url: str = SPORTMONKS_BASE_URL, raw_dir: Path = RAW_SPORTMONKS_DIR) -> None:
        self.api_token = api_token or get_credentials().sportmonks_api_token
        self.base_url = base_url.rstrip("/")
        self.raw_dir = Path(raw_dir)

    @property
    def available(self) -> bool:
        return bool(self.api_token)

    def request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.api_token:
            raise RuntimeError("SPORTMONKS_API_TOKEN is not configured")
        query = dict(params or {})
        query["api_token"] = self.api_token
        response = requests.get(f"{self.base_url}/{endpoint.lstrip('/')}", params=query, timeout=30)
        response.raise_for_status()
        return response.json()

    def search_leagues(self, query: str) -> dict[str, Any]:
        return self.request("/leagues/search/" + query)

    def get_fixtures_by_date(self, date: str) -> dict[str, Any]:
        return self.request(f"/fixtures/date/{date}", params={"include": "participants"})

    def get_fixtures_by_date_range(self, from_date: str, to_date: str) -> dict[str, Any]:
        return self.request(f"/fixtures/between/{from_date}/{to_date}", params={"include": "participants;odds;premiumOdds"})

    def get_fixture_by_id(self, fixture_id: int, include: str | None = None) -> dict[str, Any]:
        params = {"include": include} if include else None
        return self.request(f"/fixtures/{fixture_id}", params=params)

    def fetch_worldcup_2026_odds_if_available(self) -> tuple[pd.DataFrame, Path]:
        payload = self._safe_worldcup_2026_payload(include="participants;odds;premiumOdds")
        raw_path = save_raw_payload(payload, self.raw_dir, "worldcup_2026_odds")
        return sportmonks_odds_to_market_rows(payload, raw_payload_file=str(raw_path)), raw_path

    def fetch_worldcup_injuries_if_available(self) -> tuple[pd.DataFrame, Path]:
        payload = self._safe_worldcup_2026_payload(include="participants;sidelined;lineups;expectedLineups")
        raw_path = save_raw_payload(payload, self.raw_dir, "worldcup_2026_injuries")
        return sportmonks_injuries_to_rows(payload, raw_payload_file=str(raw_path)), raw_path

    def probe_worldcup_2010_odds_if_available(self) -> tuple[pd.DataFrame, Path]:
        try:
            payload = self.get_fixtures_by_date_range("2010-06-11", "2010-07-11")
        except requests.HTTPError as exc:
            payload = {"data": [], "errors": [str(exc)], "note": "Sportmonks 2010 odds probe failed or requires paid access."}
        raw_path = save_raw_payload(payload, self.raw_dir, "worldcup_2010_odds_probe")
        return sportmonks_odds_to_market_rows(payload, raw_payload_file=str(raw_path)), raw_path

    def _safe_worldcup_2026_payload(self, *, include: str) -> dict[str, Any]:
        try:
            return self.request("/fixtures/between/2026-06-11/2026-07-19", params={"include": include})
        except requests.HTTPError as exc:
            return {"data": [], "errors": [str(exc)], "note": "Fixture include may require paid Sportmonks add-ons."}

