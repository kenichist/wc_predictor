from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.data_sources.config import RAW_THE_ODDS_API_DIR, THE_ODDS_API_BASE_URL, get_credentials
from src.data_sources.odds_transform import save_raw_payload, the_odds_api_to_market_rows


class TheOddsApiClient:
    def __init__(self, api_key: str | None = None, *, base_url: str = THE_ODDS_API_BASE_URL, raw_dir: Path = RAW_THE_ODDS_API_DIR) -> None:
        self.api_key = api_key or get_credentials().the_odds_api_key
        self.base_url = base_url.rstrip("/")
        self.raw_dir = Path(raw_dir)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        if not self.api_key:
            raise RuntimeError("THE_ODDS_API_KEY is not configured")
        query = dict(params or {})
        query["apiKey"] = self.api_key
        response = requests.get(f"{self.base_url}/{endpoint.lstrip('/')}", params=query, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_sports(self, all: bool = True) -> Any:
        return self.request("/sports", params={"all": "true" if all else "false"})

    def get_odds(self, sport: str, regions: str = "us,uk,eu,au", markets: str = "h2h", odds_format: str = "decimal") -> Any:
        return self.request(
            f"/sports/{sport}/odds",
            params={"regions": regions, "markets": markets, "oddsFormat": odds_format},
        )

    def get_events(self, sport: str) -> Any:
        return self.request(f"/sports/{sport}/events")

    def fetch_worldcup_2026_odds_if_available(self) -> tuple[pd.DataFrame, Path, str | None]:
        sports = self.get_sports(all=True)
        sport_key = _find_worldcup_sport_key(sports)
        if sport_key is None:
            payload: Any = {"response": [], "errors": ["No World Cup or international soccer sport key found"], "sports": sports}
        else:
            payload = self.get_odds(sport_key)
        raw_path = save_raw_payload(payload, self.raw_dir, "worldcup_2026_odds")
        return the_odds_api_to_market_rows(payload, raw_payload_file=str(raw_path)), raw_path, sport_key

    def explain_2010_limitation(self) -> str:
        return "The Odds API historical odds are documented as available from 2020 onward, so 2010 World Cup odds are not fetched."


def _find_worldcup_sport_key(sports_payload: Any) -> str | None:
    sports = sports_payload if isinstance(sports_payload, list) else sports_payload.get("data", []) if isinstance(sports_payload, dict) else []
    candidates: list[str] = []
    for sport in sports:
        if not isinstance(sport, dict):
            continue
        key = str(sport.get("key") or "")
        title = str(sport.get("title") or sport.get("description") or "")
        text = f"{key} {title}".lower()
        if "soccer" not in text:
            continue
        if "world cup" in text or "international" in text or "fifa" in text:
            candidates.append(key)
    return candidates[0] if candidates else None

