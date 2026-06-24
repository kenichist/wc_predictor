from __future__ import annotations

from src.normalize import normalize_key, normalize_team_name


API_TEAM_ALIASES = {
    "usa": "United States",
    "united states": "United States",
    "united states of america": "United States",
    "korea republic": "South Korea",
    "south korea": "South Korea",
    "korea dpr": "North Korea",
    "north korea": "North Korea",
    "cote d ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "dr congo": "DR Congo",
    "d r congo": "DR Congo",
    "congo dr": "DR Congo",
    "czech republic": "Czechia",
    "czechia": "Czechia",
    "curacao": "Curacao",
    "curacao": "Curacao",
    "bosnia herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bosnia herzegovina": "Bosnia and Herzegovina",
    "saudi arabia": "Saudi Arabia",
    "ksa": "Saudi Arabia",
    "cape verde": "Cape Verde",
    "cabo verde": "Cape Verde",
    "new zealand": "New Zealand",
    "south africa": "South Africa",
    "netherlands": "Netherlands",
    "holland": "Netherlands",
}


def normalize_api_team_name(value: object) -> str | None:
    normalized = normalize_team_name(value)
    if normalized is None:
        return None
    key = normalize_key(normalized)
    if key in API_TEAM_ALIASES:
        return API_TEAM_ALIASES[key]
    raw_key = normalize_key(str(value))
    if raw_key in API_TEAM_ALIASES:
        return API_TEAM_ALIASES[raw_key]
    return normalized


def normalize_api_team_pair(home_team: object, away_team: object) -> tuple[str | None, str | None]:
    return normalize_api_team_name(home_team), normalize_api_team_name(away_team)

