from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TEAM_NAME_OVERRIDES = {
    "usa": "United States",
    "u s a": "United States",
    "united states of america": "United States",
    "usmnt": "United States",
    "korea republic": "South Korea",
    "korea south": "South Korea",
    "south korea": "South Korea",
    "republic of korea": "South Korea",
    "ir iran": "Iran",
    "islamic republic of iran": "Iran",
    "iran islamic republic of": "Iran",
    "cote d ivoire": "Ivory Coast",
    "cote divoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "turkiye": "Turkey",
    "turkey": "Turkey",
    "czech republic": "Czechia",
    "czechia": "Czechia",
    "bosnia herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "cape verde islands": "Cape Verde",
    "cape verde": "Cape Verde",
    "cabo verde": "Cape Verde",
    "curacao": "Curaçao",
    "congo dr": "DR Congo",
    "d r congo": "DR Congo",
    "dr congo": "DR Congo",
    "dem rep of congo": "DR Congo",
    "democratic republic congo": "DR Congo",
    "democratic republic of congo": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "russia": "Russia",
    "england": "England",
    "scotland": "Scotland",
    "wales": "Wales",
    "northern ireland": "Northern Ireland",
}


def _is_missing(value: object) -> bool:
    return value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value))


def strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_key(value: str) -> str:
    value = strip_accents(str(value)).replace("&", " and ")
    value = re.sub(r"[^A-Za-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def normalize_team_name(name: object) -> str | None:
    """Return a canonical team name, preserving unknown names in title form."""
    if _is_missing(name):
        return None
    raw = str(name).strip()
    if not raw:
        return None
    key = normalize_key(raw)
    if key in TEAM_NAME_OVERRIDES:
        return TEAM_NAME_OVERRIDES[key]
    return re.sub(r"\s+", " ", raw).strip()


def save_team_mapping(raw_names: Iterable[object], path: str | Path) -> pd.DataFrame:
    """Persist raw-to-canonical team mappings, preserving any existing rows."""
    rows = []
    for raw in raw_names:
        if _is_missing(raw) or str(raw).strip() == "":
            continue
        rows.append({"raw_team_name": str(raw).strip(), "canonical_team_name": normalize_team_name(raw)})
    mapping = pd.DataFrame(rows)
    if mapping.empty:
        mapping = pd.DataFrame(columns=["raw_team_name", "canonical_team_name"])
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        existing = pd.read_csv(resolved)
        mapping = pd.concat([existing, mapping], ignore_index=True)
    mapping = mapping.dropna(subset=["raw_team_name"]).drop_duplicates().sort_values(
        ["canonical_team_name", "raw_team_name"], kind="stable"
    )
    mapping.to_csv(resolved, index=False)
    return mapping


def canonicalize_match_teams(
    df: pd.DataFrame,
    *,
    columns: tuple[str, str] = ("home_team", "away_team"),
    mapping_path: str | Path | None = None,
) -> pd.DataFrame:
    output = df.copy()
    raw_names: list[object] = []
    for column in columns:
        if column in output.columns:
            raw_names.extend(output[column].dropna().tolist())
            output[column] = output[column].map(normalize_team_name)
    if mapping_path is not None:
        save_team_mapping(raw_names, mapping_path)
    return output


def classify_competition_type(tournament: object) -> str:
    text = normalize_key("" if _is_missing(tournament) else str(tournament))
    if "friendly" in text:
        return "friendly"
    if "world cup" in text and ("qualification" in text or "qualifier" in text or "qualifying" in text):
        return "world_cup_qualifier"
    if "world cup" in text:
        return "world_cup"
    if "nations league" in text:
        return "nations_league"
    if "qualification" in text or "qualifier" in text or "qualifying" in text:
        return "continental_qualifier"
    continental_terms = [
        "euro",
        "copa america",
        "african cup",
        "afcon",
        "asian cup",
        "gold cup",
        "concacaf championship",
        "ofc nations",
    ]
    if any(term in text for term in continental_terms):
        return "continental_championship"
    return "other"


def add_match_outcome_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["home_score"] = pd.to_numeric(output.get("home_score"), errors="coerce")
    output["away_score"] = pd.to_numeric(output.get("away_score"), errors="coerce")
    known = output["home_score"].notna() & output["away_score"].notna()
    home_gt = output["home_score"] > output["away_score"]
    away_gt = output["away_score"] > output["home_score"]
    draw = output["home_score"] == output["away_score"]

    output["result"] = np.select(
        [known & home_gt, known & draw, known & away_gt],
        ["home_win", "draw", "away_win"],
        default=pd.NA,
    )
    output["home_win"] = pd.Series(np.where(known, home_gt, pd.NA), index=output.index)
    output["draw"] = pd.Series(np.where(known, draw, pd.NA), index=output.index)
    output["away_win"] = pd.Series(np.where(known, away_gt, pd.NA), index=output.index)
    output["goal_diff"] = np.where(known, output["home_score"] - output["away_score"], np.nan)
    output["total_goals"] = np.where(known, output["home_score"] + output["away_score"], np.nan)
    return output
