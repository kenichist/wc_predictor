from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config_path, load_config
from src.io_utils import download_file, read_json, stable_id, write_dataframe
from src.normalize import add_match_outcome_columns, canonicalize_match_teams


logger = logging.getLogger(__name__)


def download_worldcup_2026_json(config: dict[str, Any] | None = None, *, force: bool = False) -> Path:
    cfg = config or load_config()
    path = config_path(cfg, "worldcup_2026_raw")
    if path.exists() and not force:
        logger.info("Using existing World Cup 2026 raw JSON at %s", path)
        return path
    api_cfg = cfg["sources"]["football_data"]
    return download_file(
        cfg["sources"]["worldcup_2026"]["url"],
        path,
        timeout_seconds=int(api_cfg.get("timeout_seconds", 30)),
        retry_count=int(api_cfg.get("retry_count", 3)),
        sleep_seconds=1.0,
    )


def parse_worldcup_json(path_or_data: str | Path | dict[str, Any]) -> pd.DataFrame:
    data = read_json(path_or_data) if isinstance(path_or_data, (str, Path)) else path_or_data
    return normalize_worldcup_matches(data)


def normalize_worldcup_matches(
    json_data: dict[str, Any],
    *,
    mapping_path: str | Path | None = None,
    output_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = config or load_config()
    tournament = json_data.get("name") or json_data.get("title") or "FIFA World Cup"
    matches = _extract_matches(json_data, {"tournament": tournament})
    df = pd.DataFrame(matches)
    if df.empty:
        df = _empty_worldcup_matches()
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
        df["source"] = "worldcup_json"
        df["competition_type"] = "world_cup"
        df["tournament"] = df["tournament"].fillna(tournament)
        df["neutral"] = True
        df["status"] = df.apply(lambda row: "FINISHED" if pd.notna(row["home_score"]) and pd.notna(row["away_score"]) else "SCHEDULED", axis=1)
        df["last_updated"] = pd.NA
        df = canonicalize_match_teams(df, mapping_path=mapping_path or config_path(cfg, "team_name_mapping"))
        df["match_id"] = [
            stable_id("worldcup_2026", row.get("source_match_id"), row.get("date"), row.get("home_team"), row.get("away_team"), row.get("stage"), row.get("group"))
            for row in df.to_dict("records")
        ]
        df = add_match_outcome_columns(df)

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
        "result",
        "home_win",
        "draw",
        "away_win",
        "goal_diff",
        "total_goals",
        "status",
        "last_updated",
    ]
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    df = df[columns].sort_values(["date", "stage", "group"], kind="stable").reset_index(drop=True)
    if output_path is not None:
        write_dataframe(df, output_path)
    logger.info("Normalized %s World Cup 2026 fixture/result rows", len(df))
    return df


def _extract_matches(node: Any, context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(node, list):
        for item in node:
            rows.extend(_extract_matches(item, context))
        return rows
    if not isinstance(node, dict):
        return rows

    local_context = context.copy()
    name = node.get("name") or node.get("title")
    if isinstance(name, str):
        lower = name.lower()
        if "group" in lower:
            local_context["group"] = name
            local_context.setdefault("stage", "Group Stage")
        elif any(term in lower for term in ("round", "quarter", "semi", "final", "third")):
            local_context["stage"] = name

    if _looks_like_match(node):
        return [_normalize_match_node(node, local_context)]

    for key, value in node.items():
        child_context = local_context.copy()
        key_lower = str(key).lower()
        if key_lower == "groups":
            child_context.setdefault("stage", "Group Stage")
        elif key_lower in {"rounds", "knockouts", "playoffs"}:
            child_context.pop("group", None)
        rows.extend(_extract_matches(value, child_context))
    return rows


def _looks_like_match(node: dict[str, Any]) -> bool:
    home = _first_present(node, ["team1", "home_team", "homeTeam", "home", "team_a"])
    away = _first_present(node, ["team2", "away_team", "awayTeam", "away", "team_b"])
    return home is not None and away is not None and any(key in node for key in ("date", "utcDate", "datetime", "time"))


def _normalize_match_node(node: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    home = _first_present(node, ["team1", "home_team", "homeTeam", "home", "team_a"])
    away = _first_present(node, ["team2", "away_team", "awayTeam", "away", "team_b"])
    home_score, away_score = _extract_score(node)
    stadium = node.get("stadium") or node.get("venue")
    city = node.get("city")
    return {
        "source_match_id": node.get("num") or node.get("id") or node.get("matchday"),
        "date": node.get("date") or node.get("utcDate") or node.get("datetime") or node.get("time"),
        "home_team": _extract_name(home),
        "away_team": _extract_name(away),
        "tournament": context.get("tournament", "FIFA World Cup"),
        "stage": node.get("stage") or node.get("round") or context.get("stage"),
        "group": node.get("group") or context.get("group"),
        "country": _extract_name(node.get("country")),
        "city": _extract_name(city),
        "venue": _extract_name(stadium),
        "home_score": home_score,
        "away_score": away_score,
    }


def _extract_score(node: dict[str, Any]) -> tuple[Any, Any]:
    direct_home = _first_present(node, ["score1", "home_score", "goals1", "homeGoals"])
    direct_away = _first_present(node, ["score2", "away_score", "goals2", "awayGoals"])
    if direct_home is not None or direct_away is not None:
        return direct_home, direct_away

    score = node.get("score")
    if isinstance(score, dict):
        full_time = score.get("fullTime") or score.get("ft") or score
        if isinstance(full_time, dict):
            return _first_present(full_time, ["home", "team1", "score1"]), _first_present(full_time, ["away", "team2", "score2"])
        if isinstance(full_time, list) and len(full_time) >= 2:
            return full_time[0], full_time[1]
    if isinstance(score, str) and "-" in score:
        left, right = score.split("-", 1)
        return left.strip(), right.strip()
    return None, None


def _extract_name(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, dict):
        for key in ("name", "fullName", "shortName", "title", "city"):
            candidate = value.get(key)
            if candidate:
                return str(candidate)
        return None
    return str(value)


def _first_present(node: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in node and node[key] not in ("", None):
            return node[key]
    return None


def _empty_worldcup_matches() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
            "result",
            "home_win",
            "draw",
            "away_win",
            "goal_diff",
            "total_goals",
            "status",
            "last_updated",
        ]
    )
