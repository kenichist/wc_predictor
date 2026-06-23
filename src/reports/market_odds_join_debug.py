from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config_path, load_config, resolve_project_path
from src.io_utils import read_dataframe
from src.sources.market_odds import (
    MARKET_FEATURE_COLUMNS,
    MARKET_JOIN_DATE_TOLERANCE_DAYS,
    _market_join_keys,
    add_market_odds_features,
    load_market_odds,
)


def write_market_odds_join_debug_report(config: dict[str, Any] | None = None) -> Path:
    cfg = config or load_config()
    report_path = _path_or_default(cfg, "market_odds_join_debug_md", "data/reports/market_odds_join_debug.md")
    odds_path = config_path(cfg, "betting_odds")
    fixture_path = config_path(cfg, "worldcup_2026_prediction_input")
    advanced_path = config_path(cfg, "worldcup_2026_prediction_input_advanced")

    odds = load_market_odds(odds_path)
    raw_prediction = read_dataframe(fixture_path) if fixture_path.exists() else pd.DataFrame()
    active_prediction = read_dataframe(advanced_path) if advanced_path.exists() else raw_prediction
    raw_prediction = _active_2026_rows(raw_prediction)
    active_prediction = _active_2026_rows(active_prediction)
    odds_2026 = odds[pd.to_datetime(odds.get("date"), errors="coerce", format="mixed").dt.year.eq(2026)].copy() if not odds.empty else odds

    before_matches = _strict_join_count(raw_prediction, odds_2026)
    after_joined = add_market_odds_features(raw_prediction, odds_path) if not raw_prediction.empty else raw_prediction
    after_matches = _market_complete_count(after_joined)
    active_matches = _market_complete_count(active_prediction)
    unmatched = _unmatched_odds_rows(raw_prediction, odds_2026)
    missing_prediction = _missing_prediction_rows(after_joined)

    total_prediction = len(raw_prediction)
    lines = [
        "# Market Odds Join Debug",
        "",
        f"- Odds file: `{odds_path}`",
        f"- Active 2026 fixture rows inspected: `{total_prediction}`",
        f"- 2026 odds rows: `{len(odds_2026)}`",
        f"- Join key: `date`, normalized `home_team`, normalized `away_team`",
        f"- Fallback: same normalized home/away team pair within `{MARKET_JOIN_DATE_TOLERANCE_DAYS}` day when strict date join misses and the best candidate is unique.",
        "",
        "## 2026 Coverage",
        "",
        f"- Before fix, strict same-date coverage: `{_coverage_text(before_matches, total_prediction)}`",
        f"- After fix, join-function coverage: `{_coverage_text(after_matches, total_prediction)}`",
        f"- Current generated advanced prediction coverage: `{_coverage_text(active_matches, len(active_prediction))}`",
        "",
        "## Main Findings",
        "",
        "- The previous zero-coverage report was caused by mixed date formats coercing prediction dates to `NaT` during advanced feature generation.",
        "- Most missing strict joins are one-day date shifts between `market_odds.csv` and the active fixture input.",
        "- Team-name normalization also needed the `D.R. Congo` alias mapped to `DR Congo`.",
        "- One 2026 odds row does not correspond to an active fixture: `Spain` vs `Cape Verde`.",
        "",
        "## Unmatched 2026 Odds Rows After Fallback",
        "",
        _markdown_table(unmatched, ["date", "home_team", "away_team", "reason"]),
        "",
        "## Active 2026 Fixture Rows Without Odds After Fallback",
        "",
        _markdown_table(missing_prediction, ["date", "home_team", "away_team", "match_id"]),
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _active_2026_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    output = df.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce", format="mixed")
    output = output[output["date"].dt.year.eq(2026)]
    if {"home_team", "away_team"}.issubset(output.columns):
        placeholder = output["home_team"].astype(str).str.match(r"^[123][A-L]|^[WL]\d+", na=False) | output["away_team"].astype(str).str.match(r"^[123][A-L]|^[WL]\d+", na=False)
        output = output[~placeholder]
    return output.reset_index(drop=True)


def _strict_join_count(prediction: pd.DataFrame, odds: pd.DataFrame) -> int:
    if prediction.empty or odds.empty:
        return 0
    left = _market_join_keys(prediction)
    right = _market_join_keys(odds).drop_duplicates()
    merged = left.merge(right.assign(_strict_match=True), on=["_market_date", "_market_home_team", "_market_away_team"], how="left")
    return int(merged["_strict_match"].fillna(False).sum())


def _market_complete_count(df: pd.DataFrame) -> int:
    if df.empty or not set(MARKET_FEATURE_COLUMNS).issubset(df.columns):
        return 0
    return int(df[["market_home_prob", "market_draw_prob", "market_away_prob"]].notna().all(axis=1).sum())


def _unmatched_odds_rows(prediction: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    if prediction.empty or odds.empty:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "reason"])
    left = _market_join_keys(prediction)
    fixture_pairs = set(zip(left["_market_home_team"], left["_market_away_team"]))
    fixture_tolerant = set()
    for _, row in left.iterrows():
        for delta in range(-MARKET_JOIN_DATE_TOLERANCE_DAYS, MARKET_JOIN_DATE_TOLERANCE_DAYS + 1):
            fixture_tolerant.add((row["_market_date"] + pd.Timedelta(days=delta), row["_market_home_team"], row["_market_away_team"]))
    right = _market_join_keys(odds)
    rows = []
    for idx, row in odds.reset_index(drop=True).iterrows():
        key_row = right.iloc[idx]
        tolerant_key = (key_row["_market_date"], key_row["_market_home_team"], key_row["_market_away_team"])
        pair = (key_row["_market_home_team"], key_row["_market_away_team"])
        if tolerant_key in fixture_tolerant:
            continue
        reason = "team pair not in active 2026 fixtures" if pair not in fixture_pairs else "outside date tolerance"
        rows.append(
            {
                "date": pd.Timestamp(row["date"]).date().isoformat() if pd.notna(row["date"]) else "",
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _missing_prediction_rows(joined_prediction: pd.DataFrame) -> pd.DataFrame:
    if joined_prediction.empty or not set(MARKET_FEATURE_COLUMNS).issubset(joined_prediction.columns):
        return pd.DataFrame(columns=["date", "home_team", "away_team", "match_id"])
    missing = joined_prediction[~joined_prediction[["market_home_prob", "market_draw_prob", "market_away_prob"]].notna().all(axis=1)].copy()
    if missing.empty:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "match_id"])
    missing["date"] = pd.to_datetime(missing["date"], errors="coerce", format="mixed").dt.date.astype(str)
    return missing[["date", "home_team", "away_team", "match_id"]].reset_index(drop=True)


def _coverage_text(matches: int, total: int) -> str:
    coverage = 0.0 if total == 0 else matches / total
    return f"{matches}/{total} ({coverage:.3f})"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in df.columns]
    if df.empty or not available:
        return "_No rows._"
    rows = ["| " + " | ".join(available) + " |", "| " + " | ".join(["---"] * len(available)) + " |"]
    for record in df[available].to_dict("records"):
        rows.append("| " + " | ".join("" if pd.isna(record[column]) else str(record[column]) for column in available) + " |")
    return "\n".join(rows)


def _path_or_default(config: dict[str, Any], key: str, default: str) -> Path:
    try:
        return config_path(config, key)
    except KeyError:
        return resolve_project_path(default)
