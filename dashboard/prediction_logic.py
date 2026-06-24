from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dashboard.scenario import advancement_probabilities, infer_probability_columns, normalize_probabilities
from src.normalize import normalize_key, normalize_team_name


SAFE_MODEL = "football_only_ensemble"
SAFE_FEATURE_SET = "core_football_only"
DEFAULT_MARKET_ALPHA = 0.3

LIVE_FIXTURES_SCHEMA = [
    "fixture_id",
    "date",
    "kickoff_time",
    "status",
    "minute",
    "home_team",
    "away_team",
    "venue",
    "stage",
    "group",
    "source",
    "updated_at",
]

LIVE_SCORES_SCHEMA = [
    "fixture_id",
    "date",
    "home_team",
    "away_team",
    "status",
    "minute",
    "home_score",
    "away_score",
    "home_red_cards",
    "away_red_cards",
    "source",
    "updated_at",
]

LIVE_ODDS_SCHEMA = [
    "fixture_id",
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
]

LIVE_INJURIES_SCHEMA = [
    "fixture_id",
    "date",
    "team",
    "player",
    "status",
    "reason",
    "importance",
    "source",
    "updated_at",
    "api_provider",
    "scenario_only",
]

LIVE_LINEUPS_SCHEMA = [
    "fixture_id",
    "date",
    "team",
    "player",
    "starter",
    "position",
    "formation",
    "status",
    "source",
    "updated_at",
    "api_provider",
    "scenario_only",
]

LIVE_PREDICTIONS_SCHEMA = [
    "fixture_id",
    "date",
    "home_team",
    "away_team",
    "status",
    "minute",
    "stage",
    "base_home_prob",
    "base_draw_prob",
    "base_away_prob",
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
    "live_home_prob",
    "live_draw_prob",
    "live_away_prob",
    "home_advances_prob",
    "away_advances_prob",
    "home_score",
    "away_score",
    "actual_result",
    "prediction_correct",
    "predicted_result",
    "confidence",
    "mode",
    "last_updated",
]

LIVE_FACTORS_SCHEMA = [
    "fixture_id",
    "factor_type",
    "team",
    "description",
    "impact_direction",
    "impact_strength",
    "source",
    "updated_at",
]

TEAM_ALIASES = {
    "usa": "United States",
    "united states of america": "United States",
    "korea republic": "South Korea",
    "korea dpr": "North Korea",
    "cote d ivoire": "Ivory Coast",
    "cote divoire": "Ivory Coast",
    "d r congo": "DR Congo",
    "congo dr": "DR Congo",
    "czech republic": "Czechia",
    "bosnia herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "ksa": "Saudi Arabia",
    "cabo verde": "Cape Verde",
    "holland": "Netherlands",
    "curacao": "Curacao",
}


def normalize_dashboard_team(value: object) -> str | None:
    normalized = normalize_team_name(value)
    if normalized is None:
        return None
    key = normalize_key(normalized)
    raw_key = normalize_key(str(value))
    return TEAM_ALIASES.get(raw_key) or TEAM_ALIASES.get(key) or normalized


def empty_frame(schema: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=schema)


def ensure_schema(df: pd.DataFrame, schema: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in schema:
        if column not in output.columns:
            output[column] = pd.NA
    return output[schema]


def odds_to_implied_probabilities(home_odds: Any, draw_odds: Any, away_odds: Any) -> tuple[float, float, float] | None:
    try:
        odds = np.array([float(home_odds), float(draw_odds), float(away_odds)], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(odds).all() or np.any(odds <= 1):
        return None
    probs = normalize_probabilities(1.0 / odds)
    return float(probs[0]), float(probs[1]), float(probs[2])


def blend_probabilities(
    base: tuple[float, float, float] | None,
    market: tuple[float, float, float] | None,
    *,
    alpha: float = DEFAULT_MARKET_ALPHA,
) -> tuple[float, float, float] | None:
    if base is None and market is None:
        return None
    if base is None:
        return _tuple_probs(market)
    if market is None:
        return _tuple_probs(base)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = alpha * np.asarray(base, dtype=float) + (1.0 - alpha) * np.asarray(market, dtype=float)
    return _tuple_probs(tuple(normalize_probabilities(blended)))


def apply_live_adjustments(
    probabilities: tuple[float, float, float],
    *,
    home_adjustment: float = 0.0,
    away_adjustment: float = 0.0,
) -> tuple[float, float, float]:
    base = normalize_probabilities(probabilities)
    logits = np.log(np.clip(base, 1e-9, None))
    logits[0] += float(home_adjustment)
    logits[2] += float(away_adjustment)
    shifted = logits - np.max(logits)
    adjusted = np.exp(shifted)
    adjusted = adjusted / adjusted.sum()
    return float(adjusted[0]), float(adjusted[1]), float(adjusted[2])


def build_live_predictions(
    *,
    fixtures: pd.DataFrame,
    official_predictions: pd.DataFrame,
    live_odds: pd.DataFrame | None = None,
    external_odds: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    lineups: pd.DataFrame | None = None,
    scores: pd.DataFrame | None = None,
    alpha: float = DEFAULT_MARKET_ALPHA,
    apply_unknown_impact: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    fixtures = _prepare_fixtures(fixtures)
    if fixtures.empty:
        fixtures = _fixtures_from_official_predictions(official_predictions)
    live_odds = _prepare_odds(live_odds if live_odds is not None else pd.DataFrame())
    external_odds = _prepare_odds(external_odds if external_odds is not None else pd.DataFrame())
    injuries = _prepare_injuries(injuries if injuries is not None else pd.DataFrame())
    lineups = _prepare_lineups(lineups if lineups is not None else pd.DataFrame())
    scores = _prepare_scores(scores if scores is not None else pd.DataFrame())

    rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    unmatched_teams: set[str] = set()
    invalid_odds = 0
    missing_base = 0

    for _, fixture in fixtures.iterrows():
        home = normalize_dashboard_team(fixture.get("home_team"))
        away = normalize_dashboard_team(fixture.get("away_team"))
        if home is None or away is None:
            unmatched_teams.update(str(value) for value in [fixture.get("home_team"), fixture.get("away_team")] if pd.notna(value))
            continue
        fixture_id = fixture.get("fixture_id")
        date = _date_key(fixture.get("date"))
        score_row = _lookup_score(scores, fixture_id, date, home, away)
        status = _first_present(score_row.get("status") if score_row is not None else None, fixture.get("status"), "Upcoming")
        minute = _first_present(score_row.get("minute") if score_row is not None else None, fixture.get("minute"), pd.NA)
        base = find_base_probabilities(official_predictions, date, home, away)
        if base is None:
            missing_base += 1
        market, odds_valid = find_market_probabilities(live_odds, date, home, away)
        if market is None:
            market, odds_valid = find_market_probabilities(external_odds, date, home, away)
        if not odds_valid:
            invalid_odds += 1
        blended = blend_probabilities(base, market, alpha=alpha)
        if blended is None:
            continue
        home_adjustment, away_adjustment, factors = live_factor_adjustments(
            fixture_id=fixture_id,
            home_team=home,
            away_team=away,
            injuries=injuries,
            lineups=lineups,
            score_row=score_row,
            updated_at=now,
            apply_unknown_impact=apply_unknown_impact,
        )
        factor_rows.extend(factors)
        live = apply_live_adjustments(blended, home_adjustment=home_adjustment, away_adjustment=away_adjustment)
        home_adv, away_adv = (pd.NA, pd.NA)
        if _is_knockout_stage(fixture.get("stage")):
            home_adv, away_adv = advancement_probabilities(live[0], live[2])
        predicted_result = _prediction_label(home, away, live)
        home_score = score_row.get("home_score") if score_row is not None else pd.NA
        away_score = score_row.get("away_score") if score_row is not None else pd.NA
        actual_result = _actual_result_label(home, away, home_score, away_score) if _is_finished_status(status) else pd.NA
        prediction_correct = pd.NA if pd.isna(actual_result) else predicted_result == actual_result
        rows.append(
            {
                "fixture_id": fixture_id,
                "date": date,
                "home_team": home,
                "away_team": away,
                "status": status,
                "minute": minute,
                "stage": fixture.get("stage"),
                "base_home_prob": None if base is None else base[0],
                "base_draw_prob": None if base is None else base[1],
                "base_away_prob": None if base is None else base[2],
                "market_home_prob": None if market is None else market[0],
                "market_draw_prob": None if market is None else market[1],
                "market_away_prob": None if market is None else market[2],
                "live_home_prob": live[0],
                "live_draw_prob": live[1],
                "live_away_prob": live[2],
                "home_advances_prob": home_adv,
                "away_advances_prob": away_adv,
                "home_score": home_score,
                "away_score": away_score,
                "actual_result": actual_result,
                "prediction_correct": prediction_correct,
                "predicted_result": predicted_result,
                "confidence": max(live),
                "mode": "live_scenario",
                "last_updated": now,
            }
        )

    predictions = ensure_schema(pd.DataFrame(rows), LIVE_PREDICTIONS_SCHEMA)
    factors_df = ensure_schema(pd.DataFrame(factor_rows), LIVE_FACTORS_SCHEMA)
    summary = {
        "LIVE_FIXTURES": int(len(fixtures)),
        "LIVE_ODDS_ROWS": int(len(live_odds)),
        "LIVE_INJURY_ROWS": int(len(injuries)),
        "LIVE_LINEUP_ROWS": int(len(lineups)),
        "LIVE_PREDICTION_ROWS": int(len(predictions)),
        "INVALID_ODDS": int(invalid_odds),
        "UNMATCHED_TEAMS": int(len(unmatched_teams)),
        "MISSING_BASE_PREDICTIONS": int(missing_base),
        "LAST_UPDATED": now,
        "SUCCESS": True,
    }
    return predictions, factors_df, summary


def write_live_prediction_outputs(
    predictions: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    live_dir: str | Path = "data/live",
) -> tuple[Path, Path]:
    live_path = Path(live_dir)
    live_path.mkdir(parents=True, exist_ok=True)
    prediction_path = live_path / "live_predictions.csv"
    factors_path = live_path / "live_prediction_factors.csv"
    ensure_schema(predictions, LIVE_PREDICTIONS_SCHEMA).to_csv(prediction_path, index=False)
    ensure_schema(factors, LIVE_FACTORS_SCHEMA).to_csv(factors_path, index=False)
    return prediction_path, factors_path


def find_base_probabilities(predictions: pd.DataFrame, date: str | None, home_team: str, away_team: str) -> tuple[float, float, float] | None:
    if predictions.empty or not {"home_team", "away_team"}.issubset(predictions.columns):
        return None
    prob_cols = infer_probability_columns(predictions)
    if prob_cols is None:
        return None
    frame = predictions.copy()
    frame["_home_norm"] = frame["home_team"].map(normalize_dashboard_team)
    frame["_away_norm"] = frame["away_team"].map(normalize_dashboard_team)
    frame["_date_key"] = frame["date"].map(_date_key) if "date" in frame.columns else None
    exact = frame[frame["_home_norm"].eq(home_team) & frame["_away_norm"].eq(away_team)]
    if date and "_date_key" in exact.columns:
        dated = exact[exact["_date_key"].eq(date)]
        if not dated.empty:
            exact = dated
    if not exact.empty:
        row = exact.iloc[0]
        return _tuple_probs((row[prob_cols.home], row[prob_cols.draw], row[prob_cols.away]))
    reverse = frame[frame["_home_norm"].eq(away_team) & frame["_away_norm"].eq(home_team)]
    if date and "_date_key" in reverse.columns:
        dated = reverse[reverse["_date_key"].eq(date)]
        if not dated.empty:
            reverse = dated
    if not reverse.empty:
        row = reverse.iloc[0]
        return _tuple_probs((row[prob_cols.away], row[prob_cols.draw], row[prob_cols.home]))
    return None


def find_market_probabilities(odds: pd.DataFrame, date: str | None, home_team: str, away_team: str) -> tuple[tuple[float, float, float] | None, bool]:
    if odds.empty or not {"home_team", "away_team", "home_odds", "draw_odds", "away_odds"}.issubset(odds.columns):
        return None, True
    frame = odds.copy()
    frame["_home_norm"] = frame["home_team"].map(normalize_dashboard_team)
    frame["_away_norm"] = frame["away_team"].map(normalize_dashboard_team)
    frame["_date_key"] = frame["date"].map(_date_key) if "date" in frame.columns else None
    frame["_updated"] = pd.to_datetime(
        frame.get("updated_at"),
        errors="coerce",
        format="mixed",
        utc=True,
    )
    direct = frame[frame["_home_norm"].eq(home_team) & frame["_away_norm"].eq(away_team)]
    if date and "_date_key" in direct.columns:
        dated = direct[direct["_date_key"].eq(date)]
        if not dated.empty:
            direct = dated
    if not direct.empty:
        row = direct.sort_values("_updated", kind="stable").iloc[-1]
        return odds_to_implied_probabilities(row["home_odds"], row["draw_odds"], row["away_odds"]), odds_to_implied_probabilities(row["home_odds"], row["draw_odds"], row["away_odds"]) is not None
    reverse = frame[frame["_home_norm"].eq(away_team) & frame["_away_norm"].eq(home_team)]
    if date and "_date_key" in reverse.columns:
        dated = reverse[reverse["_date_key"].eq(date)]
        if not dated.empty:
            reverse = dated
    if not reverse.empty:
        row = reverse.sort_values("_updated", kind="stable").iloc[-1]
        probs = odds_to_implied_probabilities(row["home_odds"], row["draw_odds"], row["away_odds"])
        if probs is None:
            return None, False
        return (probs[2], probs[1], probs[0]), True
    return None, True


def live_factor_adjustments(
    *,
    fixture_id: Any,
    home_team: str,
    away_team: str,
    injuries: pd.DataFrame,
    lineups: pd.DataFrame,
    score_row: pd.Series | None,
    updated_at: str,
    apply_unknown_impact: bool = False,
) -> tuple[float, float, list[dict[str, Any]]]:
    home_adjustment = 0.0
    away_adjustment = 0.0
    factors: list[dict[str, Any]] = []
    for _, injury in injuries.iterrows():
        team = normalize_dashboard_team(injury.get("team"))
        if team not in {home_team, away_team}:
            continue
        impact = _importance_impact(injury.get("importance"), apply_unknown_impact=apply_unknown_impact)
        if impact <= 0:
            continue
        if team == home_team:
            home_adjustment -= impact
        else:
            away_adjustment -= impact
        factors.append(
            _factor(
                fixture_id,
                "injury_suspension",
                team,
                f"{injury.get('player', 'Player')} {injury.get('status', 'unavailable')}: {injury.get('reason', '')}",
                "negative",
                -impact,
                injury.get("source"),
                updated_at,
            )
        )
    if not lineups.empty:
        missing = lineups[lineups.get("status", pd.Series(dtype=str)).astype(str).str.lower().isin(["out", "missing", "unavailable", "not_starting"])]
        for _, row in missing.iterrows():
            team = normalize_dashboard_team(row.get("team"))
            if team not in {home_team, away_team}:
                continue
            impact = 0.02
            if team == home_team:
                home_adjustment -= impact
            else:
                away_adjustment -= impact
            factors.append(_factor(fixture_id, "lineup", team, f"{row.get('player', 'Player')} lineup status: {row.get('status')}", "negative", -impact, row.get("source"), updated_at))
    if score_row is not None:
        home_red = _numeric(score_row.get("home_red_cards"))
        away_red = _numeric(score_row.get("away_red_cards"))
        if home_red > away_red:
            home_adjustment -= 0.15
            away_adjustment += 0.08
            factors.append(_factor(fixture_id, "red_card", home_team, "Home team has more red cards", "negative", -0.15, score_row.get("source"), updated_at))
        elif away_red > home_red:
            away_adjustment -= 0.15
            home_adjustment += 0.08
            factors.append(_factor(fixture_id, "red_card", away_team, "Away team has more red cards", "negative", -0.15, score_row.get("source"), updated_at))
        minute = _numeric(score_row.get("minute"))
        home_score = _numeric(score_row.get("home_score"))
        away_score = _numeric(score_row.get("away_score"))
        if minute >= 70 and home_score != away_score:
            leader = home_team if home_score > away_score else away_team
            if leader == home_team:
                home_adjustment += 0.08
            else:
                away_adjustment += 0.08
            factors.append(_factor(fixture_id, "live_score", leader, f"Leading after {int(minute)} minutes", "positive", 0.08, score_row.get("source"), updated_at))
    return home_adjustment, away_adjustment, factors


def probability_sums_valid(df: pd.DataFrame, prefix: str = "live") -> pd.Series:
    cols = [f"{prefix}_home_prob", f"{prefix}_draw_prob", f"{prefix}_away_prob"]
    if not set(cols).issubset(df.columns):
        return pd.Series(dtype=bool)
    sums = df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    return sums.sub(1.0).abs().le(1e-6)


def _prepare_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_frame(LIVE_FIXTURES_SCHEMA)
    output = ensure_schema(df, LIVE_FIXTURES_SCHEMA)
    output["date"] = output["date"].map(_date_key)
    output["home_team"] = output["home_team"].map(normalize_dashboard_team)
    output["away_team"] = output["away_team"].map(normalize_dashboard_team)
    return output.dropna(subset=["date", "home_team", "away_team"]).reset_index(drop=True)


def _prepare_odds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_frame(LIVE_ODDS_SCHEMA)
    output = df.copy()
    for column in LIVE_ODDS_SCHEMA:
        if column not in output.columns:
            output[column] = pd.NA
    output["date"] = output["date"].map(_date_key)
    output["home_team"] = output["home_team"].map(normalize_dashboard_team)
    output["away_team"] = output["away_team"].map(normalize_dashboard_team)
    return output[LIVE_ODDS_SCHEMA].dropna(subset=["home_team", "away_team"]).reset_index(drop=True)


def _prepare_injuries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_frame(LIVE_INJURIES_SCHEMA)
    output = ensure_schema(df, LIVE_INJURIES_SCHEMA)
    output["team"] = output["team"].map(normalize_dashboard_team)
    output["scenario_only"] = True
    return output


def _prepare_lineups(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_frame(LIVE_LINEUPS_SCHEMA)
    output = ensure_schema(df, LIVE_LINEUPS_SCHEMA)
    output["team"] = output["team"].map(normalize_dashboard_team)
    output["scenario_only"] = True
    return output


def _prepare_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_frame(LIVE_SCORES_SCHEMA)
    output = ensure_schema(df, LIVE_SCORES_SCHEMA)
    output["date"] = output["date"].map(_date_key)
    output["home_team"] = output["home_team"].map(normalize_dashboard_team)
    output["away_team"] = output["away_team"].map(normalize_dashboard_team)
    return output


def _fixtures_from_official_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or not {"home_team", "away_team"}.issubset(predictions.columns):
        return empty_frame(LIVE_FIXTURES_SCHEMA)
    frame = predictions.copy()
    if "date" not in frame.columns:
        frame["date"] = pd.NA
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    output = pd.DataFrame(
        {
            "fixture_id": frame.get("match_id", pd.Series(range(len(frame)))),
            "date": frame["date"].map(_date_key),
            "kickoff_time": "",
            "status": frame.get("status", pd.Series("Upcoming", index=frame.index)).fillna("Upcoming"),
            "minute": pd.NA,
            "home_team": frame["home_team"],
            "away_team": frame["away_team"],
            "venue": frame.get("venue", pd.Series("", index=frame.index)),
            "stage": frame.get("stage", frame.get("round", pd.Series("", index=frame.index))),
            "group": frame.get("group", pd.Series("", index=frame.index)),
            "source": "official_prediction_file",
            "updated_at": now,
        }
    )
    return _prepare_fixtures(output)


def _lookup_score(scores: pd.DataFrame, fixture_id: Any, date: str | None, home: str, away: str) -> pd.Series | None:
    if scores.empty:
        return None
    by_fixture = scores[scores["fixture_id"].astype(str).eq(str(fixture_id))] if pd.notna(fixture_id) else pd.DataFrame()
    if not by_fixture.empty:
        return by_fixture.iloc[-1]
    candidates = scores[scores["home_team"].eq(home) & scores["away_team"].eq(away)]
    if date:
        dated = candidates[candidates["date"].eq(date)]
        if not dated.empty:
            candidates = dated
    return candidates.iloc[-1] if not candidates.empty else None


def _importance_impact(value: Any, *, apply_unknown_impact: bool) -> float:
    if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
        return 0.01 if apply_unknown_impact else 0.0
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "null", "<na>"}:
        return 0.01 if apply_unknown_impact else 0.0
    if text in {"high", "key", "major"}:
        return 0.05
    if text in {"medium", "moderate"}:
        return 0.03
    if text in {"low", "minor"}:
        return 0.01
    try:
        numeric = float(text)
    except ValueError:
        return 0.01 if apply_unknown_impact else 0.0
    if not np.isfinite(numeric):
        return 0.01 if apply_unknown_impact else 0.0
    return float(np.clip(numeric, 0.0, 0.08))


def _factor(fixture_id: Any, factor_type: str, team: str, description: str, direction: str, strength: float, source: Any, updated_at: str) -> dict[str, Any]:
    return {
        "fixture_id": fixture_id,
        "factor_type": factor_type,
        "team": team,
        "description": description,
        "impact_direction": direction,
        "impact_strength": strength,
        "source": source,
        "updated_at": updated_at,
    }


def _date_key(value: Any) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce", format="mixed", utc=True)
    if pd.isna(timestamp):
        return None
    return timestamp.strftime("%Y-%m-%d")


def _tuple_probs(values: tuple[Any, Any, Any] | None) -> tuple[float, float, float] | None:
    if values is None:
        return None
    probs = normalize_probabilities(values)
    return float(probs[0]), float(probs[1]), float(probs[2])


def _prediction_label(home: str, away: str, probs: tuple[float, float, float]) -> str:
    index = int(np.argmax(np.asarray(probs, dtype=float)))
    return [f"{home} win", "Draw", f"{away} win"][index]


def _actual_result_label(home: str, away: str, home_score: Any, away_score: Any) -> str | Any:
    home_number = _score_number(home_score)
    away_number = _score_number(away_score)
    if home_number is None or away_number is None:
        return pd.NA
    if home_number > away_number:
        return f"{home} win"
    if away_number > home_number:
        return f"{away} win"
    return "Draw"


def _is_finished_status(status: Any) -> bool:
    text = str(status or "").lower()
    return any(term in text for term in ["finished", "full time", "fulltime", "ft"])


def _is_knockout_stage(stage: Any) -> bool:
    if stage is None or (not isinstance(stage, (list, dict, tuple)) and pd.isna(stage)):
        return False
    text = str(stage).lower()
    return any(term in text for term in ["round", "quarter", "semi", "final", "knockout", "third"])


def _score_number(value: Any) -> float | None:
    try:
        if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _numeric(value: Any) -> float:
    try:
        if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
            return 0.0
        number = float(value)
        if math.isnan(number):
            return 0.0
        return number
    except (TypeError, ValueError):
        return 0.0


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and not (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
            return value
    return pd.NA
