from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dashboard.scenario import infer_probability_columns, normalize_probabilities
from src.config import PROJECT_ROOT
from src.normalize import normalize_team_name


RESPONSIBLE_USE_WARNING = (
    "Warning: This tool is for quantitative risk education only. It does not place bets and does not provide "
    "financial or gambling advice. Betting may be illegal in some jurisdictions. Check your local laws. "
    "Never risk money you cannot afford to lose."
)
LIVE_MARKET_DISCLAIMER = (
    "DISCLAIMER: This simulator ingests live market data strictly for quantitative risk modeling and educational "
    "purposes. It does not facilitate wagering or constitute financial advice. Online gambling is illegal in Indonesia."
)
IDR_WARNING = "IDR display is for budgeting/risk visualization only."

RISK_PROFILES = {
    "Very conservative": 0.05,
    "Conservative": 0.10,
    "Balanced": 0.25,
    "Aggressive": 0.50,
}

OUTCOME_OPTIONS = ["Home win", "Draw", "Away win"]
OUTCOME_KEYS = {"Home win": "home", "Draw": "draw", "Away win": "away"}
OUTCOME_INDEX = {"home": 0, "draw": 1, "away": 2}

LIVE_RISK_ANALYSIS_PATH = PROJECT_ROOT / "data" / "live" / "live_risk_analysis.csv"
LIVE_RISK_REPORT_PATH = PROJECT_ROOT / "data" / "live" / "reports" / "live_risk_report.md"

RISK_OUTPUT_SCHEMA = [
    "timestamp",
    "fixture_id",
    "home_team",
    "away_team",
    "outcome",
    "model_prob",
    "market_prob",
    "decimal_odds",
    "edge",
    "ev_per_idr",
    "bankroll_idr",
    "risk_profile",
    "full_kelly_fraction",
    "fractional_kelly_fraction",
    "capped_hypothetical_stake_idr",
    "worst_case_loss_idr",
    "potential_profit_idr",
    "expected_profit_idr",
    "risk_label",
    "scenario_mode",
]


@dataclass(frozen=True)
class MonteCarloResult:
    summary: pd.DataFrame
    final_bankrolls: pd.DataFrame
    drawdowns: pd.DataFrame
    sample_paths: pd.DataFrame


def decimal_odds_to_market_probabilities(home_odds: Any, draw_odds: Any, away_odds: Any) -> tuple[float, float, float] | None:
    try:
        odds = np.array([float(home_odds), float(draw_odds), float(away_odds)], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(odds).all() or np.any(odds <= 1.0):
        return None
    probs = normalize_probabilities(1.0 / odds)
    return float(probs[0]), float(probs[1]), float(probs[2])


def raw_implied_probability(decimal_odds: Any) -> float | None:
    odds = _positive_odds_or_none(decimal_odds)
    if odds is None:
        return None
    return 1.0 / odds


def kelly_fraction(model_prob: Any, decimal_odds: Any) -> float:
    p = _probability_or_none(model_prob)
    odds = _positive_odds_or_none(decimal_odds)
    if p is None or odds is None:
        return 0.0
    b = odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    fraction = (b * p - q) / b
    return float(max(0.0, fraction))


def calculate_risk_metrics(
    *,
    model_prob: Any,
    market_prob: Any,
    decimal_odds: Any,
    bankroll_idr: Any,
    risk_profile: str = "Very conservative",
    max_stake_cap_idr: Any = 0,
    max_stake_percent: Any = 0.01,
    max_daily_loss_idr: Any = 0,
    max_tournament_loss_idr: Any = 0,
    enable_hypothetical_simulation: bool = False,
    fixture_id: Any = pd.NA,
    home_team: Any = pd.NA,
    away_team: Any = pd.NA,
    outcome: str = "Home win",
    scenario_mode: bool = True,
) -> dict[str, Any]:
    p = _probability_or_none(model_prob)
    market = _probability_or_none(market_prob)
    odds = _positive_odds_or_none(decimal_odds)
    bankroll = max(0.0, _float_or_zero(bankroll_idr))
    max_cap = max(0.0, _float_or_zero(max_stake_cap_idr))
    max_percent = max(0.0, _float_or_zero(max_stake_percent))
    daily_loss_cap = max(0.0, _float_or_zero(max_daily_loss_idr))
    tournament_loss_cap = max(0.0, _float_or_zero(max_tournament_loss_idr))
    selected_kelly = float(RISK_PROFILES.get(risk_profile, RISK_PROFILES["Very conservative"]))

    reason = ""
    ev_per_idr = np.nan
    edge = np.nan
    break_even_probability = np.nan
    full_kelly = 0.0
    fractional_kelly = 0.0
    stake = 0.0

    if p is None or market is None or odds is None:
        reason = "Insufficient data."
        risk_label = "No data"
    else:
        edge = p - market
        ev_per_idr = p * (odds - 1.0) - (1.0 - p)
        break_even_probability = 1.0 / odds
        full_kelly = kelly_fraction(p, odds)
        fractional_kelly = full_kelly * selected_kelly
        if not enable_hypothetical_simulation:
            reason = "Hypothetical stake simulation disabled."
        elif bankroll <= 0:
            reason = "Bankroll must be greater than 0 IDR."
        elif ev_per_idr <= 0 or edge <= 0 or full_kelly <= 0:
            reason = "No positive expected value under current model assumptions."
        else:
            caps = [bankroll, bankroll * max_percent]
            if max_cap > 0:
                caps.append(max_cap)
            if daily_loss_cap > 0:
                caps.append(daily_loss_cap)
            if tournament_loss_cap > 0:
                caps.append(tournament_loss_cap)
            stake = min(bankroll * fractional_kelly, *caps)
            stake = max(0.0, min(stake, bankroll))
            stake = round_down_idr(stake)
            if stake < 1000:
                stake = 0.0
                reason = "Calculated stake is below the 1,000 IDR display threshold."
        risk_label = risk_label_for_stake(stake, bankroll, ev_per_idr, edge)

    potential_profit = stake * (odds - 1.0) if odds is not None else 0.0
    expected_profit = stake * ev_per_idr if np.isfinite(ev_per_idr) else 0.0
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "timestamp": timestamp,
        "fixture_id": fixture_id,
        "home_team": home_team,
        "away_team": away_team,
        "outcome": outcome,
        "model_prob": p,
        "market_prob": market,
        "decimal_odds": odds,
        "edge": edge,
        "ev_per_idr": ev_per_idr,
        "expected_return_percent": ev_per_idr * 100 if np.isfinite(ev_per_idr) else np.nan,
        "bankroll_idr": bankroll,
        "risk_profile": risk_profile,
        "selected_kelly_fraction": selected_kelly,
        "full_kelly_fraction": full_kelly,
        "fractional_kelly_fraction": fractional_kelly,
        "capped_hypothetical_stake_idr": stake,
        "worst_case_loss_idr": stake,
        "potential_profit_idr": potential_profit,
        "expected_profit_idr": expected_profit,
        "break_even_probability": break_even_probability,
        "risk_label": risk_label,
        "reason_for_zero_stake": reason if stake <= 0 else "",
        "scenario_mode": bool(scenario_mode),
    }


def risk_label_for_stake(stake_idr: float, bankroll_idr: float, ev_per_idr: float, edge: float) -> str:
    if not np.isfinite(ev_per_idr) or not np.isfinite(edge):
        return "No data"
    if ev_per_idr <= 0 or edge <= 0:
        return "No positive edge"
    if stake_idr <= 0 or bankroll_idr <= 0:
        return "Lower risk"
    fraction = stake_idr / bankroll_idr
    if fraction > 0.02:
        return "Very high risk"
    if fraction > 0.01:
        return "High risk"
    if fraction > 0.005:
        return "Medium risk"
    return "Lower risk"


def round_down_idr(value: Any, increment: int = 1000) -> int:
    amount = max(0.0, _float_or_zero(value))
    if increment <= 0:
        return int(amount)
    return int(amount // increment * increment)


def select_outcome_value(values: tuple[Any, Any, Any] | None, outcome: str) -> float | None:
    if values is None:
        return None
    key = OUTCOME_KEYS.get(outcome, outcome.lower())
    index = OUTCOME_INDEX.get(key)
    if index is None:
        return None
    return _probability_or_none(values[index])


def select_outcome_odds(values: tuple[Any, Any, Any] | None, outcome: str) -> float | None:
    if values is None:
        return None
    key = OUTCOME_KEYS.get(outcome, outcome.lower())
    index = OUTCOME_INDEX.get(key)
    if index is None:
        return None
    return _positive_odds_or_none(values[index])


def model_probabilities_for_match(match_row: pd.Series, official_predictions: pd.DataFrame | None = None) -> tuple[float, float, float] | None:
    live_cols = ("live_home_prob", "live_draw_prob", "live_away_prob")
    base_cols = ("base_home_prob", "base_draw_prob", "base_away_prob")
    for cols in (live_cols, base_cols):
        if all(col in match_row.index for col in cols):
            values = _tuple_probabilities([match_row.get(col) for col in cols])
            if values is not None:
                return values
    if official_predictions is None or official_predictions.empty:
        return None
    home = normalize_team_name(match_row.get("home_team"))
    away = normalize_team_name(match_row.get("away_team"))
    if home is None or away is None:
        return None
    return _probabilities_from_frame(official_predictions, home, away, _date_key(match_row.get("date")))


def market_odds_for_match(match_row: pd.Series, live_odds: pd.DataFrame | None = None, external_odds: pd.DataFrame | None = None) -> tuple[float, float, float] | None:
    for odds in (live_odds, external_odds):
        found = _odds_from_frame(odds if odds is not None else pd.DataFrame(), match_row)
        if found is not None:
            return found
    return None


def calculate_match_outcome_risk(
    match_row: pd.Series,
    outcome: str,
    *,
    official_predictions: pd.DataFrame | None = None,
    live_odds: pd.DataFrame | None = None,
    external_odds: pd.DataFrame | None = None,
    bankroll_idr: Any = 0,
    risk_profile: str = "Very conservative",
    max_stake_cap_idr: Any = 0,
    max_stake_percent: Any = 0.01,
    max_daily_loss_idr: Any = 0,
    max_tournament_loss_idr: Any = 0,
    enable_hypothetical_simulation: bool = False,
) -> dict[str, Any]:
    model_probs = model_probabilities_for_match(match_row, official_predictions)
    odds = market_odds_for_match(match_row, live_odds, external_odds)
    market_probs = decimal_odds_to_market_probabilities(*odds) if odds is not None else None
    return calculate_risk_metrics(
        model_prob=select_outcome_value(model_probs, outcome),
        market_prob=select_outcome_value(market_probs, outcome),
        decimal_odds=select_outcome_odds(odds, outcome),
        bankroll_idr=bankroll_idr,
        risk_profile=risk_profile,
        max_stake_cap_idr=max_stake_cap_idr,
        max_stake_percent=max_stake_percent,
        max_daily_loss_idr=max_daily_loss_idr,
        max_tournament_loss_idr=max_tournament_loss_idr,
        enable_hypothetical_simulation=enable_hypothetical_simulation,
        fixture_id=match_row.get("fixture_id", pd.NA),
        home_team=match_row.get("home_team", pd.NA),
        away_team=match_row.get("away_team", pd.NA),
        outcome=outcome,
        scenario_mode=True,
    )


def monte_carlo_bankroll_simulation(
    *,
    initial_bankroll_idr: Any,
    stake_idr: Any,
    model_prob: Any,
    decimal_odds: Any,
    number_of_bets: int = 50,
    number_of_simulations: int = 10_000,
    stop_loss_threshold: Any | None = None,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> MonteCarloResult:
    initial = max(0.0, _float_or_zero(initial_bankroll_idr))
    stake = max(0.0, min(_float_or_zero(stake_idr), initial))
    p = _probability_or_none(model_prob) or 0.0
    odds = _positive_odds_or_none(decimal_odds) or 1.0
    n_bets = max(1, int(number_of_bets))
    n_sims = max(1, int(number_of_simulations))
    stop_loss = initial * 0.5 if stop_loss_threshold is None else max(0.0, _float_or_zero(stop_loss_threshold))
    confidence = float(np.clip(confidence_level, 0.5, 0.999))
    rng = np.random.default_rng(seed)

    bankrolls = np.full(n_sims, initial, dtype=float)
    peaks = bankrolls.copy()
    max_drawdowns = np.zeros(n_sims, dtype=float)
    sample_count = min(25, n_sims)
    sample_paths = np.zeros((n_bets + 1, sample_count), dtype=float)
    sample_paths[0, :] = initial

    for bet_index in range(1, n_bets + 1):
        active = (bankrolls > stop_loss) & (bankrolls > 0) & (stake > 0)
        wins = rng.random(n_sims) < p
        bankrolls = np.where(active & wins, bankrolls + stake * (odds - 1.0), bankrolls)
        bankrolls = np.where(active & ~wins, bankrolls - stake, bankrolls)
        bankrolls = np.maximum(bankrolls, 0.0)
        peaks = np.maximum(peaks, bankrolls)
        drawdowns = np.where(peaks > 0, (peaks - bankrolls) / peaks, 0.0)
        max_drawdowns = np.maximum(max_drawdowns, drawdowns)
        sample_paths[bet_index, :] = bankrolls[:sample_count]

    losses = initial - bankrolls
    percentile = confidence * 100.0
    var_loss = float(np.percentile(losses, percentile))
    tail_losses = losses[losses >= var_loss]
    cvar_loss = float(tail_losses.mean()) if len(tail_losses) else var_loss
    summary = pd.DataFrame(
        [
            {
                "median_final_bankroll": float(np.median(bankrolls)),
                "mean_final_bankroll": float(np.mean(bankrolls)),
                "p5_final_bankroll": float(np.percentile(bankrolls, 5)),
                "p1_final_bankroll": float(np.percentile(bankrolls, 1)),
                "probability_losing_money": float(np.mean(bankrolls < initial)),
                "probability_losing_more_than_10pct": float(np.mean(bankrolls < initial * 0.90)),
                "probability_losing_more_than_25pct": float(np.mean(bankrolls < initial * 0.75)),
                "probability_losing_more_than_50pct": float(np.mean(bankrolls < initial * 0.50)),
                "var_confidence_level": confidence,
                "var_loss": var_loss,
                "cvar_loss": cvar_loss,
                "var_95": float(np.percentile(losses, 95)),
                "cvar_95": float(losses[losses >= np.percentile(losses, 95)].mean()) if len(losses) else 0.0,
                "risk_of_ruin": float(np.mean(bankrolls <= 0)),
            }
        ]
    )
    final_bankrolls = pd.DataFrame({"simulation": np.arange(n_sims), "final_bankroll": bankrolls})
    drawdowns_frame = pd.DataFrame({"simulation": np.arange(n_sims), "max_drawdown": max_drawdowns})
    paths = pd.DataFrame(sample_paths, columns=[f"sim_{index + 1}" for index in range(sample_count)])
    paths.insert(0, "bet_number", np.arange(n_bets + 1))
    return MonteCarloResult(summary=summary, final_bankrolls=final_bankrolls, drawdowns=drawdowns_frame, sample_paths=paths)


def build_portfolio_table(
    matches: pd.DataFrame,
    *,
    outcomes: list[str] | None = None,
    official_predictions: pd.DataFrame | None = None,
    live_odds: pd.DataFrame | None = None,
    external_odds: pd.DataFrame | None = None,
    bankroll_idr: Any = 0,
    risk_profile: str = "Very conservative",
    total_exposure_cap_percent: float = 0.05,
    per_match_cap_percent: float = 0.01,
    daily_loss_cap_percent: float = 0.02,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outcomes = outcomes or OUTCOME_OPTIONS
    bankroll = max(0.0, _float_or_zero(bankroll_idr))
    rows = []
    for _, match in matches.iterrows():
        for outcome in outcomes:
            rows.append(
                calculate_match_outcome_risk(
                    match,
                    outcome,
                    official_predictions=official_predictions,
                    live_odds=live_odds,
                    external_odds=external_odds,
                    bankroll_idr=bankroll,
                    risk_profile=risk_profile,
                    max_stake_cap_idr=bankroll * per_match_cap_percent,
                    max_stake_percent=per_match_cap_percent,
                    max_daily_loss_idr=bankroll * daily_loss_cap_percent,
                    max_tournament_loss_idr=bankroll * total_exposure_cap_percent,
                    enable_hypothetical_simulation=True,
                )
            )
    table = pd.DataFrame(rows)
    if table.empty:
        return table, _portfolio_summary(table, bankroll)
    total_cap = bankroll * max(0.0, total_exposure_cap_percent)
    exposure = pd.to_numeric(table["capped_hypothetical_stake_idr"], errors="coerce").fillna(0.0)
    positive_total = float(exposure.sum())
    scale = 1.0
    if positive_total > total_cap > 0:
        scale = total_cap / positive_total
        table["capped_hypothetical_stake_idr"] = table["capped_hypothetical_stake_idr"].map(lambda value: round_down_idr(float(value) * scale))
        table["worst_case_loss_idr"] = table["capped_hypothetical_stake_idr"]
        table["potential_profit_idr"] = table["capped_hypothetical_stake_idr"] * (table["decimal_odds"].fillna(1.0) - 1.0)
        table["expected_profit_idr"] = table["capped_hypothetical_stake_idr"] * table["ev_per_idr"].fillna(0.0)
    summary = _portfolio_summary(table, bankroll)
    summary["scaling_factor"] = scale
    summary["correlation_warning"] = "Correlation between tournament outcomes is not fully modeled. Portfolio risk may be underestimated."
    return table, summary


def write_risk_outputs(rows: list[dict[str, Any]] | pd.DataFrame) -> tuple[Path, Path]:
    frame = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    for column in RISK_OUTPUT_SCHEMA:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[RISK_OUTPUT_SCHEMA]
    LIVE_RISK_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_RISK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(LIVE_RISK_ANALYSIS_PATH, index=False)
    positive = int((pd.to_numeric(frame["ev_per_idr"], errors="coerce") > 0).sum()) if not frame.empty else 0
    exposure = float(pd.to_numeric(frame["capped_hypothetical_stake_idr"], errors="coerce").fillna(0.0).sum()) if not frame.empty else 0.0
    max_loss = float(pd.to_numeric(frame["worst_case_loss_idr"], errors="coerce").fillna(0.0).sum()) if not frame.empty else 0.0
    lines = [
        "# Live Risk Report",
        "",
        LIVE_MARKET_DISCLAIMER,
        "",
        IDR_WARNING,
        "",
        "## Summary",
        "",
        f"- evaluated_outcomes: `{len(frame)}`",
        f"- positive_ev_count: `{positive}`",
        f"- total_hypothetical_exposure_idr: `{exposure:.0f}`",
        f"- max_worst_case_loss_idr: `{max_loss:.0f}`",
        "",
        "## Warnings",
        "",
        "- This is an educational risk model, not betting advice.",
        "- No bet slips, deposits, withdrawals, sportsbook links, or betting actions are provided.",
        "- Correlation between tournament outcomes is not fully modeled.",
    ]
    LIVE_RISK_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return LIVE_RISK_ANALYSIS_PATH, LIVE_RISK_REPORT_PATH


def _portfolio_summary(table: pd.DataFrame, bankroll: float) -> dict[str, Any]:
    if table.empty:
        return {
            "total_exposure_idr": 0.0,
            "total_worst_case_loss_idr": 0.0,
            "expected_portfolio_profit_idr": 0.0,
            "positive_ev_candidates": 0,
            "bankroll_idr": bankroll,
        }
    exposure = pd.to_numeric(table["capped_hypothetical_stake_idr"], errors="coerce").fillna(0.0)
    ev = pd.to_numeric(table["ev_per_idr"], errors="coerce").fillna(0.0)
    return {
        "total_exposure_idr": float(exposure.sum()),
        "total_worst_case_loss_idr": float(exposure.sum()),
        "expected_portfolio_profit_idr": float((exposure * ev).sum()),
        "positive_ev_candidates": int((ev > 0).sum()),
        "bankroll_idr": bankroll,
    }


def _probabilities_from_frame(frame: pd.DataFrame, home: str, away: str, date: str | None) -> tuple[float, float, float] | None:
    if frame.empty or not {"home_team", "away_team"}.issubset(frame.columns):
        return None
    prob_cols = infer_probability_columns(frame)
    if prob_cols is None:
        return None
    output = frame.copy()
    output["_home_norm"] = output["home_team"].map(normalize_team_name)
    output["_away_norm"] = output["away_team"].map(normalize_team_name)
    output["_date_key"] = output["date"].map(_date_key) if "date" in output.columns else pd.NA
    direct = output[output["_home_norm"].eq(home) & output["_away_norm"].eq(away)]
    if date:
        dated = direct[direct["_date_key"].eq(date)]
        if not dated.empty:
            direct = dated
    if not direct.empty:
        row = direct.iloc[0]
        return _tuple_probabilities([row[prob_cols.home], row[prob_cols.draw], row[prob_cols.away]])
    reverse = output[output["_home_norm"].eq(away) & output["_away_norm"].eq(home)]
    if date:
        dated = reverse[reverse["_date_key"].eq(date)]
        if not dated.empty:
            reverse = dated
    if not reverse.empty:
        row = reverse.iloc[0]
        return _tuple_probabilities([row[prob_cols.away], row[prob_cols.draw], row[prob_cols.home]])
    return None


def _odds_from_frame(frame: pd.DataFrame, match_row: pd.Series) -> tuple[float, float, float] | None:
    if frame.empty or not {"home_odds", "draw_odds", "away_odds"}.issubset(frame.columns):
        return None
    output = frame.copy()
    if "fixture_id" in output.columns and "fixture_id" in match_row.index and pd.notna(match_row.get("fixture_id")):
        by_fixture = output[output["fixture_id"].astype(str).eq(str(match_row.get("fixture_id")))]
        if not by_fixture.empty:
            return _odds_tuple(by_fixture.iloc[-1], reverse=False)
    if not {"home_team", "away_team"}.issubset(output.columns):
        return None
    home = normalize_team_name(match_row.get("home_team"))
    away = normalize_team_name(match_row.get("away_team"))
    date = _date_key(match_row.get("date"))
    if home is None or away is None:
        return None
    output["_home_norm"] = output["home_team"].map(normalize_team_name)
    output["_away_norm"] = output["away_team"].map(normalize_team_name)
    output["_date_key"] = output["date"].map(_date_key) if "date" in output.columns else pd.NA
    if "updated_at" in output.columns:
        output["_updated"] = pd.to_datetime(output["updated_at"], errors="coerce", format="mixed", utc=True)
    else:
        output["_updated"] = pd.NaT
    direct = output[output["_home_norm"].eq(home) & output["_away_norm"].eq(away)]
    if date:
        dated = direct[direct["_date_key"].eq(date)]
        if not dated.empty:
            direct = dated
    if not direct.empty:
        return _odds_tuple(direct.sort_values("_updated", kind="stable").iloc[-1], reverse=False)
    reverse = output[output["_home_norm"].eq(away) & output["_away_norm"].eq(home)]
    if date:
        dated = reverse[reverse["_date_key"].eq(date)]
        if not dated.empty:
            reverse = dated
    if not reverse.empty:
        return _odds_tuple(reverse.sort_values("_updated", kind="stable").iloc[-1], reverse=True)
    return None


def _odds_tuple(row: pd.Series, *, reverse: bool) -> tuple[float, float, float] | None:
    odds = (
        _positive_odds_or_none(row.get("home_odds")),
        _positive_odds_or_none(row.get("draw_odds")),
        _positive_odds_or_none(row.get("away_odds")),
    )
    if any(value is None for value in odds):
        return None
    return (odds[2], odds[1], odds[0]) if reverse else odds


def _tuple_probabilities(values: list[Any]) -> tuple[float, float, float] | None:
    probs = [_probability_or_none(value) for value in values]
    if any(value is None for value in probs):
        return None
    normalized = normalize_probabilities(np.asarray(probs, dtype=float))
    return float(normalized[0]), float(normalized[1]), float(normalized[2])


def _probability_or_none(value: Any) -> float | None:
    try:
        if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number < 0 or number > 1:
        return None
    return number


def _positive_odds_or_none(value: Any) -> float | None:
    try:
        if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 1:
        return None
    return number


def _float_or_zero(value: Any) -> float:
    try:
        if value is None or (not isinstance(value, (list, dict, tuple)) and pd.isna(value)):
            return 0.0
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _date_key(value: Any) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce", format="mixed", utc=True)
    if pd.isna(timestamp):
        return None
    return timestamp.strftime("%Y-%m-%d")
