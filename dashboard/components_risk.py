from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dashboard.risk import (
    IDR_WARNING,
    LIVE_MARKET_DISCLAIMER,
    OUTCOME_OPTIONS,
    RESPONSIBLE_USE_WARNING,
    RISK_PROFILES,
    build_portfolio_table,
    calculate_match_outcome_risk,
    calculate_risk_metrics,
    decimal_odds_to_market_probabilities,
    market_odds_for_match,
    monte_carlo_bankroll_simulation,
    model_probabilities_for_match,
    raw_implied_probability,
    select_outcome_odds,
    select_outcome_value,
    write_risk_outputs,
)


def compact_risk_summary(match_row: pd.Series, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    official_predictions = frames.get("actual_team_match_predictions", pd.DataFrame())
    live_odds = frames.get("live_odds", pd.DataFrame())
    external_odds = frames.get("market_odds", pd.DataFrame())
    model_probs = model_probabilities_for_match(match_row, official_predictions)
    odds = market_odds_for_match(match_row, live_odds, external_odds)
    has_model_probs = model_probs is not None
    has_market_odds = odds is not None
    results = [
        calculate_match_outcome_risk(
            match_row,
            outcome,
            official_predictions=official_predictions,
            live_odds=live_odds,
            external_odds=external_odds,
            bankroll_idr=0,
            risk_profile="Very conservative",
            max_stake_cap_idr=0,
            max_stake_percent=0.01,
            max_daily_loss_idr=0,
            max_tournament_loss_idr=0,
            enable_hypothetical_simulation=False,
        )
        for outcome in OUTCOME_OPTIONS
    ]
    valid = [result for result in results if _is_finite(result.get("model_prob")) and _is_finite(result.get("market_prob")) and _is_finite(result.get("decimal_odds"))]
    if not valid:
        if not has_model_probs and not has_market_odds:
            reason = "model probabilities and market odds are missing for this match"
        elif not has_model_probs:
            reason = "model probabilities are missing for this match"
        elif not has_market_odds:
            reason = "market odds are missing for this fixture"
        else:
            reason = "available model or market values are not valid for EV calculation"
        return {
            "status": "No data",
            "message": f"Risk note: {reason}. EV/IDR cannot be calculated. Hypothetical stake: 0 IDR. Not betting advice.",
            "outcome": "",
            "edge": np.nan,
            "ev_per_idr": np.nan,
            "has_model_probs": has_model_probs,
            "has_market_odds": has_market_odds,
        }
    positive = [result for result in valid if float(result.get("edge", 0.0)) > 0 and float(result.get("ev_per_idr", 0.0)) > 0]
    if not positive:
        return {
            "status": "No positive edge",
            "message": "Risk note: no positive edge under current model assumptions. Hypothetical stake: 0 IDR. Not betting advice.",
            "outcome": "",
            "edge": max(float(result.get("edge", 0.0)) for result in valid),
            "ev_per_idr": max(float(result.get("ev_per_idr", 0.0)) for result in valid),
            "has_model_probs": has_model_probs,
            "has_market_odds": has_market_odds,
        }
    best = max(positive, key=lambda result: float(result.get("ev_per_idr", 0.0)))
    return {
        "status": "Positive model edge",
        "message": (
            f"Risk note: positive model edge for {best.get('outcome')} "
            f"({ _format_percent(best.get('edge'), signed=True) } edge, "
            f"{ _format_percent(best.get('ev_per_idr'), signed=True) } EV/IDR). "
            "Hypothetical stake remains 0 IDR here; open Match Detail to opt in. Not betting advice."
        ),
        "outcome": best.get("outcome", ""),
        "edge": best.get("edge"),
        "ev_per_idr": best.get("ev_per_idr"),
        "has_model_probs": has_model_probs,
        "has_market_odds": has_market_odds,
    }


def add_compact_risk_columns(matches: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    output = matches.copy()
    if output.empty:
        for column in ("risk_status", "best_ev_outcome", "best_edge", "best_ev_per_idr", "has_positive_ev"):
            output[column] = pd.Series(dtype=object)
        return output
    summaries = [compact_risk_summary(row, frames) for _, row in output.iterrows()]
    output["risk_status"] = [summary.get("status") for summary in summaries]
    output["best_ev_outcome"] = [summary.get("outcome") for summary in summaries]
    output["best_edge"] = [summary.get("edge") for summary in summaries]
    output["best_ev_per_idr"] = [summary.get("ev_per_idr") for summary in summaries]
    output["risk_has_model_probs"] = [summary.get("has_model_probs") for summary in summaries]
    output["risk_has_market_odds"] = [summary.get("has_market_odds") for summary in summaries]
    output["has_positive_ev"] = output["risk_status"].eq("Positive model edge")
    return output


def render_live_match_risk_note(match_row: pd.Series, frames: dict[str, pd.DataFrame]) -> None:
    import streamlit as st

    summary = compact_risk_summary(match_row, frames)
    st.caption(summary["message"])


def _render_market_event_selector(default_match_row: pd.Series, frames: dict[str, pd.DataFrame], *, key_prefix: str) -> pd.Series:
    import streamlit as st

    live_predictions = frames.get("live_predictions", pd.DataFrame())
    if live_predictions.empty:
        st.info("No live market event list is available. Using the selected match context.")
        return default_match_row
    labels = _match_labels(live_predictions)
    default_index = _default_match_index(live_predictions, default_match_row)
    selected = st.selectbox("Market event", labels, index=default_index, key=f"{key_prefix}_market_event")
    return live_predictions.iloc[labels.index(selected)]


def _render_market_ingestion_summary(match_row: pd.Series, outcome: str, frames: dict[str, pd.DataFrame], *, market_line: str) -> None:
    import streamlit as st

    model_probs = model_probabilities_for_match(match_row, frames.get("actual_team_match_predictions", pd.DataFrame()))
    odds = market_odds_for_match(match_row, frames.get("live_odds", pd.DataFrame()), frames.get("market_odds", pd.DataFrame()))
    market_probs = decimal_odds_to_market_probabilities(*odds) if odds is not None else None
    selected_model = select_outcome_value(model_probs, outcome)
    selected_odds = select_outcome_odds(odds, outcome)
    raw_implied = raw_implied_probability(selected_odds)
    normalized_implied = select_outcome_value(market_probs, outcome)

    st.caption("Live market data is read-only and used only to populate quantitative risk fields.")
    if selected_odds is None:
        st.info("No current decimal odds are available for this event/market line.")
        return
    cols = st.columns(4)
    cols[0].metric("Market line", market_line)
    cols[1].metric("Decimal odds", _format_decimal(selected_odds))
    cols[2].metric("Implied win probability", _format_percent(raw_implied))
    cols[3].metric("Model chance", _format_percent(selected_model))
    st.caption(f"Normalized market probability after overround adjustment: {_format_percent(normalized_implied)}")


def render_user_risk_panel(match_row: pd.Series, frames: dict[str, pd.DataFrame]) -> None:
    import streamlit as st

    st.subheader("Risk & Bankroll Lab")
    st.warning(LIVE_MARKET_DISCLAIMER)
    st.caption(IDR_WARNING)
    col_refresh, col_note = st.columns([1, 3])
    if col_refresh.button("Refresh live market data", key="user_risk_refresh"):
        with st.spinner("Refreshing live market data..."):
            from dashboard.live_api import refresh_live_data

            summary = refresh_live_data()
        st.cache_data.clear()
        col_note.success(f"Live market refresh complete. Rows: {summary.get('LIVE_PREDICTION_ROWS', 0)}")

    selected_match = _render_market_event_selector(match_row, frames, key_prefix="user_risk")

    col1, col2 = st.columns(2)
    market_line = col1.selectbox("Market line", ["Match Winner (1X2)"], key="user_risk_market_line")
    outcome = col1.selectbox("Outcome", OUTCOME_OPTIONS, key="user_risk_outcome")
    risk_profile = col2.selectbox("Risk profile", list(RISK_PROFILES), index=0, key="user_risk_profile")
    _render_market_ingestion_summary(selected_match, outcome, frames, market_line=market_line)

    bankroll = st.number_input("Bankroll in IDR", min_value=0, value=0, step=100_000, key="user_risk_bankroll")
    enable = st.toggle("Enable hypothetical stake simulation", value=False, key="user_risk_enable")
    cap_default = int(max(0, bankroll * 0.01))
    daily_default = int(max(0, bankroll * 0.02))
    tournament_default = int(max(0, bankroll * 0.05))

    cap_col, pct_col = st.columns(2)
    max_cap = cap_col.number_input("Maximum stake cap in IDR", min_value=0, value=cap_default, step=50_000, key="user_risk_cap")
    max_percent = pct_col.number_input("Maximum stake percentage of bankroll", min_value=0.0, max_value=100.0, value=1.0, step=0.25, key="user_risk_pct")
    loss_col1, loss_col2 = st.columns(2)
    daily_loss = loss_col1.number_input("Max daily loss in IDR", min_value=0, value=daily_default, step=50_000, key="user_risk_daily_loss")
    tournament_loss = loss_col2.number_input("Max tournament loss in IDR", min_value=0, value=tournament_default, step=50_000, key="user_risk_tournament_loss")

    if bankroll <= 0:
        st.info("Enter a bankroll greater than 0 IDR to calculate hypothetical stake sizing.")

    result = calculate_match_outcome_risk(
        selected_match,
        outcome,
        official_predictions=frames.get("actual_team_match_predictions", pd.DataFrame()),
        live_odds=frames.get("live_odds", pd.DataFrame()),
        external_odds=frames.get("market_odds", pd.DataFrame()),
        bankroll_idr=bankroll,
        risk_profile=risk_profile,
        max_stake_cap_idr=max_cap,
        max_stake_percent=max_percent / 100.0,
        max_daily_loss_idr=daily_loss,
        max_tournament_loss_idr=tournament_loss,
        enable_hypothetical_simulation=enable,
    )
    _render_user_metric_cards(result)
    if result.get("reason_for_zero_stake"):
        st.info(str(result["reason_for_zero_stake"]))


def render_developer_risk_quant_lab(frames: dict[str, pd.DataFrame]) -> None:
    import streamlit as st

    st.warning(LIVE_MARKET_DISCLAIMER)
    st.caption(IDR_WARNING)
    live_predictions = frames.get("live_predictions", pd.DataFrame())
    if live_predictions.empty:
        st.warning("No live prediction rows are available for risk analysis.")
        return

    col_refresh, col_note = st.columns([1, 3])
    if col_refresh.button("Refresh live market data", key="dev_risk_refresh"):
        with st.spinner("Refreshing live market data..."):
            from dashboard.live_api import refresh_live_data

            summary = refresh_live_data()
        st.cache_data.clear()
        col_note.success(f"Live market refresh complete. Rows: {summary.get('LIVE_PREDICTION_ROWS', 0)}")

    match_row = _render_market_event_selector(live_predictions.iloc[0], frames, key_prefix="dev_risk")

    st.markdown("### Single-Match Risk")
    inputs = _developer_risk_inputs(prefix="dev_single")
    _render_market_ingestion_summary(match_row, inputs["outcome"], frames, market_line=inputs["market_line"])
    result = calculate_match_outcome_risk(
        match_row,
        inputs["outcome"],
        official_predictions=frames.get("actual_team_match_predictions", pd.DataFrame()),
        live_odds=frames.get("live_odds", pd.DataFrame()),
        external_odds=frames.get("market_odds", pd.DataFrame()),
        bankroll_idr=inputs["bankroll"],
        risk_profile=inputs["risk_profile"],
        max_stake_cap_idr=inputs["max_cap"],
        max_stake_percent=inputs["max_percent"] / 100.0,
        max_daily_loss_idr=inputs["daily_loss"],
        max_tournament_loss_idr=inputs["tournament_loss"],
        enable_hypothetical_simulation=inputs["enable"],
    )
    _render_raw_metrics(result)

    st.markdown("### Kelly Criterion Details")
    st.markdown(
        """
Kelly: `f* = (bp - q) / b`

EV: `EV = p(odds - 1) - (1 - p)`

Break-even: `p_break_even = 1 / odds`
"""
    )
    kelly_table = pd.DataFrame(
        [
            {
                "full_kelly_fraction": result.get("full_kelly_fraction"),
                "half_kelly_fraction": result.get("full_kelly_fraction") * 0.50 if pd.notna(result.get("full_kelly_fraction")) else np.nan,
                "quarter_kelly_fraction": result.get("full_kelly_fraction") * 0.25 if pd.notna(result.get("full_kelly_fraction")) else np.nan,
                "selected_fractional_kelly": result.get("selected_kelly_fraction"),
                "fractional_kelly_fraction": result.get("fractional_kelly_fraction"),
                "capped_hypothetical_stake_idr": result.get("capped_hypothetical_stake_idr"),
                "reason_for_zero_stake": result.get("reason_for_zero_stake"),
            }
        ]
    )
    st.dataframe(kelly_table, use_container_width=True)

    st.markdown("### Market Edge Details")
    edge_table = pd.DataFrame(
        [
            {
                "model_prob": result.get("model_prob"),
                "market_implied_prob": result.get("market_prob"),
                "edge": result.get("edge"),
                "decimal_odds": result.get("decimal_odds"),
                "break_even_probability": result.get("break_even_probability"),
                "ev_per_idr": result.get("ev_per_idr"),
                "expected_return_percent": result.get("expected_return_percent"),
            }
        ]
    )
    st.dataframe(edge_table, use_container_width=True)

    st.markdown("### Sensitivity Analysis")
    st.write("Probability sensitivity")
    st.dataframe(_probability_sensitivity(result, inputs), use_container_width=True)
    st.write("Odds sensitivity")
    st.dataframe(_odds_sensitivity(result, inputs), use_container_width=True)

    st.markdown("### Monte Carlo Bankroll Simulation")
    _render_monte_carlo(result, inputs)

    st.markdown("### Portfolio Exposure")
    portfolio_table, portfolio_summary = _render_portfolio(frames, live_predictions, inputs)

    st.markdown("### Raw Risk Output Table")
    raw = pd.DataFrame([result])
    st.dataframe(raw, use_container_width=True)
    if st.button("Save session risk output", key="save_risk_output"):
        output = raw
        if not portfolio_table.empty:
            output = pd.concat([raw, portfolio_table], ignore_index=True, sort=False)
        csv_path, report_path = write_risk_outputs(output)
        st.success(f"Saved risk outputs to {csv_path} and {report_path}.")

    st.markdown("### Methodology Notes")
    st.info(
        "This is an educational risk model. It uses model probabilities, market-implied probabilities, "
        "Kelly-style hypothetical stake sizing, caps, and simulation diagnostics. Correlation between "
        "tournament outcomes is not fully modeled. Portfolio risk may be underestimated."
    )
    st.json(portfolio_summary)


def _developer_risk_inputs(*, prefix: str) -> dict[str, Any]:
    import streamlit as st

    col1, col2, col3 = st.columns(3)
    market_line = col1.selectbox("Market line", ["Match Winner (1X2)"], key=f"{prefix}_market_line")
    outcome = col1.selectbox("Outcome", OUTCOME_OPTIONS, key=f"{prefix}_outcome")
    risk_profile = col2.selectbox("Risk profile", list(RISK_PROFILES), index=0, key=f"{prefix}_profile")
    enable = col3.toggle("Enable hypothetical stake simulation", value=False, key=f"{prefix}_enable")
    bankroll = st.number_input("Bankroll in IDR", min_value=0, value=10_000_000, step=100_000, key=f"{prefix}_bankroll")
    cap_default = int(bankroll * 0.01)
    daily_default = int(bankroll * 0.02)
    tournament_default = int(bankroll * 0.05)
    c1, c2, c3 = st.columns(3)
    max_cap = c1.number_input("Maximum stake cap in IDR", min_value=0, value=cap_default, step=50_000, key=f"{prefix}_cap")
    max_percent = c2.number_input("Maximum stake percentage of bankroll", min_value=0.0, max_value=100.0, value=1.0, step=0.25, key=f"{prefix}_pct")
    daily_loss = c3.number_input("Max daily loss in IDR", min_value=0, value=daily_default, step=50_000, key=f"{prefix}_daily_loss")
    tournament_loss = st.number_input("Max tournament loss in IDR", min_value=0, value=tournament_default, step=50_000, key=f"{prefix}_tournament_loss")
    return {
        "outcome": outcome,
        "market_line": market_line,
        "risk_profile": risk_profile,
        "enable": enable,
        "bankroll": bankroll,
        "max_cap": max_cap,
        "max_percent": max_percent,
        "daily_loss": daily_loss,
        "tournament_loss": tournament_loss,
    }


def _render_user_metric_cards(result: dict[str, Any]) -> None:
    import streamlit as st

    cols = st.columns(4)
    cols[0].metric("Model chance", _format_percent(result.get("model_prob")))
    cols[1].metric("Market implied chance", _format_percent(result.get("market_prob")))
    cols[2].metric("Edge", _format_percent(result.get("edge"), signed=True))
    cols[3].metric("Expected value", _format_percent(result.get("ev_per_idr"), signed=True))
    cols = st.columns(4)
    cols[0].metric("Hypothetical max stake", _format_idr(result.get("capped_hypothetical_stake_idr")))
    cols[1].metric("Worst-case loss", _format_idr(result.get("worst_case_loss_idr")))
    cols[2].metric("Potential profit", _format_idr(result.get("potential_profit_idr")))
    cols[3].metric("Risk label", str(result.get("risk_label") or "No data"))


def _render_raw_metrics(result: dict[str, Any]) -> None:
    import streamlit as st

    cols = st.columns(5)
    cols[0].metric("Model probability", _format_percent(result.get("model_prob")))
    cols[1].metric("Market probability", _format_percent(result.get("market_prob")))
    cols[2].metric("Edge", _format_percent(result.get("edge"), signed=True))
    cols[3].metric("EV per IDR", _format_number(result.get("ev_per_idr")))
    cols[4].metric("Hypothetical stake", _format_idr(result.get("capped_hypothetical_stake_idr")))


def _render_monte_carlo(result: dict[str, Any], inputs: dict[str, Any]) -> None:
    import streamlit as st

    p = _finite_or_default(result.get("model_prob"), 0.5)
    odds = _finite_or_default(result.get("decimal_odds"), 2.0)
    default_stake = int(_finite_or_default(result.get("capped_hypothetical_stake_idr"), 0.0))
    c1, c2, c3 = st.columns(3)
    initial = c1.number_input("Initial bankroll IDR", min_value=0, value=int(inputs["bankroll"]), step=100_000, key="mc_initial")
    stake = c2.number_input("Stake IDR", min_value=0, value=default_stake, step=50_000, key="mc_stake")
    bets = c3.number_input("Number of bets", min_value=1, max_value=500, value=50, step=1, key="mc_bets")
    c4, c5, c6 = st.columns(3)
    simulations = c4.number_input("Number of simulations", min_value=100, max_value=100_000, value=10_000, step=1_000, key="mc_sims")
    stop_loss = c5.number_input("Stop-loss threshold IDR", min_value=0, value=int(initial * 0.5), step=100_000, key="mc_stop")
    seed = c6.number_input("Random seed", min_value=0, value=42, step=1, key="mc_seed")
    confidence = st.selectbox("Confidence Level for VaR", [0.90, 0.95, 0.99], index=1, format_func=lambda value: f"{value:.0%}", key="mc_confidence")
    if st.button("Run bankroll simulation", key="run_mc"):
        simulation = monte_carlo_bankroll_simulation(
            initial_bankroll_idr=initial,
            stake_idr=stake,
            model_prob=p,
            decimal_odds=odds,
            number_of_bets=int(bets),
            number_of_simulations=int(simulations),
            stop_loss_threshold=stop_loss,
            confidence_level=float(confidence),
            seed=int(seed),
        )
        st.dataframe(simulation.summary, use_container_width=True)
        st.bar_chart(simulation.final_bankrolls["final_bankroll"])
        st.bar_chart(simulation.drawdowns["max_drawdown"])
        st.line_chart(simulation.sample_paths.set_index("bet_number"))


def _render_portfolio(frames: dict[str, pd.DataFrame], live_predictions: pd.DataFrame, inputs: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    import streamlit as st

    labels = _match_labels(live_predictions)
    selected = st.multiselect("Select multiple matches/outcomes for portfolio exposure", labels, default=labels[: min(3, len(labels))], key="portfolio_matches")
    c1, c2, c3 = st.columns(3)
    total_cap = c1.number_input("Total exposure cap (% bankroll)", min_value=0.0, max_value=100.0, value=5.0, step=0.5, key="portfolio_total_cap")
    per_match_cap = c2.number_input("Per-match cap (% bankroll)", min_value=0.0, max_value=100.0, value=1.0, step=0.25, key="portfolio_per_match_cap")
    daily_cap = c3.number_input("Daily loss cap (% bankroll)", min_value=0.0, max_value=100.0, value=2.0, step=0.25, key="portfolio_daily_cap")
    if not selected:
        st.info("Select at least one match for portfolio exposure.")
        return pd.DataFrame(), {}
    indexes = [labels.index(label) for label in selected]
    table, summary = build_portfolio_table(
        live_predictions.iloc[indexes],
        official_predictions=frames.get("actual_team_match_predictions", pd.DataFrame()),
        live_odds=frames.get("live_odds", pd.DataFrame()),
        external_odds=frames.get("market_odds", pd.DataFrame()),
        bankroll_idr=inputs["bankroll"],
        risk_profile=inputs["risk_profile"],
        total_exposure_cap_percent=total_cap / 100.0,
        per_match_cap_percent=per_match_cap / 100.0,
        daily_loss_cap_percent=daily_cap / 100.0,
    )
    st.warning("Correlation between tournament outcomes is not fully modeled. Portfolio risk may be underestimated.")
    if not table.empty:
        display_cols = [
            "home_team",
            "away_team",
            "outcome",
            "model_prob",
            "market_prob",
            "edge",
            "ev_per_idr",
            "capped_hypothetical_stake_idr",
            "expected_profit_idr",
            "risk_label",
        ]
        st.dataframe(table[[column for column in display_cols if column in table.columns]], use_container_width=True)
    return table, summary


def _probability_sensitivity(result: dict[str, Any], inputs: dict[str, Any]) -> pd.DataFrame:
    p = _finite_or_default(result.get("model_prob"), np.nan)
    if not np.isfinite(p):
        return pd.DataFrame()
    rows = []
    for delta in np.linspace(-0.10, 0.10, 9):
        candidate_p = float(np.clip(p + delta, 0.001, 0.999))
        rows.append(
            calculate_risk_metrics(
                model_prob=candidate_p,
                market_prob=result.get("market_prob"),
                decimal_odds=result.get("decimal_odds"),
                bankroll_idr=inputs["bankroll"],
                risk_profile=inputs["risk_profile"],
                max_stake_cap_idr=inputs["max_cap"],
                max_stake_percent=inputs["max_percent"] / 100.0,
                max_daily_loss_idr=inputs["daily_loss"],
                max_tournament_loss_idr=inputs["tournament_loss"],
                enable_hypothetical_simulation=inputs["enable"],
                outcome=result.get("outcome", ""),
            )
        )
    return pd.DataFrame(rows)[["model_prob", "edge", "ev_per_idr", "full_kelly_fraction", "capped_hypothetical_stake_idr", "risk_label"]]


def _odds_sensitivity(result: dict[str, Any], inputs: dict[str, Any]) -> pd.DataFrame:
    odds = _finite_or_default(result.get("decimal_odds"), np.nan)
    if not np.isfinite(odds):
        return pd.DataFrame()
    rows = []
    for multiplier in np.linspace(0.90, 1.10, 9):
        candidate_odds = max(1.01, float(odds * multiplier))
        rows.append(
            calculate_risk_metrics(
                model_prob=result.get("model_prob"),
                market_prob=result.get("market_prob"),
                decimal_odds=candidate_odds,
                bankroll_idr=inputs["bankroll"],
                risk_profile=inputs["risk_profile"],
                max_stake_cap_idr=inputs["max_cap"],
                max_stake_percent=inputs["max_percent"] / 100.0,
                max_daily_loss_idr=inputs["daily_loss"],
                max_tournament_loss_idr=inputs["tournament_loss"],
                enable_hypothetical_simulation=inputs["enable"],
                outcome=result.get("outcome", ""),
            )
        )
    return pd.DataFrame(rows)[["decimal_odds", "break_even_probability", "edge", "ev_per_idr", "full_kelly_fraction", "capped_hypothetical_stake_idr"]]


def _match_labels(frame: pd.DataFrame) -> list[str]:
    labels = []
    for row in frame.itertuples(index=False):
        labels.append(f"{getattr(row, 'home_team', 'Home')} vs {getattr(row, 'away_team', 'Away')} ({getattr(row, 'date', '')})")
    return labels


def _default_match_index(frame: pd.DataFrame, match_row: pd.Series) -> int:
    if frame.empty:
        return 0
    fixture_id = match_row.get("fixture_id")
    if pd.notna(fixture_id) and "fixture_id" in frame.columns:
        matched = frame[frame["fixture_id"].astype(str).eq(str(fixture_id))]
        if not matched.empty:
            return int(frame.index.get_loc(matched.index[0]))
    home = str(match_row.get("home_team", ""))
    away = str(match_row.get("away_team", ""))
    date = str(match_row.get("date", ""))
    candidates = frame[
        frame.get("home_team", pd.Series(dtype=str)).astype(str).eq(home)
        & frame.get("away_team", pd.Series(dtype=str)).astype(str).eq(away)
        & frame.get("date", pd.Series(dtype=str)).astype(str).eq(date)
    ]
    if not candidates.empty:
        return int(frame.index.get_loc(candidates.index[0]))
    return 0


def _format_idr(value: Any) -> str:
    try:
        if pd.isna(value):
            return "0 IDR"
        return f"{float(value):,.0f} IDR"
    except (TypeError, ValueError):
        return "0 IDR"


def _format_percent(value: Any, *, signed: bool = False) -> str:
    try:
        if pd.isna(value):
            return "--"
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    prefix = "+" if signed and number > 0 else ""
    return f"{prefix}{number:.1%}"


def _format_number(value: Any) -> str:
    try:
        if pd.isna(value):
            return "--"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "--"


def _format_decimal(value: Any) -> str:
    try:
        if pd.isna(value):
            return "--"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "--"


def _finite_or_default(value: Any, default: float) -> float:
    try:
        if pd.isna(value):
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _is_finite(value: Any) -> bool:
    try:
        if pd.isna(value):
            return False
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
