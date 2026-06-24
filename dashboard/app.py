from __future__ import annotations

import re
import sys
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.bracket_view import add_matchup_label, display_stage, format_percent_columns, stage_probability_columns, validate_stage_sums
from dashboard.components import render_match_card, render_probability_comparison
from dashboard.components_risk import add_compact_risk_columns, render_developer_risk_quant_lab, render_live_match_risk_note, render_user_risk_panel
from dashboard.data_loader import file_presence, odds_rows_by_year, read_csv_safe, read_markdown_safe, validate_market_odds
from dashboard.live_api import fetch_api_football_status, generate_live_predictions, refresh_live_data
from dashboard.parsers import extract_regex_bool, extract_regex_float, extract_regex_value, markdown_table_after_heading
from dashboard.prediction_logic import build_live_predictions, probability_sums_valid
from dashboard.scenario import (
    advancement_probabilities,
    decimal_odds_to_implied,
    find_fixture_prediction,
    infer_probability_columns,
    team_options,
    apply_scenario_adjustments,
)
from dashboard.styles import inject_styles
from src.normalize import normalize_team_name


DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
PREDICTIONS_DIR = DATA_DIR / "predictions"
SIMULATION_DIR = DATA_DIR / "simulation"
BACKTESTS_DIR = DATA_DIR / "backtests"
EXTERNAL_DIR = DATA_DIR / "external"
STAGING_DIR = DATA_DIR / "staging"
LIVE_DIR = DATA_DIR / "live"
LIVE_REPORTS_DIR = LIVE_DIR / "reports"

SAFE_MODEL = "football_only_ensemble"
SAFE_FEATURE_SET = "core_football_only"
MARKET_BLEND_STATUS = "benchmark-only"
SOTA_STATUS = "SOTA-inspired, not true SOTA"
MARKET_PRODUCTION_THRESHOLD = 0.90

REPORT_FILES = {
    "final_project_summary": REPORTS_DIR / "final_project_summary.md",
    "final_model_recommendation": REPORTS_DIR / "final_model_recommendation.md",
    "market_odds_join_debug": REPORTS_DIR / "market_odds_join_debug.md",
    "feature_null_rate_report": REPORTS_DIR / "feature_null_rate_report.md",
    "market_blend_report": REPORTS_DIR / "market_blend_report.md",
    "benchmark_significance_report": REPORTS_DIR / "benchmark_significance_report.md",
    "worldcup_backtest_report": REPORTS_DIR / "worldcup_backtest_report.md",
    "worldcup_tournament_simulation_backtest_report": REPORTS_DIR / "worldcup_tournament_simulation_backtest_report.md",
    "odds_conversion_benchmark_report": REPORTS_DIR / "odds_conversion_benchmark_report.md",
    "odds_provider_competition_discovery": REPORTS_DIR / "odds_provider_competition_discovery.md",
    "refresh_market_odds_report": REPORTS_DIR / "refresh_market_odds_report.md",
    "missing_market_odds_template_report": REPORTS_DIR / "missing_market_odds_template_report.md",
    "manual_market_odds_validation_errors": REPORTS_DIR / "manual_market_odds_validation_errors.md",
    "market_odds_api_merge_report": REPORTS_DIR / "market_odds_api_merge_report.md",
    "live_api_status_report": LIVE_REPORTS_DIR / "live_api_status_report.md",
    "api_football_quota_report": LIVE_REPORTS_DIR / "api_football_quota_report.md",
    "live_prediction_report": LIVE_REPORTS_DIR / "live_prediction_report.md",
    "live_risk_report": LIVE_REPORTS_DIR / "live_risk_report.md",
}

CSV_FILES = {
    "actual_team_match_predictions": PREDICTIONS_DIR / "actual_team_match_predictions.csv",
    "worldcup_2026_simulation_results": SIMULATION_DIR / "worldcup_2026_simulation_results.csv",
    "worldcup_2026_matchup_probabilities": SIMULATION_DIR / "worldcup_2026_matchup_probabilities.csv",
    "worldcup_2026_bracket_path_samples": SIMULATION_DIR / "worldcup_2026_bracket_path_samples.csv",
    "worldcup_backtest_metrics": BACKTESTS_DIR / "worldcup_backtest_metrics.csv",
    "worldcup_tournament_simulation_backtest": BACKTESTS_DIR / "worldcup_tournament_simulation_backtest.csv",
    "market_odds": EXTERNAL_DIR / "market_odds.csv",
    "api_football_injuries_live": STAGING_DIR / "api_football_injuries_live.csv",
    "sportmonks_injuries_live": STAGING_DIR / "sportmonks_injuries_live.csv",
    "live_fixtures": LIVE_DIR / "live_fixtures.csv",
    "live_odds": LIVE_DIR / "live_odds.csv",
    "live_injuries": LIVE_DIR / "live_injuries.csv",
    "live_lineups": LIVE_DIR / "live_lineups.csv",
    "live_scores": LIVE_DIR / "live_scores.csv",
    "live_predictions": LIVE_DIR / "live_predictions.csv",
    "live_prediction_factors": LIVE_DIR / "live_prediction_factors.csv",
    "live_risk_analysis": LIVE_DIR / "live_risk_analysis.csv",
    "missing_market_odds_template": STAGING_DIR / "missing_market_odds_template_2026.csv",
}

OPTIONAL_FILES = {
    "worldcup_2026_bracket_sample": SIMULATION_DIR / "worldcup_2026_bracket_sample.json",
    "worldcup_2026_simulation_metadata": SIMULATION_DIR / "worldcup_2026_simulation_metadata.json",
}


@st.cache_data(show_spinner=False)
def cached_csv(path: str) -> pd.DataFrame:
    return read_csv_safe(Path(path))


@st.cache_data(show_spinner=False)
def cached_text(path: str) -> str:
    return read_markdown_safe(Path(path))



def _metric_or_unknown(value: object) -> str:
    if value is None or str(value).strip() == "":
        return "unknown"
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    st.set_page_config(page_title="World Cup 2026 Prediction Research Dashboard", layout="wide")
    inject_styles()
    reports = {name: cached_text(str(path)) for name, path in REPORT_FILES.items()}
    frames = {name: cached_csv(str(path)) for name, path in CSV_FILES.items()}
    values = parsed_overview_values(reports)

    mode = render_sidebar(values)
    st.title("World Cup 2026 Prediction Research Dashboard")
    if mode == "Developer Mode" and developer_mode_allowed():
        render_developer_mode(reports, frames, values)
    elif mode == "Developer Mode":
        st.warning("Developer Mode password was not accepted. Showing User Mode only.")
        render_user_mode(reports, frames, values)
    else:
        render_user_mode(reports, frames, values)


def render_sidebar(values: dict[str, object]) -> str:
    mode = st.sidebar.radio("Mode", ["User Mode", "Developer Mode"], index=0)
    st.sidebar.header("Official Output Policy")
    st.sidebar.write(f"Safe production model: `{SAFE_MODEL}`")
    st.sidebar.write(f"Safe production feature set: `{SAFE_FEATURE_SET}`")
    st.sidebar.write(f"Market blend status: `{MARKET_BLEND_STATUS}`")
    st.sidebar.write(f"SOTA status: `{SOTA_STATUS}`")
    st.sidebar.write("2026 market production rule: production-ready only if active fixture odds coverage >= 90%")
    st.sidebar.write(f"Last refresh: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
    if st.sidebar.button("Refresh live data"):
        with st.spinner("Refreshing live data..."):
            summary = refresh_live_data()
        st.sidebar.success(f"Live refresh complete. Rows: {summary.get('LIVE_PREDICTION_ROWS', 0)}")
        st.cache_data.clear()
    active_coverage = values.get("active_fixture_coverage_2026")
    full_coverage = values.get("coverage_2026")
    missing_active = values.get("missing_active_fixtures")
    if isinstance(active_coverage, float):
        st.sidebar.metric("Active fixture odds coverage", f"{active_coverage:.1%}")
        if missing_active is not None:
            st.sidebar.caption(f"Missing active fixtures: {missing_active}")
    elif isinstance(full_coverage, float):
        st.sidebar.metric("2026 odds coverage", f"{full_coverage:.1%}")
    if isinstance(full_coverage, float):
        st.sidebar.caption(f"Full 2026 prediction-file coverage: {full_coverage:.1%}")

    st.sidebar.divider()
    st.sidebar.subheader("API-Football quota")
    quota = st.session_state.get("api_football_quota")
    if st.sidebar.button("Check API-Football quota"):
        with st.spinner("Checking API-Football quota..."):
            quota = fetch_api_football_status()
        st.session_state["api_football_quota"] = quota
    if isinstance(quota, dict):
        used = quota.get("used_today")
        limit = quota.get("daily_limit")
        remaining = quota.get("remaining_today")
        if used is not None and limit is not None:
            st.sidebar.metric("Requests used today", f"{used}/{limit}")
        if remaining is not None:
            st.sidebar.metric("Requests remaining", str(remaining))
        if quota.get("errors"):
            st.sidebar.warning("Quota check failed. See Developer Mode → Live API Status.")
    else:
        st.sidebar.caption("Click to check quota. This calls /status only.")
    return mode

def developer_mode_allowed() -> bool:
    password = os.getenv("DASHBOARD_DEV_PASSWORD", "").strip()
    if not password:
        st.warning("Developer Mode is unprotected. Set DASHBOARD_DEV_PASSWORD for access control.")
        return True
    supplied = st.sidebar.text_input("Developer password", type="password")
    return bool(supplied and supplied == password)


def parsed_overview_values(reports: dict[str, str]) -> dict[str, object]:
    recommendation = reports.get("final_model_recommendation", "")
    market_debug = reports.get("market_odds_join_debug", "")
    refresh_report = reports.get("refresh_market_odds_report", "")
    blend = reports.get("market_blend_report", "")
    significance = reports.get("benchmark_significance_report", "")
    coverage = _extract_coverage(market_debug) or extract_regex_float(
        recommendation,
        [r"World Cup 2026 market feature coverage\s*[:|]\s*([0-9.]+)"],
    )
    active_coverage = extract_regex_float(
        refresh_report,
        [r"ACTIVE_FIXTURE_COVERAGE_AFTER\s*=\s*([0-9.]+)"],
    )
    missing_active = extract_regex_value(
        refresh_report,
        [r"MISSING_ACTIVE_FIXTURES\s*=\s*([0-9]+)"],
        default="unknown",
    )
    try:
        missing_active_value: int | str = int(missing_active)
    except (TypeError, ValueError):
        missing_active_value = missing_active
    return {
        "coverage_2026": coverage,
        "active_fixture_coverage_2026": active_coverage,
        "missing_active_fixtures": missing_active_value,
        "best_odds_conversion": extract_regex_value(
            blend + "\n" + recommendation,
            [r"Best odds conversion method\s*[:|]\s*`?([A-Za-z_\-]+)`?", r"Best odds conversion\s*[:|]\s*`?([A-Za-z_\-]+)`?"],
        ),
        "best_alpha": extract_regex_value(
            blend + "\n" + recommendation,
            [r"Best(?: market blend)? alpha\s*[:|]\s*`?([0-9.]+)`?", r"best alpha\s*[:|]\s*`?([0-9.]+)`?"],
        ),
        "statistically_meaningful": extract_regex_bool(
            significance + "\n" + recommendation,
            [r"statistically meaningful\s*[:|]\s*`?(True|False)`?", r"Meaningful\s*[:|]\s*`?(True|False)`?"],
        ),
    }

def render_user_mode(reports: dict[str, str], frames: dict[str, pd.DataFrame], values: dict[str, object]) -> None:
    st.markdown(
        """
<div class="wc-hero">
  <strong>Prediction/research dashboard.</strong> Predictions are uncertain. This is not a betting platform.
  Live-adjusted predictions are scenario-based. Official model remains football_only_ensemble / core_football_only.
</div>
""",
        unsafe_allow_html=True,
    )
    live_predictions = _live_predictions_for_display(frames)
    tabs = st.tabs(["Live / Upcoming Matches", "Match Detail", "Bracket & Knockouts", "Tournament Winner", "About"])
    with tabs[0]:
        render_user_live_matches(live_predictions, frames)
    with tabs[1]:
        render_user_match_detail(live_predictions, frames)
    with tabs[2]:
        render_user_bracket(frames)
    with tabs[3]:
        render_user_tournament_winner(frames["worldcup_2026_simulation_results"])
    with tabs[4]:
        render_user_about(values)


def render_developer_mode(reports: dict[str, str], frames: dict[str, pd.DataFrame], values: dict[str, object]) -> None:
    tabs = st.tabs(
        [
            "Dev Overview",
            "Live API Status",
            "Live Predictions Debug",
            "Risk Quant Lab",
            "Official Predictions",
            "Market Odds Coverage",
            "Market Blend Benchmark",
            "Backtests",
            "Significance",
            "Feature/Data Quality",
            "Reports",
            "Raw Files",
        ]
    )
    with tabs[0]:
        render_dev_overview(reports, frames, values)
    with tabs[1]:
        render_live_api_status(reports)
    with tabs[2]:
        render_live_predictions_debug(frames)
    with tabs[3]:
        render_developer_risk_quant_lab(frames)
    with tabs[4]:
        render_predictions(frames["actual_team_match_predictions"])
    with tabs[5]:
        render_market_odds_coverage(reports, frames["market_odds"])
    with tabs[6]:
        render_market_blend(reports)
    with tabs[7]:
        render_backtests(reports, frames)
    with tabs[8]:
        render_significance(reports["benchmark_significance_report"])
    with tabs[9]:
        render_feature_data_quality(reports)
    with tabs[10]:
        render_report_expanders(reports)
    with tabs[11]:
        render_raw_files(frames, reports)


def render_user_live_matches(live_predictions: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> None:
    if live_predictions.empty:
        st.warning("No live or official prediction rows are available.")
        return
    scores = frames.get("live_scores", pd.DataFrame())
    view = _attach_scores(live_predictions, scores)
    col1, col2, col3, col4 = st.columns(4)
    status_filter = col1.selectbox("Status", ["All", "Live only", "Upcoming only", "Finished"], key="user_match_status_filter")
    team_query = col2.text_input("Team search", key="user_match_team_search")
    sort_by = col3.selectbox("Sort", ["Kickoff time", "Confidence", "EV/IDR rate"], key="user_match_sort")
    positive_ev_only = col4.checkbox("Positive EV only", value=False, key="user_match_positive_ev_only")
    status_text = view["status"].astype(str).str.lower()
    is_finished = status_text.str.contains("finished|full", na=False)
    is_upcoming = status_text.str.contains("upcoming|not started|scheduled|time", na=False)
    is_live = status_text.str.contains("live|first|second|half|extra|penalty", na=False) | (view["minute"].notna() & ~is_finished & ~is_upcoming)
    if status_filter == "Live only":
        view = view[is_live]
    elif status_filter == "Upcoming only":
        view = view[is_upcoming]
    elif status_filter == "Finished":
        view = view[is_finished]
    if team_query:
        query = team_query.strip().lower()
        view = view[view["home_team"].astype(str).str.lower().str.contains(query, na=False) | view["away_team"].astype(str).str.lower().str.contains(query, na=False)]
    view = add_compact_risk_columns(view, frames)
    ev_eligible_count = int(view["risk_has_market_odds"].fillna(False).sum()) if "risk_has_market_odds" in view.columns else 0
    visible_count = len(view)
    if positive_ev_only:
        view = view[view["has_positive_ev"]]
    if sort_by == "Confidence" and "confidence" in view.columns:
        view = view.sort_values("confidence", ascending=False, kind="stable")
    elif sort_by == "EV/IDR rate" and "best_ev_per_idr" in view.columns:
        view = view.sort_values("best_ev_per_idr", ascending=False, kind="stable", na_position="last")
    elif "date" in view.columns:
        view = view.sort_values("date", kind="stable")
    if sort_by == "EV/IDR rate" or positive_ev_only:
        st.caption(f"EV/IDR eligibility: {ev_eligible_count}/{visible_count} visible matches have market odds. EV/IDR cannot be calculated without odds.")
    if view.empty:
        st.info("No matches match the current filters. Switch Status to All or Upcoming only to see the current prediction rows.")
        if status_filter != "All":
            fallback = _attach_scores(live_predictions, scores).sort_values("date", kind="stable").head(8)
            if not fallback.empty:
                st.caption("Previewing the next available matches:")
                for _, row in fallback.iterrows():
                    render_match_card(row)
                    render_live_match_risk_note(row, frames)
        return
    for _, row in view.head(40).iterrows():
        render_match_card(row)
        render_live_match_risk_note(row, frames)


def render_user_match_detail(live_predictions: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> None:
    if live_predictions.empty:
        st.warning("No match predictions are available.")
        return
    view = _attach_scores(live_predictions, frames.get("live_scores", pd.DataFrame()))
    labels = [
        f"{row.home_team} vs {row.away_team} ({row.date})"
        for row in view.itertuples(index=False)
    ]
    selected = st.selectbox("Select match", labels)
    row = view.iloc[labels.index(selected)]
    st.subheader(f"{row.get('home_team')} vs {row.get('away_team')}")
    st.write(f"Status: `{row.get('status')}`  Minute: `{row.get('minute')}`")
    render_probability_comparison(row)
    if pd.notna(row.get("home_advances_prob")) and pd.notna(row.get("away_advances_prob")):
        c1, c2 = st.columns(2)
        c1.metric(f"{row.get('home_team')} advances", f"{float(row.get('home_advances_prob')):.0%}")
        c2.metric(f"{row.get('away_team')} advances", f"{float(row.get('away_advances_prob')):.0%}")
    factors = frames.get("live_prediction_factors", pd.DataFrame())
    if not factors.empty and "fixture_id" in factors.columns:
        match_factors = factors[factors["fixture_id"].astype(str).eq(str(row.get("fixture_id")))]
        if not match_factors.empty:
            st.subheader("Live Factors")
            for _, factor in match_factors.iterrows():
                st.write(f"- {factor.get('team')}: {factor.get('description')}")
    explanation = _plain_english_explanation(row, factors)
    st.info(explanation)
    render_user_risk_panel(row, frames)


def render_user_bracket(frames: dict[str, pd.DataFrame]) -> None:
    st.info("These are simulation-derived possible matchups, not confirmed fixtures.")
    matchup = frames["worldcup_2026_matchup_probabilities"]
    simulation = frames["worldcup_2026_simulation_results"]
    if not matchup.empty:
        stage = st.selectbox("Stage", sorted(matchup["stage"].dropna().unique()), format_func=display_stage, key="user_bracket_stage")
        top = add_matchup_label(matchup[matchup["stage"].eq(stage)]).sort_values("matchup_probability", ascending=False).head(20)
        st.dataframe(format_percent_columns(top, ["matchup_probability", "team_a_win_probability", "team_b_win_probability"]), use_container_width=True)
    elif not simulation.empty:
        st.warning("Matchup probability file is missing. Showing stage probabilities instead.")
        render_tournament_simulation(simulation)
    else:
        st.warning("Simulation outputs are missing.")


def render_user_tournament_winner(simulation: pd.DataFrame) -> None:
    if simulation.empty:
        st.warning("Tournament simulation results are missing.")
        return
    team_search = st.text_input("Search team")
    view = simulation.copy()
    if team_search:
        view = view[view["team"].astype(str).str.lower().str.contains(team_search.strip().lower(), na=False)]
    stage_cols = stage_probability_columns(view)
    top = view.sort_values("champion_probability", ascending=False).head(10) if "champion_probability" in view.columns else view.head(10)
    st.dataframe(format_percent_columns(top[["team", *stage_cols]], stage_cols), use_container_width=True)
    if "champion_probability" in top.columns:
        st.bar_chart(top.set_index("team")["champion_probability"])


def render_user_about(values: dict[str, object]) -> None:
    st.markdown(
        f"""
### About This Dashboard

This is a World Cup 2026 prediction and research dashboard. It is not a betting platform, does not take wagers, and does not provide betting actions.

- Official model: `{SAFE_MODEL}`
- Official feature set: `{SAFE_FEATURE_SET}`
- Project label: SOTA-inspired, historically backtested, benchmarked against bookmaker odds, but not true SOTA
- Market blend status: benchmark-only unless 2026 odds coverage reaches 90%
- Best market alpha currently shown by reports: `{values.get("best_alpha", "unknown")}`

Live-adjusted predictions are scenario-based and may change when API data updates. Predictions are uncertain.
"""
    )


def render_dev_overview(reports: dict[str, str], frames: dict[str, pd.DataFrame], values: dict[str, object]) -> None:
    cols = st.columns(4)
    cols[0].metric("Official model", SAFE_MODEL)
    cols[1].metric("Feature set", SAFE_FEATURE_SET)
    cols[2].metric("Official rows", len(frames["actual_team_match_predictions"]))
    cols[3].metric("Live prediction rows", len(frames["live_predictions"]))
    cols = st.columns(4)
    cols[0].metric("Live fixtures", len(frames["live_fixtures"]))
    cols[1].metric("Live odds", len(frames["live_odds"]))
    cols[2].metric("Live injuries", len(frames["live_injuries"]))
    cols[3].metric("Live lineups", len(frames["live_lineups"]))
    coverage = values.get("coverage_2026")
    st.write(f"Market blend status: `{MARKET_BLEND_STATUS}`")
    st.write(f"2026 odds coverage: `{coverage if coverage is not None else 'unknown'}`")
    st.write(f"Statistical significance: `{values.get('statistically_meaningful', 'unknown')}`")
    st.subheader("File Timestamps")
    st.dataframe(file_presence({**REPORT_FILES, **CSV_FILES}), use_container_width=True)


def render_live_api_status(reports: dict[str, str]) -> None:
    st.subheader("Live API Controls")
    fetch_lineups = st.checkbox("Fetch lineups on refresh", value=False, help="Lineups can use extra API quota.")
    if st.button("Refresh live data now"):
        with st.spinner("Fetching live API data and regenerating live predictions..."):
            summary = refresh_live_data(fetch_lineups=fetch_lineups)
        st.json(summary)
        st.cache_data.clear()
    if st.button("Generate live predictions from local live CSVs"):
        summary = generate_live_predictions()
        st.json(summary)
        st.cache_data.clear()

    st.subheader("API Key Status")
    st.write(f"API_FOOTBALL_KEY detected: `{bool(os.getenv('API_FOOTBALL_KEY', '').strip())}`")
    st.write(f"THE_ODDS_API_KEY detected: `{bool(os.getenv('THE_ODDS_API_KEY', '').strip())}`")
    st.write(f"SPORTMONKS_API_TOKEN detected: `{bool(os.getenv('SPORTMONKS_API_TOKEN', '').strip())}`")

    st.subheader("API-Football Daily Quota")
    if st.button("Check API-Football quota now"):
        with st.spinner("Calling API-Football /status..."):
            quota = fetch_api_football_status()
        st.session_state["api_football_quota"] = quota
        st.json(quota)
        st.cache_data.clear()
    quota = st.session_state.get("api_football_quota")
    if isinstance(quota, dict):
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Used today", _metric_or_unknown(quota.get("used_today")))
        q2.metric("Daily limit", _metric_or_unknown(quota.get("daily_limit")))
        q3.metric("Remaining", _metric_or_unknown(quota.get("remaining_today")))
        usage = quota.get("usage_percent")
        q4.metric("Usage", f"{float(usage):.1%}" if isinstance(usage, (int, float)) else "unknown")
        if quota.get("errors"):
            st.warning("Quota check returned errors:")
            for error in quota.get("errors", []):
                st.write(f"- {error}")
    else:
        st.info("Click ‘Check API-Football quota now’ to fetch daily request usage. This does not refresh match data.")

    with st.expander("API-Football quota report"):
        st.markdown(reports.get("api_football_quota_report") or "Missing report. Click the quota check button to generate it.")
    with st.expander("Live API status report", expanded=True):
        st.markdown(reports.get("live_api_status_report") or "Missing report.")
    with st.expander("Live prediction report"):
        st.markdown(reports.get("live_prediction_report") or "Missing report.")

def render_live_predictions_debug(frames: dict[str, pd.DataFrame]) -> None:
    predictions = frames["live_predictions"]
    factors = frames["live_prediction_factors"]
    if predictions.empty:
        st.warning("live_predictions.csv is missing or empty.")
    else:
        st.subheader("Live Predictions")
        st.dataframe(predictions, use_container_width=True)
        valid = probability_sums_valid(predictions)
        if not valid.empty:
            st.write(f"Probability sum checks passed: `{int(valid.sum())}/{len(valid)}`")
        missing_base = predictions[["base_home_prob", "base_draw_prob", "base_away_prob"]].isna().any(axis=1).sum()
        st.write(f"Rows missing base prediction: `{missing_base}`")
    st.subheader("Live Prediction Factors")
    st.dataframe(factors, use_container_width=True)
    for name in ["live_fixtures", "live_odds", "live_injuries", "live_lineups", "live_scores"]:
        with st.expander(name):
            st.dataframe(frames[name], use_container_width=True)


def render_feature_data_quality(reports: dict[str, str]) -> None:
    st.warning("Official model features remain separate from scenario-only live injuries, lineups, and live odds.")
    for heading in ("Feature Null Rate Summary", "market", "injuries", "squad", "xG"):
        table = markdown_table_after_heading(reports.get("feature_null_rate_report", ""), heading)
        if not table.empty:
            st.subheader(heading)
            st.dataframe(table, use_container_width=True)
    with st.expander("Raw feature null-rate report", expanded=True):
        st.markdown(reports.get("feature_null_rate_report") or "Missing report.")


def render_report_expanders(reports: dict[str, str]) -> None:
    for name in [
        "final_project_summary",
        "final_model_recommendation",
        "market_odds_join_debug",
        "feature_null_rate_report",
        "market_blend_report",
        "benchmark_significance_report",
        "worldcup_backtest_report",
        "worldcup_tournament_simulation_backtest_report",
    ]:
        with st.expander(name):
            st.markdown(reports.get(name) or "Missing report.")


def render_raw_files(frames: dict[str, pd.DataFrame], reports: dict[str, str]) -> None:
    st.subheader("File Presence")
    st.dataframe(file_presence({**REPORT_FILES, **CSV_FILES, **OPTIONAL_FILES}), use_container_width=True)
    st.subheader("CSV Previews")
    for name, frame in frames.items():
        with st.expander(name):
            if frame.empty:
                st.warning("Missing or empty.")
            else:
                st.dataframe(frame.head(200), use_container_width=True)


def render_overview(reports: dict[str, str], frames: dict[str, pd.DataFrame], values: dict[str, object]) -> None:
    coverage = values.get("coverage_2026")
    production_ready = isinstance(coverage, float) and coverage >= MARKET_PRODUCTION_THRESHOLD
    prediction_rows = len(frames["actual_team_match_predictions"])
    simulation_teams = len(frames["worldcup_2026_simulation_results"])
    cols = st.columns(5)
    cols[0].metric("Safe model", SAFE_MODEL)
    cols[1].metric("Safe feature set", SAFE_FEATURE_SET)
    cols[2].metric("Prediction rows", prediction_rows)
    cols[3].metric("Simulation teams", simulation_teams)
    cols[4].metric("Market blend", MARKET_BLEND_STATUS)
    cols = st.columns(5)
    cols[0].metric("2026 odds coverage", f"{coverage:.1%}" if isinstance(coverage, float) else "unknown")
    cols[1].metric("Odds conversion", str(values.get("best_odds_conversion", "unknown")))
    cols[2].metric("Best alpha", str(values.get("best_alpha", "unknown")))
    cols[3].metric("Stat meaningful", str(values.get("statistically_meaningful", "unknown")))
    cols[4].metric("Production-ready", str(production_ready))
    st.warning("Market blend is benchmark-only, not production-ready for 2026, unless 2026 market odds coverage is at least 90%.")
    st.warning("This dashboard is a research viewer and scenario lab. It is not a betting tool.")
    with st.expander("Final Model Recommendation"):
        st.markdown(reports.get("final_model_recommendation") or "Missing report.")
    with st.expander("Final Project Summary"):
        st.markdown(reports.get("final_project_summary") or "Missing report.")


def render_bracket_view(frames: dict[str, pd.DataFrame]) -> None:
    st.info("These are simulation-derived probabilities, not confirmed scheduled fixtures.")
    sample = frames["worldcup_2026_bracket_path_samples"]
    matchup = frames["worldcup_2026_matchup_probabilities"]
    simulation = frames["worldcup_2026_simulation_results"]
    if not sample.empty:
        simulation_ids = sorted(sample["simulation_id"].dropna().unique())
        selected = st.selectbox("Simulation sample", simulation_ids)
        filtered = sample[sample["simulation_id"].eq(selected)]
        for stage, stage_df in filtered.groupby("stage", sort=False):
            st.subheader(display_stage(stage))
            st.dataframe(stage_df.drop(columns=["simulation_id"], errors="ignore"), use_container_width=True)
        return
    if not matchup.empty:
        st.warning("Bracket path samples are missing. Showing most likely matchup tiers instead.")
        stage = st.selectbox("Stage", sorted(matchup["stage"].dropna().unique()), format_func=display_stage, key="bracket_stage")
        top_n = st.slider("Top matchups", 5, 50, 15, key="bracket_top_n")
        view = add_matchup_label(matchup[matchup["stage"].eq(stage)]).head(top_n)
        st.dataframe(format_percent_columns(view, ["matchup_probability", "team_a_win_probability", "team_b_win_probability"]), use_container_width=True)
        return
    if not simulation.empty:
        st.warning("Exact matchup probabilities require data/simulation/worldcup_2026_matchup_probabilities.csv.")
        columns = stage_probability_columns(simulation)
        stage_col = st.selectbox("Stage probability", columns)
        st.dataframe(simulation[["team", stage_col]].sort_values(stage_col, ascending=False).head(32), use_container_width=True)
        st.bar_chart(simulation.set_index("team")[stage_col].sort_values(ascending=False).head(16))
        return
    st.warning("No simulation outputs are available.")


def render_knockout_matchups(frames: dict[str, pd.DataFrame]) -> None:
    matchup = frames["worldcup_2026_matchup_probabilities"]
    if matchup.empty:
        st.warning("Matchup probability file not found. Rerun the full simulation after enabling matchup logging.")
        st.code("python -m src.cli full-sota-pipeline --model catboost --feature-set core_football_only --n-sims 100000")
        return
    matchup = add_matchup_label(matchup)
    stage = st.selectbox("Stage", sorted(matchup["stage"].dropna().unique()), format_func=display_stage)
    teams = sorted(set(matchup["team_a"]).union(set(matchup["team_b"])))
    team_filter = st.selectbox("Team filter", ["All teams"] + teams)
    top_n = st.slider("Top N", 5, 100, 25)
    view = matchup[matchup["stage"].eq(stage)].copy()
    if team_filter != "All teams":
        view = view[view["team_a"].eq(team_filter) | view["team_b"].eq(team_filter)]
    view = view.sort_values("matchup_probability", ascending=False).head(top_n)
    display = format_percent_columns(view, ["matchup_probability", "team_a_win_probability", "team_b_win_probability"])
    st.dataframe(display, use_container_width=True)
    if not view.empty:
        st.bar_chart(view.set_index("matchup")["matchup_probability"])
    st.subheader("Team Path Explorer")
    selected_team = st.selectbox("Team", teams, key="path_team")
    path = matchup[matchup["team_a"].eq(selected_team) | matchup["team_b"].eq(selected_team)].copy()
    path["selected_team_win_probability"] = path.apply(
        lambda row: row["team_a_win_probability"] if row["team_a"] == selected_team else row["team_b_win_probability"],
        axis=1,
    )
    st.dataframe(
        format_percent_columns(path.sort_values("matchup_probability", ascending=False), ["matchup_probability", "selected_team_win_probability"]),
        use_container_width=True,
    )
    simulation = frames["worldcup_2026_simulation_results"]
    if not simulation.empty and "team" in simulation.columns:
        stage_cols = stage_probability_columns(simulation)
        team_stage = simulation[simulation["team"].eq(selected_team)][["team", *stage_cols]]
        if not team_stage.empty:
            st.dataframe(format_percent_columns(team_stage, stage_cols), use_container_width=True)
    st.caption("Argentina vs France appearing in a semifinal at 8% means 8% of simulations produced that exact semifinal matchup. It does not mean the match is scheduled.")


def render_matchup_lab(frames: dict[str, pd.DataFrame]) -> None:
    st.warning("Matchup Lab outputs are research estimates. Scenario-adjusted outputs are not validated official model outputs.")
    predictions = frames["actual_team_match_predictions"]
    simulation = frames["worldcup_2026_simulation_results"]
    matchup = frames["worldcup_2026_matchup_probabilities"]
    market = frames["market_odds"]
    teams = team_options(simulation, predictions, market)
    if len(teams) < 2:
        st.warning("No team list is available from current outputs.")
        return
    col1, col2, col3 = st.columns(3)
    team_a = col1.selectbox("Team A", teams, index=0)
    team_b_index = 1 if len(teams) > 1 else 0
    team_b = col2.selectbox("Team B", teams, index=team_b_index)
    stage = col3.selectbox("Stage", ["Group stage", "Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Third-place", "Final"])
    neutral = st.toggle("Neutral venue", value=True)
    use_market = st.toggle("Use market odds if exact uploaded/live odds are available", value=False)
    use_scenario = st.toggle("Use scenario adjustments", value=False)
    if team_a == team_b:
        st.warning("Choose two different teams.")
        return
    base = find_fixture_prediction(predictions, team_a, team_b)
    if base is None:
        st.warning("No direct fixture prediction available. Rerun or add a supported arbitrary-match predictor.")
        return
    p_a, p_draw, p_b = base["probabilities"]
    st.write(f"Base source: `{base['orientation']}` fixture lookup. Neutral venue: `{neutral}`")
    render_probability_triplet(team_a, team_b, (p_a, p_draw, p_b), "Official model probabilities")
    if stage != "Group stage":
        adv_a, adv_b = advancement_probabilities(p_a, p_b)
        st.write(f"Advancement probability: **{team_a} {adv_a:.1%}**, **{team_b} {adv_b:.1%}**")
    if use_market:
        market_probs = lookup_market_probability(market, team_a, team_b)
        live_probs = lookup_market_probability(st.session_state.get("live_market_odds", pd.DataFrame()), team_a, team_b)
        chosen = live_probs or market_probs
        if chosen is None:
            st.warning("No exact market odds row found for this matchup.")
        else:
            render_probability_triplet(team_a, team_b, chosen, "Market-implied probabilities")
    if use_scenario:
        col_a, col_b = st.columns(2)
        adj_a = col_a.number_input(f"{team_a} overall logit adjustment", value=0.0, step=0.01, min_value=-1.0, max_value=1.0)
        adj_b = col_b.number_input(f"{team_b} overall logit adjustment", value=0.0, step=0.01, min_value=-1.0, max_value=1.0)
        adjusted = apply_scenario_adjustments((p_a, p_draw, p_b), team_a_adjustment=adj_a, team_b_adjustment=adj_b)
        render_probability_triplet(team_a, team_b, adjusted, "SCENARIO MODE ONLY - adjusted probabilities")
        scenario_df = pd.DataFrame(
            [
                {"team": team_a, "base": p_a, "scenario": adjusted[0], "delta": adjusted[0] - p_a},
                {"team": "Draw", "base": p_draw, "scenario": adjusted[1], "delta": adjusted[1] - p_draw},
                {"team": team_b, "base": p_b, "scenario": adjusted[2], "delta": adjusted[2] - p_b},
            ]
        )
        st.download_button("Download scenario result", scenario_df.to_csv(index=False), "scenario_matchup_result.csv", "text/csv")
    render_matchup_probability_context(matchup, team_a, team_b)


def render_probability_triplet(team_a: str, team_b: str, probabilities: tuple[float, float, float], title: str) -> None:
    st.subheader(title)
    cols = st.columns(3)
    cols[0].metric(f"{team_a} win", f"{probabilities[0]:.1%}")
    cols[1].metric("Draw", f"{probabilities[1]:.1%}")
    cols[2].metric(f"{team_b} win", f"{probabilities[2]:.1%}")


def lookup_market_probability(market: pd.DataFrame, team_a: str, team_b: str) -> tuple[float, float, float] | None:
    if market is None or market.empty or not {"home_team", "away_team", "home_odds", "draw_odds", "away_odds"}.issubset(market.columns):
        return None
    output = market.copy()
    output["_home_norm"] = output["home_team"].map(normalize_team_name)
    output["_away_norm"] = output["away_team"].map(normalize_team_name)
    if "date" in output.columns:
        output["_date"] = pd.to_datetime(output["date"], errors="coerce", format="mixed")
        output = output.sort_values("_date", kind="stable")
    team_a_norm = normalize_team_name(team_a)
    team_b_norm = normalize_team_name(team_b)
    direct = output[output["_home_norm"].eq(team_a_norm) & output["_away_norm"].eq(team_b_norm)]
    reverse = output[output["_home_norm"].eq(team_b_norm) & output["_away_norm"].eq(team_a_norm)]
    if not direct.empty:
        row = direct.iloc[-1]
        return decimal_odds_to_implied(row["home_odds"], row["draw_odds"], row["away_odds"])
    if not reverse.empty:
        row = reverse.iloc[-1]
        probs = decimal_odds_to_implied(row["home_odds"], row["draw_odds"], row["away_odds"])
        return (probs[2], probs[1], probs[0]) if probs else None
    return None


def render_matchup_probability_context(matchup: pd.DataFrame, team_a: str, team_b: str) -> None:
    if matchup.empty:
        return
    team_a_norm = normalize_team_name(team_a)
    team_b_norm = normalize_team_name(team_b)
    left, right = sorted([team_a_norm, team_b_norm])
    view = matchup[matchup["team_a"].eq(left) & matchup["team_b"].eq(right)].copy()
    if view.empty:
        st.info("This team pair does not appear in the current matchup probability output.")
        return
    st.subheader("Simulation matchup context")
    st.dataframe(format_percent_columns(view, ["matchup_probability", "team_a_win_probability", "team_b_win_probability"]), use_container_width=True)


def render_live_scenario_mode(frames: dict[str, pd.DataFrame]) -> None:
    st.error("SCENARIO MODE ONLY - not validated, not official production output.")
    staged_injuries = [
        ("API-Football staged injuries", frames.get("api_football_injuries_live", pd.DataFrame())),
        ("Sportmonks staged injuries", frames.get("sportmonks_injuries_live", pd.DataFrame())),
    ]
    for label, staged in staged_injuries:
        if not staged.empty:
            with st.expander(label, expanded=False):
                st.dataframe(staged, use_container_width=True)
    odds_upload = st.file_uploader("Live market odds CSV", type=["csv"], key="live_market_upload")
    if odds_upload is not None:
        live_market = pd.read_csv(odds_upload)
        st.session_state["live_market_odds"] = live_market
        st.write(validate_market_odds(live_market))
        implied = live_market.copy()
        probs = implied.apply(lambda row: decimal_odds_to_implied(row.get("home_odds"), row.get("draw_odds"), row.get("away_odds")), axis=1)
        prob_rows = [prob if prob is not None else (pd.NA, pd.NA, pd.NA) for prob in probs.tolist()]
        implied[["market_home_prob", "market_draw_prob", "market_away_prob"]] = pd.DataFrame(prob_rows, index=implied.index)
        st.dataframe(implied, use_container_width=True)
    injuries_upload = st.file_uploader("Injuries/suspensions live CSV", type=["csv"], key="injury_upload")
    if injuries_upload is not None:
        injuries = pd.read_csv(injuries_upload)
        st.session_state["live_injuries"] = injuries
        st.dataframe(injuries, use_container_width=True)
        st.info("Injuries are displayed as context only unless manual scenario adjustments are supplied in Matchup Lab.")
    adjustments_upload = st.file_uploader("Manual team adjustment CSV", type=["csv"], key="adjustment_upload")
    if adjustments_upload is not None:
        adjustments = pd.read_csv(adjustments_upload)
        st.session_state["manual_adjustments"] = adjustments
        st.dataframe(adjustments, use_container_width=True)
        st.info("Manual adjustments are held in session state and are not written into official files.")


def render_predictions(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        st.warning("Prediction file is missing or empty.")
        return
    view = predictions.copy()
    teams = ["All teams"] + team_options(view)
    selected = st.selectbox("Team", teams)
    if selected != "All teams" and {"home_team", "away_team"}.issubset(view.columns):
        norm = normalize_team_name(selected)
        view = view[view["home_team"].map(normalize_team_name).eq(norm) | view["away_team"].map(normalize_team_name).eq(norm)]
    for column in ("model_name", "feature_set"):
        if column in view.columns:
            options = ["All"] + sorted(view[column].dropna().astype(str).unique())
            selected_value = st.selectbox(column, options)
            if selected_value != "All":
                view = view[view[column].astype(str).eq(selected_value)]
    prob_cols = infer_probability_columns(view)
    if prob_cols is not None:
        probs = view[[prob_cols.home, prob_cols.draw, prob_cols.away]].apply(pd.to_numeric, errors="coerce")
        view["confidence"] = probs.max(axis=1)
        view["predicted_outcome"] = probs.idxmax(axis=1).map({prob_cols.home: "home", prob_cols.draw: "draw", prob_cols.away: "away"})
        st.subheader("Top 10 Most Confident Predictions")
        st.dataframe(view.sort_values("confidence", ascending=False).head(10), use_container_width=True)
    if "model_name" in predictions.columns and "feature_set" in predictions.columns:
        official = predictions["model_name"].eq(SAFE_MODEL).all() and predictions["feature_set"].eq(SAFE_FEATURE_SET).all()
        st.write(f"All rows use official safe model/feature set: `{official}`")
    st.write(f"Prediction row count: `{len(predictions)}`")
    st.dataframe(view, use_container_width=True)


def render_tournament_simulation(simulation: pd.DataFrame) -> None:
    if simulation.empty:
        st.warning("Simulation results file is missing or empty.")
        return
    view = simulation.sort_values("champion_probability", ascending=False) if "champion_probability" in simulation.columns else simulation
    st.dataframe(format_percent_columns(view, stage_probability_columns(view)), use_container_width=True)
    if "champion_probability" in simulation.columns and "team" in simulation.columns:
        top = simulation.set_index("team")["champion_probability"].sort_values(ascending=False).head(10)
        st.subheader("Top 10 Champion Probabilities")
        st.bar_chart(top)
    sums = validate_stage_sums(simulation)
    if not sums.empty:
        st.subheader("Stage Probability Sum Validation")
        st.dataframe(sums, use_container_width=True)
        bad = sums[~sums["within_tolerance"]]
        if not bad.empty:
            st.warning("Some stage probability sums deviate from expected tournament counts.")


def render_market_odds_coverage(reports: dict[str, str], market_odds: pd.DataFrame) -> None:
    validation = validate_market_odds(market_odds)
    st.subheader("Provider Discovery")
    discovery = reports.get("odds_provider_competition_discovery", "")
    if discovery:
        if "club world cup" in discovery.lower():
            st.warning("Discovery includes Club World Cup candidates. Do not use Club World Cup odds for national-team World Cup predictions.")
        st.markdown(discovery)
    else:
        st.info("Run `python -m src.cli discover-odds-competitions` to create provider discovery output.")

    st.subheader("CSV Validation")
    st.json(validation)
    if validation["duplicate_rows"]:
        st.warning("Duplicate date/home_team/away_team rows are present.")
    if validation["bad_odds"]:
        st.warning("Some odds are missing, non-numeric, or <= 1.")
    rows_by_year = odds_rows_by_year(market_odds)
    if not rows_by_year.empty:
        st.subheader("Odds Rows By Year")
        st.dataframe(rows_by_year, use_container_width=True)
    debug = reports.get("market_odds_join_debug", "")
    coverage = _extract_coverage(debug)
    if coverage is not None:
        st.metric("2026 active fixture coverage", f"{coverage:.1%}")
        if coverage < MARKET_PRODUCTION_THRESHOLD:
            st.warning("2026 odds coverage is below 90%; market_blend remains benchmark-only.")
    st.subheader("Parsed Tables")
    for heading in ("Historical Coverage", "Unmatched Odds", "Missing Active Fixtures"):
        table = markdown_table_after_heading(debug, heading)
        if not table.empty:
            st.write(heading)
            st.dataframe(table, use_container_width=True)
    st.subheader("Missing Active Fixtures")
    missing_template = cached_csv(str(CSV_FILES["missing_market_odds_template"]))
    if missing_template.empty:
        st.info("Run `python -m src.cli generate-missing-odds-template` to create a missing odds template.")
    else:
        st.dataframe(missing_template, use_container_width=True)
        st.download_button(
            "Download missing odds template",
            missing_template.to_csv(index=False),
            "missing_market_odds_template_2026.csv",
            "text/csv",
        )
    st.subheader("Provider Fetch Results")
    refresh_report = reports.get("refresh_market_odds_report", "")
    st.markdown(refresh_report or "Run `python -m src.cli refresh-market-odds` to create this report.")
    st.subheader("Merge Status")
    st.markdown(reports.get("market_odds_api_merge_report") or "No merge report is available yet.")
    with st.expander("Raw market odds join debug report"):
        st.markdown(debug or "Missing report.")
    with st.expander("Feature null-rate report"):
        st.markdown(reports.get("feature_null_rate_report") or "Missing report.")
    with st.expander("Market blend report"):
        st.markdown(reports.get("market_blend_report") or "Missing report.")
    with st.expander("Missing odds template report"):
        st.markdown(reports.get("missing_market_odds_template_report") or "Missing report.")
    with st.expander("Manual odds validation errors"):
        st.markdown(reports.get("manual_market_odds_validation_errors") or "No manual validation errors report.")


def render_market_blend(reports: dict[str, str]) -> None:
    blend = reports.get("market_blend_report", "")
    odds = reports.get("odds_conversion_benchmark_report", "")
    values = parsed_overview_values(reports)
    cols = st.columns(2)
    cols[0].metric("Best odds conversion", str(values.get("best_odds_conversion", "unknown")))
    cols[1].metric("Best alpha", str(values.get("best_alpha", "unknown")))
    for label, markdown in [("Alpha Search", blend), ("Odds-Covered Metrics", blend), ("Average Odds-Covered Metrics", blend), ("Odds Conversion Benchmark", odds)]:
        table = markdown_table_after_heading(markdown, label)
        if not table.empty:
            st.subheader(label)
            st.dataframe(table, use_container_width=True)
    st.info("Market blend may be numerically better; the Significance tab determines whether it is statistically meaningful. It remains benchmark-only unless 2026 coverage reaches the production threshold.")
    with st.expander("Raw market blend report"):
        st.markdown(blend or "Missing report.")
    with st.expander("Raw odds conversion benchmark report"):
        st.markdown(odds or "Missing report.")


def render_backtests(reports: dict[str, str], frames: dict[str, pd.DataFrame]) -> None:
    metrics = frames["worldcup_backtest_metrics"]
    tournament = frames["worldcup_tournament_simulation_backtest"]
    if not metrics.empty:
        st.subheader("Match-Level Backtest Metrics")
        st.dataframe(metrics, use_container_width=True)
        if {"model_name", "log_loss"}.issubset(metrics.columns):
            ranking = metrics.groupby("model_name", as_index=False)["log_loss"].mean().sort_values("log_loss")
            st.subheader("Model Ranking By Average Log Loss")
            st.dataframe(ranking, use_container_width=True)
    else:
        st.warning("Match-level backtest metrics are missing.")
    if not tournament.empty:
        st.subheader("Tournament Simulation Backtest")
        st.dataframe(tournament, use_container_width=True)
        if {"model_name", "group_match_log_loss"}.issubset(tournament.columns):
            st.bar_chart(tournament.groupby("model_name")["group_match_log_loss"].mean().sort_values())
    else:
        st.warning("Tournament simulation backtest file is missing.")
    with st.expander("Raw match-level backtest report"):
        st.markdown(reports.get("worldcup_backtest_report") or "Missing report.")
    with st.expander("Raw tournament simulation report"):
        st.markdown(reports.get("worldcup_tournament_simulation_backtest_report") or "Missing report.")


def render_significance(significance: str) -> None:
    if not significance:
        st.warning("Benchmark significance report is missing.")
        return
    ci_table = markdown_table_after_heading(significance, "Log Loss Confidence Intervals")
    delta_table = markdown_table_after_heading(significance, "Paired Delta Confidence Intervals")
    if not ci_table.empty:
        st.subheader("Log Loss Confidence Intervals")
        st.dataframe(ci_table, use_container_width=True)
    if not delta_table.empty:
        st.subheader("Paired Delta Confidence Intervals")
        st.dataframe(delta_table, use_container_width=True)
        row = _delta_row(delta_table, "market_blend_minus_bookmaker_odds_only")
        if row is not None:
            delta, lower, upper = row
            st.metric("market_blend - bookmaker_odds_only", f"{delta:.6f}", f"CI [{lower:.6f}, {upper:.6f}]")
            if delta > 0:
                st.warning("Market blend is worse than the comparator on log loss.")
            elif lower < 0 < upper:
                st.warning("Do not claim statistically significant improvement.")
            elif upper < 0:
                st.success("Market blend significantly improves bookmaker odds on odds-covered historical World Cup matches.")
    with st.expander("Raw benchmark significance report"):
        st.markdown(significance)


def render_reports_missing_data(reports: dict[str, str]) -> None:
    all_paths = {**REPORT_FILES, **CSV_FILES, **OPTIONAL_FILES}
    st.subheader("File Presence Checklist")
    st.dataframe(file_presence(all_paths), use_container_width=True)
    st.subheader("Recommended Next Actions")
    st.checkbox("Add 2010 1X2 market odds.", value=False)
    st.checkbox("Add remaining 2026 active fixture odds.", value=False)
    st.checkbox("Rerun market join report.", value=False)
    st.checkbox("Rerun feature null report.", value=False)
    st.checkbox("Rerun market blend benchmark.", value=False)
    st.checkbox("Rerun benchmark significance.", value=False)
    st.checkbox("Rerun full simulation to regenerate matchup probabilities.", value=False)
    st.checkbox("Only promote market_blend if 2026 market odds coverage >= 90% and reports pass.", value=False)
    st.subheader("Raw Reports")
    for name, markdown in reports.items():
        with st.expander(name):
            st.markdown(markdown or "Missing report.")


def _live_predictions_for_display(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    live_predictions = frames.get("live_predictions", pd.DataFrame())
    if not live_predictions.empty:
        return live_predictions
    predictions, _, _ = build_live_predictions(
        fixtures=frames.get("live_fixtures", pd.DataFrame()),
        official_predictions=frames.get("actual_team_match_predictions", pd.DataFrame()),
        live_odds=frames.get("live_odds", pd.DataFrame()),
        external_odds=frames.get("market_odds", pd.DataFrame()),
        injuries=frames.get("live_injuries", pd.DataFrame()),
        lineups=frames.get("live_lineups", pd.DataFrame()),
        scores=frames.get("live_scores", pd.DataFrame()),
    )
    return predictions


def _attach_scores(predictions: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    if scores.empty or "fixture_id" not in predictions.columns or "fixture_id" not in scores.columns:
        return _add_actual_outcome_columns(predictions.copy())
    score_cols = ["fixture_id", "home_score", "away_score", "home_red_cards", "away_red_cards"]
    available = [column for column in score_cols if column in scores.columns]
    merged = predictions.merge(scores[available].drop_duplicates("fixture_id", keep="last"), on="fixture_id", how="left", suffixes=("", "_score"))
    for column in score_cols[1:]:
        score_column = f"{column}_score"
        if score_column not in merged.columns:
            continue
        if column in merged.columns:
            merged[column] = merged[column].where(merged[column].notna(), merged[score_column])
        else:
            merged[column] = merged[score_column]
        merged = merged.drop(columns=[score_column])
    return _add_actual_outcome_columns(merged)


def _add_actual_outcome_columns(view: pd.DataFrame) -> pd.DataFrame:
    output = view.copy()
    if "actual_result" not in output.columns:
        output["actual_result"] = output.apply(_actual_result_from_row, axis=1)
    else:
        output["actual_result"] = output.apply(_actual_result_from_row, axis=1)
    if "prediction_correct" not in output.columns:
        output["prediction_correct"] = output.apply(_prediction_correct_from_row, axis=1)
    else:
        output["prediction_correct"] = output.apply(_prediction_correct_from_row, axis=1)
    return output


def _actual_result_from_row(row: pd.Series) -> object:
    if not _is_finished_match(row.get("status")):
        return pd.NA
    home_score = _score_number_or_none(row.get("home_score"))
    away_score = _score_number_or_none(row.get("away_score"))
    if home_score is None or away_score is None:
        return pd.NA
    if home_score > away_score:
        return f"{row.get('home_team', 'Home')} win"
    if away_score > home_score:
        return f"{row.get('away_team', 'Away')} win"
    return "Draw"


def _prediction_correct_from_row(row: pd.Series) -> object:
    actual = row.get("actual_result")
    predicted = row.get("predicted_result")
    if pd.isna(actual) or pd.isna(predicted):
        return pd.NA
    return str(actual) == str(predicted)


def _score_number_or_none(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_finished_match(status: object) -> bool:
    text = str(status or "").lower()
    return any(term in text for term in ["finished", "full time", "fulltime", "ft"])


def _plain_english_explanation(row: pd.Series, factors: pd.DataFrame) -> str:
    home = row.get("home_team", "Home")
    away = row.get("away_team", "Away")
    prediction = row.get("predicted_result", "the most likely outcome")
    parts = [f"The live-adjusted prediction currently leans toward {prediction}."]
    if pd.notna(row.get("base_home_prob")):
        parts.append("The official base model is available for this matchup.")
    if pd.notna(row.get("market_home_prob")):
        parts.append("Market-implied probabilities are included in the scenario blend.")
    else:
        parts.append("No valid market odds were available for this matchup.")
    if not factors.empty and "fixture_id" in factors.columns:
        match_factors = factors[factors["fixture_id"].astype(str).eq(str(row.get("fixture_id")))]
        if not match_factors.empty:
            teams = ", ".join(sorted(match_factors["team"].dropna().astype(str).unique()))
            parts.append(f"Detected live factors for: {teams}.")
        else:
            parts.append("No high-impact live factors were detected.")
    parts.append("Live adjustments are scenario-based and not official model outputs.")
    return " ".join(parts)


def _extract_coverage(text: str) -> float | None:
    patterns = [
        r"Current generated advanced prediction coverage.*?(\d+)\s*/\s*(\d+).*?\(([0-9.]+)\)",
        r"After fix.*?coverage.*?(\d+)\s*/\s*(\d+).*?\(([0-9.]+)\)",
        r"2026 active fixture odds coverage.*?(\d+)\s*/\s*(\d+).*?\(([0-9.]+)\)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            return float(match.group(3))
    return None


def _delta_row(df: pd.DataFrame, label: str) -> tuple[float, float, float] | None:
    lower_columns = {str(column).lower(): column for column in df.columns}
    label_column = next((column for column in df.columns if "comparison" in str(column).lower() or "delta" in str(column).lower() or "model" in str(column).lower()), None)
    if label_column is None:
        return None
    row = df[df[label_column].astype(str).str.contains(label, case=False, na=False)]
    if row.empty:
        return None
    record = row.iloc[0]
    delta_col = next((lower_columns[name] for name in lower_columns if name in {"delta", "log_loss_delta", "mean_delta"}), None)
    lower_col = next((lower_columns[name] for name in lower_columns if name in {"ci_lower", "lower", "lower_95", "ci_low"}), None)
    upper_col = next((lower_columns[name] for name in lower_columns if name in {"ci_upper", "upper", "upper_95", "ci_high"}), None)
    if not all([delta_col, lower_col, upper_col]):
        return None
    try:
        return float(record[delta_col]), float(record[lower_col]), float(record[upper_col])
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
