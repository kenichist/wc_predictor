from __future__ import annotations

from typing import Any

import pandas as pd


def percent(value: Any) -> str:
    try:
        if pd.isna(value):
            return "--"
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "--"


def status_badge(status: Any, minute: Any = None) -> str:
    text = str(status or "Upcoming")
    lower = text.lower()
    finished = any(term in lower for term in ["finished", "full"])
    upcoming = any(term in lower for term in ["not started", "scheduled", "upcoming", "time to be defined"])
    live = not finished and (any(term in lower for term in ["live", "first", "second", "half", "extra", "penalty"]) or (_is_number(minute) and not upcoming))
    label = f"LIVE {int(float(minute))}'" if live and _is_number(minute) else text
    klass = "status-badge live-badge" if live else "status-badge"
    return f'<span class="{klass}">{label}</span>'


def render_match_card(row: pd.Series, *, key_prefix: str = "match") -> None:
    import streamlit as st

    home = row.get("home_team", "Home")
    away = row.get("away_team", "Away")
    probs = [
        ("Home Win", _first_numeric(row.get("live_home_prob"), row.get("base_home_prob"), row.get("market_home_prob"))),
        ("Draw", _first_numeric(row.get("live_draw_prob"), row.get("base_draw_prob"), row.get("market_draw_prob"))),
        ("Away Win", _first_numeric(row.get("live_away_prob"), row.get("base_away_prob"), row.get("market_away_prob"))),
    ]
    best_index = _best_probability_index([value for _, value in probs])
    confidence = _first_numeric(row.get("confidence"), max(value for _, value in probs if value is not None) if any(value is not None for _, value in probs) else None)
    market_available = pd.notna(row.get("market_home_prob")) and pd.notna(row.get("market_draw_prob")) and pd.notna(row.get("market_away_prob"))
    score = _score_text(row)
    actual = _actual_outcome_text(row)
    badges = [
        status_badge(row.get("status"), row.get("minute")),
        f'<span class="status-badge">Confidence {confidence_label(confidence)}</span>',
        f'<span class="status-badge">Market {"Available" if market_available else "Missing"}</span>',
    ]
    chips = []
    for index, (label, value) in enumerate(probs):
        best_class = " best" if index == best_index else ""
        chips.append(f'<div class="prob-chip{best_class}"><span>{label}</span><strong>{percent(value)}</strong></div>')
    st.markdown(
        f"""
<div class="match-card">
  <div class="match-head">
    <div>
      <div class="teams">{home} vs {away}</div>
      <div class="muted-note">{row.get('date', '')} {score}</div>
    </div>
    <div>{" ".join(badges)}</div>
  </div>
  <div class="prob-grid">{"".join(chips)}</div>
  <div class="factor-row">Most likely: {row.get('predicted_result', 'Unavailable')}</div>
  {actual}
</div>
""",
        unsafe_allow_html=True,
    )


def confidence_label(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Unknown"
    if pd.isna(number):
        return "Unknown"
    if number >= 0.60:
        return "High"
    if number >= 0.45:
        return "Medium"
    return "Low"


def render_probability_comparison(row: pd.Series) -> None:
    import streamlit as st

    groups = [
        ("Base model", ("base_home_prob", "base_draw_prob", "base_away_prob")),
        ("Market", ("market_home_prob", "market_draw_prob", "market_away_prob")),
        ("Live adjusted", ("live_home_prob", "live_draw_prob", "live_away_prob")),
    ]
    for title, cols in groups:
        st.subheader(title)
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{row.get('home_team', 'Home')} win", percent(row.get(cols[0])))
        c2.metric("Draw", percent(row.get(cols[1])))
        c3.metric(f"{row.get('away_team', 'Away')} win", percent(row.get(cols[2])))


def _best_probability_index(values: list[Any]) -> int | None:
    numeric = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            numeric.append(float("nan"))
    if all(pd.isna(value) for value in numeric):
        return None
    return int(pd.Series(numeric).idxmax())


def _score_text(row: pd.Series) -> str:
    home_score = row.get("home_score")
    away_score = row.get("away_score")
    if pd.notna(home_score) and pd.notna(away_score):
        label = "Final" if _is_finished_status(row.get("status")) else "Current score"
        return f"{label}: {int(float(home_score))}-{int(float(away_score))}"
    return ""


def _actual_outcome_text(row: pd.Series) -> str:
    if not _is_finished_status(row.get("status")):
        return ""
    actual = row.get("actual_result")
    if pd.isna(actual):
        return ""
    correctness = _correctness_label(row.get("prediction_correct"))
    suffix = f" | Prediction: {correctness}" if correctness else ""
    return f'<div class="factor-row">Actual: {actual}{suffix}</div>'


def _correctness_label(value: Any) -> str:
    if isinstance(value, bool):
        return "Correct" if value else "Missed"
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "Correct"
    if text in {"false", "0", "no"}:
        return "Missed"
    return ""


def _is_finished_status(status: Any) -> bool:
    text = str(status or "").lower()
    return any(term in text for term in ["finished", "full time", "fulltime", "ft"])


def _is_number(value: Any) -> bool:
    try:
        return pd.notna(value) and str(value).strip() != "" and float(value) >= 0
    except (TypeError, ValueError):
        return False


def _first_numeric(*values: Any) -> float | None:
    for value in values:
        try:
            if pd.isna(value):
                continue
            number = float(value)
        except (TypeError, ValueError):
            continue
        if pd.notna(number):
            return number
    return None
