import pandas as pd
import pytest

from dashboard.components import _actual_outcome_text, _score_text, confidence_label, percent, status_badge
from dashboard.parsers import markdown_table_after_heading
from dashboard.scenario import (
    advancement_probabilities,
    apply_scenario_adjustments,
    decimal_odds_to_implied,
    find_fixture_prediction,
    infer_probability_columns,
)


def test_markdown_table_after_heading_parses_numeric_values() -> None:
    markdown = """
## Alpha Search

| alpha | log_loss |
| --- | --- |
| 0.3 | 0.962 |
"""

    table = markdown_table_after_heading(markdown, "Alpha Search")

    assert table.loc[0, "alpha"] == 0.3
    assert table.loc[0, "log_loss"] == 0.962


def test_find_fixture_prediction_handles_project_probability_orientation() -> None:
    predictions = pd.DataFrame(
        [
            {
                "home_team": "Argentina",
                "away_team": "France",
                "p_home_loss": 0.25,
                "p_draw": 0.20,
                "p_home_win": 0.55,
            }
        ]
    )

    direct = find_fixture_prediction(predictions, "Argentina", "France")
    reverse = find_fixture_prediction(predictions, "France", "Argentina")

    assert infer_probability_columns(predictions) is not None
    assert direct["probabilities"] == (0.55, 0.20, 0.25)
    assert reverse["probabilities"] == (0.25, 0.20, 0.55)


def test_decimal_odds_to_implied_sums_to_one() -> None:
    probs = decimal_odds_to_implied(2.0, 3.5, 4.0)

    assert probs is not None
    assert abs(sum(probs) - 1.0) < 1e-9


def test_scenario_adjustments_keep_probabilities_normalized() -> None:
    adjusted = apply_scenario_adjustments((0.4, 0.25, 0.35), team_a_adjustment=0.1, team_b_adjustment=-0.05)

    assert abs(sum(adjusted) - 1.0) < 1e-9
    assert adjusted[0] > 0.4


def test_advancement_probabilities_remove_draw_mass() -> None:
    assert advancement_probabilities(0.6, 0.2) == pytest.approx((0.75, 0.25))


def test_dashboard_percent_and_confidence_handle_missing_values() -> None:
    assert percent(float("nan")) == "--"
    assert confidence_label(float("nan")) == "Unknown"


def test_finished_status_with_minute_is_not_live_badge() -> None:
    badge = status_badge("Match Finished", 90)

    assert "Match Finished" in badge
    assert "LIVE 90" not in badge


def test_live_score_does_not_render_actual_outcome() -> None:
    row = pd.Series({"status": "First Half", "home_score": 0, "away_score": 0, "actual_result": "Draw", "prediction_correct": False})

    assert _actual_outcome_text(row) == ""
    assert _score_text(row) == "Current score: 0-0"


def test_finished_score_renders_final_actual_outcome() -> None:
    row = pd.Series({"status": "Match Finished", "home_score": 0, "away_score": 0, "actual_result": "Draw", "prediction_correct": True})

    assert "Actual: Draw" in _actual_outcome_text(row)
    assert _score_text(row) == "Final: 0-0"
