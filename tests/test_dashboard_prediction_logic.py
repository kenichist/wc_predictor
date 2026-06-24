import pandas as pd
import pytest

from dashboard.prediction_logic import (
    apply_live_adjustments,
    blend_probabilities,
    build_live_predictions,
    find_base_probabilities,
    odds_to_implied_probabilities,
)
from dashboard.scenario import advancement_probabilities


def test_odds_to_implied_probabilities_sum_to_one() -> None:
    probs = odds_to_implied_probabilities(2.0, 3.5, 4.0)

    assert probs is not None
    assert sum(probs) == pytest.approx(1.0)


def test_knockout_advancement_removes_draw_mass() -> None:
    assert advancement_probabilities(0.45, 0.30) == pytest.approx((0.6, 0.4))


def test_probability_blending_sums_to_one() -> None:
    blended = blend_probabilities((0.5, 0.25, 0.25), (0.4, 0.30, 0.30), alpha=0.3)

    assert blended is not None
    assert sum(blended) == pytest.approx(1.0)
    assert blended[0] == pytest.approx(0.43)


def test_injury_adjustment_keeps_probabilities_valid() -> None:
    adjusted = apply_live_adjustments((0.5, 0.25, 0.25), home_adjustment=-0.05, away_adjustment=0.03)

    assert sum(adjusted) == pytest.approx(1.0)
    assert all(0 <= value <= 1 for value in adjusted)


def test_reverse_fixture_probability_swapping() -> None:
    predictions = pd.DataFrame(
        [
            {
                "date": "2026-06-11",
                "home_team": "Argentina",
                "away_team": "France",
                "p_home_win": 0.50,
                "p_draw": 0.25,
                "p_home_loss": 0.25,
            }
        ]
    )

    probs = find_base_probabilities(predictions, "2026-06-11", "France", "Argentina")

    assert probs == pytest.approx((0.25, 0.25, 0.50))


def test_build_live_predictions_applies_high_importance_injury_factor() -> None:
    fixtures = pd.DataFrame(
        [
            {
                "fixture_id": "m1",
                "date": "2026-06-11",
                "home_team": "Argentina",
                "away_team": "France",
                "status": "Upcoming",
                "stage": "Group stage",
            }
        ]
    )
    official = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2026-06-11",
                "home_team": "Argentina",
                "away_team": "France",
                "p_home_win": 0.50,
                "p_draw": 0.25,
                "p_home_loss": 0.25,
            }
        ]
    )
    injuries = pd.DataFrame(
        [
            {
                "fixture_id": "m1",
                "date": "2026-06-11",
                "team": "Argentina",
                "player": "Key Player",
                "status": "injured",
                "reason": "test",
                "importance": "high",
            }
        ]
    )

    live, factors, summary = build_live_predictions(
        fixtures=fixtures,
        official_predictions=official,
        injuries=injuries,
    )

    assert summary["LIVE_PREDICTION_ROWS"] == 1
    assert live.loc[0, "live_home_prob"] < live.loc[0, "base_home_prob"]
    assert len(factors) == 1


def test_unknown_nan_injury_importance_does_not_nan_live_probabilities() -> None:
    fixtures = pd.DataFrame(
        [{"fixture_id": "m1", "date": "2026-06-11", "home_team": "Argentina", "away_team": "France", "status": "Upcoming"}]
    )
    official = pd.DataFrame(
        [{"date": "2026-06-11", "home_team": "Argentina", "away_team": "France", "p_home_win": 0.5, "p_draw": 0.25, "p_home_loss": 0.25}]
    )
    injuries = pd.DataFrame(
        [{"fixture_id": "m1", "date": "2026-06-11", "team": "Argentina", "player": "Unknown", "status": "injured", "importance": float("nan")}]
    )

    live, factors, _ = build_live_predictions(fixtures=fixtures, official_predictions=official, injuries=injuries)

    assert live[["live_home_prob", "live_draw_prob", "live_away_prob"]].notna().all(axis=None)
    assert factors.empty


def test_build_live_predictions_records_finished_actual_outcome() -> None:
    fixtures = pd.DataFrame(
        [{"fixture_id": "m1", "date": "2026-06-11", "home_team": "Argentina", "away_team": "France", "status": "Match Finished"}]
    )
    official = pd.DataFrame(
        [{"date": "2026-06-11", "home_team": "Argentina", "away_team": "France", "p_home_win": 0.60, "p_draw": 0.20, "p_home_loss": 0.20}]
    )
    scores = pd.DataFrame(
        [{"fixture_id": "m1", "date": "2026-06-11", "home_team": "Argentina", "away_team": "France", "status": "Match Finished", "home_score": 0, "away_score": 1}]
    )

    live, _, _ = build_live_predictions(fixtures=fixtures, official_predictions=official, scores=scores)

    assert live.loc[0, "predicted_result"] == "Argentina win"
    assert live.loc[0, "actual_result"] == "France win"
    assert live.loc[0, "prediction_correct"] == False
    assert live.loc[0, "home_score"] == 0
    assert live.loc[0, "away_score"] == 1


def test_build_live_predictions_does_not_mark_live_score_as_actual_outcome() -> None:
    fixtures = pd.DataFrame(
        [{"fixture_id": "m1", "date": "2026-06-11", "home_team": "Argentina", "away_team": "France", "status": "First Half"}]
    )
    official = pd.DataFrame(
        [{"date": "2026-06-11", "home_team": "Argentina", "away_team": "France", "p_home_win": 0.60, "p_draw": 0.20, "p_home_loss": 0.20}]
    )
    scores = pd.DataFrame(
        [{"fixture_id": "m1", "date": "2026-06-11", "home_team": "Argentina", "away_team": "France", "status": "First Half", "minute": 53, "home_score": 0, "away_score": 0}]
    )

    live, _, _ = build_live_predictions(fixtures=fixtures, official_predictions=official, scores=scores)

    assert pd.isna(live.loc[0, "actual_result"])
    assert pd.isna(live.loc[0, "prediction_correct"])
    assert live.loc[0, "home_score"] == 0
    assert live.loc[0, "away_score"] == 0
