import pandas as pd
import pytest

from dashboard.components_risk import add_compact_risk_columns, compact_risk_summary
from dashboard.risk import (
    calculate_match_outcome_risk,
    calculate_risk_metrics,
    decimal_odds_to_market_probabilities,
    kelly_fraction,
    model_probabilities_for_match,
    monte_carlo_bankroll_simulation,
    raw_implied_probability,
    round_down_idr,
)
from dashboard.scenario import advancement_probabilities


def test_decimal_odds_implied_probabilities_sum_to_one() -> None:
    probs = decimal_odds_to_market_probabilities(2.0, 3.5, 4.0)

    assert probs is not None
    assert sum(probs) == pytest.approx(1.0)


def test_raw_implied_probability_uses_decimal_odds_inverse() -> None:
    assert raw_implied_probability(2.5) == pytest.approx(0.4)


def test_negative_ev_gives_zero_stake() -> None:
    result = calculate_risk_metrics(
        model_prob=0.40,
        market_prob=0.50,
        decimal_odds=2.0,
        bankroll_idr=1_000_000,
        max_stake_cap_idr=100_000,
        max_stake_percent=0.05,
        max_daily_loss_idr=100_000,
        enable_hypothetical_simulation=True,
    )

    assert result["capped_hypothetical_stake_idr"] == 0
    assert result["risk_label"] == "No positive edge"


def test_positive_ev_gives_positive_kelly() -> None:
    assert kelly_fraction(0.60, 2.10) > 0


def test_fractional_kelly_is_lower_than_full_kelly() -> None:
    result = calculate_risk_metrics(
        model_prob=0.60,
        market_prob=0.45,
        decimal_odds=2.10,
        bankroll_idr=1_000_000,
        risk_profile="Very conservative",
        max_stake_cap_idr=1_000_000,
        max_stake_percent=1.0,
        max_daily_loss_idr=1_000_000,
        enable_hypothetical_simulation=True,
    )

    assert result["fractional_kelly_fraction"] < result["full_kelly_fraction"]


def test_stake_cap_works() -> None:
    result = calculate_risk_metrics(
        model_prob=0.70,
        market_prob=0.45,
        decimal_odds=2.20,
        bankroll_idr=1_000_000,
        risk_profile="Aggressive",
        max_stake_cap_idr=20_000,
        max_stake_percent=0.50,
        max_daily_loss_idr=1_000_000,
        enable_hypothetical_simulation=True,
    )

    assert result["capped_hypothetical_stake_idr"] <= 20_000


def test_stake_never_exceeds_bankroll() -> None:
    result = calculate_risk_metrics(
        model_prob=0.90,
        market_prob=0.30,
        decimal_odds=5.00,
        bankroll_idr=50_000,
        risk_profile="Aggressive",
        max_stake_cap_idr=1_000_000,
        max_stake_percent=10.0,
        max_daily_loss_idr=1_000_000,
        enable_hypothetical_simulation=True,
    )

    assert result["capped_hypothetical_stake_idr"] <= 50_000


def test_knockout_advancement_removes_draw_mass_if_reused() -> None:
    assert advancement_probabilities(0.45, 0.30) == pytest.approx((0.6, 0.4))


def test_monte_carlo_returns_expected_columns() -> None:
    result = monte_carlo_bankroll_simulation(
        initial_bankroll_idr=1_000_000,
        stake_idr=10_000,
        model_prob=0.55,
        decimal_odds=2.0,
        number_of_bets=10,
        number_of_simulations=200,
        confidence_level=0.90,
        seed=7,
    )

    expected = {
        "median_final_bankroll",
        "mean_final_bankroll",
        "p5_final_bankroll",
        "p1_final_bankroll",
        "probability_losing_money",
        "probability_losing_more_than_10pct",
        "probability_losing_more_than_25pct",
        "probability_losing_more_than_50pct",
        "var_95",
        "cvar_95",
        "var_confidence_level",
        "var_loss",
        "cvar_loss",
        "risk_of_ruin",
    }
    assert expected.issubset(result.summary.columns)
    assert result.summary.loc[0, "var_confidence_level"] == pytest.approx(0.90)
    assert {"simulation", "final_bankroll"}.issubset(result.final_bankrolls.columns)
    assert {"simulation", "max_drawdown"}.issubset(result.drawdowns.columns)


def test_idr_rounding_rounds_down_to_nearest_1000() -> None:
    assert round_down_idr(12_999) == 12_000


def test_probability_rows_remain_valid() -> None:
    live_row = pd.Series(
        {
            "fixture_id": "m1",
            "date": "2026-06-11",
            "home_team": "Argentina",
            "away_team": "France",
            "live_home_prob": 0.50,
            "live_draw_prob": 0.25,
            "live_away_prob": 0.25,
        }
    )
    probs = model_probabilities_for_match(live_row)

    assert probs is not None
    assert sum(probs) == pytest.approx(1.0)


def test_match_outcome_risk_uses_live_odds() -> None:
    match = pd.Series(
        {
            "fixture_id": "m1",
            "date": "2026-06-11",
            "home_team": "Argentina",
            "away_team": "France",
            "live_home_prob": 0.60,
            "live_draw_prob": 0.20,
            "live_away_prob": 0.20,
        }
    )
    live_odds = pd.DataFrame(
        [
            {
                "fixture_id": "m1",
                "date": "2026-06-11",
                "home_team": "Argentina",
                "away_team": "France",
                "home_odds": 2.1,
                "draw_odds": 3.4,
                "away_odds": 4.2,
            }
        ]
    )

    result = calculate_match_outcome_risk(
        match,
        "Home win",
        live_odds=live_odds,
        bankroll_idr=1_000_000,
        risk_profile="Very conservative",
        max_stake_cap_idr=100_000,
        max_stake_percent=0.05,
        max_daily_loss_idr=100_000,
        enable_hypothetical_simulation=True,
    )

    assert result["decimal_odds"] == 2.1
    assert result["model_prob"] == pytest.approx(0.60)


def test_compact_risk_summary_stays_non_actionable() -> None:
    match = pd.Series(
        {
            "fixture_id": "m1",
            "date": "2026-06-11",
            "home_team": "Argentina",
            "away_team": "France",
            "live_home_prob": 0.60,
            "live_draw_prob": 0.20,
            "live_away_prob": 0.20,
        }
    )
    frames = {
        "live_odds": pd.DataFrame(
            [
                {
                    "fixture_id": "m1",
                    "date": "2026-06-11",
                    "home_team": "Argentina",
                    "away_team": "France",
                    "home_odds": 2.1,
                    "draw_odds": 3.4,
                    "away_odds": 4.2,
                }
            ]
        )
    }

    summary = compact_risk_summary(match, frames)

    assert summary["status"] == "Positive model edge"
    assert "Hypothetical stake remains 0 IDR" in summary["message"]
    assert "Not betting advice" in summary["message"]


def test_compact_risk_summary_explains_missing_market_odds() -> None:
    match = pd.Series(
        {
            "fixture_id": "m1",
            "date": "2026-06-11",
            "home_team": "Argentina",
            "away_team": "France",
            "live_home_prob": 0.60,
            "live_draw_prob": 0.20,
            "live_away_prob": 0.20,
        }
    )

    summary = compact_risk_summary(match, {})

    assert summary["status"] == "No data"
    assert summary["has_model_probs"] is True
    assert summary["has_market_odds"] is False
    assert "market odds are missing" in summary["message"]


def test_compact_risk_columns_support_ev_sorting() -> None:
    matches = pd.DataFrame(
        [
            {
                "fixture_id": "m1",
                "date": "2026-06-11",
                "home_team": "Argentina",
                "away_team": "France",
                "live_home_prob": 0.60,
                "live_draw_prob": 0.20,
                "live_away_prob": 0.20,
            },
            {
                "fixture_id": "m2",
                "date": "2026-06-12",
                "home_team": "Brazil",
                "away_team": "Spain",
                "live_home_prob": 0.52,
                "live_draw_prob": 0.24,
                "live_away_prob": 0.24,
            },
        ]
    )
    frames = {
        "live_odds": pd.DataFrame(
            [
                {"fixture_id": "m1", "home_odds": 2.1, "draw_odds": 3.4, "away_odds": 4.2},
                {"fixture_id": "m2", "home_odds": 1.7, "draw_odds": 3.5, "away_odds": 5.0},
            ]
        )
    }

    annotated = add_compact_risk_columns(matches, frames)
    sorted_view = annotated.sort_values("best_ev_per_idr", ascending=False, na_position="last")

    assert {"best_ev_per_idr", "best_ev_outcome", "has_positive_ev"}.issubset(annotated.columns)
    assert sorted_view.iloc[0]["fixture_id"] == "m1"
    assert annotated["has_positive_ev"].any()
