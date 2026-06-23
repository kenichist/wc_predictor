from __future__ import annotations

import pandas as pd

from src.backtesting.worldcup_backtest import run_worldcup_backtests
from src.features.external_feature_joiner import add_market_features
from src.sources.market_odds import ODDS_CONVERSION_METHODS, decimal_odds_to_probabilities
from tests.test_worldcup_backtest import _synthetic_backtest_frame, _tmp_config


def test_decimal_odds_convert_to_probabilities_summing_to_one() -> None:
    converted = decimal_odds_to_probabilities(pd.DataFrame([{"home_odds": 2.0, "draw_odds": 4.0, "away_odds": 4.0}]))

    total = converted[["market_home_prob", "market_draw_prob", "market_away_prob"]].iloc[0].sum()
    assert abs(total - 1.0) < 1e-12
    assert converted.loc[0, "market_home_prob"] == 0.5


def test_all_odds_conversion_methods_sum_to_one() -> None:
    odds = pd.DataFrame([{"home_odds": 1.8, "draw_odds": 3.5, "away_odds": 5.2}])

    for method in ODDS_CONVERSION_METHODS:
        converted = decimal_odds_to_probabilities(odds, method=method)
        total = converted[["market_home_prob", "market_draw_prob", "market_away_prob"]].iloc[0].sum()
        assert abs(total - 1.0) < 1e-9


def test_market_join_uses_team_name_normalization(tmp_path) -> None:
    odds_path = tmp_path / "market_odds.csv"
    pd.DataFrame(
        [
            {
                "date": "2014-06-15",
                "home_team": "Argentina",
                "away_team": "Bosnia & Herzegovina",
                "home_odds": 1.31,
                "draw_odds": 5.56,
                "away_odds": 10.24,
                "bookmaker": "test",
                "source": "unit",
                "updated_at": "2026-06-23",
            }
        ]
    ).to_csv(odds_path, index=False)
    matches = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2014-06-15",
                "home_team": "Argentina",
                "away_team": "Bosnia and Herzegovina",
            }
        ]
    )

    result = add_market_features(matches, odds_path)

    assert pd.notna(result.loc[0, "market_home_prob"])


def test_market_join_recovers_one_day_fixture_shift_and_dr_congo_alias(tmp_path) -> None:
    odds_path = tmp_path / "market_odds.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-18",
                "home_team": "Portugal",
                "away_team": "D.R. Congo",
                "home_odds": 1.5,
                "draw_odds": 4.0,
                "away_odds": 7.0,
                "bookmaker": "test",
                "source": "unit",
                "updated_at": "2026-06-23",
            }
        ]
    ).to_csv(odds_path, index=False)
    matches = pd.DataFrame([{"match_id": "m1", "date": "2026-06-17", "home_team": "Portugal", "away_team": "DR Congo"}])

    result = add_market_features(matches, odds_path)

    assert pd.notna(result.loc[0, "market_home_prob"])


def test_incomplete_odds_coverage_is_reported(tmp_path) -> None:
    frame = _synthetic_backtest_frame()
    mask = frame["tournament"].eq("FIFA World Cup")
    frame.loc[mask, ["market_home_prob", "market_draw_prob", "market_away_prob"]] = pd.NA
    first_world_cup_index = frame[mask].index[0]
    frame.loc[first_world_cup_index, ["market_home_prob", "market_draw_prob", "market_away_prob"]] = [0.5, 0.25, 0.25]

    _, metrics, report_path = run_worldcup_backtests(
        model_name="hist_gradient_boosting",
        feature_set="v1_baseline",
        config=_tmp_config(tmp_path),
        feature_frame=frame,
        years=[2010],
    )

    bookmaker = metrics[metrics["model_name"].eq("bookmaker_odds_only")].iloc[0]
    assert bookmaker["n_matches"] == 1
    assert bookmaker["odds_coverage_rate"] == 0.25
    assert "Missing Odds Matches" in report_path.read_text(encoding="utf-8")


def test_bookmaker_and_market_blend_appear_when_odds_exist(tmp_path) -> None:
    _, metrics, _ = run_worldcup_backtests(
        model_name="hist_gradient_boosting",
        feature_set="v1_baseline",
        config=_tmp_config(tmp_path),
        feature_frame=_synthetic_backtest_frame(),
        years=[2010],
    )

    assert "bookmaker_odds_only" in set(metrics["model_name"])
    assert "market_blend" in set(metrics["model_name"])


def test_empty_market_odds_file_does_not_crash(tmp_path) -> None:
    odds_path = tmp_path / "market_odds.csv"
    pd.DataFrame(columns=["date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds"]).to_csv(odds_path, index=False)
    matches = pd.DataFrame([{"match_id": "m1", "date": "2026-06-11", "home_team": "Mexico", "away_team": "South Africa"}])

    result = add_market_features(matches, odds_path)

    assert result["market_home_prob"].isna().all()
