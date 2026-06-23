import pandas as pd

from src.features.feature_sets import ABLATION_FEATURE_SETS, columns_for_feature_set, groups_for_feature_set
from src.features.internal_historical_elo import INTERNAL_HISTORICAL_ELO_COLUMNS, compute_internal_historical_elo


def _cfg() -> dict:
    return {
        "internal_historical_elo": {
            "default_elo": 1500,
            "k_factor": 20,
            "home_advantage": 50,
            "goal_diff_cap": 4,
            "shootout_win_score": 0.75,
            "k_multipliers": {"friendly": 1.0, "other": 1.0},
        }
    }


def test_internal_elo_uses_pre_match_values_and_no_future_leakage() -> None:
    matches = pd.DataFrame(
        [
            {"match_id": "m2", "date": "2020-02-01", "home_team": "A", "away_team": "B", "home_score": 0, "away_score": 0, "neutral": True, "competition_type": "friendly"},
            {"match_id": "m1", "date": "2020-01-01", "home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0, "neutral": True, "competition_type": "friendly"},
        ]
    )

    result = compute_internal_historical_elo(matches, config=_cfg())
    first = result.features[result.features["match_id"].eq("m1")].iloc[0]
    second = result.features[result.features["match_id"].eq("m2")].iloc[0]

    assert first["home_internal_elo"] == 1500
    assert first["away_internal_elo"] == 1500
    assert second["home_internal_elo"] > 1500
    assert second["away_internal_elo"] < 1500


def test_internal_elo_ratings_update_after_same_date_matches() -> None:
    matches = pd.DataFrame(
        [
            {"match_id": "m1", "date": "2020-01-01", "home_team": "A", "away_team": "B", "home_score": 3, "away_score": 0, "neutral": True, "competition_type": "friendly"},
            {"match_id": "m2", "date": "2020-01-01", "home_team": "A", "away_team": "C", "home_score": 1, "away_score": 0, "neutral": True, "competition_type": "friendly"},
            {"match_id": "m3", "date": "2020-01-02", "home_team": "A", "away_team": "D", "home_score": 0, "away_score": 0, "neutral": True, "competition_type": "friendly"},
        ]
    )

    result = compute_internal_historical_elo(matches, config=_cfg())
    same_day = result.features[result.features["match_id"].isin(["m1", "m2"])]
    next_day = result.features[result.features["match_id"].eq("m3")].iloc[0]

    assert same_day["home_internal_elo"].tolist() == [1500, 1500]
    assert next_day["home_internal_elo"] > 1500


def test_internal_elo_available_for_both_teams_and_new_teams_start_default() -> None:
    matches = pd.DataFrame(
        [
            {"match_id": "m1", "date": "2020-01-01", "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0, "neutral": True, "competition_type": "friendly"},
            {"match_id": "m2", "date": "2020-02-01", "home_team": "C", "away_team": "A", "home_score": None, "away_score": None, "neutral": True, "competition_type": "friendly"},
        ]
    )

    result = compute_internal_historical_elo(matches, config=_cfg())
    second = result.features[result.features["match_id"].eq("m2")].iloc[0]

    assert second[INTERNAL_HISTORICAL_ELO_COLUMNS].notna().all()
    assert second["home_internal_elo"] == 1500
    assert second["away_internal_elo"] > 1500


def test_internal_elo_neutral_venue_does_not_apply_home_advantage() -> None:
    matches = pd.DataFrame(
        [
            {"match_id": "m1", "date": "2020-01-01", "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0, "neutral": True, "competition_type": "friendly"},
            {"match_id": "m2", "date": "2020-01-02", "home_team": "A", "away_team": "B", "home_score": 0, "away_score": 0, "neutral": True, "competition_type": "friendly"},
        ]
    )
    non_neutral = matches.copy()
    non_neutral.loc[0, "neutral"] = False

    neutral_result = compute_internal_historical_elo(matches, config=_cfg(), home_advantage=100)
    non_neutral_result = compute_internal_historical_elo(non_neutral, config=_cfg(), home_advantage=100)
    neutral_second = neutral_result.features[neutral_result.features["match_id"].eq("m2")].iloc[0]
    non_neutral_second = non_neutral_result.features[non_neutral_result.features["match_id"].eq("m2")].iloc[0]

    assert neutral_second["home_internal_elo"] == 1510
    assert non_neutral_second["home_internal_elo"] < neutral_second["home_internal_elo"]


def test_internal_elo_penalty_shootout_handling_if_available() -> None:
    matches = pd.DataFrame(
        [
            {"match_id": "m1", "date": "2020-01-01", "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 1, "neutral": True, "competition_type": "world_cup"},
            {"match_id": "m2", "date": "2020-01-02", "home_team": "A", "away_team": "B", "home_score": 0, "away_score": 0, "neutral": True, "competition_type": "friendly"},
        ]
    )
    shootouts = pd.DataFrame([{"date": "2020-01-01", "home_team": "A", "away_team": "B", "winner": "A"}])

    result = compute_internal_historical_elo(matches, config=_cfg(), shootouts=shootouts, k_multipliers={"world_cup": 1.0})
    second = result.features[result.features["match_id"].eq("m2")].iloc[0]

    assert second["home_internal_elo"] == 1505
    assert second["away_internal_elo"] == 1495


def test_internal_elo_coverage_is_near_complete() -> None:
    matches = pd.DataFrame(
        [
            {"match_id": "m1", "date": "2020-01-01", "home_team": "A", "away_team": "B", "home_score": 1, "away_score": 0, "neutral": True, "competition_type": "friendly"},
            {"match_id": "m2", "date": "2020-02-01", "home_team": "C", "away_team": "D", "home_score": None, "away_score": None, "neutral": True, "competition_type": "friendly"},
        ]
    )

    result = compute_internal_historical_elo(matches, config=_cfg())
    coverage = result.features[INTERNAL_HISTORICAL_ELO_COLUMNS].notna().all(axis=1).mean()

    assert coverage == 1.0


def test_internal_elo_ablation_can_compare_against_core() -> None:
    assert list(ABLATION_FEATURE_SETS).index("core_plus_internal_elo") == list(ABLATION_FEATURE_SETS).index("core_football_only") + 1
    assert "internal_historical_elo" in groups_for_feature_set("core_plus_internal_elo")
    assert "home_internal_elo" in columns_for_feature_set("core_plus_internal_elo")
