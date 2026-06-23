from src.features.feature_sets import ABLATION_FEATURE_SETS, columns_for_feature_set, groups_for_feature_set
from src.sources.fifa_rankings import FIFA_RANKING_MODEL_COLUMNS
from src.sources.injury_interface import INJURY_FEATURE_COLUMNS
from src.sources.player_stats_interface import PLAYER_FEATURE_COLUMNS
from src.sources.world_football_elo import WORLD_FOOTBALL_ELO_MODEL_COLUMNS
from src.sources.xg_interface import XG_FEATURE_COLUMNS


def test_ablation_runner_defines_all_versions() -> None:
    expected = [
        "v1_baseline",
        "core_football_only",
        "core_plus_internal_elo",
        "v2_baseline_plus_fifa_rankings",
        "v3_plus_external_elo",
        "v4_plus_dynamic_ratings",
        "v5_plus_pi_ratings",
        "v6_full_rating_stack",
        "rating_stack_experimental",
    ]
    assert list(ABLATION_FEATURE_SETS) == expected
    assert "elo_diff" in columns_for_feature_set("v1_baseline")
    assert "home_fifa_rank" in columns_for_feature_set("v2_baseline_plus_fifa_rankings")
    assert "home_external_elo" in columns_for_feature_set("v3_plus_external_elo")
    assert "home_dynamic_rating_pre" in columns_for_feature_set("v4_plus_dynamic_ratings")
    assert "home_internal_elo" in columns_for_feature_set("core_plus_internal_elo")


def test_core_football_only_excludes_sparse_external_groups() -> None:
    columns = set(columns_for_feature_set("core_football_only"))
    excluded = set(FIFA_RANKING_MODEL_COLUMNS) | set(WORLD_FOOTBALL_ELO_MODEL_COLUMNS) | set(XG_FEATURE_COLUMNS)
    excluded |= set(PLAYER_FEATURE_COLUMNS) | set(INJURY_FEATURE_COLUMNS)
    excluded |= {"home_odds", "draw_odds", "away_odds", "market_home_prob", "market_draw_prob", "market_away_prob"}

    assert {"baseline", "basic_elo", "rolling_form", "dynamic_ratings", "pi_ratings"} <= set(groups_for_feature_set("core_football_only"))
    assert columns.isdisjoint(excluded)
    assert columns_for_feature_set("full_football_only") == columns_for_feature_set("core_football_only")
    assert "home_fifa_rank" in columns_for_feature_set("rating_stack_experimental")
    assert "internal_historical_elo" in set(groups_for_feature_set("core_plus_internal_elo"))
