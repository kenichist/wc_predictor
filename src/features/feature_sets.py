from __future__ import annotations

from src.features.dynamic_ratings import DYNAMIC_RATING_COLUMNS
from src.features.internal_historical_elo import INTERNAL_HISTORICAL_ELO_COLUMNS
from src.features.pi_ratings import PI_RATING_COLUMNS
from src.features.rolling_team_form import ROLLING_FEATURE_COLUMNS
from src.sources.fifa_rankings import FIFA_RANKING_MODEL_COLUMNS
from src.sources.injury_interface import INJURY_FEATURE_COLUMNS
from src.sources.player_stats_interface import PLAYER_FEATURE_COLUMNS
from src.sources.world_football_elo import WORLD_FOOTBALL_ELO_MODEL_COLUMNS
from src.sources.xg_interface import XG_FEATURE_COLUMNS


BASELINE_FEATURES = [
    *ROLLING_FEATURE_COLUMNS,
    "home_elo_pre_match",
    "away_elo_pre_match",
    "elo_diff",
    "neutral",
    "is_world_cup",
    "is_qualifier",
    "is_continental_tournament",
    "is_friendly",
    "match_weight",
]

BASIC_ELO_FEATURES = ["home_elo_pre_match", "away_elo_pre_match", "elo_diff"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "baseline": BASELINE_FEATURES,
    "basic_elo": BASIC_ELO_FEATURES,
    "rolling_form": ROLLING_FEATURE_COLUMNS,
    "fifa_rankings": FIFA_RANKING_MODEL_COLUMNS,
    "external_elo": WORLD_FOOTBALL_ELO_MODEL_COLUMNS,
    "internal_historical_elo": INTERNAL_HISTORICAL_ELO_COLUMNS,
    "dynamic_ratings": DYNAMIC_RATING_COLUMNS,
    "pi_ratings": PI_RATING_COLUMNS,
    "xg": XG_FEATURE_COLUMNS,
    "squad": PLAYER_FEATURE_COLUMNS,
    "injuries": INJURY_FEATURE_COLUMNS,
    "market": [
        "home_odds",
        "draw_odds",
        "away_odds",
        "market_home_prob",
        "market_draw_prob",
        "market_away_prob",
    ],
}

ABLATION_FEATURE_SETS: dict[str, list[str]] = {
    "v1_baseline": ["baseline"],
    "core_football_only": ["baseline", "basic_elo", "rolling_form", "dynamic_ratings", "pi_ratings"],
    "core_plus_internal_elo": ["baseline", "basic_elo", "rolling_form", "dynamic_ratings", "pi_ratings", "internal_historical_elo"],
    "v2_baseline_plus_fifa_rankings": ["baseline", "fifa_rankings"],
    "v3_plus_external_elo": ["baseline", "fifa_rankings", "external_elo"],
    "v4_plus_dynamic_ratings": ["baseline", "fifa_rankings", "external_elo", "dynamic_ratings"],
    "v5_plus_pi_ratings": ["baseline", "fifa_rankings", "external_elo", "dynamic_ratings", "pi_ratings"],
    "v6_full_rating_stack": ["baseline", "fifa_rankings", "external_elo", "dynamic_ratings", "pi_ratings"],
    "rating_stack_experimental": ["baseline", "basic_elo", "rolling_form", "fifa_rankings", "external_elo", "dynamic_ratings", "pi_ratings"],
}

FEATURE_SET_ALIASES = {
    "baseline": "v1_baseline",
    "full_football_only": "core_football_only",
    "market_assisted_optional": "core_football_only",
}

EXPERIMENTAL_FEATURE_SETS = {"rating_stack_experimental", "v6_full_rating_stack"}
EXPERIMENTAL_GROUPS = {"fifa_rankings", "external_elo", "xg", "squad", "injuries", "market"}


def resolved_feature_set_name(feature_set: str) -> str:
    return FEATURE_SET_ALIASES.get(feature_set, feature_set)


def groups_for_feature_set(feature_set: str) -> list[str]:
    name = resolved_feature_set_name(feature_set)
    groups = ABLATION_FEATURE_SETS.get(name)
    if groups is None:
        if feature_set in FEATURE_GROUPS:
            return [feature_set]
        raise KeyError(f"Unknown feature set: {feature_set}")
    return groups


def is_experimental_feature_set(feature_set: str) -> bool:
    name = resolved_feature_set_name(feature_set)
    return name in EXPERIMENTAL_FEATURE_SETS or bool(set(groups_for_feature_set(feature_set)) & EXPERIMENTAL_GROUPS)


def columns_for_feature_set(feature_set: str, available_columns: list[str] | set[str] | None = None) -> list[str]:
    groups = groups_for_feature_set(feature_set)
    columns: list[str] = []
    for group in groups:
        for column in FEATURE_GROUPS[group]:
            if column not in columns:
                columns.append(column)
    if available_columns is not None:
        available = set(available_columns)
        columns = [column for column in columns if column in available]
    return columns
