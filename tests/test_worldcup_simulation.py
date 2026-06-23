import random

import pandas as pd

from src.config import load_config
from src.simulation.knockout_stage import knockout_winner
from src.simulation.worldcup_simulator import (
    _build_group_context,
    _compute_group_rankings,
    _is_placeholder_team,
    _load_worldcup_schedule,
    _overlay_completed_group_scores,
    _resolve_round_of_32,
    _score_for_match,
    _tournament_teams,
    simulate_worldcup,
)


def test_completed_match_score_is_preserved_in_simulation() -> None:
    match = {"match_id": "m1", "home_score": 2, "away_score": 1}

    score = _score_for_match(match, predictions=[], rng=random.Random(1))

    assert score == (2, 1)


def test_completed_worldcup_group_fixture_is_overlaid_from_real_scores() -> None:
    config = load_config()
    schedule = _load_worldcup_schedule(config)

    overlaid = _overlay_completed_group_scores(schedule["group_matches"], config)
    mexico = [
        match
        for match in overlaid
        if match["home_team"] == "Mexico" and match["away_team"] == "South Africa"
    ][0]

    assert mexico["home_score"] == 2
    assert mexico["away_score"] == 0
    assert mexico["status"] == "FINISHED"


def test_knockout_match_always_produces_winner() -> None:
    winner = knockout_winner("A", "B", 0.1, 0.8, 0.1, rng=random.Random(1))

    assert winner in {"A", "B"}


def test_worldcup_2026_schedule_has_48_unique_teams() -> None:
    schedule = _load_worldcup_schedule(load_config())

    teams = _tournament_teams(schedule["group_matches"])

    assert len(teams) == 48
    assert "DR Congo" in teams
    assert "Congo DR" not in teams
    assert not any(_is_placeholder_team(team) for team in teams)


def test_round_of_32_resolution_has_32_teams() -> None:
    schedule = _load_worldcup_schedule(load_config())
    simulated_scores = {match["match_id"]: (1, 0) for match in schedule["group_matches"]}
    rankings = _compute_group_rankings(schedule["group_matches"], simulated_scores, random.Random(2))
    context = _build_group_context(rankings)

    round_of_32 = _resolve_round_of_32(schedule["bracket_matches"]["round_of_32"], context)

    assert len(round_of_32) == 32
    assert len(set(round_of_32)) == 32
    assert not any(_is_placeholder_team(team) for team in round_of_32)


def test_worldcup_2026_stage_probability_sums(tmp_path) -> None:
    results, _ = simulate_worldcup(n_sims=20, config=_simulation_config(tmp_path), seed=3)

    assert len(results) == 48
    expected = {
        "champion_probability": 1,
        "final_probability": 2,
        "semifinal_probability": 4,
        "quarterfinal_probability": 8,
        "round_of_16_probability": 16,
        "round_of_32_probability": 32,
    }
    for column, target in expected.items():
        assert abs(results[column].sum() - target) <= 1e-6


def test_worldcup_2026_elimination_stage_counts_and_no_placeholders(tmp_path) -> None:
    _, stage_df = simulate_worldcup(n_sims=1, config=_simulation_config(tmp_path), seed=4)

    assert int((stage_df["round_of_32_probability"] > 0).sum()) == 32
    assert int((stage_df["round_of_16_probability"] > 0).sum()) == 16
    assert int((stage_df["quarterfinal_probability"] > 0).sum()) == 8
    assert int((stage_df["semifinal_probability"] > 0).sum()) == 4
    assert int((stage_df["final_probability"] > 0).sum()) == 2
    assert int((stage_df["champion_probability"] > 0).sum()) == 1
    assert not any(_is_placeholder_team(team) for team in stage_df["team"])


def _simulation_config(tmp_path):
    cfg = load_config()
    paths = cfg["paths"].copy()
    paths["simulation_results"] = str(tmp_path / "worldcup_2026_simulation_results.csv")
    paths["simulation_stage_probabilities"] = str(tmp_path / "worldcup_2026_stage_probabilities.csv")
    paths["simulation_bracket_sample"] = str(tmp_path / "worldcup_2026_bracket_sample.json")
    paths["simulation_metadata"] = str(tmp_path / "worldcup_2026_simulation_metadata.json")
    paths["simulation_dynamic_predictions_parquet"] = str(tmp_path / "simulation_dynamic_predictions.parquet")
    paths["simulation_dynamic_predictions_csv"] = str(tmp_path / "simulation_dynamic_predictions.csv")
    cfg["paths"] = paths
    return cfg
