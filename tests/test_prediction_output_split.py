import pandas as pd

from src.models.prediction_outputs import is_placeholder_team_name, split_prediction_outputs


def test_placeholder_names_are_detected() -> None:
    placeholders = ["1A", "2B", "3C/E/F/H/I", "W73", "L102", "TBD", "", None]
    real_teams = ["Brazil", "Spain", "DR Congo", "South Korea"]

    assert all(is_placeholder_team_name(name) for name in placeholders)
    assert not any(is_placeholder_team_name(name) for name in real_teams)


def test_prediction_outputs_are_split(tmp_path) -> None:
    cfg = {
        "paths": {
            "ensemble_predictions": str(tmp_path / "ensemble.parquet"),
            "calibrated_predictions": str(tmp_path / "missing_calibrated.parquet"),
            "match_predictions": str(tmp_path / "missing_match.parquet"),
            "poisson_predictions": str(tmp_path / "missing_poisson.parquet"),
            "worldcup_2026_prediction_input_advanced": str(tmp_path / "input.parquet"),
            "worldcup_2026_prediction_input": str(tmp_path / "missing_input.parquet"),
            "group_stage_predictions_parquet": str(tmp_path / "group.parquet"),
            "group_stage_predictions_csv": str(tmp_path / "group.csv"),
            "actual_team_match_predictions_parquet": str(tmp_path / "actual.parquet"),
            "actual_team_match_predictions_csv": str(tmp_path / "actual.csv"),
            "placeholder_fixture_predictions_parquet": str(tmp_path / "placeholder.parquet"),
            "placeholder_fixture_predictions_csv": str(tmp_path / "placeholder.csv"),
            "simulation_dynamic_predictions_parquet": str(tmp_path / "dynamic.parquet"),
            "simulation_dynamic_predictions_csv": str(tmp_path / "dynamic.csv"),
        }
    }
    predictions = pd.DataFrame(
        [
            {"match_id": "m1", "date": "2026-06-11", "home_team": "Brazil", "away_team": "Spain", "p_home_loss": 0.2, "p_draw": 0.2, "p_home_win": 0.6},
            {"match_id": "m2", "date": "2026-06-28", "home_team": "1A", "away_team": "3C/E/F/H/I", "p_home_loss": 0.3, "p_draw": 0.2, "p_home_win": 0.5},
        ]
    )
    prediction_input = pd.DataFrame(
        [
            {"match_id": "m1", "stage": "Matchday 1", "group": "Group A"},
            {"match_id": "m2", "stage": "Round of 32", "group": pd.NA},
        ]
    )
    predictions.to_parquet(cfg["paths"]["ensemble_predictions"], index=False)
    prediction_input.to_parquet(cfg["paths"]["worldcup_2026_prediction_input_advanced"], index=False)

    result = split_prediction_outputs(cfg)

    assert len(result["actual_team_match_predictions"]) == 1
    assert len(result["group_stage_predictions"]) == 1
    assert len(result["placeholder_fixture_predictions"]) == 1
    assert not result["actual_team_match_predictions"]["home_team"].map(is_placeholder_team_name).any()
    assert result["placeholder_fixture_predictions"]["away_team"].map(is_placeholder_team_name).all()
