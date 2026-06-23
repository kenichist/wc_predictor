from __future__ import annotations

import pandas as pd

from src import cli
from src.backtesting.worldcup_tournament_simulation import run_worldcup_tournament_simulation_backtest
from src.features.rolling_team_form import ROLLING_FEATURE_COLUMNS


def test_worldcup_tournament_simulation_backtest_outputs_metrics_and_report(tmp_path) -> None:
    results, report_path = run_worldcup_tournament_simulation_backtest(
        model_name="hist_gradient_boosting",
        feature_set="v1_baseline",
        n_sims=20,
        config=_tmp_config(tmp_path),
        feature_frame=_synthetic_tournament_frame(),
        years=[2010],
        seed=7,
    )

    assert (tmp_path / "backtests" / "worldcup_tournament_simulation_backtest.csv").exists()
    assert report_path.exists()
    assert set(results["model_name"]) == {"final_model", "elo_only"}
    assert {
        "champion_probability_assigned",
        "actual_finalists_probability",
        "semifinalist_probability_recall",
        "round_of_16_brier_score",
        "group_qualification_accuracy",
        "group_match_log_loss",
    } <= set(results.columns)


def test_worldcup_tournament_simulation_cli_command_is_registered() -> None:
    args = cli.build_parser().parse_args(
        ["backtest-worldcup-tournaments", "--model", "catboost", "--feature-set", "core_football_only", "--n-sims", "25"]
    )

    assert args.command == "backtest-worldcup-tournaments"
    assert args.n_sims == 25


def _synthetic_tournament_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    teams = {group: [f"{group}{idx}" for idx in range(1, 5)] for group in "ABCDEFGH"}
    classes = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]
    for index, actual_class in enumerate(classes):
        rows.append(_row(index=index, date=f"2009-{(index % 12) + 1:02d}-01", home=f"T{index % 6}", away=f"T{(index + 1) % 6}", actual_class=actual_class, tournament="Friendly"))

    match_index = 100
    for group, group_teams in teams.items():
        pairings = [(0, 3), (1, 2), (0, 2), (1, 3), (0, 1), (2, 3)]
        for home_idx, away_idx in pairings:
            actual_class = 2 if home_idx < away_idx else 0
            rows.append(
                _row(
                    index=match_index,
                    date=f"2010-06-{10 + ((match_index - 100) // 8):02d}",
                    home=group_teams[home_idx],
                    away=group_teams[away_idx],
                    actual_class=actual_class,
                    tournament="FIFA World Cup",
                )
            )
            match_index += 1

    r16_pairs = [("A1", "B2"), ("C1", "D2"), ("E1", "F2"), ("G1", "H2"), ("B1", "A2"), ("D1", "C2"), ("F1", "E2"), ("H1", "G2")]
    qf_pairs = [("A1", "C1"), ("E1", "G1"), ("B1", "D1"), ("F1", "H1")]
    sf_pairs = [("A1", "E1"), ("B1", "F1")]
    third_pair = [("E1", "F1")]
    final_pair = [("A1", "B1")]
    for home, away in [*r16_pairs, *qf_pairs, *sf_pairs, *third_pair, *final_pair]:
        rows.append(_row(index=match_index, date=f"2010-07-{1 + (match_index - 148):02d}", home=home, away=away, actual_class=2, tournament="FIFA World Cup"))
        match_index += 1
    return pd.DataFrame(rows)


def _row(*, index: int, date: str, home: str, away: str, actual_class: int, tournament: str) -> dict[str, object]:
    home_score, away_score = {2: (2, 0), 1: (1, 1), 0: (0, 2)}[actual_class]
    row: dict[str, object] = {
        "match_id": f"m_{index}",
        "date": date,
        "home_team": home,
        "away_team": away,
        "tournament": tournament,
        "competition_type": "world_cup" if tournament == "FIFA World Cup" else "friendly",
        "neutral": True,
        "home_score": home_score,
        "away_score": away_score,
        "target_result_class": actual_class,
        "is_world_cup": tournament == "FIFA World Cup",
        "is_qualifier": False,
        "is_continental_tournament": False,
        "is_friendly": tournament != "FIFA World Cup",
        "match_weight": 1.0,
        "home_elo_pre_match": 1500 + index,
        "away_elo_pre_match": 1490 - index,
        "elo_diff": 10 + (2 * index),
    }
    for column in ROLLING_FEATURE_COLUMNS:
        row[column] = float((index % 5) + 1)
    return row


def _tmp_config(tmp_path) -> dict[str, object]:
    return {
        "paths": {
            "worldcup_tournament_simulation_backtest_csv": str(tmp_path / "backtests" / "worldcup_tournament_simulation_backtest.csv"),
            "worldcup_tournament_simulation_backtest_report_md": str(tmp_path / "reports" / "worldcup_tournament_simulation_backtest_report.md"),
        },
        "modeling": {"random_seed": 42},
    }
