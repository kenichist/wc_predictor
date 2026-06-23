from __future__ import annotations

import pandas as pd

from src import cli
from src.backtesting.worldcup_backtest import run_worldcup_backtests, split_worldcup_backtest_fold
from src.features.elo import ELO_COLUMNS
from src.features.rolling_team_form import ROLLING_FEATURE_COLUMNS
from src.sources.fifa_rankings import FIFA_RANKING_MODEL_COLUMNS


def test_training_cutoff_is_before_tournament_start() -> None:
    frame = _synthetic_backtest_frame()

    train, target = split_worldcup_backtest_fold(frame, 2010)

    assert not target.empty
    assert pd.to_datetime(train["date"]).max() < pd.Timestamp("2010-06-11")


def test_target_tournament_rows_do_not_leak_into_training() -> None:
    frame = _synthetic_backtest_frame(include_early_world_cup_row=True)

    train, target = split_worldcup_backtest_fold(frame, 2010)

    assert "early_wc_row" in set(target["match_id"])
    assert "early_wc_row" not in set(train["match_id"])


def test_worldcup_backtest_creates_metrics(tmp_path) -> None:
    _, metrics, _ = run_worldcup_backtests(
        model_name="hist_gradient_boosting",
        feature_set="v1_baseline",
        config=_tmp_config(tmp_path),
        feature_frame=_synthetic_backtest_frame(),
        years=[2010],
    )

    assert not metrics.empty
    assert {"accuracy", "log_loss", "brier_score", "ranked_probability_score", "calibration_error"} <= set(metrics.columns)
    assert (tmp_path / "backtests" / "worldcup_backtest_metrics.csv").exists()


def test_worldcup_backtest_creates_report(tmp_path) -> None:
    _, _, report_path = run_worldcup_backtests(
        model_name="hist_gradient_boosting",
        feature_set="v1_baseline",
        config=_tmp_config(tmp_path),
        feature_frame=_synthetic_backtest_frame(),
        years=[2010],
    )

    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Historical World Cup Backtest Report" in text
    assert "Model Ranking By Log Loss" in text


def test_worldcup_backtest_runs_baselines(tmp_path) -> None:
    predictions, metrics, _ = run_worldcup_backtests(
        model_name="hist_gradient_boosting",
        feature_set="v1_baseline",
        config=_tmp_config(tmp_path),
        feature_frame=_synthetic_backtest_frame(),
        years=[2010],
    )

    expected = {
        "final_model",
        "elo_only",
        "fifa_ranking_only",
        "rolling_form_only",
        "majority_class_baseline",
        "bookmaker_odds_only",
        "market_blend",
    }
    assert expected <= set(metrics["model_name"])
    assert expected <= set(predictions["model_name"])


def test_backtest_world_cups_cli_command_is_registered() -> None:
    args = cli.build_parser().parse_args(["backtest-world-cups", "--model", "catboost", "--feature-set", "core_football_only"])

    assert args.command == "backtest-world-cups"
    assert args.model == "catboost"
    assert args.feature_set == "core_football_only"


def _synthetic_backtest_frame(*, include_early_world_cup_row: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    classes = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]
    for index, actual_class in enumerate(classes):
        rows.append(_row(index=index, date=f"2009-{(index % 12) + 1:02d}-01", actual_class=actual_class, tournament="Friendly"))
    if include_early_world_cup_row:
        rows.append(_row(index=100, date="2010-06-01", actual_class=2, tournament="FIFA World Cup", match_id="early_wc_row"))
    for offset, actual_class in enumerate([2, 1, 0, 2], start=200):
        rows.append(_row(index=offset, date=f"2010-06-{offset - 189:02d}", actual_class=actual_class, tournament="FIFA World Cup"))
    return pd.DataFrame(rows)


def _row(*, index: int, date: str, actual_class: int, tournament: str, match_id: str | None = None) -> dict[str, object]:
    home_score, away_score = {2: (2, 0), 1: (1, 1), 0: (0, 2)}[actual_class]
    row: dict[str, object] = {
        "match_id": match_id or f"match_{index}",
        "date": date,
        "home_team": f"Team {index % 5}",
        "away_team": f"Team {(index + 1) % 5}",
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
        "away_elo_pre_match": 1480 - index,
        "elo_diff": 20 + (2 * index),
        "market_home_prob": 0.45,
        "market_draw_prob": 0.25,
        "market_away_prob": 0.30,
    }
    for column in ROLLING_FEATURE_COLUMNS:
        row[column] = float((index % 7) + 1)
    for column in FIFA_RANKING_MODEL_COLUMNS:
        row[column] = float((index % 10) + 1)
    for column in ELO_COLUMNS:
        row.setdefault(column, float(index))
    return row


def _tmp_config(tmp_path) -> dict[str, object]:
    return {
        "paths": {
            "worldcup_backtest_predictions": str(tmp_path / "backtests" / "worldcup_backtest_predictions.csv"),
            "worldcup_backtest_metrics": str(tmp_path / "backtests" / "worldcup_backtest_metrics.csv"),
            "worldcup_backtest_report_md": str(tmp_path / "reports" / "worldcup_backtest_report.md"),
            "market_blend_report_md": str(tmp_path / "reports" / "market_blend_report.md"),
            "odds_conversion_benchmark_report_md": str(tmp_path / "reports" / "odds_conversion_benchmark_report.md"),
            "benchmark_significance_report_md": str(tmp_path / "reports" / "benchmark_significance_report.md"),
            "market_blend_params": str(tmp_path / "models" / "market_blend_params.json"),
        },
        "modeling": {"random_seed": 42},
    }
