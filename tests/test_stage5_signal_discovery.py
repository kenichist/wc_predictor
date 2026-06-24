from __future__ import annotations

import pandas as pd
import pytest

import src.cli as cli
from src.backtesting.stage5_signal_discovery import (
    add_signal_segments,
    build_segment_table,
    paired_official_bookmaker,
    run_stage5_signal_discovery,
)


def test_signal_discovery_pairs_rows_and_delta_direction() -> None:
    predictions = _synthetic_predictions(n_matches=4)

    paired = paired_official_bookmaker(predictions)

    assert len(paired) == 4
    first = paired[paired["match_id"].eq("m0")].iloc[0]
    assert first["log_loss_delta"] == pytest.approx(0.4)
    assert first["brier_delta"] == pytest.approx(0.2)
    assert first["rps_delta"] == pytest.approx(0.1)


def test_signal_discovery_segment_table_flags_low_sample() -> None:
    paired = add_signal_segments(paired_official_bookmaker(_synthetic_predictions(n_matches=6)))

    segments = build_segment_table(paired, n_bootstrap=25)

    assert not segments.empty
    low_sample = segments[segments["segment_name"].eq("tournament_year")].iloc[0]
    assert low_sample["n_matches"] == 6
    assert "low_sample_size" in low_sample["interpretation"]
    assert low_sample["statistically_meaningful"] in {False, pd.NA}


def test_signal_discovery_creates_outputs(tmp_path) -> None:
    _synthetic_predictions(n_matches=24).to_csv(tmp_path / "stage5_match_predictions.csv", index=False)
    pd.DataFrame([{"model_name": "official_model", "test_year": 2022, "n_matches": 24}]).to_csv(
        tmp_path / "stage5_benchmark_metrics.csv",
        index=False,
    )

    result = run_stage5_signal_discovery(
        output_dir=tmp_path,
        n_bootstrap=25,
        report_path=tmp_path / "stage5_signal_discovery_report.md",
    )

    assert (tmp_path / "stage5_signal_by_segment.csv").exists()
    assert (tmp_path / "stage5_signal_winning_segments.csv").exists()
    assert (tmp_path / "stage5_signal_losing_segments.csv").exists()
    assert (tmp_path / "stage5_signal_worst_matches.csv").exists()
    assert (tmp_path / "stage5_signal_feature_hypotheses.csv").exists()
    assert result.report_path.exists()
    assert result.stage5_achieved is False
    assert not result.by_segment.empty
    assert not result.hypotheses.empty


def test_stage5_signal_discovery_cli_parser() -> None:
    args = cli.build_parser().parse_args(["stage5-signal-discovery", "--n-bootstrap", "500"])

    assert args.command == "stage5-signal-discovery"
    assert args.n_bootstrap == 500


def _synthetic_predictions(*, n_matches: int) -> pd.DataFrame:
    rows = []
    for idx in range(n_matches):
        actual = ["home_win", "draw", "away_win"][idx % 3]
        official_loss = 1.0 if idx % 2 == 0 else 0.5
        bookmaker_loss = 0.6 if idx % 2 == 0 else 0.7
        rows.append(_row(idx, "official_model", actual, official_loss, 0.7, 0.4))
        rows.append(_row(idx, "bookmaker_odds_only", actual, bookmaker_loss, 0.5, 0.3))
    return pd.DataFrame(rows)


def _row(idx: int, model_name: str, actual: str, log_loss: float, brier: float, rps: float) -> dict[str, object]:
    probs = {
        "home_win": (0.55, 0.25, 0.20),
        "draw": (0.35, 0.35, 0.30),
        "away_win": (0.20, 0.25, 0.55),
    }[actual]
    predicted = max(zip(["home_win", "draw", "away_win"], probs), key=lambda item: item[1])[0]
    return {
        "match_id": f"m{idx}",
        "date": "2022-11-20",
        "tournament_year": 2022,
        "stage": "group_stage" if idx < 20 else "round_of_16",
        "home_team": f"Home {idx}",
        "away_team": f"Away {idx}",
        "actual_result": actual,
        "actual_class": {"away_win": 0, "draw": 1, "home_win": 2}[actual],
        "model_name": model_name,
        "feature_set": "fixture",
        "baseline_type": "fixture",
        "home_prob": probs[0],
        "draw_prob": probs[1],
        "away_prob": probs[2],
        "predicted_result": predicted,
        "confidence": max(probs),
        "log_loss": log_loss,
        "brier": brier,
        "rps": rps,
        "data_source_notes": "fixture",
    }
