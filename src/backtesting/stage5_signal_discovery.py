from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.stage5_benchmark import paired_bootstrap_ci
from src.config import load_config, resolve_project_path


STAGE5_SIGNAL_BY_SEGMENT_PATH = resolve_project_path("data/backtests/stage5_signal_by_segment.csv")
STAGE5_SIGNAL_WINNING_SEGMENTS_PATH = resolve_project_path("data/backtests/stage5_signal_winning_segments.csv")
STAGE5_SIGNAL_LOSING_SEGMENTS_PATH = resolve_project_path("data/backtests/stage5_signal_losing_segments.csv")
STAGE5_SIGNAL_WORST_MATCHES_PATH = resolve_project_path("data/backtests/stage5_signal_worst_matches.csv")
STAGE5_SIGNAL_FEATURE_HYPOTHESES_PATH = resolve_project_path("data/backtests/stage5_signal_feature_hypotheses.csv")
STAGE5_SIGNAL_DISCOVERY_REPORT_PATH = resolve_project_path("data/reports/stage5_signal_discovery_report.md")


@dataclass(frozen=True)
class Stage5SignalDiscoveryResult:
    by_segment: pd.DataFrame
    winning_segments: pd.DataFrame
    losing_segments: pd.DataFrame
    worst_matches: pd.DataFrame
    hypotheses: pd.DataFrame
    report_path: Path
    stage5_achieved: bool


def run_stage5_signal_discovery(
    *,
    output_dir: str | Path = "data/backtests",
    n_bootstrap: int = 1000,
    report_path: str | Path = STAGE5_SIGNAL_DISCOVERY_REPORT_PATH,
    config: dict[str, Any] | None = None,
) -> Stage5SignalDiscoveryResult:
    cfg = config or load_config()
    del cfg

    output_root = resolve_project_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _input_paths(output_root)
    predictions = _read_required_csv(paths["stage5_predictions"])
    metrics = _read_optional_csv(paths["stage5_metrics"])
    calibrated_predictions_present = paths["calibrated_predictions"].exists()
    calibrated_metrics_present = paths["calibrated_metrics"].exists()
    benchmark_report_present = resolve_project_path("data/reports/stage5_benchmark_report.md").exists()
    calibrated_report_present = resolve_project_path("data/reports/stage5_calibrated_market_report.md").exists()

    paired = paired_official_bookmaker(predictions)
    if paired.empty:
        raise ValueError("No paired official_model and bookmaker_odds_only rows are available in stage5_match_predictions.csv")
    paired = add_signal_segments(paired)
    by_segment = build_segment_table(paired, n_bootstrap=n_bootstrap)
    winning_segments = _winning_segments(by_segment)
    losing_segments = _losing_segments(by_segment)
    worst_matches = build_worst_matches(paired, limit=50)
    hypotheses = build_feature_hypotheses(losing_segments, worst_matches)

    output_paths = _output_paths(output_root)
    by_segment.to_csv(output_paths["by_segment"], index=False)
    winning_segments.to_csv(output_paths["winning_segments"], index=False)
    losing_segments.to_csv(output_paths["losing_segments"], index=False)
    worst_matches.to_csv(output_paths["worst_matches"], index=False)
    hypotheses.to_csv(output_paths["hypotheses"], index=False)

    report = write_stage5_signal_discovery_report(
        path=report_path,
        paired=paired,
        by_segment=by_segment,
        winning_segments=winning_segments,
        losing_segments=losing_segments,
        worst_matches=worst_matches,
        hypotheses=hypotheses,
        metrics=metrics,
        calibrated_predictions_present=calibrated_predictions_present,
        calibrated_metrics_present=calibrated_metrics_present,
        benchmark_report_present=benchmark_report_present,
        calibrated_report_present=calibrated_report_present,
    )
    return Stage5SignalDiscoveryResult(
        by_segment=by_segment,
        winning_segments=winning_segments,
        losing_segments=losing_segments,
        worst_matches=worst_matches,
        hypotheses=hypotheses,
        report_path=report,
        stage5_achieved=False,
    )


def paired_official_bookmaker(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_id",
        "date",
        "tournament_year",
        "stage",
        "home_team",
        "away_team",
        "actual_result",
        "actual_class",
        "model_name",
        "home_prob",
        "draw_prob",
        "away_prob",
        "predicted_result",
        "confidence",
        "log_loss",
        "brier",
        "rps",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"stage5_match_predictions.csv is missing required columns: {sorted(missing)}")
    key = ["match_id", "tournament_year"]
    official = predictions[predictions["model_name"].eq("official_model")].drop_duplicates(key, keep="last").copy()
    bookmaker = predictions[predictions["model_name"].eq("bookmaker_odds_only")].drop_duplicates(key, keep="last").copy()
    paired = official.merge(bookmaker, on=key, suffixes=("_official", "_bookmaker"), how="inner")
    if paired.empty:
        return pd.DataFrame()

    output = pd.DataFrame(
        {
            "match_id": paired["match_id"],
            "date": paired["date_official"],
            "tournament_year": paired["tournament_year"].astype(int),
            "stage": paired["stage_official"].fillna("unknown").astype(str),
            "home_team": paired["home_team_official"].astype(str),
            "away_team": paired["away_team_official"].astype(str),
            "actual_result": paired["actual_result_official"].astype(str),
            "actual_class": paired["actual_class_official"].astype(int),
            "official_predicted_result": paired["predicted_result_official"].astype(str),
            "bookmaker_predicted_result": paired["predicted_result_bookmaker"].astype(str),
            "official_home_prob": pd.to_numeric(paired["home_prob_official"], errors="coerce"),
            "official_draw_prob": pd.to_numeric(paired["draw_prob_official"], errors="coerce"),
            "official_away_prob": pd.to_numeric(paired["away_prob_official"], errors="coerce"),
            "bookmaker_home_prob": pd.to_numeric(paired["home_prob_bookmaker"], errors="coerce"),
            "bookmaker_draw_prob": pd.to_numeric(paired["draw_prob_bookmaker"], errors="coerce"),
            "bookmaker_away_prob": pd.to_numeric(paired["away_prob_bookmaker"], errors="coerce"),
            "official_confidence": pd.to_numeric(paired["confidence_official"], errors="coerce"),
            "bookmaker_confidence": pd.to_numeric(paired["confidence_bookmaker"], errors="coerce"),
            "official_log_loss": pd.to_numeric(paired["log_loss_official"], errors="coerce"),
            "bookmaker_log_loss": pd.to_numeric(paired["log_loss_bookmaker"], errors="coerce"),
            "official_brier": pd.to_numeric(paired["brier_official"], errors="coerce"),
            "bookmaker_brier": pd.to_numeric(paired["brier_bookmaker"], errors="coerce"),
            "official_rps": pd.to_numeric(paired["rps_official"], errors="coerce"),
            "bookmaker_rps": pd.to_numeric(paired["rps_bookmaker"], errors="coerce"),
        }
    )
    output["log_loss_delta"] = output["official_log_loss"] - output["bookmaker_log_loss"]
    output["brier_delta"] = output["official_brier"] - output["bookmaker_brier"]
    output["rps_delta"] = output["official_rps"] - output["bookmaker_rps"]
    output["official_correct"] = output["official_predicted_result"].eq(output["actual_result"]).astype(int)
    output["bookmaker_correct"] = output["bookmaker_predicted_result"].eq(output["actual_result"]).astype(int)
    _copy_optional_columns(paired, output)
    return output.sort_values(["tournament_year", "date", "match_id"], kind="stable").reset_index(drop=True)


def add_signal_segments(paired: pd.DataFrame) -> pd.DataFrame:
    output = paired.copy()
    output["actual_result_type"] = output["actual_result"].map({"home_win": "home", "draw": "draw", "away_win": "away"}).fillna(output["actual_result"])
    output["official_confidence_bucket"] = _confidence_bucket(output["official_confidence"])
    output["bookmaker_confidence_bucket"] = _confidence_bucket(output["bookmaker_confidence"])
    output["close_match_probability_bucket"] = _close_match_bucket(output["bookmaker_confidence"])
    output["market_favorite_type"] = _market_favorite_type(output)
    output["favorite_won_status"] = np.where(output["bookmaker_predicted_result"].eq(output["actual_result"]), "favorite_won", "favorite_failed")
    output["group_vs_knockout"] = np.where(output["stage"].str.lower().str.contains("group", na=False), "group_stage", "knockout")
    output["prediction_agreement"] = np.where(output["official_predicted_result"].eq(output["bookmaker_predicted_result"]), "same_prediction", "different_prediction")
    output["official_predicted_result_type"] = output["official_predicted_result"].map({"home_win": "home", "draw": "draw", "away_win": "away"}).fillna(output["official_predicted_result"])
    output["bookmaker_predicted_result_type"] = output["bookmaker_predicted_result"].map({"home_win": "home", "draw": "draw", "away_win": "away"}).fillna(output["bookmaker_predicted_result"])
    output["official_market_confidence_gap_bucket"] = _numeric_bucket(output["official_confidence"] - output["bookmaker_confidence"], [-1.0, -0.15, -0.05, 0.05, 0.15, 1.0], ["market_much_higher", "market_higher", "similar", "official_higher", "official_much_higher"])
    _add_optional_diff_buckets(output)
    _add_optional_confederation_buckets(output)
    return output


def build_segment_table(paired: pd.DataFrame, *, n_bootstrap: int = 1000) -> pd.DataFrame:
    segment_columns = [
        "tournament_year",
        "stage",
        "actual_result_type",
        "official_predicted_result_type",
        "bookmaker_predicted_result_type",
        "prediction_agreement",
        "bookmaker_confidence_bucket",
        "official_confidence_bucket",
        "market_favorite_type",
        "favorite_won_status",
        "group_vs_knockout",
        "close_match_probability_bucket",
        "official_market_confidence_gap_bucket",
        "home_team",
        "away_team",
        "elo_difference_bucket",
        "fifa_ranking_difference_bucket",
        "confederation_matchup",
    ]
    rows: list[dict[str, Any]] = []
    for column in segment_columns:
        if column not in paired.columns:
            continue
        for value, group in paired.groupby(column, dropna=False, sort=True):
            if pd.isna(value):
                continue
            rows.append(_segment_summary(str(column), str(value), group, n_bootstrap=n_bootstrap))
    if not rows:
        return pd.DataFrame(columns=_segment_columns())
    return pd.DataFrame(rows, columns=_segment_columns()).sort_values(
        ["segment_name", "mean_log_loss_delta", "segment_value"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def build_worst_matches(paired: pd.DataFrame, *, limit: int = 50) -> pd.DataFrame:
    columns = [
        "date",
        "tournament_year",
        "stage",
        "home_team",
        "away_team",
        "actual_result",
        "official_home_prob",
        "official_draw_prob",
        "official_away_prob",
        "bookmaker_home_prob",
        "bookmaker_draw_prob",
        "bookmaker_away_prob",
        "official_log_loss",
        "bookmaker_log_loss",
        "log_loss_delta",
        "possible_failure_reason",
    ]
    worst = paired[paired["log_loss_delta"].gt(0)].copy().sort_values("log_loss_delta", ascending=False, kind="stable").head(limit)
    if worst.empty:
        return pd.DataFrame(columns=columns)
    worst["possible_failure_reason"] = worst.apply(_failure_reason, axis=1)
    return worst.reindex(columns=columns).reset_index(drop=True)


def build_feature_hypotheses(losing_segments: pd.DataFrame, worst_matches: pd.DataFrame) -> pd.DataFrame:
    top_losing = losing_segments.head(12)
    top_segment = _evidence_segment(top_losing)
    draw_evidence = _find_evidence(top_losing, "actual_result_type", "draw") or _find_worst_reason(worst_matches, "draw")
    upset_evidence = _find_evidence(top_losing, "favorite_won_status", "favorite_failed") or _find_worst_reason(worst_matches, "upset")
    low_confidence_evidence = _find_evidence(top_losing, "close_match_probability_bucket", "very_close") or _find_evidence(top_losing, "bookmaker_confidence_bucket", "0.30")
    rows = [
        {
            "failure_pattern": "draw handling",
            "evidence_segment": draw_evidence or top_segment,
            "proposed_feature_or_model_change": "draw-specific calibration or draw-aware ordinal/scoreline layer",
            "expected_help": "Reduce overconfident non-draw probabilities when historical tournament draws are underweighted.",
            "leakage_risk": "Low if calibrated only on prior tournament folds.",
            "data_needed": "Historical no-leakage match probabilities and draw outcomes by tournament year.",
            "priority": "high" if draw_evidence else "medium",
        },
        {
            "failure_pattern": "favorite or underdog upset handling",
            "evidence_segment": upset_evidence or top_segment,
            "proposed_feature_or_model_change": "underdog/favorite upset feature using disagreement between model, market, Elo, and form",
            "expected_help": "Improve cases where the model is too confident in the wrong side of an upset.",
            "leakage_risk": "Medium; all market/rating inputs must be pre-match as-of data.",
            "data_needed": "Pre-match market odds, internal Elo, ranking snapshots, and opponent-adjusted form.",
            "priority": "high" if upset_evidence else "medium",
        },
        {
            "failure_pattern": "market low-confidence segment modeling",
            "evidence_segment": low_confidence_evidence or top_segment,
            "proposed_feature_or_model_change": "separate calibration for low market-confidence matches",
            "expected_help": "Target matches where bookmaker probabilities are flat and model disagreement may contain signal.",
            "leakage_risk": "Low if bucket thresholds are fixed or tuned on prior folds only.",
            "data_needed": "Complete historical 1X2 odds coverage and prior-fold calibration rows.",
            "priority": "medium",
        },
        {
            "failure_pattern": "opponent-adjusted recent form",
            "evidence_segment": top_segment,
            "proposed_feature_or_model_change": "strength-of-schedule adjusted rolling form features",
            "expected_help": "Separate strong recent form against weak opponents from stronger signal against elite opponents.",
            "leakage_risk": "Low if rolling windows close before match date.",
            "data_needed": "Historical results with internal Elo/opponent strength as-of match date.",
            "priority": "high",
        },
        {
            "failure_pattern": "rest/travel/venue context missing",
            "evidence_segment": top_segment,
            "proposed_feature_or_model_change": "rest days, travel distance, host/venue effect, and tournament-stage interaction features",
            "expected_help": "Capture World Cup-specific fatigue and venue context not present in generic match form.",
            "leakage_risk": "Medium; fixture schedule and venue data must be known before match kickoff.",
            "data_needed": "Venue locations, team travel bases, match kickoff dates, rest days, host flags.",
            "priority": "medium",
        },
        {
            "failure_pattern": "confederation matchup effects",
            "evidence_segment": _find_evidence(top_losing, "confederation_matchup", "") or top_segment,
            "proposed_feature_or_model_change": "confederation matchup and inter-confederation tournament calibration",
            "expected_help": "Model cross-region matchup priors that generic team ratings may miss.",
            "leakage_risk": "Low if team confederations are static metadata.",
            "data_needed": "Team-to-confederation mapping for all historical World Cup teams.",
            "priority": "medium",
        },
        {
            "failure_pattern": "Elo/FIFA disagreement",
            "evidence_segment": _find_evidence(top_losing, "elo_difference_bucket", "") or _find_evidence(top_losing, "fifa_ranking_difference_bucket", "") or top_segment,
            "proposed_feature_or_model_change": "feature capturing disagreement between internal Elo, FIFA/ranking, and market favorite",
            "expected_help": "Identify matches where independent rating systems disagree and raw football model may be overconfident.",
            "leakage_risk": "Medium; ranking snapshots must be as-of pre-match.",
            "data_needed": "Internal Elo and FIFA/ranking snapshots joined to historical tournament matches.",
            "priority": "medium",
        },
        {
            "failure_pattern": "low-scoring match structure",
            "evidence_segment": top_segment,
            "proposed_feature_or_model_change": "defensive-strength and proper Dixon-Coles scoreline model",
            "expected_help": "Improve 1X2 probabilities for low-scoring matches where draw and one-goal outcomes dominate.",
            "leakage_risk": "Low if trained only on pre-tournament history.",
            "data_needed": "Historical goals, team attack/defense strengths, and time-decayed scoreline fitting.",
            "priority": "high",
        },
    ]
    return pd.DataFrame(rows)


def write_stage5_signal_discovery_report(
    *,
    path: str | Path,
    paired: pd.DataFrame,
    by_segment: pd.DataFrame,
    winning_segments: pd.DataFrame,
    losing_segments: pd.DataFrame,
    worst_matches: pd.DataFrame,
    hypotheses: pd.DataFrame,
    metrics: pd.DataFrame,
    calibrated_predictions_present: bool,
    calibrated_metrics_present: bool,
    benchmark_report_present: bool,
    calibrated_report_present: bool,
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    meaningful_wins = winning_segments[winning_segments["statistically_meaningful"].eq(True)] if not winning_segments.empty else pd.DataFrame()
    overall = _overall_summary(paired)
    lines = [
        "# Stage 5 Signal Discovery Report",
        "",
        "## Executive Summary",
        "",
        "- Stage 5 status: `Stage 5 not achieved`.",
        "- This is research/evaluation output only.",
        "- Official production model logic was not changed.",
        "- Negative deltas mean the official model is better than bookmaker odds-only.",
        f"- Paired official/bookmaker matches analyzed: `{len(paired)}`.",
        f"- Statistically meaningful winning segments found: `{len(meaningful_wins)}`.",
        "",
        "## Overall Official vs Bookmaker",
        "",
        _markdown_table(pd.DataFrame([overall]), ["n_matches", "official_log_loss", "bookmaker_log_loss", "mean_log_loss_delta", "official_brier", "bookmaker_brier", "brier_delta", "official_rps", "bookmaker_rps", "rps_delta", "official_accuracy", "bookmaker_accuracy"]),
        "",
        "## Top Winning Segments",
        "",
        _markdown_table(winning_segments.head(10), _report_segment_columns()),
        "",
        "## Top Losing Segments",
        "",
        _markdown_table(losing_segments.head(10), _report_segment_columns()),
        "",
        "## Worst Official-Model Misses",
        "",
        _markdown_table(worst_matches.head(20), ["date", "tournament_year", "stage", "home_team", "away_team", "actual_result", "official_log_loss", "bookmaker_log_loss", "log_loss_delta", "possible_failure_reason"]),
        "",
        "## Feature Hypotheses",
        "",
        _markdown_table(hypotheses, ["failure_pattern", "evidence_segment", "proposed_feature_or_model_change", "expected_help", "leakage_risk", "data_needed", "priority"]),
        "",
        "## Optional Segment Availability",
        "",
        _segment_availability_text(by_segment),
        "",
        "## Input File Check",
        "",
        f"- Stage 5 benchmark metrics present: `{not metrics.empty}`.",
        f"- Stage 5 calibrated market predictions present: `{calibrated_predictions_present}`.",
        f"- Stage 5 calibrated market metrics present: `{calibrated_metrics_present}`.",
        f"- Stage 5 benchmark report present: `{benchmark_report_present}`.",
        f"- Stage 5 calibrated market report present: `{calibrated_report_present}`.",
        "",
        "## Recommendation",
        "",
        _recommendation_text(meaningful_wins, losing_segments, hypotheses),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _input_paths(output_root: Path) -> dict[str, Path]:
    return {
        "stage5_predictions": output_root / "stage5_match_predictions.csv",
        "stage5_metrics": output_root / "stage5_benchmark_metrics.csv",
        "calibrated_predictions": output_root / "stage5_calibrated_market_predictions.csv",
        "calibrated_metrics": output_root / "stage5_calibrated_market_metrics.csv",
    }


def _output_paths(output_root: Path) -> dict[str, Path]:
    return {
        "by_segment": output_root / "stage5_signal_by_segment.csv",
        "winning_segments": output_root / "stage5_signal_winning_segments.csv",
        "losing_segments": output_root / "stage5_signal_losing_segments.csv",
        "worst_matches": output_root / "stage5_signal_worst_matches.csv",
        "hypotheses": output_root / "stage5_signal_feature_hypotheses.csv",
    }


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Stage 5 output: {path}. Run `python -m src.cli stage5-benchmark --include-market` first.")
    return pd.read_csv(path, low_memory=False)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _copy_optional_columns(source: pd.DataFrame, output: pd.DataFrame) -> None:
    optional_columns = [
        "internal_elo_diff",
        "elo_diff",
        "home_internal_elo",
        "away_internal_elo",
        "fifa_rank_diff",
        "ranking_diff",
        "home_fifa_rank",
        "away_fifa_rank",
        "home_confederation",
        "away_confederation",
    ]
    for column in optional_columns:
        official_column = f"{column}_official"
        bookmaker_column = f"{column}_bookmaker"
        if official_column in source.columns:
            output[column] = source[official_column]
        elif bookmaker_column in source.columns:
            output[column] = source[bookmaker_column]


def _confidence_bucket(values: pd.Series) -> pd.Series:
    bins = [0.0, 0.40, 0.50, 0.60, 0.70, 1.0]
    labels = ["0.00-0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", "0.70-1.00"]
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True).astype(str)


def _close_match_bucket(values: pd.Series) -> pd.Series:
    bins = [0.0, 0.40, 0.50, 0.60, 0.70, 1.0]
    labels = ["very_close_<=0.40", "close_0.40-0.50", "moderate_0.50-0.60", "clear_0.60-0.70", "heavy_>0.70"]
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True).astype(str)


def _numeric_bucket(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True).astype(str)


def _market_favorite_type(df: pd.DataFrame) -> pd.Series:
    probs = df[["bookmaker_home_prob", "bookmaker_draw_prob", "bookmaker_away_prob"]].to_numpy(dtype=float)
    labels = np.array(["home_favorite", "draw_favorite", "away_favorite"])
    return pd.Series(labels[np.nanargmax(probs, axis=1)], index=df.index)


def _add_optional_diff_buckets(output: pd.DataFrame) -> None:
    elo_col = next((column for column in ["internal_elo_diff", "elo_diff"] if column in output.columns), None)
    if elo_col:
        output["elo_difference_bucket"] = _numeric_bucket(output[elo_col], [-10_000, -200, -75, 75, 200, 10_000], ["away_big_rating_edge", "away_small_rating_edge", "rating_close", "home_small_rating_edge", "home_big_rating_edge"])
    rank_col = next((column for column in ["fifa_rank_diff", "ranking_diff"] if column in output.columns), None)
    if rank_col:
        output["fifa_ranking_difference_bucket"] = _numeric_bucket(output[rank_col], [-10_000, -30, -10, 10, 30, 10_000], ["away_big_rank_edge", "away_small_rank_edge", "ranking_close", "home_small_rank_edge", "home_big_rank_edge"])


def _add_optional_confederation_buckets(output: pd.DataFrame) -> None:
    if {"home_confederation", "away_confederation"}.issubset(output.columns):
        output["confederation_matchup"] = output["home_confederation"].astype(str) + "_vs_" + output["away_confederation"].astype(str)


def _segment_summary(segment_name: str, segment_value: str, group: pd.DataFrame, *, n_bootstrap: int) -> dict[str, Any]:
    n_matches = int(len(group))
    delta = pd.to_numeric(group["log_loss_delta"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    mean_delta = float(np.mean(delta)) if len(delta) else np.nan
    if n_matches >= 20 and len(delta):
        _, ci_lower, ci_upper = paired_bootstrap_ci(delta, n_bootstrap=n_bootstrap)
        statistically_meaningful = bool(ci_upper < 0)
    else:
        ci_lower = np.nan
        ci_upper = np.nan
        statistically_meaningful = False
    return {
        "segment_name": segment_name,
        "segment_value": segment_value,
        "n_matches": n_matches,
        "official_log_loss": float(group["official_log_loss"].mean()),
        "bookmaker_log_loss": float(group["bookmaker_log_loss"].mean()),
        "mean_log_loss_delta": mean_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "statistically_meaningful": statistically_meaningful,
        "official_brier": float(group["official_brier"].mean()),
        "bookmaker_brier": float(group["bookmaker_brier"].mean()),
        "brier_delta": float(group["brier_delta"].mean()),
        "official_rps": float(group["official_rps"].mean()),
        "bookmaker_rps": float(group["bookmaker_rps"].mean()),
        "rps_delta": float(group["rps_delta"].mean()),
        "official_accuracy": float(group["official_correct"].mean()),
        "bookmaker_accuracy": float(group["bookmaker_correct"].mean()),
        "interpretation": _segment_interpretation(n_matches, mean_delta, ci_lower, ci_upper),
    }


def _segment_interpretation(n_matches: int, mean_delta: float, ci_lower: float, ci_upper: float) -> str:
    if n_matches < 20:
        if mean_delta < 0:
            return "low_sample_size_official_better_mean"
        if mean_delta > 0:
            return "low_sample_size_bookmaker_better_mean"
        return "low_sample_size_tie"
    if np.isfinite(ci_upper) and ci_upper < 0:
        return "strong_official_value"
    if np.isfinite(ci_lower) and ci_lower > 0:
        return "strong_bookmaker_advantage"
    if mean_delta < 0:
        return "official_better_mean_ci_crosses_zero"
    if mean_delta > 0:
        return "bookmaker_better_mean_ci_crosses_zero"
    return "tie"


def _segment_columns() -> list[str]:
    return [
        "segment_name",
        "segment_value",
        "n_matches",
        "official_log_loss",
        "bookmaker_log_loss",
        "mean_log_loss_delta",
        "ci_lower",
        "ci_upper",
        "statistically_meaningful",
        "official_brier",
        "bookmaker_brier",
        "brier_delta",
        "official_rps",
        "bookmaker_rps",
        "rps_delta",
        "official_accuracy",
        "bookmaker_accuracy",
        "interpretation",
    ]


def _report_segment_columns() -> list[str]:
    return [
        "segment_name",
        "segment_value",
        "n_matches",
        "official_log_loss",
        "bookmaker_log_loss",
        "mean_log_loss_delta",
        "ci_lower",
        "ci_upper",
        "statistically_meaningful",
        "official_accuracy",
        "bookmaker_accuracy",
        "interpretation",
    ]


def _winning_segments(by_segment: pd.DataFrame) -> pd.DataFrame:
    if by_segment.empty:
        return by_segment
    return (
        by_segment[
            by_segment["mean_log_loss_delta"].lt(0)
            & by_segment["n_matches"].ge(20)
        ]
        .sort_values(["statistically_meaningful", "mean_log_loss_delta", "n_matches"], ascending=[False, True, False], kind="stable")
        .reset_index(drop=True)
    )


def _losing_segments(by_segment: pd.DataFrame) -> pd.DataFrame:
    if by_segment.empty:
        return by_segment
    return (
        by_segment[
            by_segment["mean_log_loss_delta"].gt(0)
            & by_segment["n_matches"].ge(20)
        ]
        .sort_values(["mean_log_loss_delta", "n_matches"], ascending=[False, False], kind="stable")
        .reset_index(drop=True)
    )


def _failure_reason(row: pd.Series) -> str:
    actual = str(row.get("actual_result", ""))
    official_pred = str(row.get("official_predicted_result", ""))
    bookmaker_pred = str(row.get("bookmaker_predicted_result", ""))
    official_actual_prob = _prob_for_result(row, "official", actual)
    bookmaker_actual_prob = _prob_for_result(row, "bookmaker", actual)
    if actual == "draw" and row.get("official_draw_prob", 0.0) < row.get("bookmaker_draw_prob", 0.0):
        return "official underweighted draw"
    if bookmaker_pred == actual and official_pred != actual:
        return f"bookmaker selected actual outcome while official selected {official_pred}"
    if official_actual_prob + 0.15 < bookmaker_actual_prob:
        return "official assigned much lower probability to actual outcome"
    if official_pred != actual and row.get("official_confidence", 0.0) >= 0.55:
        return "official overconfident wrong-side prediction"
    if str(row.get("favorite_won_status", "")) == "favorite_failed":
        return "favorite failed or upset pattern"
    return "official less calibrated than market on actual outcome"


def _prob_for_result(row: pd.Series, prefix: str, actual: str) -> float:
    if actual == "home_win":
        return float(row.get(f"{prefix}_home_prob", np.nan))
    if actual == "draw":
        return float(row.get(f"{prefix}_draw_prob", np.nan))
    if actual == "away_win":
        return float(row.get(f"{prefix}_away_prob", np.nan))
    return float("nan")


def _overall_summary(paired: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_matches": int(len(paired)),
        "official_log_loss": float(paired["official_log_loss"].mean()),
        "bookmaker_log_loss": float(paired["bookmaker_log_loss"].mean()),
        "mean_log_loss_delta": float(paired["log_loss_delta"].mean()),
        "official_brier": float(paired["official_brier"].mean()),
        "bookmaker_brier": float(paired["bookmaker_brier"].mean()),
        "brier_delta": float(paired["brier_delta"].mean()),
        "official_rps": float(paired["official_rps"].mean()),
        "bookmaker_rps": float(paired["bookmaker_rps"].mean()),
        "rps_delta": float(paired["rps_delta"].mean()),
        "official_accuracy": float(paired["official_correct"].mean()),
        "bookmaker_accuracy": float(paired["bookmaker_correct"].mean()),
    }


def _evidence_segment(df: pd.DataFrame) -> str:
    if df.empty:
        return "No losing segment available"
    row = df.iloc[0]
    return f"{row['segment_name']}={row['segment_value']} (n={int(row['n_matches'])}, delta={float(row['mean_log_loss_delta']):.6f})"


def _find_evidence(df: pd.DataFrame, segment_name: str, contains: str) -> str:
    if df.empty:
        return ""
    subset = df[df["segment_name"].eq(segment_name)].copy()
    if contains:
        subset = subset[subset["segment_value"].astype(str).str.contains(contains, case=False, regex=False)]
    if subset.empty:
        return ""
    return _evidence_segment(subset.sort_values("mean_log_loss_delta", ascending=False, kind="stable"))


def _find_worst_reason(worst_matches: pd.DataFrame, contains: str) -> str:
    if worst_matches.empty or "possible_failure_reason" not in worst_matches.columns:
        return ""
    subset = worst_matches[worst_matches["possible_failure_reason"].astype(str).str.contains(contains, case=False, regex=False)]
    if subset.empty:
        return ""
    row = subset.iloc[0]
    return f"worst_match={row['home_team']} vs {row['away_team']} ({row['possible_failure_reason']}, delta={float(row['log_loss_delta']):.6f})"


def _segment_availability_text(by_segment: pd.DataFrame) -> str:
    segment_names = set(by_segment["segment_name"].dropna().astype(str)) if not by_segment.empty else set()
    checks = {
        "Elo difference buckets": "elo_difference_bucket" in segment_names,
        "FIFA/ranking difference buckets": "fifa_ranking_difference_bucket" in segment_names,
        "Confederation matchup buckets": "confederation_matchup" in segment_names,
        "Team buckets": {"home_team", "away_team"}.issubset(segment_names),
    }
    return "\n".join(f"- {name}: `{available}`." for name, available in checks.items())


def _recommendation_text(meaningful_wins: pd.DataFrame, losing_segments: pd.DataFrame, hypotheses: pd.DataFrame) -> str:
    lines = []
    if meaningful_wins.empty:
        lines.append("- No segment shows statistically meaningful official-model value over bookmaker odds under the current CI rule.")
    else:
        lines.append("- Some segments show statistically meaningful official-model value, but this does not make Stage 5 achieved.")
    if not losing_segments.empty:
        row = losing_segments.iloc[0]
        lines.append(f"- Biggest failure segment: `{row['segment_name']}={row['segment_value']}` with mean log-loss delta `{float(row['mean_log_loss_delta']):.6f}`.")
    if not hypotheses.empty:
        row = hypotheses.sort_values("priority", ascending=True, kind="stable").iloc[0]
        lines.append(f"- Best next hypothesis to test: `{row['proposed_feature_or_model_change']}`.")
    lines.append("- Keep production unchanged and test feature/model changes only through no-leakage historical folds.")
    return "\n".join(lines)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in df.columns]
    if df.empty or not available:
        return "_No rows._"
    rows = ["| " + " | ".join(available) + " |", "| " + " | ".join(["---"] * len(available)) + " |"]
    for record in df[available].to_dict("records"):
        rows.append("| " + " | ".join(_format_value(record.get(column)) for column in available) + " |")
    return "\n".join(rows)


def _format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("|", "\\|")
