from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_project_path


STAGE5_FAILURE_REPORT_PATH = resolve_project_path("data/reports/stage5_failure_diagnosis_report.md")
STAGE5_FAILURE_BY_YEAR_PATH = resolve_project_path("data/backtests/stage5_failure_by_year.csv")
STAGE5_FAILURE_BY_STAGE_PATH = resolve_project_path("data/backtests/stage5_failure_by_stage.csv")
STAGE5_FAILURE_BY_OUTCOME_PATH = resolve_project_path("data/backtests/stage5_failure_by_outcome.csv")
STAGE5_WORST_MATCHES_PATH = resolve_project_path("data/backtests/stage5_worst_matches.csv")
STAGE5_ABLATION_RECONCILIATION_PATH = resolve_project_path("data/backtests/stage5_ablation_reconciliation.csv")


@dataclass(frozen=True)
class Stage5FailureDiagnosisResult:
    report_path: Path
    by_year_path: Path
    by_stage_path: Path
    by_outcome_path: Path
    worst_matches_path: Path
    ablation_reconciliation_path: Path
    by_year: pd.DataFrame
    by_stage: pd.DataFrame
    by_outcome: pd.DataFrame
    worst_matches: pd.DataFrame
    ablation_reconciliation: pd.DataFrame
    stage5_achieved: bool


def run_stage5_failure_diagnosis(
    *,
    output_dir: str | Path = "data/backtests",
    report_path: str | Path = STAGE5_FAILURE_REPORT_PATH,
    config: dict[str, Any] | None = None,
) -> Stage5FailureDiagnosisResult:
    cfg = config or load_config()
    del cfg

    output_root = resolve_project_path(output_dir)
    paths = _input_paths(output_root)
    predictions = _read_required_csv(paths["predictions"])
    metrics = _read_required_csv(paths["metrics"])
    significance = _read_required_csv(paths["significance"])
    calibration = _read_optional_csv(paths["calibration"])
    decision_report = _read_optional_text(resolve_project_path("data/reports/stage5_research_decision.md"))
    benchmark_report = _read_optional_text(resolve_project_path("data/reports/stage5_benchmark_report.md"))

    paired = _paired_official_bookmaker(predictions)
    by_year = _comparison_by(paired, ["tournament_year"])
    by_stage = _comparison_by(paired, ["stage"])
    by_outcome = _comparison_by(paired, ["actual_result"])
    worst_matches = _worst_official_matches(paired, limit=30)
    market_blend = _market_blend_diagnosis(predictions, metrics)
    ablation_reconciliation = _ablation_reconciliation(metrics, paired)
    catboost_status = _catboost_status()
    stage5_achieved = _stage5_achieved_from_decision(decision_report, significance)
    output_paths = _failure_output_paths(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    by_year.to_csv(output_paths["by_year"], index=False)
    by_stage.to_csv(output_paths["by_stage"], index=False)
    by_outcome.to_csv(output_paths["by_outcome"], index=False)
    worst_matches.to_csv(output_paths["worst_matches"], index=False)
    ablation_reconciliation.to_csv(output_paths["ablation_reconciliation"], index=False)

    report = write_stage5_failure_diagnosis_report(
        path=report_path,
        by_year=by_year,
        by_stage=by_stage,
        by_outcome=by_outcome,
        worst_matches=worst_matches,
        ablation_reconciliation=ablation_reconciliation,
        significance=significance,
        calibration=calibration,
        market_blend=market_blend,
        catboost_status=catboost_status,
        stage5_achieved=stage5_achieved,
        decision_report=decision_report,
        benchmark_report=benchmark_report,
    )

    return Stage5FailureDiagnosisResult(
        report_path=report,
        by_year_path=output_paths["by_year"],
        by_stage_path=output_paths["by_stage"],
        by_outcome_path=output_paths["by_outcome"],
        worst_matches_path=output_paths["worst_matches"],
        ablation_reconciliation_path=output_paths["ablation_reconciliation"],
        by_year=by_year,
        by_stage=by_stage,
        by_outcome=by_outcome,
        worst_matches=worst_matches,
        ablation_reconciliation=ablation_reconciliation,
        stage5_achieved=stage5_achieved,
    )


def write_stage5_failure_diagnosis_report(
    *,
    path: str | Path,
    by_year: pd.DataFrame,
    by_stage: pd.DataFrame,
    by_outcome: pd.DataFrame,
    worst_matches: pd.DataFrame,
    ablation_reconciliation: pd.DataFrame,
    significance: pd.DataFrame,
    calibration: pd.DataFrame,
    market_blend: dict[str, Any],
    catboost_status: dict[str, Any],
    stage5_achieved: bool,
    decision_report: str,
    benchmark_report: str,
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    official_vs_book = significance[
        significance["candidate_model"].eq("official_model")
        & significance["baseline_model"].eq("bookmaker_odds_only")
        & significance["metric"].eq("log_loss")
    ].copy()
    calibration_summary = _calibration_gap_summary(calibration)

    lines = [
        "# Stage 5 Failure Diagnosis Report",
        "",
        "## Executive Summary",
        "",
        f"- Stage 5 status remains: `{'Stage 5 achieved' if stage5_achieved else 'Stage 5 not yet achieved'}`.",
        "- This diagnosis does not change the official production model.",
        "- Official production model remains `football_only_ensemble` with `core_football_only`.",
        "- All official-vs-bookmaker comparisons below use paired matches where both prediction rows exist.",
        "",
        "## Main Evidence",
        "",
        _markdown_table(
            official_vs_book,
            ["candidate_model", "baseline_model", "metric", "mean_delta", "ci_lower", "ci_upper", "statistically_meaningful", "n_matches"],
        ),
        "",
        "Negative delta means the candidate is better. The Stage 5 benchmark requires the upper confidence bound to be below 0.",
        "",
        "## Official Model vs Bookmaker Odds By Year",
        "",
        _markdown_table(by_year, _comparison_columns(["tournament_year"])),
        "",
        "## Official Model vs Bookmaker Odds By Stage",
        "",
        _markdown_table(by_stage, _comparison_columns(["stage"])),
        "",
        "## Official Model vs Bookmaker Odds By Actual Result",
        "",
        _markdown_table(by_outcome, _comparison_columns(["actual_result"])),
        "",
        "## Worst Official-Model Misses Relative To Bookmaker Odds",
        "",
        _markdown_table(
            worst_matches.head(30),
            [
                "date",
                "match_id",
                "tournament_year",
                "stage",
                "home_team",
                "away_team",
                "actual_result",
                "official_predicted_result",
                "bookmaker_predicted_result",
                "official_log_loss",
                "bookmaker_log_loss",
                "delta_log_loss",
                "official_home_prob",
                "official_draw_prob",
                "official_away_prob",
                "bookmaker_home_prob",
                "bookmaker_draw_prob",
                "bookmaker_away_prob",
            ],
        ),
        "",
        "## Why Market Blend Equals Bookmaker Odds",
        "",
        f"- Prediction rows identical: `{market_blend['prediction_rows_identical']}`.",
        f"- Metric rows identical: `{market_blend['metric_rows_identical']}`.",
        f"- Stored blend alpha: `{market_blend['alpha']}`.",
        f"- Stored odds conversion method: `{market_blend['odds_conversion_method']}`.",
        f"- Explanation: {market_blend['explanation']}",
        "",
        "## Ablation vs Stage 5 Reconciliation",
        "",
        _markdown_table(
            ablation_reconciliation,
            [
                "source",
                "evaluation_scope",
                "model_name",
                "feature_set",
                "n_matches",
                "log_loss",
                "brier_score",
                "ranked_probability_score",
                "calibration_error",
                "explanation",
            ],
        ),
        "",
        "The ablation log loss and Stage 5 official-model log loss are not measuring the same experiment. Ablation uses the project's broad time-aware validation split across the full historical match dataset, while Stage 5 uses held-out World Cup tournaments only with no-leakage tournament cutoffs. World Cup-only matches are a smaller, harder, and differently distributed sample.",
        "",
        "## CatBoost Status",
        "",
        _markdown_table(
            pd.DataFrame([catboost_status]),
            [
                "catboost_installed",
                "catboost_being_used_if_rerun_now",
                "saved_model_name",
                "expected_training_backend",
                "catboost_gpu_enabled",
                "stage5_outputs_store_estimator_backend",
                "notes",
            ],
        ),
        "",
        "## Calibration Notes",
        "",
        _markdown_table(calibration_summary, ["model_name", "rows", "mean_abs_calibration_gap", "max_abs_calibration_gap"]),
        "",
        "## Next Recommended Improvement",
        "",
        _next_recommendation(by_year, by_stage, by_outcome, market_blend, catboost_status),
        "",
        "## Input Reports Used",
        "",
        f"- Stage 5 decision report present: `{bool(decision_report)}`.",
        f"- Stage 5 benchmark report present: `{bool(benchmark_report)}`.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _input_paths(output_root: Path) -> dict[str, Path]:
    return {
        "metrics": output_root / "stage5_benchmark_metrics.csv",
        "predictions": output_root / "stage5_match_predictions.csv",
        "significance": output_root / "stage5_significance_tests.csv",
        "calibration": output_root / "stage5_calibration_table.csv",
    }


def _failure_output_paths(output_root: Path) -> dict[str, Path]:
    return {
        "by_year": output_root / "stage5_failure_by_year.csv",
        "by_stage": output_root / "stage5_failure_by_stage.csv",
        "by_outcome": output_root / "stage5_failure_by_outcome.csv",
        "worst_matches": output_root / "stage5_worst_matches.csv",
        "ablation_reconciliation": output_root / "stage5_ablation_reconciliation.csv",
    }


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Stage 5 output: {path}. Run `python -m src.cli stage5-benchmark` first.")
    return pd.read_csv(path, low_memory=False)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _paired_official_bookmaker(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"match_id", "tournament_year", "model_name", "log_loss", "brier", "rps"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"stage5_match_predictions.csv is missing required columns: {sorted(missing)}")

    key = ["match_id", "tournament_year"]
    official = predictions[predictions["model_name"].eq("official_model")].copy()
    bookmaker = predictions[predictions["model_name"].eq("bookmaker_odds_only")].copy()
    if official.empty or bookmaker.empty:
        return pd.DataFrame()

    official = official.drop_duplicates(key, keep="last")
    bookmaker = bookmaker.drop_duplicates(key, keep="last")
    paired = official.merge(bookmaker, on=key, suffixes=("_official", "_bookmaker"), how="inner")
    if paired.empty:
        return paired

    paired["date"] = paired.get("date_official", paired.get("date_bookmaker"))
    paired["stage"] = paired.get("stage_official", paired.get("stage_bookmaker")).fillna("unknown")
    paired["home_team"] = paired.get("home_team_official", paired.get("home_team_bookmaker"))
    paired["away_team"] = paired.get("away_team_official", paired.get("away_team_bookmaker"))
    paired["actual_result"] = paired.get("actual_result_official", paired.get("actual_result_bookmaker")).fillna("unknown")
    paired["official_predicted_result"] = paired.get("predicted_result_official", pd.Series(index=paired.index, dtype=object))
    paired["bookmaker_predicted_result"] = paired.get("predicted_result_bookmaker", pd.Series(index=paired.index, dtype=object))
    paired["official_log_loss"] = pd.to_numeric(paired["log_loss_official"], errors="coerce")
    paired["bookmaker_log_loss"] = pd.to_numeric(paired["log_loss_bookmaker"], errors="coerce")
    paired["delta_log_loss"] = paired["official_log_loss"] - paired["bookmaker_log_loss"]
    paired["official_brier"] = pd.to_numeric(paired["brier_official"], errors="coerce")
    paired["bookmaker_brier"] = pd.to_numeric(paired["brier_bookmaker"], errors="coerce")
    paired["delta_brier"] = paired["official_brier"] - paired["bookmaker_brier"]
    paired["official_rps"] = pd.to_numeric(paired["rps_official"], errors="coerce")
    paired["bookmaker_rps"] = pd.to_numeric(paired["rps_bookmaker"], errors="coerce")
    paired["delta_rps"] = paired["official_rps"] - paired["bookmaker_rps"]

    for prefix in ["official", "bookmaker"]:
        source_suffix = "official" if prefix == "official" else "bookmaker"
        paired[f"{prefix}_home_prob"] = pd.to_numeric(paired[f"home_prob_{source_suffix}"], errors="coerce")
        paired[f"{prefix}_draw_prob"] = pd.to_numeric(paired[f"draw_prob_{source_suffix}"], errors="coerce")
        paired[f"{prefix}_away_prob"] = pd.to_numeric(paired[f"away_prob_{source_suffix}"], errors="coerce")
    paired["official_correct"] = paired["official_predicted_result"].eq(paired["actual_result"]).astype(float)
    paired["bookmaker_correct"] = paired["bookmaker_predicted_result"].eq(paired["actual_result"]).astype(float)
    return paired


def _comparison_by(paired: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame(columns=_comparison_columns(group_cols))
    grouped = paired.groupby(group_cols, dropna=False)
    result = grouped.agg(
        n_matches=("match_id", "nunique"),
        official_log_loss=("official_log_loss", "mean"),
        bookmaker_log_loss=("bookmaker_log_loss", "mean"),
        delta_log_loss=("delta_log_loss", "mean"),
        official_brier=("official_brier", "mean"),
        bookmaker_brier=("bookmaker_brier", "mean"),
        delta_brier=("delta_brier", "mean"),
        official_rps=("official_rps", "mean"),
        bookmaker_rps=("bookmaker_rps", "mean"),
        delta_rps=("delta_rps", "mean"),
        official_accuracy=("official_correct", "mean"),
        bookmaker_accuracy=("bookmaker_correct", "mean"),
    ).reset_index()
    result["winner_by_log_loss"] = np.select(
        [result["delta_log_loss"].lt(0), result["delta_log_loss"].gt(0)],
        ["official_model", "bookmaker_odds_only"],
        default="tie",
    )
    return result.sort_values(group_cols, kind="stable").reset_index(drop=True)


def _comparison_columns(group_cols: list[str]) -> list[str]:
    return group_cols + [
        "n_matches",
        "official_log_loss",
        "bookmaker_log_loss",
        "delta_log_loss",
        "official_brier",
        "bookmaker_brier",
        "delta_brier",
        "official_rps",
        "bookmaker_rps",
        "delta_rps",
        "official_accuracy",
        "bookmaker_accuracy",
        "winner_by_log_loss",
    ]


def _worst_official_matches(paired: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    columns = [
        "date",
        "match_id",
        "tournament_year",
        "stage",
        "home_team",
        "away_team",
        "actual_result",
        "official_predicted_result",
        "bookmaker_predicted_result",
        "official_log_loss",
        "bookmaker_log_loss",
        "delta_log_loss",
        "official_home_prob",
        "official_draw_prob",
        "official_away_prob",
        "bookmaker_home_prob",
        "bookmaker_draw_prob",
        "bookmaker_away_prob",
    ]
    if paired.empty:
        return pd.DataFrame(columns=columns)
    worst = paired[paired["delta_log_loss"].gt(0)].sort_values("delta_log_loss", ascending=False, kind="stable").head(limit)
    return worst.reindex(columns=columns).reset_index(drop=True)


def _market_blend_diagnosis(predictions: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Any]:
    blend_params = _read_blend_params()
    key = ["match_id", "tournament_year"]
    bookmaker = predictions[predictions["model_name"].eq("bookmaker_odds_only")].drop_duplicates(key, keep="last")
    blend = predictions[predictions["model_name"].eq("market_blend")].drop_duplicates(key, keep="last")
    merged = bookmaker.merge(blend, on=key, suffixes=("_bookmaker", "_blend"), how="inner")
    probability_columns = ["home_prob", "draw_prob", "away_prob"]
    prediction_rows_identical = False
    if not merged.empty:
        checks = []
        for column in probability_columns:
            left = pd.to_numeric(merged[f"{column}_bookmaker"], errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(merged[f"{column}_blend"], errors="coerce").to_numpy(dtype=float)
            checks.append(np.allclose(left, right, atol=1e-12, rtol=1e-12, equal_nan=True))
        prediction_rows_identical = bool(all(checks))

    book_metrics = metrics[metrics["model_name"].eq("bookmaker_odds_only")].copy()
    blend_metrics = metrics[metrics["model_name"].eq("market_blend")].copy()
    metric_rows_identical = False
    if not book_metrics.empty and not blend_metrics.empty:
        metric_merge = book_metrics.merge(blend_metrics, on="test_year", suffixes=("_bookmaker", "_blend"), how="inner")
        metric_checks = []
        for column in ["log_loss", "brier", "rps", "ece", "accuracy", "mean_confidence"]:
            left = pd.to_numeric(metric_merge[f"{column}_bookmaker"], errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(metric_merge[f"{column}_blend"], errors="coerce").to_numpy(dtype=float)
            metric_checks.append(np.allclose(left, right, atol=1e-12, rtol=1e-12, equal_nan=True))
        metric_rows_identical = bool(metric_checks and all(metric_checks))

    alpha = blend_params.get("alpha")
    if prediction_rows_identical and str(alpha) in {"0", "0.0"}:
        explanation = "The stored market-blend alpha is 0.0, so blend = alpha * model + (1 - alpha) * market collapses to pure bookmaker market probabilities."
    elif prediction_rows_identical:
        explanation = "Market-blend and bookmaker rows are numerically identical in the Stage 5 outputs. Check the blend parameter file and whether the benchmark used market-only probabilities."
    else:
        explanation = "Market-blend predictions differ from bookmaker odds-only in the Stage 5 outputs."

    return {
        "prediction_rows_identical": prediction_rows_identical,
        "metric_rows_identical": metric_rows_identical,
        "alpha": alpha if alpha is not None else "unknown",
        "odds_conversion_method": blend_params.get("odds_conversion_method", "unknown"),
        "n_paired_rows": int(len(merged)),
        "explanation": explanation,
    }


def _read_blend_params() -> dict[str, Any]:
    path = resolve_project_path("models/market_blend_params.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _ablation_reconciliation(metrics: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ablation_path = resolve_project_path("data/experiments/ablation_results.csv")
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path, low_memory=False)
        if not ablation.empty and "feature_set" in ablation.columns:
            core = ablation[ablation["feature_set"].eq("core_football_only")].head(1)
            if not core.empty:
                row = core.iloc[0]
                rows.append(
                    {
                        "source": "ablation_results.csv",
                        "evaluation_scope": "project time-aware validation split",
                        "model_name": "ablation_estimator",
                        "feature_set": "core_football_only",
                        "n_matches": np.nan,
                        "log_loss": row.get("log_loss"),
                        "brier_score": row.get("brier_score"),
                        "ranked_probability_score": row.get("ranked_probability_score"),
                        "calibration_error": row.get("calibration_error"),
                        "explanation": "Ablation ranking uses the full historical match validation split, not held-out World Cup-only tournaments.",
                    }
                )

    official = _weighted_metric_row(
        metrics[metrics["model_name"].eq("official_model")],
        source="stage5_benchmark_metrics.csv",
        evaluation_scope="held-out World Cup tournaments, all matches",
        model_name="official_model",
        feature_set="football_only_ensemble/core_football_only",
        explanation="Stage 5 official-model score is tournament-only and no-leakage by World Cup year.",
    )
    if official:
        rows.append(official)

    if not paired.empty:
        rows.append(
            {
                "source": "stage5_match_predictions.csv",
                "evaluation_scope": "paired odds-covered World Cup matches",
                "model_name": "official_model",
                "feature_set": "football_only_ensemble/core_football_only",
                "n_matches": int(paired["match_id"].nunique()),
                "log_loss": float(paired["official_log_loss"].mean()),
                "brier_score": float(paired["official_brier"].mean()),
                "ranked_probability_score": float(paired["official_rps"].mean()),
                "calibration_error": np.nan,
                "explanation": "This is the exact official-model subset used against bookmaker odds-only.",
            }
        )
        rows.append(
            {
                "source": "stage5_match_predictions.csv",
                "evaluation_scope": "paired odds-covered World Cup matches",
                "model_name": "bookmaker_odds_only",
                "feature_set": "best_market_odds_conversion",
                "n_matches": int(paired["match_id"].nunique()),
                "log_loss": float(paired["bookmaker_log_loss"].mean()),
                "brier_score": float(paired["bookmaker_brier"].mean()),
                "ranked_probability_score": float(paired["bookmaker_rps"].mean()),
                "calibration_error": np.nan,
                "explanation": "Bookmaker baseline is evaluated only where pre-match 1X2 odds are available.",
            }
        )

    return pd.DataFrame(rows)


def _weighted_metric_row(
    df: pd.DataFrame,
    *,
    source: str,
    evaluation_scope: str,
    model_name: str,
    feature_set: str,
    explanation: str,
) -> dict[str, Any]:
    valid = df[df["n_matches"].fillna(0).astype(float).gt(0)].copy()
    if valid.empty:
        return {}
    weights = pd.to_numeric(valid["n_matches"], errors="coerce").fillna(0).to_numpy(dtype=float)
    total = float(weights.sum())
    if total <= 0:
        return {}

    def weighted(column: str) -> float:
        values = pd.to_numeric(valid[column], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not mask.any():
            return float("nan")
        return float(np.average(values[mask], weights=weights[mask]))

    return {
        "source": source,
        "evaluation_scope": evaluation_scope,
        "model_name": model_name,
        "feature_set": feature_set,
        "n_matches": int(total),
        "log_loss": weighted("log_loss"),
        "brier_score": weighted("brier"),
        "ranked_probability_score": weighted("rps"),
        "calibration_error": weighted("ece"),
        "explanation": explanation,
    }


def _catboost_status() -> dict[str, Any]:
    installed = importlib.util.find_spec("catboost") is not None
    saved_model_name = _saved_model_name()
    cfg = load_config()
    modeling = cfg.get("modeling", {})
    gpu_enabled = _as_bool(modeling.get("catboost_use_gpu", False))
    expected_backend = "catboost" if installed else "hist_gradient_boosting fallback"
    if installed:
        notes = "CatBoost is importable in the current environment. Stage 5 reruns that request --model catboost should use CatBoost. Existing Stage 5 CSVs do not store the fold estimator backend."
    else:
        notes = "CatBoost is not importable in the current environment. Stage 5 reruns that request --model catboost will fall back to HistGradientBoostingClassifier. Existing Stage 5 CSVs do not store the fold estimator backend."
    return {
        "catboost_installed": installed,
        "catboost_being_used_if_rerun_now": installed,
        "saved_model_name": saved_model_name,
        "expected_training_backend": expected_backend,
        "catboost_gpu_enabled": gpu_enabled,
        "stage5_outputs_store_estimator_backend": False,
        "notes": notes,
    }


def _saved_model_name() -> str:
    path = resolve_project_path("models/match_model.joblib")
    if path.exists():
        return "present_not_inspected"
    return "not_found"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _stage5_achieved_from_decision(decision_report: str, significance: pd.DataFrame) -> bool:
    if "Decision: **Stage 5 achieved**" in decision_report:
        return True
    row = significance[
        significance["candidate_model"].eq("official_model")
        & significance["baseline_model"].eq("bookmaker_odds_only")
        & significance["metric"].eq("log_loss")
    ]
    if row.empty:
        return False
    return bool(row.iloc[0].get("statistically_meaningful", False)) and float(row.iloc[0].get("ci_upper", 0.0)) < 0


def _calibration_gap_summary(calibration: pd.DataFrame) -> pd.DataFrame:
    if calibration.empty or "calibration_gap" not in calibration.columns:
        return pd.DataFrame(columns=["model_name", "rows", "mean_abs_calibration_gap", "max_abs_calibration_gap"])
    valid = calibration.copy()
    valid["calibration_gap"] = pd.to_numeric(valid["calibration_gap"], errors="coerce")
    valid = valid.dropna(subset=["calibration_gap"])
    if valid.empty:
        return pd.DataFrame(columns=["model_name", "rows", "mean_abs_calibration_gap", "max_abs_calibration_gap"])
    valid["abs_gap"] = valid["calibration_gap"].abs()
    return (
        valid.groupby("model_name", as_index=False)
        .agg(rows=("abs_gap", "size"), mean_abs_calibration_gap=("abs_gap", "mean"), max_abs_calibration_gap=("abs_gap", "max"))
        .sort_values("mean_abs_calibration_gap", kind="stable")
        .reset_index(drop=True)
    )


def _next_recommendation(
    by_year: pd.DataFrame,
    by_stage: pd.DataFrame,
    by_outcome: pd.DataFrame,
    market_blend: dict[str, Any],
    catboost_status: dict[str, Any],
) -> str:
    parts = [
        "- Keep the production model unchanged until a rerun shows statistically meaningful improvement over bookmaker odds-only.",
        "- Focus on the largest positive official-minus-bookmaker log-loss slices in the by-year, by-stage, and by-outcome CSVs.",
    ]
    if market_blend.get("prediction_rows_identical") and str(market_blend.get("alpha")) in {"0", "0.0"}:
        parts.append("- Retune market blend only after the model side improves; alpha 0.0 currently means the blend is just bookmaker odds.")
    if not bool(catboost_status.get("catboost_installed")):
        parts.append("- Install CatBoost in the benchmark environment if you want the requested catboost estimator instead of the sklearn fallback.")
    if not by_stage.empty:
        worst_stage = by_stage.sort_values("delta_log_loss", ascending=False, kind="stable").head(1)
        if not worst_stage.empty:
            parts.append(f"- First diagnostic slice to inspect: stage `{worst_stage.iloc[0]['stage']}` with delta_log_loss `{float(worst_stage.iloc[0]['delta_log_loss']):.6f}`.")
    if not by_outcome.empty:
        worst_outcome = by_outcome.sort_values("delta_log_loss", ascending=False, kind="stable").head(1)
        if not worst_outcome.empty:
            parts.append(f"- Outcome slice to inspect: `{worst_outcome.iloc[0]['actual_result']}` with delta_log_loss `{float(worst_outcome.iloc[0]['delta_log_loss']):.6f}`.")
    if by_year.empty:
        parts.append("- Re-run `python -m src.cli stage5-benchmark --include-market` if paired official/bookmaker rows are missing.")
    return "\n".join(parts)


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
