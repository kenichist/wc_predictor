from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.backtesting.stage5_benchmark import (
    normalize_probability_rows,
    paired_bootstrap_ci,
    per_row_brier,
    per_row_log_loss,
    per_row_rps,
)
from src.config import load_config, resolve_project_path
from src.models.evaluate import expected_calibration_error


CLASS_LABELS = {0: "away_win", 1: "draw", 2: "home_win"}
PROB_COLUMNS = ["away_prob", "draw_prob", "home_prob"]
FRAME_PROB_COLUMNS = ["home_prob", "draw_prob", "away_prob"]

STAGE5_CALIBRATED_MARKET_METRICS_PATH = resolve_project_path("data/backtests/stage5_calibrated_market_metrics.csv")
STAGE5_CALIBRATED_MARKET_PREDICTIONS_PATH = resolve_project_path("data/backtests/stage5_calibrated_market_predictions.csv")
STAGE5_MARKET_VALUE_SIGNIFICANCE_PATH = resolve_project_path("data/backtests/stage5_market_value_significance.csv")
STAGE5_ALPHA_TUNING_PATH = resolve_project_path("data/backtests/stage5_alpha_tuning.csv")
STAGE5_STACKER_COEFFICIENTS_PATH = resolve_project_path("data/backtests/stage5_stacker_coefficients.csv")
STAGE5_CALIBRATED_MARKET_REPORT_PATH = resolve_project_path("data/reports/stage5_calibrated_market_report.md")
STAGE5_CALIBRATED_MARKET_DECISION_PATH = resolve_project_path("data/reports/stage5_calibrated_market_decision.md")


@dataclass(frozen=True)
class Stage5CalibratedMarketResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    significance: pd.DataFrame
    alpha_tuning: pd.DataFrame
    stacker_coefficients: pd.DataFrame
    report_path: Path
    decision_path: Path
    stage5_achieved: bool


def run_stage5_calibrated_market(
    *,
    years: list[int] | None = None,
    n_bootstrap: int = 1000,
    alpha_grid_step: float = 0.05,
    include_stacker: bool = False,
    include_draw_adjustment: bool = False,
    output_dir: str | Path = "data/backtests",
    report_path: str | Path = STAGE5_CALIBRATED_MARKET_REPORT_PATH,
    decision_path: str | Path = STAGE5_CALIBRATED_MARKET_DECISION_PATH,
    config: dict[str, Any] | None = None,
) -> Stage5CalibratedMarketResult:
    cfg = config or load_config()
    del cfg

    selected_years = sorted(int(year) for year in (years or [2014, 2018, 2022]))
    output_root = resolve_project_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    source_predictions = _read_required_csv(output_root / "stage5_match_predictions.csv")
    paired = paired_official_bookmaker(source_predictions, selected_years)
    if paired.empty:
        raise ValueError("No paired official_model and bookmaker_odds_only rows are available. Run stage5-benchmark with market outputs first.")

    alpha_grid = _alpha_grid(alpha_grid_step)
    prediction_frames: list[pd.DataFrame] = []
    metric_skip_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    stacker_rows: list[dict[str, Any]] = []

    for year in selected_years:
        target = paired[paired["tournament_year"].eq(year)].copy()
        prior = paired[paired["tournament_year"].lt(year)].copy()
        target_count = int(len(target))
        if target.empty:
            continue

        prediction_frames.append(_frame_from_probabilities(target, target[["official_away_prob", "official_draw_prob", "official_home_prob"]].to_numpy(dtype=float), "official_model", "existing official model probabilities", "none", "official model from Stage 5 benchmark"))
        prediction_frames.append(_frame_from_probabilities(target, target[["bookmaker_away_prob", "bookmaker_draw_prob", "bookmaker_home_prob"]].to_numpy(dtype=float), "bookmaker_odds_only", "existing bookmaker odds-only probabilities", "none", "bookmaker odds-only from Stage 5 benchmark"))

        official_calibrated, _ = _time_safe_temperature_calibration(
            prior,
            target,
            source_prefix="official",
            model_name="official_model_calibrated",
        )
        prediction_frames.append(official_calibrated)

        bookmaker_calibrated, _ = _time_safe_temperature_calibration(
            prior,
            target,
            source_prefix="bookmaker",
            model_name="bookmaker_calibrated",
        )
        prediction_frames.append(bookmaker_calibrated)

        blend_frame, tuning_rows = _market_blend_tuned_for_year(target, prior, alpha_grid)
        prediction_frames.append(blend_frame)
        alpha_rows.extend(tuning_rows)

        if include_stacker:
            stacker_frame, coefficient_rows, skip_row = _stacked_model_for_year(target, prior)
            if not stacker_frame.empty:
                prediction_frames.append(stacker_frame)
            if coefficient_rows:
                stacker_rows.extend(coefficient_rows)
            if skip_row:
                metric_skip_rows.append({**skip_row, "n_skipped": target_count})

        if include_draw_adjustment:
            draw_frame, skip_row = _draw_adjusted_for_year(target, prior)
            if not draw_frame.empty:
                prediction_frames.append(draw_frame)
            if skip_row:
                metric_skip_rows.append({**skip_row, "n_skipped": target_count})

    predictions = pd.concat([frame for frame in prediction_frames if not frame.empty], ignore_index=True, sort=False)
    predictions = _deduplicate_predictions(predictions)
    predictions = _add_prediction_scores(predictions)
    target_counts = paired.groupby("tournament_year").size().to_dict()
    metrics = _metrics_table(predictions, target_counts)
    if metric_skip_rows:
        metrics = pd.concat([metrics, _skipped_metrics(metric_skip_rows, target_counts)], ignore_index=True, sort=False)
    significance = _market_value_significance(predictions, n_bootstrap=n_bootstrap)
    alpha_tuning = pd.DataFrame(alpha_rows)
    stacker_coefficients = pd.DataFrame(stacker_rows)
    stage5_achieved = _stage5_achieved(significance)

    output_paths = _output_paths(output_root)
    metrics.to_csv(output_paths["metrics"], index=False)
    predictions.to_csv(output_paths["predictions"], index=False)
    significance.to_csv(output_paths["significance"], index=False)
    alpha_tuning.to_csv(output_paths["alpha_tuning"], index=False)
    stacker_coefficients.to_csv(output_paths["stacker_coefficients"], index=False)

    report_path = write_stage5_calibrated_market_report(
        metrics=metrics,
        significance=significance,
        alpha_tuning=alpha_tuning,
        stacker_coefficients=stacker_coefficients,
        stage5_achieved=stage5_achieved,
        include_stacker=include_stacker,
        include_draw_adjustment=include_draw_adjustment,
        path=report_path,
    )
    decision_path = write_stage5_calibrated_market_decision(
        metrics=metrics,
        significance=significance,
        stage5_achieved=stage5_achieved,
        path=decision_path,
    )

    return Stage5CalibratedMarketResult(
        metrics=metrics,
        predictions=predictions,
        significance=significance,
        alpha_tuning=alpha_tuning,
        stacker_coefficients=stacker_coefficients,
        report_path=report_path,
        decision_path=decision_path,
        stage5_achieved=stage5_achieved,
    )


def paired_official_bookmaker(predictions: pd.DataFrame, years: list[int]) -> pd.DataFrame:
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
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"stage5_match_predictions.csv is missing required columns: {sorted(missing)}")
    source = predictions[predictions["tournament_year"].astype(int).isin([int(year) for year in years])].copy()
    key = ["match_id", "tournament_year"]
    official = source[source["model_name"].eq("official_model")].drop_duplicates(key, keep="last").copy()
    bookmaker = source[source["model_name"].eq("bookmaker_odds_only")].drop_duplicates(key, keep="last").copy()
    paired = official.merge(bookmaker, on=key, suffixes=("_official", "_bookmaker"), how="inner")
    if paired.empty:
        return paired
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
            "official_home_prob": pd.to_numeric(paired["home_prob_official"], errors="coerce"),
            "official_draw_prob": pd.to_numeric(paired["draw_prob_official"], errors="coerce"),
            "official_away_prob": pd.to_numeric(paired["away_prob_official"], errors="coerce"),
            "bookmaker_home_prob": pd.to_numeric(paired["home_prob_bookmaker"], errors="coerce"),
            "bookmaker_draw_prob": pd.to_numeric(paired["draw_prob_bookmaker"], errors="coerce"),
            "bookmaker_away_prob": pd.to_numeric(paired["away_prob_bookmaker"], errors="coerce"),
        }
    )
    for prefix in ["official", "bookmaker"]:
        ordered = normalize_probability_rows(output[[f"{prefix}_away_prob", f"{prefix}_draw_prob", f"{prefix}_home_prob"]].to_numpy(dtype=float))
        output[f"{prefix}_away_prob"] = ordered[:, 0]
        output[f"{prefix}_draw_prob"] = ordered[:, 1]
        output[f"{prefix}_home_prob"] = ordered[:, 2]
    return output.sort_values(["tournament_year", "date", "match_id"], kind="stable").reset_index(drop=True)


def blend_probabilities(model_probabilities: np.ndarray, market_probabilities: np.ndarray, alpha: float) -> np.ndarray:
    model = normalize_probability_rows(model_probabilities)
    market = normalize_probability_rows(market_probabilities)
    return normalize_probability_rows(float(alpha) * model + (1.0 - float(alpha)) * market)


def select_alpha_for_year(paired: pd.DataFrame, test_year: int, alpha_grid: list[float]) -> tuple[float, pd.DataFrame]:
    prior = paired[paired["tournament_year"].lt(int(test_year))].copy()
    rows: list[dict[str, Any]] = []
    tuning_years = _years_label(prior)
    if prior.empty:
        selected_alpha = 0.0
        for alpha in alpha_grid:
            rows.append(
                {
                    "test_year": int(test_year),
                    "alpha": float(alpha),
                    "tuning_years": "none",
                    "tuning_log_loss": np.nan,
                    "selected_alpha": selected_alpha,
                    "selected": bool(float(alpha) == selected_alpha),
                    "notes": "limited: no prior tournament with paired model and market rows; safe default alpha=0.0",
                }
            )
        return selected_alpha, pd.DataFrame(rows)

    y = prior["actual_class"].astype(int).to_numpy()
    model_prob = prior[["official_away_prob", "official_draw_prob", "official_home_prob"]].to_numpy(dtype=float)
    market_prob = prior[["bookmaker_away_prob", "bookmaker_draw_prob", "bookmaker_home_prob"]].to_numpy(dtype=float)
    best_alpha = float(alpha_grid[0])
    best_loss = float("inf")
    losses: list[tuple[float, float]] = []
    for alpha in alpha_grid:
        blended = blend_probabilities(model_prob, market_prob, float(alpha))
        loss = float(log_loss(y, blended, labels=[0, 1, 2]))
        losses.append((float(alpha), loss))
        if loss < best_loss - 1e-12:
            best_alpha = float(alpha)
            best_loss = loss
    for alpha, loss in losses:
        rows.append(
            {
                "test_year": int(test_year),
                "alpha": alpha,
                "tuning_years": tuning_years,
                "tuning_log_loss": loss,
                "selected_alpha": best_alpha,
                "selected": bool(abs(alpha - best_alpha) < 1e-12),
                "notes": "alpha selected using only prior tournament rows",
            }
        )
    return best_alpha, pd.DataFrame(rows)


def write_stage5_calibrated_market_report(
    *,
    metrics: pd.DataFrame,
    significance: pd.DataFrame,
    alpha_tuning: pd.DataFrame,
    stacker_coefficients: pd.DataFrame,
    stage5_achieved: bool,
    include_stacker: bool,
    include_draw_adjustment: bool,
    path: str | Path = STAGE5_CALIBRATED_MARKET_REPORT_PATH,
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    average = _average_metrics(metrics)
    best = _best_models(average)
    lines = [
        "# Stage 5 Calibrated Market Report",
        "",
        "## Executive Summary",
        "",
        f"- Stage 5 status after this attempt: `{'Stage 5 achieved' if stage5_achieved else 'Stage 5 not yet achieved'}`.",
        "- This is research/evaluation output only.",
        "- Official production model logic was not changed.",
        "- Official production model remains `football_only_ensemble` with `core_football_only`.",
        "- Market blend remains benchmark-only unless statistically validated.",
        "",
        "## Tuning Protocol",
        "",
        "- All calibration, alpha tuning, and stacking use only prior tournament years.",
        "- 2014 has no prior Stage 5 paired market/model tournament rows, so tuned models use safe defaults or are skipped.",
        "- 2018 tunes on 2014 only.",
        "- 2022 tunes on 2014 and 2018 only.",
        "- All fair market-value comparisons use only rows where both official_model and bookmaker_odds_only are present.",
        "",
        "## Calibration Method",
        "",
        "- `official_model_calibrated` uses temperature/exponent scaling over prior tournament probabilities.",
        "- `bookmaker_calibrated` uses the same time-safe temperature method on bookmaker probabilities.",
        "- The selected exponent minimizes prior-year log loss; no target tournament rows are used for tuning.",
        "",
        "## Alpha Tuning Results",
        "",
        _markdown_table(alpha_tuning[alpha_tuning.get("selected", pd.Series(False, index=alpha_tuning.index)).eq(True)], ["test_year", "alpha", "tuning_years", "tuning_log_loss", "selected_alpha", "selected", "notes"]),
        "",
        "## Stacker Results",
        "",
        f"- Stacker requested: `{include_stacker}`.",
        f"- Stacker coefficient rows: `{len(stacker_coefficients)}`.",
        _markdown_table(stacker_coefficients.head(40), ["test_year", "class_label", "feature_name", "coefficient", "intercept", "tuning_source_years"]),
        "",
        "## Metrics Table",
        "",
        _markdown_table(metrics, ["model_name", "test_year", "n_matches", "n_skipped", "accuracy", "log_loss", "brier", "rps", "ece", "mean_confidence", "tuning_source_years", "notes"]),
        "",
        "## Average Metrics",
        "",
        _markdown_table(average, ["model_name", "years", "total_matches", "log_loss", "brier", "rps", "ece", "accuracy", "mean_confidence"]),
        "",
        "## Best Models",
        "",
        f"- Best log loss: `{best['log_loss']}`",
        f"- Best Brier: `{best['brier']}`",
        f"- Best RPS: `{best['rps']}`",
        f"- Best ECE: `{best['ece']}`",
        "",
        "## Significance Table",
        "",
        _markdown_table(significance, ["candidate_model", "baseline_model", "metric", "mean_delta", "ci_lower", "ci_upper", "statistically_meaningful", "n_matches", "interpretation"]),
        "",
        "## Does The Model Add Value Over Bookmaker Odds?",
        "",
        _market_value_text(significance),
        "",
        "## Limitations",
        "",
        "- 2014 tuned variants are limited because no earlier Stage 5 paired market/model tournament rows are available.",
        "- Stacking has very small training samples: 2018 uses 2014 only, and 2022 uses 2014 plus 2018.",
        "- This suite evaluates historical World Cup market-covered rows only; it does not promote market features into production.",
        "- Accuracy is secondary; log loss, Brier, RPS, calibration, and paired significance drive conclusions.",
        f"- Draw adjustment included: `{include_draw_adjustment}`.",
        "",
        "## Final Recommendation",
        "",
        _final_recommendation(stage5_achieved, significance),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_stage5_calibrated_market_decision(
    *,
    metrics: pd.DataFrame,
    significance: pd.DataFrame,
    stage5_achieved: bool,
    path: str | Path = STAGE5_CALIBRATED_MARKET_DECISION_PATH,
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    best = _best_models(_average_metrics(metrics))
    key_evidence = significance[
        significance["baseline_model"].eq("bookmaker_odds_only")
        & significance["metric"].eq("log_loss")
        & significance["candidate_model"].isin(["official_model_calibrated", "market_blend_tuned", "stacked_model_market"])
    ].copy()
    lines = [
        "# Stage 5 Calibrated Market Decision",
        "",
        f"Decision: **{'Stage 5 achieved' if stage5_achieved else 'Stage 5 not yet achieved'}**",
        "",
        "## Main Evidence",
        "",
        _markdown_table(key_evidence, ["candidate_model", "baseline_model", "metric", "mean_delta", "ci_lower", "ci_upper", "statistically_meaningful", "n_matches", "interpretation"]),
        "",
        "## Best Average Models",
        "",
        f"- Best log loss: `{best['log_loss']}`",
        f"- Best Brier: `{best['brier']}`",
        f"- Best RPS: `{best['rps']}`",
        f"- Best ECE: `{best['ece']}`",
        "",
        "## Decision Rule",
        "",
        "- Negative delta means the candidate is better than the baseline.",
        "- A candidate is statistically meaningful only when the 95% CI upper bound is below 0.",
        "- Stage 5 is not promoted unless a time-safe calibrated/blended/stacked candidate significantly beats bookmaker odds-only.",
        "",
        "## Next Step",
        "",
        _decision_next_step(stage5_achieved, significance),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _output_paths(output_root: Path) -> dict[str, Path]:
    return {
        "metrics": output_root / "stage5_calibrated_market_metrics.csv",
        "predictions": output_root / "stage5_calibrated_market_predictions.csv",
        "significance": output_root / "stage5_market_value_significance.csv",
        "alpha_tuning": output_root / "stage5_alpha_tuning.csv",
        "stacker_coefficients": output_root / "stage5_stacker_coefficients.csv",
    }


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Stage 5 output: {path}. Run `python -m src.cli stage5-benchmark --include-market` first.")
    return pd.read_csv(path, low_memory=False)


def _alpha_grid(step: float) -> list[float]:
    if step <= 0 or step > 1:
        raise ValueError("--alpha-grid-step must be in (0, 1]")
    count = int(round(1.0 / step))
    values = [round(i * step, 10) for i in range(count + 1)]
    if values[-1] != 1.0:
        values.append(1.0)
    return sorted(set(float(np.clip(value, 0.0, 1.0)) for value in values))


def _time_safe_temperature_calibration(
    prior: pd.DataFrame,
    target: pd.DataFrame,
    *,
    source_prefix: str,
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_prob = target[[f"{source_prefix}_away_prob", f"{source_prefix}_draw_prob", f"{source_prefix}_home_prob"]].to_numpy(dtype=float)
    tuning_years = _years_label(prior)
    if len(prior) < 20 or prior["actual_class"].nunique() < 2:
        frame = _frame_from_probabilities(
            target,
            target_prob,
            model_name,
            f"{source_prefix} probabilities with identity calibration",
            "none",
            "limited: no prior tournament rows; identity calibration used",
        )
        return frame, {"model_name": model_name, "limited": True, "notes": "limited: identity calibration due to insufficient prior tournament rows"}
    train_prob = prior[[f"{source_prefix}_away_prob", f"{source_prefix}_draw_prob", f"{source_prefix}_home_prob"]].to_numpy(dtype=float)
    y = prior["actual_class"].astype(int).to_numpy()
    gamma, train_loss = _fit_temperature(y, train_prob)
    calibrated = _temperature_calibrate(target_prob, gamma)
    frame = _frame_from_probabilities(
        target,
        calibrated,
        model_name,
        f"time-safe {source_prefix} temperature calibration gamma={gamma:.3f}",
        tuning_years,
        f"temperature/exponent calibration gamma={gamma:.3f}; train_log_loss={train_loss:.6f}",
    )
    return frame, {"model_name": model_name, "limited": False, "notes": f"gamma={gamma:.3f}"}


def _fit_temperature(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best_gamma = 1.0
    best_loss = float("inf")
    for gamma in np.linspace(0.45, 2.25, 37):
        calibrated = _temperature_calibrate(probabilities, float(gamma))
        loss = float(log_loss(y_true, calibrated, labels=[0, 1, 2]))
        if loss < best_loss - 1e-12:
            best_gamma = float(gamma)
            best_loss = loss
    return best_gamma, best_loss


def _temperature_calibrate(probabilities: np.ndarray, gamma: float) -> np.ndarray:
    return normalize_probability_rows(np.power(np.clip(probabilities, 1e-12, 1.0), float(gamma)))


def _market_blend_tuned_for_year(target: pd.DataFrame, prior: pd.DataFrame, alpha_grid: list[float]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    paired = pd.concat([prior, target], ignore_index=True, sort=False)
    selected_alpha, tuning = select_alpha_for_year(paired, int(target["tournament_year"].iloc[0]), alpha_grid)
    model_prob = target[["official_away_prob", "official_draw_prob", "official_home_prob"]].to_numpy(dtype=float)
    market_prob = target[["bookmaker_away_prob", "bookmaker_draw_prob", "bookmaker_home_prob"]].to_numpy(dtype=float)
    blended = blend_probabilities(model_prob, market_prob, selected_alpha)
    frame = _frame_from_probabilities(
        target,
        blended,
        "market_blend_tuned",
        f"alpha={selected_alpha:.2f}",
        _years_label(prior),
        f"time-safe alpha tuned blend alpha={selected_alpha:.2f}; p=alpha*model+(1-alpha)*market",
    )
    return frame, tuning.to_dict("records")


def _stacked_model_for_year(target: pd.DataFrame, prior: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any] | None]:
    year = int(target["tournament_year"].iloc[0])
    tuning_years = _years_label(prior)
    if len(prior) < 30 or prior["actual_class"].nunique() < 3:
        return (
            pd.DataFrame(),
            [],
            {
                "model_name": "stacked_model_market",
                "test_year": year,
                "notes": "skipped: insufficient prior paired tournament rows/classes for time-safe stacker",
                "tuning_source_years": tuning_years,
            },
        )
    feature_names = _stacker_feature_names()
    x_train = _stacker_features(prior)
    y_train = prior["actual_class"].astype(int).to_numpy()
    x_target = _stacker_features(target)
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(x_train, y_train)
    probabilities = np.zeros((len(target), 3), dtype=float)
    raw = model.predict_proba(x_target)
    for idx, cls in enumerate(model.classes_):
        probabilities[:, int(cls)] = raw[:, idx]
    probabilities = normalize_probability_rows(probabilities)
    frame = _frame_from_probabilities(
        target,
        probabilities,
        "stacked_model_market",
        "logistic meta-model over official and bookmaker probabilities",
        tuning_years,
        "time-safe logistic stacker trained on prior paired World Cup rows",
    )
    coefficient_rows = []
    for class_idx, cls in enumerate(model.classes_):
        for feature_name, coefficient in zip(feature_names, model.coef_[class_idx]):
            coefficient_rows.append(
                {
                    "test_year": year,
                    "class_label": CLASS_LABELS.get(int(cls), str(cls)),
                    "feature_name": feature_name,
                    "coefficient": float(coefficient),
                    "intercept": float(model.intercept_[class_idx]),
                    "tuning_source_years": tuning_years,
                }
            )
    return frame, coefficient_rows, None


def _draw_adjusted_for_year(target: pd.DataFrame, prior: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    year = int(target["tournament_year"].iloc[0])
    tuning_years = _years_label(prior)
    if len(prior) < 20 or prior["actual_class"].nunique() < 2:
        return pd.DataFrame(), {"model_name": "draw_adjusted_model", "test_year": year, "notes": "skipped: insufficient prior rows for draw adjustment", "tuning_source_years": tuning_years}
    y = prior["actual_class"].astype(int).to_numpy()
    train_prob = prior[["official_away_prob", "official_draw_prob", "official_home_prob"]].to_numpy(dtype=float)
    best_multiplier = 1.0
    best_loss = float("inf")
    for multiplier in np.linspace(0.60, 1.60, 21):
        adjusted = _draw_adjust(train_prob, float(multiplier))
        loss = float(log_loss(y, adjusted, labels=[0, 1, 2]))
        if loss < best_loss - 1e-12:
            best_multiplier = float(multiplier)
            best_loss = loss
    target_prob = target[["official_away_prob", "official_draw_prob", "official_home_prob"]].to_numpy(dtype=float)
    frame = _frame_from_probabilities(
        target,
        _draw_adjust(target_prob, best_multiplier),
        "draw_adjusted_model",
        f"draw multiplier={best_multiplier:.2f}",
        tuning_years,
        f"time-safe draw multiplier={best_multiplier:.2f}; train_log_loss={best_loss:.6f}",
    )
    return frame, None


def _draw_adjust(probabilities: np.ndarray, multiplier: float) -> np.ndarray:
    adjusted = normalize_probability_rows(probabilities).copy()
    adjusted[:, 1] *= float(multiplier)
    return normalize_probability_rows(adjusted)


def _stacker_feature_names() -> list[str]:
    return [
        "official_away_prob",
        "official_draw_prob",
        "official_home_prob",
        "bookmaker_away_prob",
        "bookmaker_draw_prob",
        "bookmaker_home_prob",
        "prob_diff_away",
        "prob_diff_draw",
        "prob_diff_home",
        "official_confidence",
        "bookmaker_confidence",
    ]


def _stacker_features(df: pd.DataFrame) -> np.ndarray:
    official = df[["official_away_prob", "official_draw_prob", "official_home_prob"]].to_numpy(dtype=float)
    bookmaker = df[["bookmaker_away_prob", "bookmaker_draw_prob", "bookmaker_home_prob"]].to_numpy(dtype=float)
    diff = official - bookmaker
    confidence = np.column_stack([official.max(axis=1), bookmaker.max(axis=1)])
    return np.column_stack([official, bookmaker, diff, confidence])


def _frame_from_probabilities(
    target: pd.DataFrame,
    probabilities: np.ndarray,
    model_name: str,
    feature_set: str,
    tuning_source_years: str,
    notes: str,
) -> pd.DataFrame:
    probabilities = normalize_probability_rows(probabilities)
    predicted_class = probabilities.argmax(axis=1)
    return pd.DataFrame(
        {
            "match_id": target["match_id"].astype(str).values,
            "date": target["date"].astype(str).values,
            "tournament_year": target["tournament_year"].astype(int).values,
            "stage": target["stage"].astype(str).values,
            "home_team": target["home_team"].astype(str).values,
            "away_team": target["away_team"].astype(str).values,
            "actual_result": target["actual_result"].astype(str).values,
            "actual_class": target["actual_class"].astype(int).values,
            "model_name": model_name,
            "feature_set": feature_set,
            "home_prob": probabilities[:, 2],
            "draw_prob": probabilities[:, 1],
            "away_prob": probabilities[:, 0],
            "predicted_result": [CLASS_LABELS[int(cls)] for cls in predicted_class],
            "confidence": probabilities.max(axis=1),
            "tuning_source_years": tuning_source_years,
            "data_source_notes": notes,
        }
    )


def _deduplicate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    return predictions.drop_duplicates(["tournament_year", "match_id", "model_name"], keep="last").reset_index(drop=True)


def _add_prediction_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    output = predictions.copy()
    probabilities = output[["away_prob", "draw_prob", "home_prob"]].to_numpy(dtype=float)
    probabilities = normalize_probability_rows(probabilities)
    y = output["actual_class"].astype(int).to_numpy()
    output["away_prob"] = probabilities[:, 0]
    output["draw_prob"] = probabilities[:, 1]
    output["home_prob"] = probabilities[:, 2]
    output["log_loss"] = per_row_log_loss(y, probabilities)
    output["brier"] = per_row_brier(y, probabilities)
    output["rps"] = per_row_rps(y, probabilities)
    output["confidence"] = probabilities.max(axis=1)
    output["predicted_result"] = [CLASS_LABELS[int(cls)] for cls in probabilities.argmax(axis=1)]
    columns = [
        "match_id",
        "date",
        "tournament_year",
        "stage",
        "home_team",
        "away_team",
        "actual_result",
        "actual_class",
        "model_name",
        "feature_set",
        "home_prob",
        "draw_prob",
        "away_prob",
        "predicted_result",
        "confidence",
        "log_loss",
        "brier",
        "rps",
        "tuning_source_years",
        "data_source_notes",
    ]
    return output[columns]


def _metrics_table(predictions: pd.DataFrame, target_counts: dict[int, int]) -> pd.DataFrame:
    rows = []
    if predictions.empty:
        return pd.DataFrame()
    for (model_name, year), group in predictions.groupby(["model_name", "tournament_year"], sort=True):
        probabilities = group[["away_prob", "draw_prob", "home_prob"]].to_numpy(dtype=float)
        y = group["actual_class"].astype(int).to_numpy()
        total = int(target_counts.get(int(year), len(group)))
        rows.append(
            {
                "model_name": model_name,
                "test_year": int(year),
                "n_matches": int(len(group)),
                "n_skipped": max(0, total - int(len(group))),
                "accuracy": float((group["predicted_result"].to_numpy() == group["actual_result"].to_numpy()).mean()),
                "log_loss": float(group["log_loss"].mean()),
                "brier": float(group["brier"].mean()),
                "rps": float(group["rps"].mean()),
                "ece": float(expected_calibration_error(y, probabilities)),
                "mean_confidence": float(group["confidence"].mean()),
                "tuning_source_years": _coalesce_tuning_years(group["tuning_source_years"]),
                "notes": _coalesce_notes(group["data_source_notes"]),
            }
        )
    columns = [
        "model_name",
        "test_year",
        "n_matches",
        "n_skipped",
        "accuracy",
        "log_loss",
        "brier",
        "rps",
        "ece",
        "mean_confidence",
        "tuning_source_years",
        "notes",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(["test_year", "model_name"], kind="stable").reset_index(drop=True)


def _skipped_metrics(skips: list[dict[str, Any]], target_counts: dict[int, int]) -> pd.DataFrame:
    rows = []
    for skip in skips:
        year = int(skip["test_year"])
        rows.append(
            {
                "model_name": skip["model_name"],
                "test_year": year,
                "n_matches": 0,
                "n_skipped": int(skip.get("n_skipped", target_counts.get(year, 0))),
                "accuracy": np.nan,
                "log_loss": np.nan,
                "brier": np.nan,
                "rps": np.nan,
                "ece": np.nan,
                "mean_confidence": np.nan,
                "tuning_source_years": skip.get("tuning_source_years", "none"),
                "notes": skip.get("notes", "skipped"),
            }
        )
    return pd.DataFrame(rows)


def _market_value_significance(predictions: pd.DataFrame, *, n_bootstrap: int) -> pd.DataFrame:
    comparisons = [
        ("official_model_calibrated", "bookmaker_odds_only"),
        ("market_blend_tuned", "bookmaker_odds_only"),
        ("stacked_model_market", "bookmaker_odds_only"),
        ("market_blend_tuned", "official_model"),
        ("stacked_model_market", "official_model"),
        ("official_model_calibrated", "official_model"),
    ]
    metrics = ["log_loss", "brier", "rps"]
    rows = []
    for candidate, baseline in comparisons:
        paired = _paired_model_rows(predictions, candidate, baseline)
        for metric in metrics:
            if paired.empty:
                rows.append(_empty_significance_row(candidate, baseline, metric, "not evaluated: missing paired rows"))
                continue
            values = pd.to_numeric(paired[f"{metric}_candidate"], errors="coerce") - pd.to_numeric(paired[f"{metric}_baseline"], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if len(values) == 0:
                rows.append(_empty_significance_row(candidate, baseline, metric, "not evaluated: no finite paired metric deltas"))
                continue
            mean_delta, lower, upper = paired_bootstrap_ci(values, n_bootstrap=n_bootstrap)
            meaningful = bool(upper < 0)
            rows.append(
                {
                    "candidate_model": candidate,
                    "baseline_model": baseline,
                    "metric": metric,
                    "mean_delta": mean_delta,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "statistically_meaningful": meaningful,
                    "n_matches": int(len(values)),
                    "interpretation": _interpret_delta(mean_delta, upper, meaningful),
                }
            )
    return pd.DataFrame(rows)


def _paired_model_rows(predictions: pd.DataFrame, candidate: str, baseline: str) -> pd.DataFrame:
    key = ["match_id", "tournament_year"]
    candidate_rows = predictions[predictions["model_name"].eq(candidate)].drop_duplicates(key, keep="last")
    baseline_rows = predictions[predictions["model_name"].eq(baseline)].drop_duplicates(key, keep="last")
    if candidate_rows.empty or baseline_rows.empty:
        return pd.DataFrame()
    return candidate_rows.merge(baseline_rows, on=key, suffixes=("_candidate", "_baseline"), how="inner")


def _empty_significance_row(candidate: str, baseline: str, metric: str, interpretation: str) -> dict[str, Any]:
    return {
        "candidate_model": candidate,
        "baseline_model": baseline,
        "metric": metric,
        "mean_delta": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "statistically_meaningful": False,
        "n_matches": 0,
        "interpretation": interpretation,
    }


def _interpret_delta(mean_delta: float, upper: float, meaningful: bool) -> str:
    if meaningful:
        return "candidate significantly improves baseline"
    if mean_delta < 0:
        return "candidate improves mean metric, but CI crosses 0"
    if mean_delta > 0:
        return "candidate is worse on mean metric"
    if not np.isfinite(upper):
        return "insufficient finite comparison rows"
    return "candidate ties baseline on mean metric"


def _stage5_achieved(significance: pd.DataFrame) -> bool:
    if significance.empty:
        return False
    candidates = {"official_model_calibrated", "market_blend_tuned", "stacked_model_market"}
    rows = significance[
        significance["candidate_model"].isin(candidates)
        & significance["baseline_model"].eq("bookmaker_odds_only")
        & significance["metric"].eq("log_loss")
        & significance["statistically_meaningful"].eq(True)
        & pd.to_numeric(significance["ci_upper"], errors="coerce").lt(0)
    ]
    return not rows.empty


def _average_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    valid = metrics[pd.to_numeric(metrics["n_matches"], errors="coerce").fillna(0).gt(0)].copy()
    if valid.empty:
        return pd.DataFrame()
    rows = []
    for model_name, group in valid.groupby("model_name", sort=True):
        weights = pd.to_numeric(group["n_matches"], errors="coerce").to_numpy(dtype=float)
        rows.append(
            {
                "model_name": model_name,
                "years": int(group["test_year"].nunique()),
                "total_matches": int(weights.sum()),
                "log_loss": _weighted(group, "log_loss", weights),
                "brier": _weighted(group, "brier", weights),
                "rps": _weighted(group, "rps", weights),
                "ece": _weighted(group, "ece", weights),
                "accuracy": _weighted(group, "accuracy", weights),
                "mean_confidence": _weighted(group, "mean_confidence", weights),
            }
        )
    return pd.DataFrame(rows).sort_values(["log_loss", "model_name"], kind="stable").reset_index(drop=True)


def _weighted(group: pd.DataFrame, column: str, weights: np.ndarray) -> float:
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _best_models(average: pd.DataFrame) -> dict[str, str]:
    return {metric: _best_model(average, metric) for metric in ["log_loss", "brier", "rps", "ece"]}


def _best_model(average: pd.DataFrame, metric: str) -> str:
    if average.empty or metric not in average.columns:
        return "not available"
    valid = average.dropna(subset=[metric]).sort_values([metric, "model_name"], kind="stable")
    if valid.empty:
        return "not available"
    row = valid.iloc[0]
    return f"{row['model_name']} ({metric}={float(row[metric]):.6f})"


def _market_value_text(significance: pd.DataFrame) -> str:
    rows = significance[
        significance["baseline_model"].eq("bookmaker_odds_only")
        & significance["metric"].eq("log_loss")
        & significance["candidate_model"].isin(["official_model_calibrated", "market_blend_tuned", "stacked_model_market"])
    ].copy()
    if rows.empty:
        return "- No calibrated/blended/stacked candidate had paired bookmaker odds-only rows."
    lines = []
    for _, row in rows.iterrows():
        meaningful = bool(row.get("statistically_meaningful", False))
        lines.append(
            f"- `{row['candidate_model']}` vs bookmaker log-loss delta `{float(row['mean_delta']):.6f}` "
            f"with CI `[{float(row['ci_lower']):.6f}, {float(row['ci_upper']):.6f}]`; significant: `{meaningful}`."
        )
    if not rows["statistically_meaningful"].astype(bool).any():
        lines.append("- No candidate significantly beats bookmaker odds-only on log loss.")
    return "\n".join(lines)


def _final_recommendation(stage5_achieved: bool, significance: pd.DataFrame) -> str:
    if stage5_achieved:
        return "A time-safe calibrated/blended/stacked benchmark candidate significantly beats bookmaker odds-only. Treat this as benchmark evidence only; production promotion still requires a separate policy decision."
    return "Stage 5 is still not achieved. Keep the production model unchanged and continue treating market blend/stacking as benchmark-only research."


def _decision_next_step(stage5_achieved: bool, significance: pd.DataFrame) -> str:
    if stage5_achieved:
        return "Audit the winning candidate for stability by tournament year and decide whether to keep it benchmark-only or design a separate production-readiness gate."
    rows = significance[
        significance["baseline_model"].eq("bookmaker_odds_only")
        & significance["metric"].eq("log_loss")
        & significance["candidate_model"].isin(["official_model_calibrated", "market_blend_tuned", "stacked_model_market"])
    ].copy()
    if rows.empty:
        return "Generate more paired historical market/model rows before tuning blends or stackers."
    best = rows.sort_values("mean_delta", kind="stable").head(1)
    if best.empty:
        return "Improve historical market coverage and add stronger pre-match football features validated by tournament-year folds."
    return f"Focus next on `{best.iloc[0]['candidate_model']}`, which had the best mean log-loss delta but did not satisfy the confidence interval rule."


def _years_label(df: pd.DataFrame) -> str:
    if df.empty or "tournament_year" not in df.columns:
        return "none"
    years = sorted(df["tournament_year"].dropna().astype(int).unique())
    return ",".join(str(year) for year in years) if years else "none"


def _coalesce_tuning_years(series: pd.Series) -> str:
    values = [str(value) for value in series.dropna().astype(str).unique() if str(value)]
    if not values:
        return "none"
    if len(values) == 1:
        return values[0]
    return "; ".join(values)


def _coalesce_notes(series: pd.Series) -> str:
    values = [str(value) for value in series.dropna().astype(str).unique() if str(value)]
    return values[0] if len(values) == 1 else "; ".join(values[:3])


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
