from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
OUTCOME_TO_CLASS = {"away_win": 0, "draw": 1, "home_win": 2}

STAGE5_DRAW_CALIBRATION_METRICS_PATH = resolve_project_path("data/backtests/stage5_draw_calibration_metrics.csv")
STAGE5_DRAW_CALIBRATION_PREDICTIONS_PATH = resolve_project_path("data/backtests/stage5_draw_calibration_predictions.csv")
STAGE5_DRAW_CALIBRATION_SIGNIFICANCE_PATH = resolve_project_path("data/backtests/stage5_draw_calibration_significance.csv")
STAGE5_DRAW_CALIBRATION_TUNING_PATH = resolve_project_path("data/backtests/stage5_draw_calibration_tuning.csv")
STAGE5_DRAW_CALIBRATION_REPORT_PATH = resolve_project_path("data/reports/stage5_draw_calibration_report.md")
STAGE5_DRAW_CALIBRATION_DECISION_PATH = resolve_project_path("data/reports/stage5_draw_calibration_decision.md")

CONFIDENCE_BINS = [0.0, 0.40, 0.50, 0.60, 0.70, 1.0]
CONFIDENCE_LABELS = ["0.00-0.40", "0.40-0.50", "0.50-0.60", "0.60-0.70", "0.70-1.00"]


@dataclass(frozen=True)
class Stage5DrawCalibrationResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    significance: pd.DataFrame
    tuning: pd.DataFrame
    report_path: Path
    decision_path: Path
    stage5_achieved: bool


def run_stage5_draw_calibration(
    *,
    years: list[int] | None = None,
    n_bootstrap: int = 1000,
    output_dir: str | Path = "data/backtests",
    report_path: str | Path = STAGE5_DRAW_CALIBRATION_REPORT_PATH,
    decision_path: str | Path = STAGE5_DRAW_CALIBRATION_DECISION_PATH,
    config: dict[str, Any] | None = None,
) -> Stage5DrawCalibrationResult:
    cfg = config or load_config()
    del cfg

    selected_years = sorted(int(year) for year in (years or [2014, 2018, 2022]))
    output_root = resolve_project_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    stage5_predictions = _read_required_csv(output_root / "stage5_match_predictions.csv")
    paired = paired_official_bookmaker(stage5_predictions, selected_years)
    if paired.empty:
        raise ValueError("No paired official_model and bookmaker_odds_only rows are available.")

    calibrated_market_predictions = _read_optional_csv(output_root / "stage5_calibrated_market_predictions.csv")
    calibrated_market = _standardize_optional_market_blend_tuned(calibrated_market_predictions, selected_years)

    prediction_frames: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    draw_conf_by_year: dict[int, pd.DataFrame] = {}

    for year in selected_years:
        target = paired[paired["tournament_year"].eq(year)].copy()
        prior = paired[paired["tournament_year"].lt(year)].copy()
        if target.empty:
            continue

        prediction_frames.append(
            _frame_from_probabilities(
                target,
                target[["official_away_prob", "official_draw_prob", "official_home_prob"]].to_numpy(dtype=float),
                "official_model",
                "existing official model probabilities",
                "none",
                "official model from Stage 5 benchmark",
            )
        )
        prediction_frames.append(
            _frame_from_probabilities(
                target,
                target[["bookmaker_away_prob", "bookmaker_draw_prob", "bookmaker_home_prob"]].to_numpy(dtype=float),
                "bookmaker_odds_only",
                "existing bookmaker odds-only probabilities",
                "none",
                "bookmaker odds-only from Stage 5 benchmark",
            )
        )

        draw_multiplier, draw_tuning = tune_draw_multiplier(prior, year)
        tuning_rows.extend(draw_tuning)
        target_official = target[["official_away_prob", "official_draw_prob", "official_home_prob"]].to_numpy(dtype=float)
        draw_probs = apply_draw_multiplier(target_official, draw_multiplier)
        prediction_frames.append(
            _frame_from_probabilities(
                target,
                draw_probs,
                "official_model_draw_calibrated",
                f"draw_multiplier={draw_multiplier:.3f}",
                _years_label(prior),
                f"time-safe draw multiplier={draw_multiplier:.3f}",
            )
        )

        confidence_params, confidence_tuning = tune_confidence_calibration(prior, year, source_probabilities=None)
        tuning_rows.extend(confidence_tuning)
        confidence_probs = apply_confidence_calibration(
            target_official,
            target["official_confidence_bucket"].astype(str),
            confidence_params,
        )
        prediction_frames.append(
            _frame_from_probabilities(
                target,
                confidence_probs,
                "official_model_confidence_calibrated",
                "confidence_bucket_temperature",
                _years_label(prior),
                "time-safe confidence-bucket exponent calibration",
            )
        )

        prior_draw_probs = None
        if not prior.empty:
            prior_official = prior[["official_away_prob", "official_draw_prob", "official_home_prob"]].to_numpy(dtype=float)
            prior_draw_probs = apply_draw_multiplier(prior_official, draw_multiplier)
        combined_params, combined_tuning = tune_confidence_calibration(
            prior,
            year,
            source_probabilities=prior_draw_probs,
            model_name="official_model_draw_confidence_calibrated",
        )
        tuning_rows.extend(combined_tuning)
        draw_conf_probs = apply_confidence_calibration(
            draw_probs,
            target["official_confidence_bucket"].astype(str),
            combined_params,
        )
        draw_conf_frame = _frame_from_probabilities(
            target,
            draw_conf_probs,
            "official_model_draw_confidence_calibrated",
            f"draw_multiplier={draw_multiplier:.3f}+confidence_bucket_temperature",
            _years_label(prior),
            "time-safe draw multiplier plus confidence-bucket calibration",
        )
        draw_conf_by_year[year] = draw_conf_frame.copy()
        prediction_frames.append(draw_conf_frame)

    calibrated_official_all = pd.concat(draw_conf_by_year.values(), ignore_index=True, sort=False) if draw_conf_by_year else pd.DataFrame()
    for year in selected_years:
        target = paired[paired["tournament_year"].eq(year)].copy()
        prior_target_rows = calibrated_official_all[calibrated_official_all["tournament_year"].lt(year)].copy()
        target_calibrated_rows = calibrated_official_all[calibrated_official_all["tournament_year"].eq(year)].copy()
        blend_frame, blend_tuning = _market_blend_draw_calibrated_for_year(target, prior_target_rows, target_calibrated_rows, year)
        if not blend_frame.empty:
            prediction_frames.append(blend_frame)
        tuning_rows.extend(blend_tuning)

    if not calibrated_market.empty:
        prediction_frames.append(calibrated_market)

    predictions = pd.concat([frame for frame in prediction_frames if not frame.empty], ignore_index=True, sort=False)
    predictions = _deduplicate_predictions(predictions)
    predictions = _add_prediction_scores(predictions)
    target_counts = paired.groupby("tournament_year").size().to_dict()
    metrics = _metrics_table(predictions, target_counts)
    significance = _significance(predictions, n_bootstrap=n_bootstrap)
    tuning = pd.DataFrame(tuning_rows)
    stage5_achieved = _stage5_achieved(significance)

    metrics.to_csv(output_root / "stage5_draw_calibration_metrics.csv", index=False)
    predictions.to_csv(output_root / "stage5_draw_calibration_predictions.csv", index=False)
    significance.to_csv(output_root / "stage5_draw_calibration_significance.csv", index=False)
    tuning.to_csv(output_root / "stage5_draw_calibration_tuning.csv", index=False)

    written_report = write_stage5_draw_calibration_report(
        metrics=metrics,
        significance=significance,
        tuning=tuning,
        stage5_achieved=stage5_achieved,
        path=report_path,
    )
    written_decision = write_stage5_draw_calibration_decision(
        metrics=metrics,
        significance=significance,
        tuning=tuning,
        stage5_achieved=stage5_achieved,
        path=decision_path,
    )
    return Stage5DrawCalibrationResult(
        metrics=metrics,
        predictions=predictions,
        significance=significance,
        tuning=tuning,
        report_path=written_report,
        decision_path=written_decision,
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
        "predicted_result",
        "confidence",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"stage5_match_predictions.csv is missing required columns: {sorted(missing)}")
    key = ["match_id", "tournament_year"]
    source = predictions[predictions["tournament_year"].astype(int).isin([int(year) for year in years])].copy()
    official = source[source["model_name"].eq("official_model")].drop_duplicates(key, keep="last").copy()
    bookmaker = source[source["model_name"].eq("bookmaker_odds_only")].drop_duplicates(key, keep="last").copy()
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
        }
    )
    output["official_confidence_bucket"] = confidence_bucket(output["official_confidence"])
    return output.sort_values(["tournament_year", "date", "match_id"], kind="stable").reset_index(drop=True)


def confidence_bucket(values: pd.Series) -> pd.Series:
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=CONFIDENCE_BINS, labels=CONFIDENCE_LABELS, include_lowest=True).astype(str)


def apply_draw_multiplier(probabilities: np.ndarray, multiplier: float) -> np.ndarray:
    output = normalize_probability_rows(probabilities).copy()
    output[:, 1] *= float(multiplier)
    return normalize_probability_rows(output)


def tune_draw_multiplier(prior: pd.DataFrame, test_year: int, grid: list[float] | None = None) -> tuple[float, list[dict[str, Any]]]:
    grid = grid or [round(value, 3) for value in np.arange(0.55, 1.81, 0.05)]
    tuning_years = _years_label(prior)
    rows: list[dict[str, Any]] = []
    if prior.empty or len(prior) < 20 or prior["actual_class"].nunique() < 2:
        selected = 1.0
        for multiplier in grid:
            rows.append(_tuning_row("official_model_draw_calibrated", int(test_year), "draw_multiplier", multiplier, tuning_years, np.nan, selected, "limited: no prior tournament rows; identity draw multiplier"))
        return selected, rows
    y_all = prior["actual_class"].astype(int).to_numpy()
    probs_all = prior[["official_away_prob", "official_draw_prob", "official_home_prob"]].to_numpy(dtype=float)
    draw_mask = y_all == OUTCOME_TO_CLASS["draw"]
    if int(draw_mask.sum()) >= 5:
        y = y_all[draw_mask]
        probs = probs_all[draw_mask]
        objective_note = "draw-specific log loss selected using prior tournament draw rows only"
    else:
        y = y_all
        probs = probs_all
        objective_note = "overall log loss selected using prior tournament rows only; insufficient prior draw rows"
    best_multiplier = 1.0
    best_loss = float("inf")
    losses: list[tuple[float, float]] = []
    for multiplier in grid:
        adjusted = apply_draw_multiplier(probs, multiplier)
        loss = float(log_loss(y, adjusted, labels=[0, 1, 2]))
        losses.append((float(multiplier), loss))
        if loss < best_loss - 1e-12:
            best_multiplier = float(multiplier)
            best_loss = loss
    for multiplier, loss in losses:
        rows.append(_tuning_row("official_model_draw_calibrated", int(test_year), "draw_multiplier", multiplier, tuning_years, loss, best_multiplier, objective_note))
    return best_multiplier, rows


def apply_confidence_calibration(probabilities: np.ndarray, buckets: pd.Series, params: dict[str, float]) -> np.ndarray:
    output = normalize_probability_rows(probabilities).copy()
    bucket_values = buckets.astype(str).to_numpy()
    for bucket, gamma in params.items():
        mask = bucket_values == str(bucket)
        if mask.any():
            output[mask] = _temperature(output[mask], gamma)
    return normalize_probability_rows(output)


def tune_confidence_calibration(
    prior: pd.DataFrame,
    test_year: int,
    *,
    source_probabilities: np.ndarray | None,
    model_name: str = "official_model_confidence_calibrated",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    tuning_years = _years_label(prior)
    params = {label: 1.0 for label in CONFIDENCE_LABELS}
    rows: list[dict[str, Any]] = []
    if prior.empty or len(prior) < 20 or prior["actual_class"].nunique() < 2:
        for bucket in CONFIDENCE_LABELS:
            rows.append(_tuning_row(model_name, int(test_year), f"confidence_gamma:{bucket}", 1.0, tuning_years, np.nan, 1.0, "limited: no prior rows; identity confidence calibration"))
        return params, rows

    probs = source_probabilities
    if probs is None:
        probs = prior[["official_away_prob", "official_draw_prob", "official_home_prob"]].to_numpy(dtype=float)
    probs = normalize_probability_rows(probs)
    y = prior["actual_class"].astype(int).to_numpy()
    buckets = prior["official_confidence_bucket"].astype(str).to_numpy()
    for bucket in CONFIDENCE_LABELS:
        mask = buckets == bucket
        if int(mask.sum()) < 10 or len(np.unique(y[mask])) < 2:
            rows.append(_tuning_row(model_name, int(test_year), f"confidence_gamma:{bucket}", 1.0, tuning_years, np.nan, 1.0, "identity: insufficient rows/classes in bucket"))
            continue
        mean_confidence = float(probs[mask].max(axis=1).mean())
        empirical_accuracy = float((probs[mask].argmax(axis=1) == y[mask]).mean())
        if empirical_accuracy < mean_confidence:
            gamma_grid = np.linspace(0.55, 1.0, 10)
        else:
            gamma_grid = np.linspace(1.0, 1.35, 8)
        best_gamma = 1.0
        best_loss = float("inf")
        losses: list[tuple[float, float]] = []
        for gamma in gamma_grid:
            adjusted = _temperature(probs[mask], float(gamma))
            loss = float(log_loss(y[mask], adjusted, labels=[0, 1, 2]))
            losses.append((float(gamma), loss))
            if loss < best_loss - 1e-12:
                best_gamma = float(gamma)
                best_loss = loss
        params[bucket] = best_gamma
        for gamma, loss in losses:
            rows.append(
                {
                    **_tuning_row(model_name, int(test_year), f"confidence_gamma:{bucket}", gamma, tuning_years, loss, best_gamma, "bucket gamma selected using prior tournament rows only"),
                    "bucket_n": int(mask.sum()),
                    "bucket_mean_confidence": mean_confidence,
                    "bucket_empirical_accuracy": empirical_accuracy,
                }
            )
    return params, rows


def _temperature(probabilities: np.ndarray, gamma: float) -> np.ndarray:
    return normalize_probability_rows(np.power(np.clip(probabilities, 1e-12, 1.0), float(gamma)))


def _market_blend_draw_calibrated_for_year(
    target: pd.DataFrame,
    prior_calibrated_rows: pd.DataFrame,
    target_calibrated_rows: pd.DataFrame,
    year: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    grid = [round(value, 2) for value in np.arange(0.0, 1.0001, 0.05)]
    tuning_years = _years_label(prior_calibrated_rows)
    rows: list[dict[str, Any]] = []
    if target.empty or target_calibrated_rows.empty:
        return pd.DataFrame(), rows
    selected_alpha = 0.0
    if prior_calibrated_rows.empty:
        for alpha in grid:
            rows.append(_tuning_row("market_blend_draw_calibrated", int(year), "alpha", alpha, "none", np.nan, selected_alpha, "limited: no prior calibrated rows; safe default alpha=0.0"))
    else:
        prior = _join_calibrated_with_bookmaker(prior_calibrated_rows, target_context=None)
        y = prior["actual_class"].astype(int).to_numpy()
        model_prob = prior[["away_prob", "draw_prob", "home_prob"]].to_numpy(dtype=float)
        market_prob = prior[["bookmaker_away_prob", "bookmaker_draw_prob", "bookmaker_home_prob"]].to_numpy(dtype=float)
        best_loss = float("inf")
        losses: list[tuple[float, float]] = []
        for alpha in grid:
            blended = _blend(model_prob, market_prob, alpha)
            loss = float(log_loss(y, blended, labels=[0, 1, 2]))
            losses.append((alpha, loss))
            if loss < best_loss - 1e-12:
                selected_alpha = float(alpha)
                best_loss = loss
        for alpha, loss in losses:
            rows.append(_tuning_row("market_blend_draw_calibrated", int(year), "alpha", alpha, tuning_years, loss, selected_alpha, "alpha selected using prior tournament rows only"))
    target_joined = _join_calibrated_with_bookmaker(target_calibrated_rows, target_context=target)
    model_prob = target_joined[["away_prob", "draw_prob", "home_prob"]].to_numpy(dtype=float)
    market_prob = target_joined[["bookmaker_away_prob", "bookmaker_draw_prob", "bookmaker_home_prob"]].to_numpy(dtype=float)
    blended = _blend(model_prob, market_prob, selected_alpha)
    frame = _frame_from_probabilities(
        target,
        blended,
        "market_blend_draw_calibrated",
        f"alpha={selected_alpha:.2f}+draw_confidence_calibrated_official",
        tuning_years,
        f"time-safe blend of bookmaker and draw/confidence calibrated official alpha={selected_alpha:.2f}",
    )
    return frame, rows


def _join_calibrated_with_bookmaker(calibrated_rows: pd.DataFrame, target_context: pd.DataFrame | None) -> pd.DataFrame:
    rows = calibrated_rows.copy()
    if target_context is None:
        return rows
    key = ["match_id", "tournament_year"]
    bookmaker_cols = key + ["bookmaker_away_prob", "bookmaker_draw_prob", "bookmaker_home_prob"]
    return rows.drop(columns=[col for col in ["bookmaker_away_prob", "bookmaker_draw_prob", "bookmaker_home_prob"] if col in rows.columns]).merge(
        target_context[bookmaker_cols].drop_duplicates(key),
        on=key,
        how="left",
    )


def _blend(model_prob: np.ndarray, market_prob: np.ndarray, alpha: float) -> np.ndarray:
    return normalize_probability_rows(float(alpha) * normalize_probability_rows(model_prob) + (1.0 - float(alpha)) * normalize_probability_rows(market_prob))


def _frame_from_probabilities(
    target: pd.DataFrame,
    probabilities: np.ndarray,
    model_name: str,
    feature_set: str,
    tuning_source_years: str,
    notes: str,
) -> pd.DataFrame:
    probabilities = normalize_probability_rows(probabilities)
    predicted = probabilities.argmax(axis=1)
    frame = pd.DataFrame(
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
            "predicted_result": [CLASS_LABELS[int(cls)] for cls in predicted],
            "confidence": probabilities.max(axis=1),
            "tuning_source_years": tuning_source_years,
            "data_source_notes": notes,
        }
    )
    for column in ["bookmaker_away_prob", "bookmaker_draw_prob", "bookmaker_home_prob", "official_confidence_bucket"]:
        if column in target.columns:
            frame[column] = target[column].values
    return frame


def _standardize_optional_market_blend_tuned(predictions: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    if predictions.empty or "model_name" not in predictions.columns:
        return pd.DataFrame()
    rows = predictions[
        predictions["model_name"].eq("market_blend_tuned")
        & predictions["tournament_year"].astype(int).isin([int(year) for year in years])
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["tuning_source_years"] = rows.get("tuning_source_years", pd.Series("external", index=rows.index)).fillna("external")
    rows["data_source_notes"] = rows.get("data_source_notes", pd.Series("market_blend_tuned from calibrated-market suite", index=rows.index)).fillna("market_blend_tuned from calibrated-market suite")
    return rows[
        [
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
            "tuning_source_years",
            "data_source_notes",
        ]
    ].copy()


def _deduplicate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions.drop_duplicates(["match_id", "tournament_year", "model_name"], keep="last").reset_index(drop=True)


def _add_prediction_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    probs = normalize_probability_rows(output[["away_prob", "draw_prob", "home_prob"]].to_numpy(dtype=float))
    y = output["actual_class"].astype(int).to_numpy()
    output["away_prob"] = probs[:, 0]
    output["draw_prob"] = probs[:, 1]
    output["home_prob"] = probs[:, 2]
    output["log_loss"] = per_row_log_loss(y, probs)
    output["brier"] = per_row_brier(y, probs)
    output["rps"] = per_row_rps(y, probs)
    output["confidence"] = probs.max(axis=1)
    output["predicted_result"] = [CLASS_LABELS[int(cls)] for cls in probs.argmax(axis=1)]
    return output


def _metrics_table(predictions: pd.DataFrame, target_counts: dict[int, int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_name, year), group in predictions.groupby(["model_name", "tournament_year"], sort=True):
        y = group["actual_class"].astype(int).to_numpy()
        probs = group[["away_prob", "draw_prob", "home_prob"]].to_numpy(dtype=float)
        draw_mask = group["actual_result"].eq("draw")
        conf_mask = group["confidence"].between(0.60, 0.70, inclusive="left")
        total = int(target_counts.get(int(year), len(group)))
        rows.append(
            {
                "model_name": model_name,
                "test_year": int(year),
                "n_matches": int(len(group)),
                "n_skipped": max(0, total - int(len(group))),
                "accuracy": float(group["predicted_result"].eq(group["actual_result"]).mean()),
                "log_loss": float(group["log_loss"].mean()),
                "brier": float(group["brier"].mean()),
                "rps": float(group["rps"].mean()),
                "ece": float(expected_calibration_error(y, probs)),
                "mean_confidence": float(group["confidence"].mean()),
                "draw_log_loss": float(group.loc[draw_mask, "log_loss"].mean()) if draw_mask.any() else np.nan,
                "draw_recall": float(group.loc[draw_mask, "predicted_result"].eq("draw").mean()) if draw_mask.any() else np.nan,
                "draw_prediction_rate": float(group["predicted_result"].eq("draw").mean()),
                "confidence_060_070_n": int(conf_mask.sum()),
                "confidence_060_070_log_loss": float(group.loc[conf_mask, "log_loss"].mean()) if conf_mask.any() else np.nan,
                "tuning_source_years": _coalesce(group["tuning_source_years"]),
                "notes": _coalesce(group["data_source_notes"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["test_year", "model_name"], kind="stable").reset_index(drop=True)


def _significance(predictions: pd.DataFrame, *, n_bootstrap: int) -> pd.DataFrame:
    comparisons = [
        ("official_model_draw_calibrated", "official_model"),
        ("official_model_draw_confidence_calibrated", "official_model"),
        ("official_model_draw_confidence_calibrated", "bookmaker_odds_only"),
        ("market_blend_draw_calibrated", "bookmaker_odds_only"),
        ("market_blend_draw_calibrated", "market_blend_tuned"),
    ]
    rows = []
    for candidate, baseline in comparisons:
        paired = _paired_model_rows(predictions, candidate, baseline)
        for metric in ["log_loss", "brier", "rps"]:
            if paired.empty:
                rows.append(_empty_significance_row(candidate, baseline, metric, "not evaluated: missing paired rows"))
                continue
            deltas = pd.to_numeric(paired[f"{metric}_candidate"], errors="coerce") - pd.to_numeric(paired[f"{metric}_baseline"], errors="coerce")
            values = deltas.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
            if len(values) == 0:
                rows.append(_empty_significance_row(candidate, baseline, metric, "not evaluated: no finite deltas"))
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
                    "interpretation": _interpret_delta(mean_delta, meaningful),
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


def _interpret_delta(mean_delta: float, meaningful: bool) -> str:
    if meaningful:
        return "candidate significantly improves baseline"
    if mean_delta < 0:
        return "candidate improves mean metric, but CI crosses 0"
    if mean_delta > 0:
        return "candidate is worse on mean metric"
    return "candidate ties baseline on mean metric"


def _stage5_achieved(significance: pd.DataFrame) -> bool:
    rows = significance[
        significance["candidate_model"].isin(["official_model_draw_confidence_calibrated", "market_blend_draw_calibrated"])
        & significance["baseline_model"].eq("bookmaker_odds_only")
        & significance["metric"].eq("log_loss")
        & significance["statistically_meaningful"].eq(True)
        & pd.to_numeric(significance["ci_upper"], errors="coerce").lt(0)
    ]
    return not rows.empty


def write_stage5_draw_calibration_report(
    *,
    metrics: pd.DataFrame,
    significance: pd.DataFrame,
    tuning: pd.DataFrame,
    stage5_achieved: bool,
    path: str | Path = STAGE5_DRAW_CALIBRATION_REPORT_PATH,
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    selected = _selected_tuning(tuning)
    average = _average_metrics(metrics)
    lines = [
        "# Stage 5 Draw & Overconfidence Calibration Report",
        "",
        "## Executive Summary",
        "",
        f"- Stage 5 status after this attempt: `{'Stage 5 achieved' if stage5_achieved else 'Stage 5 not yet achieved'}`.",
        "- Official production model logic was not changed.",
        "- This is historical no-leakage research output only.",
        "- Negative deltas mean the candidate is better than the baseline.",
        "",
        "## Tuning Protocol",
        "",
        "- 2014 uses identity calibration and is marked limited because no prior tournament is available.",
        "- 2018 tunes only on 2014.",
        "- 2022 tunes only on 2014 and 2018.",
        "- No 2026 live data is used.",
        "",
        "## Selected Tuning Parameters",
        "",
        _markdown_table(selected, ["model_name", "test_year", "parameter", "value", "tuning_source_years", "tuning_log_loss", "selected", "notes"]),
        "",
        "## Metrics",
        "",
        _markdown_table(metrics, ["model_name", "test_year", "n_matches", "accuracy", "log_loss", "brier", "rps", "ece", "draw_log_loss", "draw_recall", "draw_prediction_rate", "confidence_060_070_log_loss", "tuning_source_years", "notes"]),
        "",
        "## Average Metrics",
        "",
        _markdown_table(average, ["model_name", "years", "total_matches", "log_loss", "brier", "rps", "ece", "draw_log_loss", "draw_recall", "draw_prediction_rate", "confidence_060_070_log_loss"]),
        "",
        "## Significance",
        "",
        _markdown_table(significance, ["candidate_model", "baseline_model", "metric", "mean_delta", "ci_lower", "ci_upper", "statistically_meaningful", "n_matches", "interpretation"]),
        "",
        "## Draw And Overconfidence Summary",
        "",
        _draw_overconfidence_summary(metrics),
        "",
        "## Final Recommendation",
        "",
        _final_recommendation(stage5_achieved, significance),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_stage5_draw_calibration_decision(
    *,
    metrics: pd.DataFrame,
    significance: pd.DataFrame,
    tuning: pd.DataFrame,
    stage5_achieved: bool,
    path: str | Path = STAGE5_DRAW_CALIBRATION_DECISION_PATH,
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    key = significance[
        significance["metric"].eq("log_loss")
        & significance["candidate_model"].isin(["official_model_draw_confidence_calibrated", "market_blend_draw_calibrated"])
        & significance["baseline_model"].isin(["bookmaker_odds_only", "official_model"])
    ]
    lines = [
        "# Stage 5 Draw Calibration Decision",
        "",
        f"Decision: **{'Stage 5 achieved' if stage5_achieved else 'Stage 5 not yet achieved'}**",
        "",
        "## Main Evidence",
        "",
        _markdown_table(key, ["candidate_model", "baseline_model", "metric", "mean_delta", "ci_lower", "ci_upper", "statistically_meaningful", "n_matches", "interpretation"]),
        "",
        "## Selected Draw Multipliers",
        "",
        _markdown_table(_selected_tuning(tuning, parameter_prefix="draw_multiplier"), ["model_name", "test_year", "parameter", "value", "tuning_source_years", "tuning_log_loss", "notes"]),
        "",
        "## Decision Rule",
        "",
        "- Stage 5 can only be claimed if a no-leakage candidate significantly beats bookmaker odds-only on log loss.",
        "- Statistically meaningful requires the 95% CI upper bound to be below 0.",
        "",
        "## Recommendation",
        "",
        _final_recommendation(stage5_achieved, significance),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Stage 5 output: {path}")
    return pd.read_csv(path, low_memory=False)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _tuning_row(
    model_name: str,
    test_year: int,
    parameter: str,
    value: float,
    tuning_years: str,
    tuning_log_loss: float,
    selected_value: float,
    notes: str,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "test_year": test_year,
        "parameter": parameter,
        "value": float(value),
        "tuning_source_years": tuning_years,
        "tuning_log_loss": tuning_log_loss,
        "selected_value": selected_value,
        "selected": bool(abs(float(value) - float(selected_value)) < 1e-12),
        "notes": notes,
    }


def _selected_tuning(tuning: pd.DataFrame, parameter_prefix: str | None = None) -> pd.DataFrame:
    if tuning.empty:
        return tuning
    selected = tuning[tuning["selected"].astype(bool)].copy()
    if parameter_prefix:
        selected = selected[selected["parameter"].astype(str).str.startswith(parameter_prefix)]
    return selected.sort_values(["test_year", "model_name", "parameter"], kind="stable")


def _average_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    valid = metrics[pd.to_numeric(metrics["n_matches"], errors="coerce").fillna(0).gt(0)].copy()
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
                "draw_log_loss": _weighted(group, "draw_log_loss", weights),
                "draw_recall": _weighted(group, "draw_recall", weights),
                "draw_prediction_rate": _weighted(group, "draw_prediction_rate", weights),
                "confidence_060_070_log_loss": _weighted(group, "confidence_060_070_log_loss", weights),
            }
        )
    return pd.DataFrame(rows).sort_values(["log_loss", "model_name"], kind="stable")


def _weighted(group: pd.DataFrame, column: str, weights: np.ndarray) -> float:
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _draw_overconfidence_summary(metrics: pd.DataFrame) -> str:
    average = _average_metrics(metrics)
    def metric(model: str, column: str) -> float:
        row = average[average["model_name"].eq(model)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][column])
    official_draw = metric("official_model", "draw_log_loss")
    draw_conf_draw = metric("official_model_draw_confidence_calibrated", "draw_log_loss")
    official_conf = metric("official_model", "confidence_060_070_log_loss")
    conf_cal = metric("official_model_confidence_calibrated", "confidence_060_070_log_loss")
    return "\n".join(
        [
            f"- Official draw log loss: `{official_draw:.6f}`.",
            f"- Draw+confidence calibrated draw log loss: `{draw_conf_draw:.6f}`.",
            f"- Official confidence 0.60-0.70 log loss: `{official_conf:.6f}`.",
            f"- Confidence-calibrated confidence 0.60-0.70 log loss: `{conf_cal:.6f}`.",
        ]
    )


def _final_recommendation(stage5_achieved: bool, significance: pd.DataFrame) -> str:
    if stage5_achieved:
        return "A draw/confidence-calibrated benchmark candidate significantly beats bookmaker odds-only. Treat this as research evidence only; production promotion still requires a separate policy decision."
    return "Stage 5 is still not achieved. Keep production unchanged and continue testing draw-aware scoreline/Dixon-Coles and stronger pre-match football signals through no-leakage folds."


def _years_label(df: pd.DataFrame) -> str:
    if df.empty or "tournament_year" not in df.columns:
        return "none"
    years = sorted(df["tournament_year"].dropna().astype(int).unique())
    return ",".join(str(year) for year in years) if years else "none"


def _coalesce(series: pd.Series) -> str:
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
