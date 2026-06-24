from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.backtesting.worldcup_backtest import (
    WORLD_CUP_START_DATES,
    run_worldcup_backtests,
    split_worldcup_backtest_fold,
)
from src.config import config_path, load_config, resolve_project_path
from src.io_utils import read_dataframe
from src.models.evaluate import (
    brier_score_multiclass,
    expected_calibration_error,
    ranked_probability_score,
)


STAGE5_METRICS_PATH = resolve_project_path("data/backtests/stage5_benchmark_metrics.csv")
STAGE5_PREDICTIONS_PATH = resolve_project_path("data/backtests/stage5_match_predictions.csv")
STAGE5_SIGNIFICANCE_PATH = resolve_project_path("data/backtests/stage5_significance_tests.csv")
STAGE5_CALIBRATION_PATH = resolve_project_path("data/backtests/stage5_calibration_table.csv")
STAGE5_STAGE_CALIBRATION_PATH = resolve_project_path("data/backtests/stage5_calibration_by_stage.csv")
STAGE5_REPORT_PATH = resolve_project_path("data/reports/stage5_benchmark_report.md")
STAGE5_DECISION_PATH = resolve_project_path("data/reports/stage5_research_decision.md")

CLASS_LABELS = {0: "away_win", 1: "draw", 2: "home_win"}
MODEL_BASELINE_TYPES = {
    "official_model": "official_model",
    "random_uniform": "random_baseline",
    "historical_class_frequency": "historical_frequency",
    "fifa_ranking_only": "ranking_baseline",
    "elo_only": "elo_baseline",
    "rolling_form_only": "form_baseline",
    "poisson_goal_model": "goal_model_baseline",
    "bookmaker_raw_implied": "market_baseline",
    "bookmaker_odds_only": "market_baseline",
    "bookmaker_calibrated": "market_calibration_baseline",
    "market_blend": "market_blend_benchmark",
    "dixon_coles": "skipped_or_unavailable",
}


@dataclass(frozen=True)
class Stage5Result:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    significance: pd.DataFrame
    calibration: pd.DataFrame
    report_path: Path
    decision_path: Path
    stage5_achieved: bool


def run_stage5_benchmark(
    *,
    years: list[int] | None = None,
    n_bootstrap: int = 1000,
    include_market: bool = False,
    include_poisson: bool = False,
    include_dixon_coles: bool = False,
    output_dir: str | Path = "data/backtests",
    model_name: str = "catboost",
    feature_set: str = "core_football_only",
    config: dict[str, Any] | None = None,
) -> Stage5Result:
    cfg = config or load_config()
    selected_years = years or [2014, 2018, 2022]
    output_root = resolve_project_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    base_predictions = _load_existing_backtest_predictions(selected_years)
    if base_predictions.empty:
        base_predictions, _, _ = run_worldcup_backtests(
            model_name=model_name,
            feature_set=feature_set,
            config=cfg,
            years=selected_years,
        )
    clean_matches = read_dataframe(config_path(cfg, "all_matches_clean"))

    base_eval_predictions = base_predictions[base_predictions["world_cup_year"].astype(int).isin(selected_years)].copy() if not base_predictions.empty else base_predictions
    prediction_frames = [_standardize_existing_backtest_predictions(base_eval_predictions, include_market=include_market)]
    fold_targets: dict[int, pd.DataFrame] = {}
    skipped_notes: list[dict[str, Any]] = []
    market_conversion_method = _read_market_conversion_method()

    for year in selected_years:
        train_df, target_df = split_worldcup_backtest_fold(clean_matches, year)
        if target_df.empty:
            skipped_notes.append({"model_name": "all_models", "test_year": year, "reason": "target_tournament_missing", "n_skipped": 0})
            continue
        target_df = target_df.copy()
        target_df["_stage5_stage"] = _stage_labels(target_df)
        fold_targets[year] = target_df
        prediction_frames.append(_probability_frame_for_fold(target_df, _uniform_probabilities(len(target_df)), year, "random_uniform", "none", "uniform 1/3 probabilities"))
        prediction_frames.append(
            _probability_frame_for_fold(
                target_df,
                _historical_frequency_probabilities(train_df, len(target_df)),
                year,
                "historical_class_frequency",
                "historical_outcome_frequency",
                "historical class frequencies estimated from pre-tournament matches",
            )
        )
        if include_poisson:
            poisson_prob, poisson_note = _poisson_goal_probabilities(train_df, target_df)
            prediction_frames.append(_probability_frame_for_fold(target_df, poisson_prob, year, "poisson_goal_model", "attack_defense_smoothing", poisson_note))
        if include_market:
            market_frame, market_notes = _market_extra_prediction_frames(
                base_predictions=base_predictions,
                year=year,
                odds_method=market_conversion_method,
            )
            if not market_frame.empty:
                prediction_frames.append(market_frame)
            skipped_notes.extend(market_notes)
        if include_dixon_coles:
            skipped_notes.append(
                {
                    "model_name": "dixon_coles",
                    "test_year": year,
                    "reason": "skipped: no validated time-decayed Dixon-Coles parameter fitting exists in project yet",
                    "n_skipped": len(target_df),
                }
            )

    predictions = pd.concat([frame for frame in prediction_frames if not frame.empty], ignore_index=True, sort=False)
    predictions = _deduplicate_stage5_predictions(predictions)
    predictions = _add_prediction_scores(predictions)
    total_by_year = _target_counts_by_year(predictions, fold_targets)
    metrics = _stage5_metrics(predictions, total_by_year)
    if include_dixon_coles:
        metrics = pd.concat([metrics, _skipped_dixon_coles_metrics(selected_years, total_by_year)], ignore_index=True, sort=False)
    calibration = _calibration_table(predictions)
    stage_calibration = _stage_calibration_table(predictions)
    significance = _stage5_significance(predictions, n_bootstrap=n_bootstrap)
    decision = _stage5_decision(significance)

    paths = _stage5_paths(output_root)
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    significance.to_csv(paths["significance"], index=False)
    calibration.to_csv(paths["calibration"], index=False)
    stage_calibration.to_csv(paths["stage_calibration"], index=False)
    report_path = write_stage5_benchmark_report(
        metrics=metrics,
        predictions=predictions,
        significance=significance,
        calibration=calibration,
        stage_calibration=stage_calibration,
        skipped_notes=pd.DataFrame(skipped_notes),
        years=selected_years,
        stage5_achieved=decision,
        path=STAGE5_REPORT_PATH,
    )
    decision_path = write_stage5_decision_report(
        metrics=metrics,
        significance=significance,
        skipped_notes=pd.DataFrame(skipped_notes),
        stage5_achieved=decision,
        path=STAGE5_DECISION_PATH,
    )
    return Stage5Result(
        predictions=predictions,
        metrics=metrics,
        significance=significance,
        calibration=calibration,
        report_path=report_path,
        decision_path=decision_path,
        stage5_achieved=decision,
    )


def _stage5_paths(output_root: Path) -> dict[str, Path]:
    return {
        "metrics": output_root / "stage5_benchmark_metrics.csv",
        "predictions": output_root / "stage5_match_predictions.csv",
        "significance": output_root / "stage5_significance_tests.csv",
        "calibration": output_root / "stage5_calibration_table.csv",
        "stage_calibration": output_root / "stage5_calibration_by_stage.csv",
    }


def _load_existing_backtest_predictions(years: list[int]) -> pd.DataFrame:
    path = resolve_project_path("data/backtests/worldcup_backtest_predictions.csv")
    if not path.exists():
        return pd.DataFrame()
    predictions = pd.read_csv(path, low_memory=False)
    if predictions.empty or "world_cup_year" not in predictions.columns or "model_name" not in predictions.columns:
        return pd.DataFrame()
    available_years = set(predictions["world_cup_year"].dropna().astype(int).unique())
    required_years = set(int(year) for year in years)
    required_models = {"final_model", "elo_only", "fifa_ranking_only", "majority_class_baseline"}
    available_models = set(predictions["model_name"].dropna().astype(str).unique())
    if not required_years.issubset(available_years):
        return pd.DataFrame()
    if not required_models.issubset(available_models):
        return pd.DataFrame()
    return predictions[predictions["world_cup_year"].astype(int).isin(required_years | {2010})].copy()


def _standardize_existing_backtest_predictions(predictions: pd.DataFrame, *, include_market: bool) -> pd.DataFrame:
    if predictions.empty:
        return _empty_prediction_frame()
    frames: list[pd.DataFrame] = []
    mappings = {
        "final_model": ("official_model", "football_only_ensemble/core_football_only", "official model from no-leakage World Cup backtest"),
        "elo_only": ("elo_only", "elo_rating_features", "Elo-only feature model from no-leakage World Cup backtest"),
        "fifa_ranking_only": ("fifa_ranking_only", "fifa_ranking_features", "FIFA ranking-only feature model from no-leakage World Cup backtest"),
        "rolling_form_only": ("rolling_form_only", "rolling_form_features", "rolling-form-only feature model from no-leakage World Cup backtest"),
        "majority_class_baseline": ("historical_class_frequency", "historical_outcome_frequency", "historical class-frequency baseline from pre-tournament matches"),
    }
    if include_market:
        mappings.update(
            {
                "bookmaker_odds_only": ("bookmaker_odds_only", "best_market_odds_conversion", "bookmaker odds-only baseline on odds-covered matches"),
                "market_blend": ("market_blend", "football_only_ensemble+market", "benchmark-only blend on odds-covered matches"),
            }
        )
    for source_model, (target_model, feature_set, note) in mappings.items():
        scope = "odds_covered" if source_model in {"bookmaker_odds_only", "market_blend"} else "all_matches"
        source = predictions[predictions["model_name"].eq(source_model) & predictions["evaluation_scope"].eq(scope)].copy()
        if source.empty:
            continue
        frames.append(
            _standardized_prediction_rows(
                source,
                model_name=target_model,
                feature_set=feature_set,
                baseline_type=MODEL_BASELINE_TYPES[target_model],
                data_source_notes=note,
            )
        )
    return pd.concat(frames, ignore_index=True, sort=False) if frames else _empty_prediction_frame()


def _standardized_prediction_rows(
    source: pd.DataFrame,
    *,
    model_name: str,
    feature_set: str,
    baseline_type: str,
    data_source_notes: str,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "match_id": source.get("match_id", pd.Series([pd.NA] * len(source))).astype(str),
            "date": pd.to_datetime(source["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "tournament_year": source["world_cup_year"].astype(int),
            "stage": source.get("stage", pd.Series("unknown", index=source.index)).fillna("unknown").astype(str),
            "home_team": source["home_team"].astype(str),
            "away_team": source["away_team"].astype(str),
            "actual_class": source["actual_class"].astype(int),
            "actual_result": source["actual_class"].astype(int).map(CLASS_LABELS),
            "model_name": model_name,
            "feature_set": feature_set,
            "baseline_type": baseline_type,
            "home_prob": pd.to_numeric(source["p_home_win"], errors="coerce"),
            "draw_prob": pd.to_numeric(source["p_draw"], errors="coerce"),
            "away_prob": pd.to_numeric(source["p_home_loss"], errors="coerce"),
            "data_source_notes": data_source_notes,
        }
    )
    return _with_prediction_labels(output)


def _source_probabilities(source: pd.DataFrame) -> np.ndarray | None:
    required = ["p_home_loss", "p_draw", "p_home_win"]
    if source.empty or not set(required).issubset(source.columns):
        return None
    values = source[required].apply(pd.to_numeric, errors="coerce")
    if values.isna().any(axis=None):
        return None
    return normalize_probability_rows(values.to_numpy(dtype=float))


def _probability_frame_for_fold(
    target_df: pd.DataFrame,
    probabilities: np.ndarray,
    year: int,
    model_name: str,
    feature_set: str,
    data_source_notes: str,
) -> pd.DataFrame:
    probabilities = normalize_probability_rows(probabilities)
    output = pd.DataFrame(
        {
            "match_id": target_df["match_id"].astype(str).values,
            "date": pd.to_datetime(target_df["date"], errors="coerce").dt.strftime("%Y-%m-%d").values,
            "tournament_year": year,
            "stage": target_df.get("_stage5_stage", pd.Series("unknown", index=target_df.index)).astype(str).values,
            "home_team": target_df["home_team"].astype(str).values,
            "away_team": target_df["away_team"].astype(str).values,
            "actual_class": target_df["target_result_class"].astype(int).values,
            "model_name": model_name,
            "feature_set": feature_set,
            "baseline_type": MODEL_BASELINE_TYPES.get(model_name, "benchmark"),
            "home_prob": probabilities[:, 2],
            "draw_prob": probabilities[:, 1],
            "away_prob": probabilities[:, 0],
            "data_source_notes": data_source_notes,
        }
    )
    output["actual_result"] = output["actual_class"].map(CLASS_LABELS)
    return _with_prediction_labels(output)


def _with_prediction_labels(output: pd.DataFrame) -> pd.DataFrame:
    ordered = _probabilities_away_draw_home(output)
    output = output.copy()
    output["predicted_class"] = ordered.argmax(axis=1)
    output["predicted_result"] = output["predicted_class"].map(CLASS_LABELS)
    output["confidence"] = ordered.max(axis=1)
    return output


def _empty_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "match_id",
            "date",
            "tournament_year",
            "stage",
            "home_team",
            "away_team",
            "actual_class",
            "actual_result",
            "model_name",
            "feature_set",
            "baseline_type",
            "home_prob",
            "draw_prob",
            "away_prob",
            "predicted_result",
            "confidence",
            "log_loss",
            "brier",
            "rps",
            "data_source_notes",
        ]
    )


def _deduplicate_stage5_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    return predictions.drop_duplicates(["tournament_year", "match_id", "model_name"], keep="last").reset_index(drop=True)


def _uniform_probabilities(n_rows: int) -> np.ndarray:
    return np.full((n_rows, 3), 1.0 / 3.0, dtype=float)


def _historical_frequency_probabilities(train_df: pd.DataFrame, n_rows: int) -> np.ndarray:
    counts = train_df["target_result_class"].dropna().astype(int).value_counts(normalize=True)
    probabilities = np.array([float(counts.get(0, 0.0)), float(counts.get(1, 0.0)), float(counts.get(2, 0.0))], dtype=float)
    if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
        probabilities = np.full(3, 1.0 / 3.0)
    return np.tile(probabilities / probabilities.sum(), (n_rows, 1))


def _poisson_goal_probabilities(train_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[np.ndarray, str]:
    scored = train_df.dropna(subset=["home_score", "away_score"]).copy()
    if scored.empty:
        return _uniform_probabilities(len(target_df)), "Poisson fallback to uniform because no scored training rows were available"
    global_home = max(float(scored["home_score"].mean()), 0.2)
    global_away = max(float(scored["away_score"].mean()), 0.2)
    smoothing = 12.0
    home_groups = scored.groupby("home_team")
    away_groups = scored.groupby("away_team")
    home_for = _smoothed_team_average(home_groups["home_score"].sum(), home_groups.size(), global_home, smoothing)
    home_against = _smoothed_team_average(home_groups["away_score"].sum(), home_groups.size(), global_away, smoothing)
    away_for = _smoothed_team_average(away_groups["away_score"].sum(), away_groups.size(), global_away, smoothing)
    away_against = _smoothed_team_average(away_groups["home_score"].sum(), away_groups.size(), global_home, smoothing)
    rows = []
    for _, row in target_df.iterrows():
        home = row.get("home_team")
        away = row.get("away_team")
        home_lambda = global_home * (home_for.get(home, global_home) / global_home) * (away_against.get(away, global_home) / global_home)
        away_lambda = global_away * (away_for.get(away, global_away) / global_away) * (home_against.get(home, global_away) / global_away)
        rows.append(_poisson_1x2(home_lambda, away_lambda))
    return normalize_probability_rows(np.asarray(rows, dtype=float)), "Poisson attack/defense baseline with conservative smoothing=12"


def _smoothed_team_average(sums: pd.Series, counts: pd.Series, global_mean: float, smoothing: float) -> dict[str, float]:
    values = (sums.astype(float) + smoothing * global_mean) / (counts.astype(float) + smoothing)
    return values.to_dict()


def _poisson_1x2(home_lambda: float, away_lambda: float, max_goals: int = 10) -> np.ndarray:
    home_lambda = float(np.clip(home_lambda, 0.15, 4.5))
    away_lambda = float(np.clip(away_lambda, 0.15, 4.5))
    home_probs = np.array([math.exp(-home_lambda) * home_lambda**k / math.factorial(k) for k in range(max_goals + 1)])
    away_probs = np.array([math.exp(-away_lambda) * away_lambda**k / math.factorial(k) for k in range(max_goals + 1)])
    matrix = np.outer(home_probs, away_probs)
    home_win = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, 1).sum())
    return np.array([away_win, draw, home_win], dtype=float)


def _market_extra_prediction_frames(
    *,
    base_predictions: pd.DataFrame,
    year: int,
    odds_method: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    notes: list[dict[str, Any]] = []
    market_rows = base_predictions[
        base_predictions["model_name"].eq("bookmaker_odds_only")
        & base_predictions["evaluation_scope"].eq("odds_covered")
        & base_predictions["world_cup_year"].astype(int).eq(int(year))
    ].copy()
    if market_rows.empty:
        notes.append({"model_name": "bookmaker_raw_implied", "test_year": year, "reason": "missing_market_odds", "n_skipped": 0})
        notes.append({"model_name": "bookmaker_calibrated", "test_year": year, "reason": "missing_market_odds", "n_skipped": 0})
        return _empty_prediction_frame(), notes
    frames = []
    if odds_method == "multiplicative":
        frames.append(
            _standardized_prediction_rows(
                market_rows,
                model_name="bookmaker_raw_implied",
                feature_set="market_odds_multiplicative",
                baseline_type=MODEL_BASELINE_TYPES["bookmaker_raw_implied"],
                data_source_notes="raw inverse decimal odds normalized multiplicatively; same as bookmaker_odds_only because selected odds conversion is multiplicative",
            )
        )
    else:
        notes.append(
            {
                "model_name": "bookmaker_raw_implied",
                "test_year": year,
                "reason": "raw decimal odds unavailable in cached backtest predictions; rerun full backtest artifacts to compare raw odds",
                "n_skipped": int(len(market_rows)),
            }
        )

    prior_market = base_predictions[
        base_predictions["model_name"].eq("bookmaker_odds_only")
        & base_predictions["evaluation_scope"].eq("odds_covered")
        & base_predictions["world_cup_year"].astype(int).lt(int(year))
    ].copy()
    target_prob = _source_probabilities(market_rows)
    train_prob = _source_probabilities(prior_market) if not prior_market.empty else None
    if train_prob is None or len(prior_market) < 20 or prior_market["actual_class"].nunique() < 2:
        notes.append(
            {
                "model_name": "bookmaker_calibrated",
                "test_year": year,
                "reason": "insufficient historical odds-covered training rows for time-safe calibration",
                "n_skipped": int(len(market_rows)),
            }
        )
    else:
        gamma, train_loss = _fit_market_temperature(prior_market["actual_class"].astype(int).to_numpy(), train_prob)
        calibrated = _temperature_calibrate(target_prob, gamma)
        calibrated_source = market_rows.copy()
        calibrated_source["p_home_loss"] = calibrated[:, 0]
        calibrated_source["p_draw"] = calibrated[:, 1]
        calibrated_source["p_home_win"] = calibrated[:, 2]
        frames.append(
            _standardized_prediction_rows(
                calibrated_source,
                model_name="bookmaker_calibrated",
                feature_set=f"market_odds_{odds_method}_temperature",
                baseline_type=MODEL_BASELINE_TYPES["bookmaker_calibrated"],
                data_source_notes=f"time-safe bookmaker temperature calibration gamma={gamma:.3f}; train_log_loss={train_loss:.6f}",
            )
        )
    return pd.concat(frames, ignore_index=True, sort=False) if frames else _empty_prediction_frame(), notes


def _fit_market_temperature(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best_gamma = 1.0
    best_loss = float("inf")
    for gamma in np.linspace(0.55, 1.85, 27):
        calibrated = _temperature_calibrate(probabilities, float(gamma))
        loss = float(log_loss(y_true, calibrated, labels=[0, 1, 2]))
        if loss < best_loss:
            best_gamma = float(gamma)
            best_loss = loss
    return best_gamma, best_loss


def _temperature_calibrate(probabilities: np.ndarray, gamma: float) -> np.ndarray:
    return normalize_probability_rows(np.power(np.clip(probabilities, 1e-12, 1.0), float(gamma)))


def normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Expected probability array with shape (n, 3)")
    values = np.where(np.isfinite(values), values, np.nan)
    values = np.clip(values, 1e-12, None)
    sums = np.nansum(values, axis=1, keepdims=True)
    sums[sums <= 0] = np.nan
    normalized = values / sums
    return np.where(np.isfinite(normalized), normalized, 1.0 / 3.0)


def _add_prediction_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    output = predictions.copy()
    probabilities = _probabilities_away_draw_home(output)
    y = output["actual_class"].astype(int).to_numpy()
    output["log_loss"] = per_row_log_loss(y, probabilities)
    output["brier"] = per_row_brier(y, probabilities)
    output["rps"] = per_row_rps(y, probabilities)
    return output[
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
            "baseline_type",
            "home_prob",
            "draw_prob",
            "away_prob",
            "predicted_result",
            "confidence",
            "log_loss",
            "brier",
            "rps",
            "data_source_notes",
        ]
    ]


def _probabilities_away_draw_home(df: pd.DataFrame) -> np.ndarray:
    return normalize_probability_rows(df[["away_prob", "draw_prob", "home_prob"]].to_numpy(dtype=float))


def per_row_log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true).astype(int)
    clipped = np.clip(normalize_probability_rows(probabilities), 1e-12, 1.0)
    return -np.log(clipped[np.arange(len(y)), y])


def per_row_brier(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true).astype(int)
    one_hot = np.eye(3)[y]
    return np.sum((normalize_probability_rows(probabilities) - one_hot) ** 2, axis=1)


def per_row_rps(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true).astype(int)
    one_hot_cdf = np.cumsum(np.eye(3)[y], axis=1)
    prob_cdf = np.cumsum(normalize_probability_rows(probabilities), axis=1)
    return np.sum((prob_cdf - one_hot_cdf) ** 2, axis=1) / 2.0


def _target_counts_by_year(predictions: pd.DataFrame, fold_targets: dict[int, pd.DataFrame]) -> dict[int, int]:
    counts = {int(year): len(target) for year, target in fold_targets.items()}
    official = predictions[predictions["model_name"].eq("official_model")]
    for year, group in official.groupby("tournament_year"):
        counts[int(year)] = max(counts.get(int(year), 0), int(group["match_id"].nunique()))
    return counts


def _stage5_metrics(predictions: pd.DataFrame, total_by_year: dict[int, int]) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=["model_name", "feature_set", "baseline_type", "test_year", "n_matches", "n_skipped", "accuracy", "log_loss", "brier", "rps", "ece", "mean_confidence"]
        )
    rows = []
    grouped = predictions.groupby(["model_name", "feature_set", "baseline_type", "tournament_year"], dropna=False)
    for (model_name, feature_set, baseline_type, year), group in grouped:
        probabilities = _probabilities_away_draw_home(group)
        y = group["actual_class"].astype(int).to_numpy()
        rows.append(
            {
                "model_name": model_name,
                "feature_set": feature_set,
                "baseline_type": baseline_type,
                "test_year": int(year),
                "n_matches": int(len(group)),
                "n_skipped": int(max(0, total_by_year.get(int(year), len(group)) - len(group))),
                "accuracy": float((probabilities.argmax(axis=1) == y).mean()),
                "log_loss": float(log_loss(y, probabilities, labels=[0, 1, 2])),
                "brier": brier_score_multiclass(y, probabilities),
                "rps": ranked_probability_score(y, probabilities),
                "ece": expected_calibration_error(y, probabilities),
                "mean_confidence": float(probabilities.max(axis=1).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["test_year", "log_loss", "model_name"], kind="stable").reset_index(drop=True)


def _skipped_dixon_coles_metrics(years: list[int], total_by_year: dict[int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_name": "dixon_coles",
                "feature_set": "not_fitted",
                "baseline_type": MODEL_BASELINE_TYPES["dixon_coles"],
                "test_year": int(year),
                "n_matches": 0,
                "n_skipped": int(total_by_year.get(int(year), 0)),
                "accuracy": np.nan,
                "log_loss": np.nan,
                "brier": np.nan,
                "rps": np.nan,
                "ece": np.nan,
                "mean_confidence": np.nan,
            }
            for year in years
        ]
    )


def _calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, year), group in predictions.groupby(["model_name", "tournament_year"], dropna=False):
        rows.extend(_calibration_rows(group, model_name=str(model_name), test_year=int(year)))
    for model_name, group in predictions.groupby("model_name", dropna=False):
        rows.extend(_calibration_rows(group, model_name=str(model_name), test_year="all"))
    return pd.DataFrame(rows, columns=["model_name", "test_year", "bucket", "mean_confidence", "empirical_accuracy", "count", "calibration_gap"])


def _stage_calibration_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, stage), group in predictions.groupby(["model_name", "stage"], dropna=False):
        for row in _calibration_rows(group, model_name=str(model_name), test_year="all"):
            row["stage"] = stage
            rows.append(row)
    columns = ["model_name", "test_year", "stage", "bucket", "mean_confidence", "empirical_accuracy", "count", "calibration_gap"]
    return pd.DataFrame(rows, columns=columns)


def _calibration_rows(group: pd.DataFrame, *, model_name: str, test_year: int | str) -> list[dict[str, Any]]:
    probabilities = _probabilities_away_draw_home(group)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    actual = group["actual_class"].astype(int).to_numpy()
    correct = predicted == actual
    rows = []
    bins = np.linspace(0.0, 1.0, 11)
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (confidence > left) & (confidence <= right)
        label = f"({left:.1f},{right:.1f}]"
        if not bool(mask.any()):
            rows.append(
                {
                    "model_name": model_name,
                    "test_year": test_year,
                    "bucket": label,
                    "mean_confidence": np.nan,
                    "empirical_accuracy": np.nan,
                    "count": 0,
                    "calibration_gap": np.nan,
                }
            )
            continue
        mean_confidence = float(confidence[mask].mean())
        empirical_accuracy = float(correct[mask].mean())
        rows.append(
            {
                "model_name": model_name,
                "test_year": test_year,
                "bucket": label,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
                "count": int(mask.sum()),
                "calibration_gap": empirical_accuracy - mean_confidence,
            }
        )
    return rows


def _stage5_significance(predictions: pd.DataFrame, *, n_bootstrap: int) -> pd.DataFrame:
    comparisons = [
        ("official_model", "bookmaker_odds_only"),
        ("official_model", "bookmaker_calibrated"),
        ("official_model", "elo_only"),
        ("official_model", "poisson_goal_model"),
        ("market_blend", "bookmaker_odds_only"),
        ("market_blend", "official_model"),
    ]
    rows = []
    for candidate, baseline in comparisons:
        for metric in ("log_loss", "brier"):
            rows.append(_paired_metric_delta(predictions, candidate, baseline, metric=metric, n_bootstrap=n_bootstrap))
    return pd.DataFrame(rows)


def _paired_metric_delta(
    predictions: pd.DataFrame,
    candidate: str,
    baseline: str,
    *,
    metric: str,
    n_bootstrap: int,
    seed: int = 42,
) -> dict[str, Any]:
    left = predictions[predictions["model_name"].eq(candidate)][["tournament_year", "match_id", metric]].rename(columns={metric: "candidate"})
    right = predictions[predictions["model_name"].eq(baseline)][["tournament_year", "match_id", metric]].rename(columns={metric: "baseline"})
    paired = left.merge(right, on=["tournament_year", "match_id"], how="inner").dropna()
    if paired.empty:
        return {
            "candidate_model": candidate,
            "baseline_model": baseline,
            "metric": metric,
            "mean_delta": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "statistically_meaningful": False,
            "n_matches": 0,
        }
    delta = (paired["candidate"] - paired["baseline"]).to_numpy(dtype=float)
    mean_delta, ci_lower, ci_upper = paired_bootstrap_ci(delta, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "candidate_model": candidate,
        "baseline_model": baseline,
        "metric": metric,
        "mean_delta": mean_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "statistically_meaningful": bool(ci_upper < 0),
        "n_matches": int(len(delta)),
    }


def paired_bootstrap_ci(values: np.ndarray, *, n_bootstrap: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(max(1, int(n_bootstrap)), len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _stage5_decision(significance: pd.DataFrame) -> bool:
    row = significance[
        significance["candidate_model"].eq("official_model")
        & significance["baseline_model"].eq("bookmaker_odds_only")
        & significance["metric"].eq("log_loss")
    ]
    if row.empty:
        return False
    return bool(row.iloc[0]["statistically_meaningful"])


def write_stage5_benchmark_report(
    *,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    significance: pd.DataFrame,
    calibration: pd.DataFrame,
    stage_calibration: pd.DataFrame,
    skipped_notes: pd.DataFrame,
    years: list[int],
    stage5_achieved: bool,
    path: str | Path = STAGE5_REPORT_PATH,
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    average = _average_metrics(metrics)
    coverage = _coverage_summary(predictions)
    best_log_loss = _best_model(average, "log_loss")
    best_brier = _best_model(average, "brier")
    best_rps = _best_model(average, "rps")
    best_ece = _best_model(average, "ece")
    lines = [
        "# Stage 5 Benchmark Report",
        "",
        "## Executive Summary",
        "",
        f"- Stage 5 status: `{'Stage 5 achieved' if stage5_achieved else 'Stage 5 not yet achieved'}`",
        "- Official production model remains `football_only_ensemble`.",
        "- Official production feature set remains `core_football_only`.",
        "- Market blend remains benchmark-only unless significance results support promotion.",
        "- This report is research/evaluation output, not a betting tool.",
        "",
        "## Data Coverage",
        "",
        f"- Backtest years requested: `{', '.join(str(year) for year in years)}`",
        _markdown_table(coverage, ["test_year", "total_matches", "market_covered_matches", "market_coverage_rate"]),
        "",
        "## Models Compared",
        "",
        _markdown_table(_models_compared(metrics), ["model_name", "baseline_type", "feature_set", "years", "total_matches", "total_skipped"]),
        "",
        "## Metrics Table",
        "",
        _markdown_table(metrics, ["model_name", "feature_set", "baseline_type", "test_year", "n_matches", "n_skipped", "accuracy", "log_loss", "brier", "rps", "ece", "mean_confidence"]),
        "",
        "## Average Metrics",
        "",
        _markdown_table(average, ["model_name", "baseline_type", "years", "total_matches", "log_loss", "brier", "rps", "ece", "accuracy", "mean_confidence"]),
        "",
        "## Best Models By Metric",
        "",
        f"- Best log loss: `{best_log_loss}`",
        f"- Best Brier: `{best_brier}`",
        f"- Best RPS: `{best_rps}`",
        f"- Best ECE: `{best_ece}`",
        "",
        "## Significance Results",
        "",
        _markdown_table(significance, ["candidate_model", "baseline_model", "metric", "mean_delta", "ci_lower", "ci_upper", "statistically_meaningful", "n_matches"]),
        "",
        "## Calibration Summary",
        "",
        _markdown_table(_ece_summary(metrics), ["model_name", "average_ece", "years"]),
        "",
        "## Reliability Table Preview",
        "",
        _markdown_table(calibration[calibration["count"].gt(0)].head(40), ["model_name", "test_year", "bucket", "mean_confidence", "empirical_accuracy", "count", "calibration_gap"]),
        "",
        "## Stage Calibration Preview",
        "",
        _markdown_table(_count_positive_rows(stage_calibration).head(40), ["model_name", "stage", "bucket", "mean_confidence", "empirical_accuracy", "count", "calibration_gap"]),
        "",
        "## Ablation Summary",
        "",
        _ablation_summary_text(),
        "",
        "## Skipped Or Limited Baselines",
        "",
        _markdown_table(skipped_notes, ["model_name", "test_year", "reason", "n_skipped"]),
        "",
        "## Limitations",
        "",
        "- Historical squad/player, injury, lineup, and xG data remain scenario/live-only unless historically validated as-of each match date.",
        "- Bookmaker benchmarks are evaluated only where pre-match 1X2 market odds are available.",
        "- Calibrated bookmaker output uses only odds-covered matches before each target tournament; sparse historical odds can limit calibration.",
        "- Dixon-Coles is skipped unless a validated time-decayed scoreline fitting routine is added.",
        "- Accuracy is secondary; log loss, Brier, RPS, and calibration drive the benchmark decision.",
        "",
        "## Final Recommendation",
        "",
        _recommendation_text(stage5_achieved, significance),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_stage5_decision_report(
    *,
    metrics: pd.DataFrame,
    significance: pd.DataFrame,
    skipped_notes: pd.DataFrame,
    stage5_achieved: bool,
    path: str | Path = STAGE5_DECISION_PATH,
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    official_vs_book = significance[
        significance["candidate_model"].eq("official_model")
        & significance["baseline_model"].eq("bookmaker_odds_only")
        & significance["metric"].eq("log_loss")
    ]
    lines = [
        "# Stage 5 Research Decision",
        "",
        f"Decision: **{'Stage 5 achieved' if stage5_achieved else 'Stage 5 not yet achieved'}**",
        "",
        "## Reasons",
        "",
        "- Stage 5 requires statistically meaningful benchmark evidence, not only dashboard functionality.",
        "- The official model must be compared against strong non-market and market baselines using no-leakage historical World Cup folds.",
        "- Negative log-loss delta means the candidate model is better than the baseline.",
        "",
        "## Evidence",
        "",
        _markdown_table(official_vs_book, ["candidate_model", "baseline_model", "metric", "mean_delta", "ci_lower", "ci_upper", "statistically_meaningful", "n_matches"]),
        "",
        "## What To Improve Next",
        "",
        "- Add more complete historical pre-match odds coverage.",
        "- Add a validated Dixon-Coles fitting implementation.",
        "- Validate any squad, injury, lineup, or xG data historically before treating it as production model evidence.",
        "- Improve probability calibration if ECE remains weaker than market baselines.",
        "",
        "## Skipped Or Limited Items",
        "",
        _markdown_table(skipped_notes, ["model_name", "test_year", "reason", "n_skipped"]),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _average_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    valid = metrics[metrics["n_matches"].gt(0)].copy()
    if valid.empty:
        return pd.DataFrame()
    grouped = valid.groupby(["model_name", "baseline_type"], as_index=False)
    average = grouped.agg(
        years=("test_year", "nunique"),
        total_matches=("n_matches", "sum"),
        log_loss=("log_loss", "mean"),
        brier=("brier", "mean"),
        rps=("rps", "mean"),
        ece=("ece", "mean"),
        accuracy=("accuracy", "mean"),
        mean_confidence=("mean_confidence", "mean"),
    )
    return average.sort_values(["log_loss", "model_name"], kind="stable").reset_index(drop=True)


def _count_positive_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "count" not in df.columns:
        return df
    return df[df["count"].gt(0)]


def _coverage_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    official = predictions[predictions["model_name"].eq("official_model")]
    market = predictions[predictions["model_name"].eq("bookmaker_odds_only")]
    rows = []
    for year in sorted(official["tournament_year"].dropna().astype(int).unique()):
        total = int(official[official["tournament_year"].eq(year)]["match_id"].nunique())
        covered = int(market[market["tournament_year"].eq(year)]["match_id"].nunique())
        rows.append({"test_year": year, "total_matches": total, "market_covered_matches": covered, "market_coverage_rate": 0.0 if total == 0 else covered / total})
    return pd.DataFrame(rows)


def _models_compared(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return (
        metrics.groupby(["model_name", "baseline_type", "feature_set"], as_index=False)
        .agg(years=("test_year", "nunique"), total_matches=("n_matches", "sum"), total_skipped=("n_skipped", "sum"))
        .sort_values(["model_name"], kind="stable")
    )


def _ece_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    valid = metrics[metrics["n_matches"].gt(0)].copy()
    if valid.empty:
        return pd.DataFrame()
    return valid.groupby("model_name", as_index=False).agg(average_ece=("ece", "mean"), years=("test_year", "nunique")).sort_values("average_ece", kind="stable")


def _best_model(average: pd.DataFrame, metric: str) -> str:
    if average.empty or metric not in average.columns:
        return "not available"
    row = average.dropna(subset=[metric]).sort_values([metric, "model_name"], kind="stable").head(1)
    if row.empty:
        return "not available"
    return f"{row.iloc[0]['model_name']} ({metric}={float(row.iloc[0][metric]):.6f})"


def _recommendation_text(stage5_achieved: bool, significance: pd.DataFrame) -> str:
    if stage5_achieved:
        return "Stage 5 evidence supports the official model as a statistically meaningful benchmark leader against bookmaker odds on the evaluated odds-covered historical World Cup matches. Keep market blend benchmark-only until production policy is separately updated."
    row = significance[
        significance["candidate_model"].eq("official_model")
        & significance["baseline_model"].eq("bookmaker_odds_only")
        & significance["metric"].eq("log_loss")
    ]
    if row.empty or int(row.iloc[0]["n_matches"]) == 0:
        return "Stage 5 is not achieved yet because bookmaker odds coverage is unavailable or insufficient for the required official-model comparison."
    return "Stage 5 is not achieved yet because the official model did not show a statistically meaningful log-loss improvement over bookmaker odds-only. Keep the project SOTA-inspired and research-backtested, not true SOTA."


def _ablation_summary_text() -> str:
    path = resolve_project_path("data/experiments/ablation_results.csv")
    if not path.exists():
        return "_No ablation_results.csv file was found._"
    ablation = pd.read_csv(path, low_memory=False)
    if ablation.empty:
        return "_Ablation file exists but is empty._"
    columns = [column for column in ["feature_set", "model_name", "log_loss", "brier_score", "ranked_probability_score", "calibration_error"] if column in ablation.columns]
    if not columns:
        return "_Ablation file does not contain recognized metric columns._"
    ranked = ablation.sort_values([columns[2] if len(columns) > 2 else columns[0]], kind="stable").head(10)
    return _markdown_table(ranked, columns)


def _read_market_conversion_method() -> str:
    path = resolve_project_path("models/market_blend_params.json")
    if not path.exists():
        return "multiplicative"
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "multiplicative"
    return str(payload.get("odds_conversion_method") or "multiplicative")


def _stage_labels(target_df: pd.DataFrame) -> list[str]:
    if "stage" in target_df.columns and target_df["stage"].notna().any():
        return target_df["stage"].fillna("unknown").astype(str).tolist()
    labels = []
    for idx in range(len(target_df)):
        if idx < 48:
            labels.append("group")
        elif idx < 56:
            labels.append("round_of_16")
        elif idx < 60:
            labels.append("quarterfinal")
        elif idx < 62:
            labels.append("semifinal")
        elif idx == 62:
            labels.append("third_place")
        else:
            labels.append("final")
    return labels


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in df.columns]
    if not available or df.empty:
        return "_No rows._"
    rows = ["| " + " | ".join(available) + " |", "| " + " | ".join(["---"] * len(available)) + " |"]
    for record in df[available].to_dict("records"):
        rows.append("| " + " | ".join(_format_value(record.get(column)) for column in available) + " |")
    return "\n".join(rows)


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
