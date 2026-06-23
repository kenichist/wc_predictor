from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import config_path, load_config
from src.features.feature_sets import ABLATION_FEATURE_SETS, FEATURE_GROUPS, columns_for_feature_set
from src.io_utils import read_dataframe, write_dataframe
from src.models.evaluate import evaluate_probabilities, time_aware_split
from src.models.train_match_model import _feature_matrix, _predict_proba_3


logger = logging.getLogger(__name__)


def run_ablation(config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or load_config()
    training_path = config_path(cfg, "match_training_dataset_advanced_parquet")
    if not training_path.exists():
        training_path = config_path(cfg, "match_training_dataset_parquet")
    df = read_dataframe(training_path).dropna(subset=["target_result_class"])
    split_cfg = cfg.get("modeling", {})
    train_df, val_df, test_df = time_aware_split(
        df,
        train_end=split_cfg.get("train_end", "2022-01-01"),
        validation_start=split_cfg.get("validation_start", "2022-01-01"),
        validation_end=split_cfg.get("validation_end", "2025-01-01"),
        test_start=split_cfg.get("test_start", "2025-01-01"),
    )
    if train_df.empty or val_df.empty:
        ordered = df.sort_values("date", kind="stable")
        split_idx = max(1, int(len(ordered) * 0.8))
        train_df = ordered.iloc[:split_idx].copy()
        val_df = ordered.iloc[split_idx:].copy()

    rows = []
    previous_usable_columns: list[str] = []
    previous_requested_columns: list[str] = []
    previous_metrics: dict[str, float] | None = None
    for feature_set in ABLATION_FEATURE_SETS:
        requested_columns = columns_for_feature_set(feature_set, df.columns)
        requested_columns = [column for column in requested_columns if column in df.columns]
        new_columns = [column for column in requested_columns if column not in previous_requested_columns]
        null_new = [column for column in new_columns if column not in df.columns or df[column].isna().all()]
        constant_new = [
            column
            for column in new_columns
            if column in df.columns and df[column].notna().any() and df[column].nunique(dropna=True) <= 1
        ]
        usable_new = [
            column
            for column in new_columns
            if column in df.columns and df[column].notna().any() and df[column].nunique(dropna=True) > 1
        ]
        feature_columns = [column for column in requested_columns if df[column].notna().any()]
        usable_feature_columns = [column for column in feature_columns if df[column].nunique(dropna=True) > 1]
        if not feature_columns:
            logger.warning("Skipping %s because no feature columns are available", feature_set)
            continue
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("estimator", HistGradientBoostingClassifier(random_state=int(split_cfg.get("random_seed", 42)), loss="log_loss")),
            ]
        )
        model.fit(_feature_matrix(train_df, feature_columns), train_df["target_result_class"].astype(int))
        probabilities = _predict_proba_3(model, _feature_matrix(val_df, feature_columns))
        metrics = evaluate_probabilities(val_df["target_result_class"], probabilities)
        identical_to_previous = bool(
            previous_metrics
            and all(abs(metrics[key] - previous_metrics[key]) <= 1e-12 for key in ["accuracy", "log_loss", "brier_score", "ranked_probability_score", "calibration_error"])
        )
        same_usable_columns = set(usable_feature_columns) == set(previous_usable_columns)
        warning = _ablation_warning(feature_set, new_columns, null_new, constant_new, usable_new, same_usable_columns, identical_to_previous)
        if warning:
            logger.warning("%s", warning)
        rows.append(
            {
                "feature_set": feature_set,
                "num_features": len(feature_columns),
                "new_columns_added": len(new_columns),
                "usable_new_columns": len(usable_new),
                "null_new_columns": len(null_new),
                "constant_new_columns": len(constant_new),
                "identical_to_previous": identical_to_previous,
                **metrics,
                "warning": warning,
            }
        )
        previous_requested_columns = requested_columns
        previous_usable_columns = usable_feature_columns
        previous_metrics = metrics
    result = pd.DataFrame(rows)
    ordered_columns = [
        "feature_set",
        "num_features",
        "new_columns_added",
        "usable_new_columns",
        "null_new_columns",
        "constant_new_columns",
        "identical_to_previous",
        "accuracy",
        "log_loss",
        "brier_score",
        "ranked_probability_score",
        "calibration_error",
        "warning",
    ]
    if not result.empty:
        result = _append_validation_regression_warnings(result)
        result = result[ordered_columns]
    write_dataframe(result, config_path(cfg, "ablation_results_csv"))
    _write_markdown(result, config_path(cfg, "ablation_results_md"))
    from src.reports.final_model_recommendation import write_final_model_recommendation

    write_final_model_recommendation(config=cfg)
    return result


def _write_markdown(results: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if results.empty:
        path.write_text("# Ablation Results\n\nNo ablation results were generated.\n", encoding="utf-8")
        return
    columns = [
        "feature_set",
        "accuracy",
        "log_loss",
        "brier_score",
        "ranked_probability_score",
        "calibration_error",
        "num_features",
        "new_columns_added",
        "usable_new_columns",
        "null_new_columns",
        "constant_new_columns",
        "identical_to_previous",
        "warning",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in results[columns].itertuples(index=False):
        values = [getattr(row, column) for column in columns]
        lines.append("| " + " | ".join(_format_value(value) for value in values) + " |")
    text = ["# Ablation Results", "", *lines, ""]
    path.write_text("\n".join(text), encoding="utf-8")


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _ablation_warning(
    feature_set: str,
    new_columns: list[str],
    null_new: list[str],
    constant_new: list[str],
    usable_new: list[str],
    same_usable_columns: bool,
    identical_to_previous: bool,
) -> str:
    if not new_columns:
        return "feature set has same requested columns as previous step"
    warnings: list[str] = []
    if len(null_new) == len(new_columns):
        warnings.append(f"{feature_set} added {len(new_columns)} columns, but all are fully null")
    elif len(usable_new) == 0:
        warnings.append(f"{feature_set} added {len(new_columns)} columns, but none add usable variation")
    elif null_new:
        warnings.append(f"{len(null_new)} newly added columns are fully null")
    if constant_new:
        warnings.append(f"{len(constant_new)} newly added columns are constant")
    if same_usable_columns:
        warnings.append("feature set has same usable columns as previous set")
    if identical_to_previous:
        warnings.append("metrics are identical to previous feature set")
    if warnings:
        hint = _external_data_hint(feature_set, usable_new_columns=len(usable_new))
        if hint:
            warnings.append(hint)
    return ". ".join(part for part in warnings if part)


def _external_data_hint(feature_set: str, *, usable_new_columns: int = 0) -> str:
    if "fifa_rankings" in feature_set:
        if usable_new_columns > 0:
            return "Some FIFA ranking columns are unavailable; rank-only snapshots can still be useful, but add points snapshots for points features"
        return "Add dated historical FIFA ranking snapshots or rebuild if FIFA rank columns are fully null"
    if "external_elo" in feature_set:
        return "Add data/external/world_football_elo.csv to enable external Elo features"
    return ""


def _append_validation_regression_warnings(results: pd.DataFrame) -> pd.DataFrame:
    output = results.copy()
    comparisons = [
        ("core_plus_internal_elo", "core_football_only", "internal historical Elo"),
        ("v2_baseline_plus_fifa_rankings", "v1_baseline", "FIFA rankings"),
        ("v3_plus_external_elo", "v2_baseline_plus_fifa_rankings", "external Elo"),
        ("rating_stack_experimental", "core_football_only", "experimental rating stack"),
    ]
    for candidate, baseline, label in comparisons:
        candidate_mask = output["feature_set"].eq(candidate)
        baseline_row = output[output["feature_set"].eq(baseline)]
        candidate_row = output[candidate_mask]
        if baseline_row.empty or candidate_row.empty:
            continue
        delta = float(candidate_row.iloc[0]["log_loss"] - baseline_row.iloc[0]["log_loss"])
        if delta > 1e-12:
            message = f"{label} worsened validation log_loss by {delta:.6f}"
            existing = str(candidate_row.iloc[0].get("warning", ""))
            if existing == "nan":
                existing = ""
            output.loc[candidate_mask, "warning"] = ". ".join(part for part in [existing, message] if part)
    return output
