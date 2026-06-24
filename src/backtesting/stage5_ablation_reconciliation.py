from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import resolve_project_path


FINAL_MODEL_RECOMMENDATION_PATH = resolve_project_path("data/reports/final_model_recommendation.md")
STAGE5_BENCHMARK_REPORT_PATH = resolve_project_path("data/reports/stage5_benchmark_report.md")
STAGE5_DECISION_PATH = resolve_project_path("data/reports/stage5_research_decision.md")
STAGE5_ABLATION_RECONCILIATION_PATH = resolve_project_path("data/backtests/stage5_ablation_reconciliation.csv")
STAGE5_ABLATION_RECONCILIATION_REPORT_PATH = resolve_project_path("data/reports/stage5_ablation_reconciliation_report.md")


@dataclass(frozen=True)
class Stage5AblationReconciliationResult:
    reconciliation: pd.DataFrame
    report_path: Path
    csv_path: Path
    stage5_achieved: bool


def run_stage5_ablation_reconciliation(
    *,
    output_dir: str | Path = "data/backtests",
    final_model_recommendation_path: str | Path = FINAL_MODEL_RECOMMENDATION_PATH,
    stage5_benchmark_report_path: str | Path = STAGE5_BENCHMARK_REPORT_PATH,
    stage5_decision_path: str | Path = STAGE5_DECISION_PATH,
    report_path: str | Path = STAGE5_ABLATION_RECONCILIATION_REPORT_PATH,
) -> Stage5AblationReconciliationResult:
    output_root = resolve_project_path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    final_report = _read_required_text(resolve_project_path(final_model_recommendation_path))
    benchmark_report = _read_optional_text(resolve_project_path(stage5_benchmark_report_path))
    decision_report = _read_optional_text(resolve_project_path(stage5_decision_path))
    metrics = _read_required_csv(output_root / "stage5_benchmark_metrics.csv")
    predictions = _read_optional_csv(output_root / "stage5_match_predictions.csv")
    significance = _read_optional_csv(output_root / "stage5_significance_tests.csv")

    ablation_row = _extract_core_ablation_row(final_report)
    stage5_rows = _stage5_official_rows(metrics)
    reconciliation = _build_reconciliation_rows(ablation_row, stage5_rows, predictions)
    csv_path = output_root / "stage5_ablation_reconciliation.csv"
    reconciliation.to_csv(csv_path, index=False)

    stage5_achieved = _stage5_achieved(decision_report, significance)
    written_report = write_stage5_ablation_reconciliation_report(
        path=report_path,
        reconciliation=reconciliation,
        significance=significance,
        stage5_achieved=stage5_achieved,
        benchmark_report_present=bool(benchmark_report),
        decision_report_present=bool(decision_report),
    )
    return Stage5AblationReconciliationResult(
        reconciliation=reconciliation,
        report_path=written_report,
        csv_path=csv_path,
        stage5_achieved=stage5_achieved,
    )


def write_stage5_ablation_reconciliation_report(
    *,
    path: str | Path,
    reconciliation: pd.DataFrame,
    significance: pd.DataFrame,
    stage5_achieved: bool,
    benchmark_report_present: bool,
    decision_report_present: bool,
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ablation = reconciliation[reconciliation["metric_source"].eq("final_model_recommendation.md")]
    stage5_average = reconciliation[reconciliation["model_label"].eq("official_model_weighted_average")]
    log_loss_gap = _log_loss_gap(ablation, stage5_average)
    official_vs_book = _official_vs_bookmaker(significance)

    lines = [
        "# Stage 5 Ablation Reconciliation Report",
        "",
        "## Executive Summary",
        "",
        "- The ablation and Stage 5 log-loss values are not directly comparable.",
        "- Ablation is feature-selection validation.",
        "- Stage 5 is a held-out World Cup benchmark with tournament-year cutoffs.",
        "- Stage 5 benchmark evidence is more trustworthy for near-SOTA or SOTA-style claims.",
        "- The ablation score should not be used to claim Stage 5.",
        f"- Stage 5 status remains: `{'Stage 5 achieved' if stage5_achieved else 'Stage 5 not achieved'}`.",
        "",
        "## Metric Comparison",
        "",
        _markdown_table(
            reconciliation,
            [
                "metric_source",
                "evaluation_protocol",
                "years_used",
                "row_count",
                "feature_set",
                "model_label",
                "accuracy",
                "log_loss",
                "brier_score",
                "ranked_probability_score",
                "calibration_ece",
                "comparable_to_stage5",
            ],
        ),
        "",
        "## Why The Numbers Differ",
        "",
        "- The ablation `core_football_only` row comes from the final recommendation's ablation table and is used to choose among feature sets.",
        "- The Stage 5 `official_model` row is evaluated on held-out historical World Cups only: 2014, 2018, and 2022.",
        "- Stage 5 trains before each target tournament and evaluates the tournament as a benchmark fold.",
        "- World Cup matches are a smaller and harder distribution than the broad validation set used by ablation.",
        "- The model labels differ because ablation reports a feature set, while Stage 5 reports the official prediction policy: `football_only_ensemble/core_football_only`.",
        "",
        "## Log-Loss Gap",
        "",
        log_loss_gap,
        "",
        "## Stage 5 Claim Check",
        "",
        _markdown_table(
            official_vs_book,
            ["candidate_model", "baseline_model", "metric", "mean_delta", "ci_lower", "ci_upper", "statistically_meaningful", "n_matches"],
        ),
        "",
        "The Stage 5 decision must come from held-out World Cup benchmark evidence and paired significance tests, not from ablation validation.",
        "",
        "## Conclusion",
        "",
        "- Use ablation to decide the safe production feature set.",
        "- Use Stage 5 to judge whether the project has benchmark-leading evidence.",
        "- The current evidence does not support claiming Stage 5 is achieved.",
        "- Keep the project wording as SOTA-inspired and historically benchmarked, not true SOTA.",
        "",
        "## Input File Check",
        "",
        f"- Stage 5 benchmark report present: `{benchmark_report_present}`.",
        f"- Stage 5 decision report present: `{decision_report_present}`.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _build_reconciliation_rows(ablation_row: pd.Series, stage5_rows: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "metric_source": "final_model_recommendation.md",
            "evaluation_protocol": "feature-selection validation from Ablation Summary",
            "years_used": "not reported in final_model_recommendation.md",
            "row_count": np.nan,
            "feature_set": str(ablation_row.get("feature_set", "core_football_only")),
            "model_label": "ablation_core_football_only",
            "accuracy": _float_or_nan(ablation_row.get("accuracy")),
            "log_loss": _float_or_nan(ablation_row.get("log_loss")),
            "brier_score": _float_or_nan(ablation_row.get("brier_score")),
            "ranked_probability_score": _float_or_nan(ablation_row.get("ranked_probability_score")),
            "calibration_ece": _float_or_nan(ablation_row.get("calibration_error")),
            "comparable_to_stage5": "No - different protocol, dataset, and purpose",
            "primary_use": "Select the safest feature set for production modeling",
            "notes": "Ablation is not a held-out World Cup benchmark and should not be used for Stage 5 claims.",
        }
    ]
    if not stage5_rows.empty:
        rows.append(_stage5_weighted_average_row(stage5_rows, predictions))
        for _, row in stage5_rows.sort_values("test_year", kind="stable").iterrows():
            rows.append(
                {
                    "metric_source": "stage5_benchmark_metrics.csv",
                    "evaluation_protocol": "held-out World Cup fold; train before tournament start",
                    "years_used": str(int(row["test_year"])),
                    "row_count": int(row["n_matches"]),
                    "feature_set": row.get("feature_set", "football_only_ensemble/core_football_only"),
                    "model_label": f"official_model_{int(row['test_year'])}",
                    "accuracy": _float_or_nan(row.get("accuracy")),
                    "log_loss": _float_or_nan(row.get("log_loss")),
                    "brier_score": _float_or_nan(row.get("brier")),
                    "ranked_probability_score": _float_or_nan(row.get("rps")),
                    "calibration_ece": _float_or_nan(row.get("ece")),
                    "comparable_to_stage5": "Yes - this is a Stage 5 tournament fold",
                    "primary_use": "Historical World Cup benchmark evidence",
                    "notes": "Per-year Stage 5 official-model result.",
                }
            )
    return pd.DataFrame(rows)


def _stage5_official_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"model_name", "n_matches", "log_loss", "test_year"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"stage5_benchmark_metrics.csv is missing required columns: {sorted(missing)}")
    official = metrics[metrics["model_name"].eq("official_model")].copy()
    official["n_matches"] = pd.to_numeric(official["n_matches"], errors="coerce").fillna(0)
    official = official[official["n_matches"].gt(0)].copy()
    if official.empty:
        raise ValueError("No official_model rows with n_matches > 0 were found in stage5_benchmark_metrics.csv")
    return official


def _stage5_weighted_average_row(stage5_rows: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    years = sorted(stage5_rows["test_year"].dropna().astype(int).unique())
    row_count = int(stage5_rows["n_matches"].sum())
    feature_sets = sorted(stage5_rows["feature_set"].dropna().astype(str).unique()) if "feature_set" in stage5_rows.columns else []
    feature_set = ", ".join(feature_sets) if feature_sets else "football_only_ensemble/core_football_only"
    prediction_rows = _official_prediction_count(predictions)
    notes = "Weighted by n_matches across Stage 5 official_model rows."
    if prediction_rows is not None and prediction_rows != row_count:
        notes += f" stage5_match_predictions.csv contains {prediction_rows} official rows."
    return {
        "metric_source": "stage5_benchmark_metrics.csv",
        "evaluation_protocol": "held-out World Cup benchmark; train before each target tournament",
        "years_used": ", ".join(str(year) for year in years),
        "row_count": row_count,
        "feature_set": feature_set,
        "model_label": "official_model_weighted_average",
        "accuracy": _weighted(stage5_rows, "accuracy"),
        "log_loss": _weighted(stage5_rows, "log_loss"),
        "brier_score": _weighted(stage5_rows, "brier"),
        "ranked_probability_score": _weighted(stage5_rows, "rps"),
        "calibration_ece": _weighted(stage5_rows, "ece"),
        "comparable_to_stage5": "Yes - this is the Stage 5 reference result",
        "primary_use": "Near-SOTA benchmark claim evidence",
        "notes": notes,
    }


def _official_prediction_count(predictions: pd.DataFrame) -> int | None:
    if predictions.empty or "model_name" not in predictions.columns:
        return None
    return int(predictions[predictions["model_name"].eq("official_model")].shape[0])


def _weighted(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    weights = pd.to_numeric(df["n_matches"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(weights) & (weights > 0) & np.isfinite(values)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _extract_core_ablation_row(final_report: str) -> pd.Series:
    table = _extract_markdown_table(final_report, "## Ablation Summary")
    if table.empty:
        raise ValueError("Could not find an Ablation Summary table in final_model_recommendation.md")
    if "feature_set" not in table.columns:
        raise ValueError("Ablation Summary table does not contain a feature_set column")
    core = table[table["feature_set"].eq("core_football_only")]
    if core.empty:
        raise ValueError("Ablation Summary table does not contain core_football_only")
    row = core.iloc[0].copy()
    for column in ["accuracy", "log_loss", "brier_score", "ranked_probability_score", "calibration_error"]:
        if column in row.index:
            row[column] = _float_or_nan(row[column])
    return row


def _extract_markdown_table(markdown: str, heading: str) -> pd.DataFrame:
    lines = markdown.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == heading:
            start = idx + 1
            break
    if start is None:
        return pd.DataFrame()
    table_lines: list[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)
        elif table_lines:
            break
    if len(table_lines) < 3:
        return pd.DataFrame()
    headers = _split_markdown_row(table_lines[0])
    rows = [_split_markdown_row(line) for line in table_lines[2:]]
    normalized = [row[: len(headers)] + [""] * max(0, len(headers) - len(row)) for row in rows]
    return pd.DataFrame(normalized, columns=headers)


def _split_markdown_row(line: str) -> list[str]:
    return [part.strip().replace("\\|", "|") for part in line.strip().strip("|").split("|")]


def _read_required_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required report: {path}")
    return path.read_text(encoding="utf-8")


def _read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Stage 5 output: {path}. Run `python -m src.cli stage5-benchmark` first.")
    return pd.read_csv(path, low_memory=False)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _float_or_nan(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _stage5_achieved(decision_report: str, significance: pd.DataFrame) -> bool:
    if "Decision: **Stage 5 achieved**" in decision_report:
        return True
    if significance.empty:
        return False
    row = _official_vs_bookmaker(significance)
    if row.empty:
        return False
    return bool(row.iloc[0].get("statistically_meaningful", False)) and _float_or_nan(row.iloc[0].get("ci_upper")) < 0


def _official_vs_bookmaker(significance: pd.DataFrame) -> pd.DataFrame:
    if significance.empty:
        return pd.DataFrame()
    return significance[
        significance["candidate_model"].eq("official_model")
        & significance["baseline_model"].eq("bookmaker_odds_only")
        & significance["metric"].eq("log_loss")
    ].copy()


def _log_loss_gap(ablation: pd.DataFrame, stage5_average: pd.DataFrame) -> str:
    if ablation.empty or stage5_average.empty:
        return "- Log-loss gap could not be computed because one of the source rows is missing."
    ablation_loss = _float_or_nan(ablation.iloc[0].get("log_loss"))
    stage5_loss = _float_or_nan(stage5_average.iloc[0].get("log_loss"))
    if not np.isfinite(ablation_loss) or not np.isfinite(stage5_loss):
        return "- Log-loss gap could not be computed because one of the log-loss values is missing."
    return (
        f"- Ablation `core_football_only` log_loss: `{ablation_loss:.6f}`.\n"
        f"- Stage 5 official_model weighted log_loss: `{stage5_loss:.6f}`.\n"
        f"- Raw difference: `{stage5_loss - ablation_loss:.6f}`.\n"
        "- This difference is expected because the two rows use different evaluation protocols."
    )


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
