from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import config_path, load_config, resolve_project_path
from src.features.feature_sets import FEATURE_GROUPS, groups_for_feature_set, is_experimental_feature_set


METRIC_COLUMNS = ["accuracy", "log_loss", "brier_score", "ranked_probability_score", "calibration_error"]
SAFE_BASELINE_FEATURE_SET = "core_football_only"
SAFE_INTERNAL_ELO_FEATURE_SET = "core_plus_internal_elo"
INTERNAL_ELO_STRONG_DELTA = -0.002


def write_final_model_recommendation(
    config: dict[str, Any] | None = None,
    *,
    selected_model: str | None = None,
    selected_feature_set: str | None = None,
) -> Path:
    cfg = config or load_config()
    report_path = _optional_config_path(cfg, "final_model_recommendation_md", "data/reports/final_model_recommendation.md")
    ablation = _read_csv_path(cfg, "ablation_results_csv", "data/experiments/ablation_results.csv")
    null_summary = _read_csv_path(cfg, "feature_null_rate_report_csv", "data/reports/feature_null_rate_report.csv")
    historical_coverage = _read_csv_path(cfg, "historical_rating_coverage_report_csv", "data/reports/historical_rating_coverage_report.csv")
    activation = _read_csv_path(cfg, "external_rating_activation_report_csv", "data/reports/external_rating_activation_report.csv")
    backtest = _read_csv_path(cfg, "worldcup_backtest_metrics", "data/backtests/worldcup_backtest_metrics.csv")
    tournament_backtest = _read_csv_path(
        cfg,
        "worldcup_tournament_simulation_backtest_csv",
        "data/backtests/worldcup_tournament_simulation_backtest.csv",
    )
    calibration = _read_json_path(cfg, "calibration_params", "models/calibration_params.json")
    market_params = _read_json_path(cfg, "market_blend_params", "models/market_blend_params.json")
    significance = _read_significance_summary(cfg)
    simulation_metadata = _read_json_path(cfg, "simulation_metadata", "data/simulation/worldcup_2026_simulation_metadata.json")
    predictions = _read_prediction_file(cfg)

    best_row = _best_ablation_row(ablation)
    best_feature_set = "" if best_row is None else str(best_row["feature_set"])
    best_safe_row = _best_safe_ablation_row(ablation)
    best_safe_feature_set = "" if best_safe_row is None else str(best_safe_row["feature_set"])
    safe_feature_set, safe_reason = _safe_production_feature_set(ablation)
    safe_model = _safe_prediction_model(predictions, selected_model=selected_model, calibration=calibration)
    training_model = str(calibration.get("model_name") or selected_model or "unknown")
    current_pipeline_feature_set = selected_feature_set or str(calibration.get("feature_set") or safe_feature_set)
    simulation_mode = "experimental" if _is_known_experimental(current_pipeline_feature_set) else "safe"
    validation_row = _row_for_feature_set(ablation, safe_feature_set)
    if validation_row is None:
        validation_row = best_safe_row
    if validation_row is None:
        validation_row = best_row
    validation_metrics = _validation_metrics(calibration, validation_row)
    fifa_help = _feature_help(ablation, "v1_baseline", "v2_baseline_plus_fifa_rankings")
    elo_help = _feature_help(ablation, "v2_baseline_plus_fifa_rankings", "v3_plus_external_elo")
    fifa_coverage = _coverage(historical_coverage, activation, null_summary, "fifa_rankings")
    elo_coverage = _coverage(historical_coverage, activation, null_summary, "world_football_elo")
    market_coverage_2026 = _worldcup_prediction_coverage(cfg, FEATURE_GROUPS["market"])
    market_status = "production-ready" if market_coverage_2026 >= 0.999 else "benchmark-only"
    market_alpha = _market_alpha(market_params)
    odds_method = market_params.get("odds_conversion_method", "unknown") if market_params else "unknown"
    prediction_match = _prediction_matches(predictions, safe_model, safe_feature_set)
    simulation_match = _simulation_matches(simulation_metadata, safe_model, safe_feature_set)

    lines = [
        "# Final Model Recommendation",
        "",
        "This project is SOTA-inspired, historically backtested, and benchmarked against bookmaker odds. It is not a true SOTA claim.",
        "",
        "## Recommendation",
        "",
        f"- Safe production model: `{safe_model}`",
        f"- Safe production feature set: `{safe_feature_set}`",
        f"- Training estimator: `{training_model}`",
        f"- Current pipeline feature set: `{current_pipeline_feature_set}`",
        f"- Current World Cup simulation used: `{simulation_mode}` features",
        f"- Best feature set by validation log_loss: `{best_feature_set or 'unknown'}`",
        f"- Best safe ablation feature set: `{best_safe_feature_set or 'unknown'}`",
        f"- Safe feature-set policy: {safe_reason}",
        "",
        "## Market-Assisted Benchmark",
        "",
        f"- Market-assisted status: `{market_status}`",
        f"- Best market blend alpha: `{_format_value(market_alpha)}`",
        f"- Best odds conversion method: `{odds_method}`",
        f"- Market blend formula: `alpha * model_prob + (1 - alpha) * market_prob`",
        f"- World Cup 2026 market feature coverage: `{_format_value(market_coverage_2026)}`",
        "- Policy: `market_blend` is benchmark-only until 2026 market odds successfully join to all active fixtures.",
        f"- Market blend vs bookmaker odds log_loss delta CI: `{significance.get('ci_text', 'not available')}`",
        f"- Market blend improvement statistically meaningful: `{significance.get('statistically_meaningful', 'not available')}`",
        "",
        "## Validation Metrics",
        "",
    ]
    for key in METRIC_COLUMNS:
        if key in validation_metrics:
            lines.append(f"- `{key}`: {_format_value(validation_metrics[key])}")

    lines.extend(
        [
            "",
            "## Ablation Summary",
            "",
            _ablation_table(ablation),
            "",
            "## Match-Level World Cup Backtests",
            "",
            _backtest_summary(backtest),
            "",
            "## Tournament Simulation Backtest",
            "",
            _tournament_backtest_summary(tournament_backtest),
            "",
            "## Prediction And Simulation Metadata",
            "",
            f"- Prediction file matches safe recommendation: `{prediction_match}`",
            f"- Simulation metadata matches safe recommendation: `{simulation_match}`",
            f"- Prediction model counts: `{_counts_text(_value_counts(predictions, 'model_name'))}`",
            f"- Prediction feature-set counts: `{_counts_text(_value_counts(predictions, 'feature_set'))}`",
            f"- Simulation prediction source: `{simulation_metadata.get('prediction_source_path', 'unknown') if simulation_metadata else 'unknown'}`",
            f"- Simulation model counts: `{_counts_text(simulation_metadata.get('model_name_counts', {}) if simulation_metadata else {})}`",
            f"- Simulation feature-set counts: `{_counts_text(simulation_metadata.get('feature_set_counts', {}) if simulation_metadata else {})}`",
            "",
            "## External Rating Checks",
            "",
            f"- FIFA helped validation: `{fifa_help['helped']}` (log_loss_delta={_format_value(fifa_help['delta'])})",
            f"- FIFA historical training coverage: `{_format_value(fifa_coverage)}`",
            f"- External Elo helped validation: `{elo_help['helped']}` (log_loss_delta={_format_value(elo_help['delta'])})",
            f"- External Elo historical training coverage: `{_format_value(elo_coverage)}`",
            "",
            "## Feature Policy",
            "",
            f"- `{safe_feature_set}` groups: {', '.join(groups_for_feature_set(safe_feature_set)) if safe_feature_set else 'unknown'}",
            "- Experimental and excluded from the safe production path: sparse FIFA rankings, external Elo snapshots, xG, squad/player snapshots, injury/suspension snapshots, and market odds unless coverage is complete and no-leakage validation supports promotion.",
            "- Historical FIFA/Elo ingestion is implemented, but sparse coverage means these sources are not selected for the safe final model.",
            "- Squad/player, injury/suspension, and xG interfaces are not safe historical training features with the currently available data.",
            "",
            "## Warnings",
            "",
            "- Do not use `market_blend` for 2026 production predictions until market odds join to all active 2026 fixtures.",
            "- Current 2026 market odds rows exist, but the latest feature-null report shows market World Cup 2026 coverage below production readiness.",
            "- The final simulation should be interpreted as the safe football-only path unless simulation metadata shows complete market blending.",
            "- `v1_baseline` is not recommended because current ablation results place stronger safe feature sets ahead of it.",
            "- This project is SOTA-inspired, historically backtested, and benchmarked against bookmaker odds.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _safe_production_feature_set(ablation: pd.DataFrame) -> tuple[str, str]:
    core = _row_for_feature_set(ablation, SAFE_BASELINE_FEATURE_SET)
    internal = _row_for_feature_set(ablation, SAFE_INTERNAL_ELO_FEATURE_SET)
    if core is None and internal is not None:
        return SAFE_INTERNAL_ELO_FEATURE_SET, "`core_football_only` is unavailable; using the best available internal-Elo safe set."
    if core is None:
        best_safe = _best_safe_ablation_row(ablation)
        if best_safe is not None:
            return str(best_safe["feature_set"]), "Using the best available non-experimental ablation row."
        return SAFE_BASELINE_FEATURE_SET, "Defaulting to the safe football-only feature set."
    if internal is not None:
        delta = float(internal["log_loss"] - core["log_loss"])
        if delta <= INTERNAL_ELO_STRONG_DELTA:
            return SAFE_INTERNAL_ELO_FEATURE_SET, f"`core_plus_internal_elo` improves log_loss over `core_football_only` by {abs(delta):.6f}, exceeding the promotion threshold."
        if delta < 0:
            return SAFE_BASELINE_FEATURE_SET, f"`core_plus_internal_elo` improves log_loss by only {abs(delta):.6f}; this is not strong enough to displace the simpler current production path."
        return SAFE_BASELINE_FEATURE_SET, f"`core_plus_internal_elo` worsens log_loss by {delta:.6f}; keeping the simpler current production path."
    return SAFE_BASELINE_FEATURE_SET, "`core_football_only` is the stable safe production path."


def _best_ablation_row(ablation: pd.DataFrame) -> pd.Series | None:
    if ablation.empty or "log_loss" not in ablation.columns:
        return None
    ranked = ablation.dropna(subset=["log_loss"]).sort_values("log_loss", kind="stable")
    return None if ranked.empty else ranked.iloc[0]


def _best_safe_ablation_row(ablation: pd.DataFrame) -> pd.Series | None:
    if ablation.empty or "feature_set" not in ablation.columns or "log_loss" not in ablation.columns:
        return None
    rows = []
    for _, row in ablation.dropna(subset=["log_loss"]).iterrows():
        feature_set = str(row["feature_set"])
        if not _is_known_experimental(feature_set):
            rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("log_loss", kind="stable").iloc[0]


def _row_for_feature_set(ablation: pd.DataFrame, feature_set: str) -> pd.Series | None:
    if ablation.empty or "feature_set" not in ablation.columns:
        return None
    row = ablation[ablation["feature_set"].eq(feature_set)]
    return None if row.empty else row.iloc[0]


def _feature_help(ablation: pd.DataFrame, baseline: str, candidate: str) -> dict[str, Any]:
    base = _row_for_feature_set(ablation, baseline)
    current = _row_for_feature_set(ablation, candidate)
    if base is None or current is None or "log_loss" not in base or "log_loss" not in current:
        return {"helped": False, "delta": pd.NA}
    delta = float(current["log_loss"] - base["log_loss"])
    return {"helped": delta < -1e-12, "delta": delta}


def _coverage(historical: pd.DataFrame, activation: pd.DataFrame, null_detail: pd.DataFrame, dataset: str) -> float:
    if not historical.empty and "dataset" in historical.columns and "historical_training_coverage" in historical.columns:
        row = historical[historical["dataset"].eq(dataset)]
        if not row.empty:
            return float(row.iloc[0]["historical_training_coverage"])
    if not activation.empty and "dataset" in activation.columns and "historical_training_coverage" in activation.columns:
        row = activation[activation["dataset"].eq(dataset)]
        if not row.empty:
            return float(row.iloc[0]["historical_training_coverage"])
    group = "external_elo" if dataset == "world_football_elo" else dataset
    return _feature_group_coverage(null_detail, group, "historical_training_coverage")


def _feature_group_coverage(null_detail: pd.DataFrame, group: str, column: str) -> float:
    if not null_detail.empty and "feature_group" in null_detail.columns and column in null_detail.columns:
        row = null_detail[null_detail["feature_group"].eq(group)]
        if not row.empty and pd.notna(row.iloc[0][column]):
            return float(row.iloc[0][column])
    return 0.0


def _validation_metrics(calibration: dict[str, Any], fallback_row: pd.Series | None) -> dict[str, Any]:
    after = calibration.get("after")
    if isinstance(after, dict):
        return after
    if fallback_row is not None:
        return {column: fallback_row[column] for column in METRIC_COLUMNS if column in fallback_row}
    return {}


def _safe_prediction_model(predictions: pd.DataFrame, *, selected_model: str | None, calibration: dict[str, Any]) -> str:
    counts = _value_counts(predictions, "model_name")
    if len(counts) == 1:
        return next(iter(counts))
    if "football_only_ensemble" in counts:
        return "football_only_ensemble"
    return str(calibration.get("model_name") or selected_model or "unknown")


def _prediction_matches(predictions: pd.DataFrame, safe_model: str, safe_feature_set: str) -> bool:
    if predictions.empty:
        return False
    model_counts = _value_counts(predictions, "model_name")
    feature_counts = _value_counts(predictions, "feature_set")
    return set(model_counts) == {safe_model} and set(feature_counts) == {safe_feature_set}


def _simulation_matches(metadata: dict[str, Any], safe_model: str, safe_feature_set: str) -> bool:
    if not metadata:
        return False
    model_counts = metadata.get("model_name_counts", {})
    feature_counts = metadata.get("feature_set_counts", {})
    return set(model_counts) == {safe_model} and set(feature_counts) == {safe_feature_set}


def _market_alpha(params: dict[str, Any]) -> Any:
    if not params:
        return pd.NA
    return params.get("alpha", pd.NA)


def _worldcup_prediction_coverage(cfg: dict[str, Any], columns: list[str]) -> float:
    for key in ("worldcup_2026_prediction_input_advanced", "worldcup_2026_prediction_input"):
        path = _optional_config_path(cfg, key, "")
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
        except Exception:
            csv_path = path.with_suffix(".csv")
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path, low_memory=False)
        available = [column for column in columns if column in df.columns]
        if not available or df.empty:
            continue
        if "home_team" in df.columns and "away_team" in df.columns:
            placeholder = df["home_team"].astype(str).str.match(r"^[123][A-L]|^[WL]\d+", na=False) | df["away_team"].astype(str).str.match(r"^[123][A-L]|^[WL]\d+", na=False)
            df = df[~placeholder]
        if df.empty:
            return 0.0
        return float(df[available].notna().any(axis=1).mean())
    return 0.0


def _read_significance_summary(cfg: dict[str, Any]) -> dict[str, Any]:
    path = _optional_config_path(cfg, "benchmark_significance_report_md", "data/reports/benchmark_significance_report.md")
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| market_blend_minus_bookmaker_odds_only |"):
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) >= 5:
                return {
                    "delta": parts[1],
                    "ci_lower": parts[2],
                    "ci_upper": parts[3],
                    "statistically_meaningful": parts[4],
                    "ci_text": f"delta={parts[1]}, 95% CI [{parts[2]}, {parts[3]}]",
                }
    return {}


def _ablation_table(ablation: pd.DataFrame) -> str:
    if ablation.empty:
        return "_No ablation results available._"
    cols = [column for column in ["feature_set", "accuracy", "log_loss", "brier_score", "ranked_probability_score", "calibration_error", "warning"] if column in ablation.columns]
    ranked = ablation.sort_values("log_loss", kind="stable") if "log_loss" in ablation.columns else ablation
    return _markdown_table(ranked, cols)


def _backtest_summary(backtest: pd.DataFrame) -> str:
    if backtest.empty:
        return "_No match-level World Cup backtest metrics available._"
    average = backtest.groupby(["evaluation_scope", "model_name"], as_index=False)[METRIC_COLUMNS].mean()
    average = average.sort_values(["evaluation_scope", "log_loss", "model_name"], kind="stable")
    lines = [
        _markdown_table(average, ["evaluation_scope", "model_name", "accuracy", "log_loss", "brier_score", "ranked_probability_score", "calibration_error"]),
        "",
    ]
    all_scope = average[average["evaluation_scope"].eq("all_matches")]
    odds_scope = average[average["evaluation_scope"].eq("odds_covered")]
    for baseline, label in [("elo_only", "Elo-only"), ("fifa_ranking_only", "FIFA-only"), ("rolling_form_only", "rolling-form-only"), ("majority_class_baseline", "majority-class")]:
        lines.append(f"- Final model beats {label} on all-match log_loss: `{_beats(all_scope, 'final_model', baseline)}`")
    lines.append(f"- Market blend beats final model on odds-covered log_loss: `{_beats(odds_scope, 'market_blend', 'final_model')}`")
    lines.append(f"- Market blend beats bookmaker odds on odds-covered log_loss: `{_beats(odds_scope, 'market_blend', 'bookmaker_odds_only')}`")
    return "\n".join(lines)


def _tournament_backtest_summary(tournament: pd.DataFrame) -> str:
    if tournament.empty:
        return "_No tournament simulation backtest metrics available._"
    cols = [
        "world_cup_year",
        "model_name",
        "actual_champion",
        "champion_probability_assigned",
        "actual_finalists_probability",
        "semifinalist_probability_recall",
        "round_of_16_brier_score",
        "group_qualification_accuracy",
        "group_match_log_loss",
    ]
    average = (
        tournament.groupby("model_name", as_index=False)
        .agg(
            world_cups=("world_cup_year", "nunique"),
            champion_probability_assigned=("champion_probability_assigned", "mean"),
            actual_finalists_probability=("actual_finalists_probability", "mean"),
            semifinalist_probability_recall=("semifinalist_probability_recall", "mean"),
            round_of_16_brier_score=("round_of_16_brier_score", "mean"),
            group_qualification_accuracy=("group_qualification_accuracy", "mean"),
            group_match_log_loss=("group_match_log_loss", "mean"),
        )
        .sort_values(["group_match_log_loss", "model_name"], kind="stable")
    )
    return "\n".join([_markdown_table(tournament, cols), "", "Average:", "", _markdown_table(average, list(average.columns))])


def _beats(df: pd.DataFrame, candidate: str, baseline: str) -> str:
    left = df[df["model_name"].eq(candidate)]
    right = df[df["model_name"].eq(baseline)]
    if left.empty or right.empty:
        return "not available"
    delta = float(left.iloc[0]["log_loss"] - right.iloc[0]["log_loss"])
    return f"{delta < 0} (log_loss_delta={delta:.6f})"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in df.columns]
    if df.empty or not available:
        return "_No rows._"
    rows = ["| " + " | ".join(available) + " |", "| " + " | ".join(["---"] * len(available)) + " |"]
    for record in df[available].to_dict("records"):
        rows.append("| " + " | ".join(_format_value(record[column]) for column in available) + " |")
    return "\n".join(rows)


def _read_prediction_file(cfg: dict[str, Any]) -> pd.DataFrame:
    path = _optional_config_path(cfg, "actual_team_match_predictions_csv", "data/predictions/actual_team_match_predictions.csv")
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _read_csv_path(cfg: dict[str, Any], key: str, default: str) -> pd.DataFrame:
    path = _optional_config_path(cfg, key, default)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _read_json_path(cfg: dict[str, Any], key: str, default: str) -> dict[str, Any]:
    path = _optional_config_path(cfg, key, default)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _optional_config_path(cfg: dict[str, Any], key: str, default: str) -> Path:
    try:
        return config_path(cfg, key)
    except KeyError:
        return resolve_project_path(default)


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df[column].value_counts(dropna=False).items()}


def _counts_text(counts: dict[str, int]) -> str:
    if not counts:
        return "unknown"
    return ", ".join(f"{key}:{value}" for key, value in counts.items())


def _is_known_experimental(feature_set: str) -> bool:
    try:
        return is_experimental_feature_set(feature_set)
    except KeyError:
        return False


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
