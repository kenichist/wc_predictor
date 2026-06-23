from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import config_path, load_config
from src.features.build_training_dataset import _add_flags_and_targets, _merge_feature_frame, _prepare_base_matches
from src.features.dynamic_ratings import DYNAMIC_RATING_COLUMNS, compute_dynamic_ratings
from src.features.elo import ELO_COLUMNS, compute_elo_features
from src.features.external_feature_joiner import add_market_features
from src.features.feature_sets import BASIC_ELO_FEATURES, columns_for_feature_set, groups_for_feature_set
from src.features.internal_historical_elo import INTERNAL_HISTORICAL_ELO_COLUMNS, compute_internal_historical_elo
from src.features.match_weights import compute_match_weights
from src.features.pi_ratings import PI_RATING_COLUMNS, compute_pi_ratings
from src.features.rolling_team_form import ROLLING_FEATURE_COLUMNS, compute_rolling_team_form
from src.io_utils import read_dataframe, write_dataframe, write_json
from src.models.evaluate import evaluate_probabilities
from src.models.train_match_model import _feature_matrix, _make_estimator, _predict_proba_3
from src.sources.market_odds import ODDS_CONVERSION_METHODS, decimal_odds_to_probabilities
from src.sources.fifa_rankings import FIFA_RANKING_MODEL_COLUMNS, add_fifa_ranking_features
from src.sources.injury_interface import INJURY_FEATURE_COLUMNS
from src.sources.player_stats_interface import PLAYER_FEATURE_COLUMNS
from src.sources.world_football_elo import add_world_football_elo_features
from src.sources.xg_interface import XG_FEATURE_COLUMNS
from src.validation import REQUIRED_TRAINING_COLUMNS, add_missing_columns


logger = logging.getLogger(__name__)

WORLD_CUP_START_DATES: dict[int, str] = {
    2010: "2010-06-11",
    2014: "2014-06-12",
    2018: "2018-06-14",
    2022: "2022-11-20",
}

METRIC_COLUMNS = ["accuracy", "log_loss", "brier_score", "ranked_probability_score", "calibration_error"]
PROBABILITY_COLUMNS = ["p_home_loss", "p_draw", "p_home_win"]
MARKET_PROBABILITY_COLUMNS = ["market_away_prob", "market_draw_prob", "market_home_prob"]
CURRENT_ONLY_EXTERNAL_COLUMNS = set(XG_FEATURE_COLUMNS) | set(PLAYER_FEATURE_COLUMNS) | set(INJURY_FEATURE_COLUMNS)
ALPHA_GRID = [round(value / 10.0, 1) for value in range(11)]


def run_worldcup_backtests(
    *,
    model_name: str = "catboost",
    feature_set: str = "core_football_only",
    config: dict[str, Any] | None = None,
    feature_frame: pd.DataFrame | None = None,
    years: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    cfg = config or load_config()
    tournament_years = years or list(WORLD_CUP_START_DATES)
    df = feature_frame.copy() if feature_frame is not None else build_worldcup_backtest_feature_frame(config=cfg, feature_set=feature_set)
    df = _prepare_backtest_frame(df)

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    fold_results: list[dict[str, Any]] = []
    for year in tournament_years:
        train_df, target_df = split_worldcup_backtest_fold(df, year)
        if target_df.empty:
            logger.warning("Skipping %s World Cup backtest because no target rows were found", year)
            continue
        if train_df.empty:
            logger.warning("Skipping %s World Cup backtest because no pre-tournament training rows were found", year)
            continue

        target_df["_backtest_stage"] = _infer_worldcup_stage_labels(target_df)
        tournament_start = pd.Timestamp(WORLD_CUP_START_DATES[year])
        final_prob, final_model_name, final_columns = _fit_predict_feature_model(
            train_df,
            target_df,
            feature_columns=_safe_feature_columns(feature_set, df.columns),
            model_name=model_name,
            random_seed=int(cfg.get("modeling", {}).get("random_seed", 42)),
            modeling_config=cfg.get("modeling", {}),
        )
        fold_results.append(
            {
                "year": year,
                "train_df": train_df,
                "target_df": target_df,
                "tournament_start": tournament_start,
                "final_prob": final_prob,
                "final_columns": final_columns,
            }
        )
        final_predictions = _prediction_frame(
            target_df,
            probabilities=final_prob,
            world_cup_year=year,
            model_name="final_model",
            feature_set=feature_set,
            benchmark="final_model",
            tournament_start=tournament_start,
            train_df=train_df,
            feature_count=len(final_columns),
            estimator_name=final_model_name,
            evaluation_scope="all_matches",
        )
        prediction_frames.append(final_predictions)
        metric_rows.append(_metric_row(final_predictions, train_df, target_df, tournament_start, len(final_columns), evaluation_scope="all_matches"))

        for benchmark_name, columns in _benchmark_feature_sets().items():
            probabilities, _, used_columns = _fit_predict_feature_model(
                train_df,
                target_df,
                feature_columns=[column for column in columns if column in df.columns],
                model_name="hist_gradient_boosting",
                random_seed=int(cfg.get("modeling", {}).get("random_seed", 42)),
                modeling_config=cfg.get("modeling", {}),
                fallback_name=benchmark_name,
            )
            predictions = _prediction_frame(
                target_df,
                probabilities=probabilities,
                world_cup_year=year,
                model_name=benchmark_name,
                feature_set=benchmark_name,
                benchmark=benchmark_name,
                tournament_start=tournament_start,
                train_df=train_df,
                feature_count=len(used_columns),
                evaluation_scope="all_matches",
            )
            prediction_frames.append(predictions)
            metric_rows.append(_metric_row(predictions, train_df, target_df, tournament_start, len(used_columns), evaluation_scope="all_matches"))

        majority_prob = _majority_probabilities(train_df, len(target_df))
        majority_predictions = _prediction_frame(
            target_df,
            probabilities=majority_prob,
            world_cup_year=year,
            model_name="majority_class_baseline",
            feature_set="majority_class_baseline",
            benchmark="majority_class_baseline",
            tournament_start=tournament_start,
            train_df=train_df,
            feature_count=0,
            evaluation_scope="all_matches",
        )
        prediction_frames.append(majority_predictions)
        metric_rows.append(_metric_row(majority_predictions, train_df, target_df, tournament_start, 0, evaluation_scope="all_matches"))

    odds_conversion = _odds_conversion_benchmark(fold_results)
    best_odds_method = _best_odds_conversion_method(odds_conversion)
    alpha_grid = _alpha_grid_metrics(fold_results, odds_method=best_odds_method)
    best_alpha = _best_alpha(alpha_grid)
    alpha_by_year = _alpha_grid_metrics_by_year(fold_results, odds_method=best_odds_method)
    alpha_by_stage = _alpha_grid_metrics_by_stage(fold_results, odds_method=best_odds_method)
    confidence_weighted = _confidence_weighted_blend_metrics(fold_results, odds_method=best_odds_method)
    for fold in fold_results:
        year = int(fold["year"])
        train_df = fold["train_df"]
        target_df = fold["target_df"]
        tournament_start = fold["tournament_start"]
        final_prob = fold["final_prob"]
        final_columns = fold["final_columns"]
        market_mask = _market_coverage_mask(target_df)
        covered_count = int(market_mask.sum())
        if covered_count == 0:
            continue
        covered_target = target_df[market_mask].copy().reset_index(drop=True)
        covered_final_prob = final_prob[market_mask.to_numpy()]
        market_prob = _market_probabilities(covered_target, odds_method=best_odds_method)
        if market_prob is None:
            continue

        covered_final_predictions = _prediction_frame(
            covered_target,
            probabilities=covered_final_prob,
            world_cup_year=year,
            model_name="final_model",
            feature_set=feature_set,
            benchmark="final_model",
            tournament_start=tournament_start,
            train_df=train_df,
            feature_count=len(final_columns),
            evaluation_scope="odds_covered",
        )
        prediction_frames.append(covered_final_predictions)
        metric_rows.append(
            _metric_row(
                covered_final_predictions,
                train_df,
                covered_target,
                tournament_start,
                len(final_columns),
                evaluation_scope="odds_covered",
                odds_covered_matches=covered_count,
                odds_total_matches=len(target_df),
            )
        )

        market_predictions = _prediction_frame(
            covered_target,
            probabilities=market_prob,
            world_cup_year=year,
            model_name="bookmaker_odds_only",
            feature_set="market",
            benchmark="bookmaker_odds_only",
            tournament_start=tournament_start,
            train_df=train_df,
            feature_count=3,
            evaluation_scope="odds_covered",
        )
        prediction_frames.append(market_predictions)
        metric_rows.append(
            _metric_row(
                market_predictions,
                train_df,
                covered_target,
                tournament_start,
                3,
                evaluation_scope="odds_covered",
                odds_covered_matches=covered_count,
                odds_total_matches=len(target_df),
            )
        )

        blend_prob = blend_probabilities(covered_final_prob, market_prob, best_alpha)
        blend_predictions = _prediction_frame(
            covered_target,
            probabilities=blend_prob,
            world_cup_year=year,
            model_name="market_blend",
            feature_set=f"{feature_set}+market",
            benchmark="market_blend",
            tournament_start=tournament_start,
            train_df=train_df,
            feature_count=len(final_columns) + 3,
            evaluation_scope="odds_covered",
            alpha=best_alpha,
        )
        prediction_frames.append(blend_predictions)
        metric_rows.append(
            _metric_row(
                blend_predictions,
                train_df,
                covered_target,
                tournament_start,
                len(final_columns) + 3,
                evaluation_scope="odds_covered",
                odds_covered_matches=covered_count,
                odds_total_matches=len(target_df),
                alpha=best_alpha,
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False) if prediction_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)

    predictions_path = _path_or_default(cfg, "worldcup_backtest_predictions", "data/backtests/worldcup_backtest_predictions.csv")
    metrics_path = _path_or_default(cfg, "worldcup_backtest_metrics", "data/backtests/worldcup_backtest_metrics.csv")
    report_path = _path_or_default(cfg, "worldcup_backtest_report_md", "data/reports/worldcup_backtest_report.md")
    market_report_path = _path_or_default(cfg, "market_blend_report_md", "data/reports/market_blend_report.md")
    odds_conversion_report_path = _path_or_default(cfg, "odds_conversion_benchmark_report_md", "data/reports/odds_conversion_benchmark_report.md")
    significance_report_path = _path_or_default(cfg, "benchmark_significance_report_md", "data/reports/benchmark_significance_report.md")
    params_path = _path_or_default(cfg, "market_blend_params", "models/market_blend_params.json")
    write_dataframe(predictions, predictions_path)
    write_dataframe(metrics, metrics_path)
    write_worldcup_backtest_report(metrics=metrics, predictions=predictions, path=report_path, feature_set=feature_set)
    write_odds_conversion_benchmark_report(odds_conversion, best_odds_method, odds_conversion_report_path)
    significance = write_benchmark_significance_report(predictions=predictions, path=significance_report_path)
    write_market_blend_report(
        metrics=metrics,
        predictions=predictions,
        alpha_grid=alpha_grid,
        best_alpha=best_alpha,
        best_odds_method=best_odds_method,
        alpha_by_year=alpha_by_year,
        alpha_by_stage=alpha_by_stage,
        confidence_weighted=confidence_weighted,
        significance=significance,
        path=market_report_path,
    )
    write_json(
        {
            "alpha": best_alpha,
            "source": "worldcup_backtests",
            "alpha_grid": ALPHA_GRID,
            "odds_conversion_method": best_odds_method,
        },
        params_path,
    )
    logger.info("Wrote World Cup backtest predictions to %s", predictions_path)
    logger.info("Wrote World Cup backtest metrics to %s", metrics_path)
    logger.info("Wrote World Cup backtest report to %s", report_path)
    logger.info("Wrote market blend report to %s", market_report_path)
    return predictions, metrics, report_path


def build_worldcup_backtest_feature_frame(*, config: dict[str, Any] | None = None, feature_set: str = "core_football_only") -> pd.DataFrame:
    cfg = config or load_config()
    matches = read_dataframe(config_path(cfg, "all_matches_clean"))
    if matches.empty:
        raise ValueError("No clean matches available for backtesting. Run `python -m src.cli build-clean` first.")

    base = _prepare_base_matches(matches, cfg)
    form_features = compute_rolling_team_form(base)
    elo_features = compute_elo_features(base, config=cfg)

    enriched = _merge_feature_frame(base, form_features, ROLLING_FEATURE_COLUMNS)
    enriched = _merge_feature_frame(enriched, elo_features, ELO_COLUMNS)
    enriched = add_fifa_ranking_features(enriched, config_path(cfg, "fifa_rankings"))
    enriched["match_weight"] = compute_match_weights(enriched, config=cfg)
    enriched = _add_flags_and_targets(enriched)

    requested_groups = set(groups_for_feature_set(feature_set))
    if "dynamic_ratings" in requested_groups:
        dynamic = compute_dynamic_ratings(base, config=cfg)
        enriched = _merge_feature_frame(enriched, dynamic, DYNAMIC_RATING_COLUMNS)
    if "pi_ratings" in requested_groups:
        pi = compute_pi_ratings(base)
        enriched = _merge_feature_frame(enriched, pi, PI_RATING_COLUMNS)
    if "internal_historical_elo" in requested_groups:
        internal_elo = compute_internal_historical_elo(base, config=cfg)
        enriched = _merge_feature_frame(enriched, internal_elo.features[["match_id", *INTERNAL_HISTORICAL_ELO_COLUMNS]], INTERNAL_HISTORICAL_ELO_COLUMNS)
    if "external_elo" in requested_groups:
        enriched = add_world_football_elo_features(enriched, config_path(cfg, "world_football_elo"))

    enriched = add_market_features(enriched, config_path(cfg, "betting_odds"))
    enriched = add_missing_columns(enriched, REQUIRED_TRAINING_COLUMNS)
    return enriched.reset_index(drop=True)


def split_worldcup_backtest_fold(df: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if year not in WORLD_CUP_START_DATES:
        raise KeyError(f"Unsupported World Cup backtest year: {year}")
    prepared = _prepare_backtest_frame(df)
    dates = pd.to_datetime(prepared["date"], errors="coerce")
    tournament_start = pd.Timestamp(WORLD_CUP_START_DATES[year])
    target_mask = _target_tournament_mask(prepared, year)
    scored = prepared["target_result_class"].notna()
    train = prepared[(dates < tournament_start) & scored & ~target_mask].copy().sort_values(["date", "match_id"], kind="stable")
    target = prepared[target_mask & scored].copy().sort_values(["date", "match_id"], kind="stable")
    return train.reset_index(drop=True), target.reset_index(drop=True)


def write_worldcup_backtest_report(
    *,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    path: str | Path,
    feature_set: str,
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Historical World Cup Backtest Report",
        "",
        f"- Final model feature set: `{feature_set}`",
        "- Tournaments: 2010, 2014, 2018, 2022 FIFA World Cups",
        "- Leakage rule: each fold trains only on matches dated before that tournament's opening match.",
        "- Current 2026 squad/player, injury, suspension, and xG interfaces are excluded from historical backtests unless proper historical as-of data is available.",
        "",
    ]
    if metrics.empty:
        lines.extend(["No backtest metrics were produced.", ""])
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    ordered = metrics.sort_values(["world_cup_year", "evaluation_scope", "log_loss", "model_name"], kind="stable")
    lines.extend(
        [
            "## Metrics By Tournament",
            "",
            _markdown_table(ordered, ["world_cup_year", "evaluation_scope", "model_name", *METRIC_COLUMNS, "n_matches", "n_train", "odds_coverage_rate", "alpha"]),
            "",
        ]
    )

    averages = _average_metrics(metrics)
    lines.extend(["## Average Metrics", "", _markdown_table(averages, ["evaluation_scope", "model_name", *METRIC_COLUMNS, "tournaments", "total_matches"]), ""])

    ranking = averages.sort_values(["evaluation_scope", "log_loss", "model_name"], kind="stable").reset_index(drop=True)
    lines.extend(["## Model Ranking By Log Loss", "", _markdown_table(ranking, ["evaluation_scope", "model_name", "log_loss", "accuracy", "calibration_error", "tournaments"]), ""])

    lines.extend(["## Benchmark Comparisons", ""])
    for baseline, label in [
        ("elo_only", "Elo-only"),
        ("fifa_ranking_only", "FIFA-only"),
    ]:
        lines.append(f"- Final model beats {label}: `{_comparison_text(averages, baseline, scope='all_matches')}`")
    lines.append(f"- Final model beats bookmaker odds on odds-covered matches: `{_comparison_text(averages, 'bookmaker_odds_only', scope='odds_covered')}`")
    lines.append(f"- Market blend beats final model on odds-covered matches: `{_comparison_text(averages, 'final_model', scope='odds_covered', candidate='market_blend')}`")
    lines.append(f"- Market blend beats bookmaker odds on odds-covered matches: `{_comparison_text(averages, 'bookmaker_odds_only', scope='odds_covered', candidate='market_blend')}`")
    lines.append("")

    coverage = _odds_coverage_summary(metrics)
    lines.extend(["## Odds Coverage", "", _markdown_table(coverage, ["world_cup_year", "odds_covered_matches", "odds_total_matches", "odds_missing_matches", "odds_coverage_rate"]), ""])

    missing = _missing_odds_matches(predictions)
    lines.extend(["## Missing Odds Matches", "", _markdown_table(missing, ["world_cup_year", "date", "home_team", "away_team"]), ""])

    final_average = averages[averages["model_name"].eq("final_model") & averages["evaluation_scope"].eq("all_matches")]
    best_calibration = averages.sort_values(["calibration_error", "model_name"], kind="stable").head(1)
    if final_average.empty:
        calibration_summary = "Final model calibration is unavailable."
    else:
        calibration_summary = f"Final model average calibration_error is {_format_float(final_average.iloc[0]['calibration_error'])}."
    if not best_calibration.empty:
        calibration_summary += f" Best average calibration_error is `{best_calibration.iloc[0]['model_name']}` at {_format_float(best_calibration.iloc[0]['calibration_error'])}."
    lines.extend(["## Calibration Summary", "", calibration_summary, ""])

    market_available = bool(metrics["model_name"].isin(["bookmaker_odds_only", "market_blend"]).any())
    lines.extend(
        [
            "## Known Limitations",
            "",
            "- This is a match-level backtest, not a full historical tournament simulation with group tables and knockout brackets.",
            "- Historical FIFA and external rating features are joined strictly as-of the match date; sparse snapshots can reduce benchmark strength.",
            "- Historical squad/player, injury, suspension, and xG features are excluded because the available project files are current snapshots, not historical as-of records.",
            f"- Bookmaker odds benchmarks are {'included where complete odds were available' if market_available else 'not included because no complete historical odds were available'}.",
            "- Knockout matches are evaluated using the stored match outcome class in the training data.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_market_blend_report(
    *,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    alpha_grid: pd.DataFrame,
    best_alpha: float,
    best_odds_method: str,
    alpha_by_year: pd.DataFrame,
    alpha_by_stage: pd.DataFrame,
    confidence_weighted: pd.DataFrame,
    significance: pd.DataFrame,
    path: str | Path,
) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    odds_metrics = metrics[metrics.get("evaluation_scope", pd.Series(index=metrics.index)).eq("odds_covered")].copy()
    averages = _average_metrics(metrics) if not metrics.empty else pd.DataFrame()
    lines = [
        "# Market Blend Report",
        "",
        "- Market odds join key: `date`, normalized `home_team`, normalized `away_team`",
        f"- Best odds conversion method: `{best_odds_method}`",
        f"- Alpha formula: `alpha * model_prob + (1 - alpha) * market_prob`",
        f"- Best alpha from odds-covered historical backtests: `{best_alpha:.1f}`",
        "",
        "## Alpha Search",
        "",
        _markdown_table(alpha_grid.sort_values("alpha", kind="stable") if not alpha_grid.empty else alpha_grid, ["alpha", *METRIC_COLUMNS, "n_matches"]),
        "",
        "## Tournament-Year Alpha Diagnostics",
        "",
        _markdown_table(alpha_by_year, ["world_cup_year", "alpha", *METRIC_COLUMNS, "n_matches"]),
        "",
        "## Stage-Specific Alpha Diagnostics",
        "",
        _markdown_table(alpha_by_stage, ["stage", "alpha", *METRIC_COLUMNS, "n_matches"]),
        "",
        "## Confidence-Weighted Blend Diagnostic",
        "",
        _markdown_table(confidence_weighted, ["blend_method", *METRIC_COLUMNS, "n_matches", "alpha_mean", "alpha_min", "alpha_max"]),
        "",
        "## Odds-Covered Metrics",
        "",
        _markdown_table(
            odds_metrics.sort_values(["world_cup_year", "log_loss", "model_name"], kind="stable"),
            ["world_cup_year", "model_name", *METRIC_COLUMNS, "n_matches", "odds_coverage_rate", "alpha"],
        ),
        "",
        "## Average Odds-Covered Metrics",
        "",
        _markdown_table(
            averages[averages.get("evaluation_scope", pd.Series(index=averages.index)).eq("odds_covered")] if not averages.empty else averages,
            ["model_name", *METRIC_COLUMNS, "tournaments", "total_matches"],
        ),
        "",
        "## Comparisons",
        "",
        f"- Final model beats bookmaker_odds_only: `{_comparison_text(averages, 'bookmaker_odds_only', scope='odds_covered') if not averages.empty else 'not available'}`",
        f"- Market blend beats final_model: `{_comparison_text(averages, 'final_model', scope='odds_covered', candidate='market_blend') if not averages.empty else 'not available'}`",
        f"- Market blend beats bookmaker_odds_only: `{_comparison_text(averages, 'bookmaker_odds_only', scope='odds_covered', candidate='market_blend') if not averages.empty else 'not available'}`",
        "",
        "## Statistical Significance Summary",
        "",
        _markdown_table(significance, ["comparison", "log_loss_delta", "ci_lower", "ci_upper", "statistically_meaningful"]),
        "",
        "## Odds Coverage",
        "",
        _markdown_table(_odds_coverage_summary(metrics), ["world_cup_year", "odds_covered_matches", "odds_total_matches", "odds_missing_matches", "odds_coverage_rate"]),
        "",
        "## Missing Odds Matches",
        "",
        _markdown_table(_missing_odds_matches(predictions), ["world_cup_year", "date", "home_team", "away_team"]),
        "",
        "## Notes",
        "",
        "- Bookmaker and blend metrics are evaluated only on matches with complete odds.",
        "- Coverage-adjusted final_model rows are included on the same odds-covered match set.",
        "- If odds coverage is incomplete, full-tournament final_model metrics and odds-covered metrics should not be compared as if they used the same rows.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_odds_conversion_benchmark_report(benchmark: pd.DataFrame, best_method: str, path: str | Path) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Odds Conversion Benchmark Report",
        "",
        "- Decimal odds are converted to implied probabilities and evaluated on odds-covered historical World Cup matches.",
        "- Lower log loss is better.",
        f"- Selected conversion method for market benchmarks: `{best_method}`",
        "",
        "## Methods Compared",
        "",
        "- `multiplicative`: raw inverse odds divided by their row sum.",
        "- `additive`: equal overround subtraction followed by clipping and normalization.",
        "- `power`: row-wise exponent chosen so adjusted probabilities sum to one.",
        "- `shin`: approximate Shin insider-trading adjustment with a row-wise solved parameter.",
        "- `favorite_longshot`: fixed power-style favourite-longshot-bias adjustment.",
        "",
        "## Results",
        "",
        _markdown_table(benchmark, ["odds_conversion_method", *METRIC_COLUMNS, "n_matches"]),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_benchmark_significance_report(*, predictions: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary, paired = _benchmark_significance(predictions)
    lines = [
        "# Benchmark Significance Report",
        "",
        "- Bootstrap confidence intervals use odds-covered World Cup matches only.",
        "- Log-loss deltas are paired by `world_cup_year` and `match_id`.",
        "- Negative delta means the first model in the comparison has lower log loss.",
        "",
        "## Log Loss Confidence Intervals",
        "",
        _markdown_table(summary, ["model_name", "log_loss", "ci_lower", "ci_upper", "n_matches"]),
        "",
        "## Paired Delta Confidence Intervals",
        "",
        _markdown_table(paired, ["comparison", "log_loss_delta", "ci_lower", "ci_upper", "statistically_meaningful", "n_matches"]),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return paired


def _odds_conversion_benchmark(fold_results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    y_parts = []
    target_parts = []
    for fold in fold_results:
        target_df = fold["target_df"]
        mask = _market_coverage_mask(target_df)
        if int(mask.sum()) == 0:
            continue
        covered = target_df[mask].copy()
        y_parts.append(covered["target_result_class"].astype(int).to_numpy())
        target_parts.append(covered)
    if not target_parts:
        return pd.DataFrame(columns=["odds_conversion_method", *METRIC_COLUMNS, "n_matches"])
    y = np.concatenate(y_parts)
    covered_all = pd.concat(target_parts, ignore_index=True, sort=False)
    for method in ODDS_CONVERSION_METHODS:
        probabilities = _market_probabilities(covered_all, odds_method=method)
        if probabilities is None:
            continue
        rows.append({"odds_conversion_method": method, **evaluate_probabilities(y, probabilities), "n_matches": len(y)})
    return pd.DataFrame(rows).sort_values(["log_loss", "odds_conversion_method"], kind="stable").reset_index(drop=True)


def _best_odds_conversion_method(benchmark: pd.DataFrame) -> str:
    if benchmark.empty or "odds_conversion_method" not in benchmark.columns:
        return "multiplicative"
    return str(benchmark.iloc[0]["odds_conversion_method"])


def _alpha_grid_metrics_by_year(fold_results: list[dict[str, Any]], *, odds_method: str) -> pd.DataFrame:
    rows = []
    for fold in fold_results:
        target_df = fold["target_df"]
        mask = _market_coverage_mask(target_df)
        if int(mask.sum()) == 0:
            continue
        covered = target_df[mask].copy()
        market_prob = _market_probabilities(covered, odds_method=odds_method)
        if market_prob is None:
            continue
        y = covered["target_result_class"].astype(int).to_numpy()
        model_prob = fold["final_prob"][mask.to_numpy()]
        for alpha in ALPHA_GRID:
            probabilities = blend_probabilities(model_prob, market_prob, alpha)
            rows.append({"world_cup_year": int(fold["year"]), "alpha": alpha, **evaluate_probabilities(y, probabilities), "n_matches": len(y)})
    if not rows:
        return pd.DataFrame(columns=["world_cup_year", "alpha", *METRIC_COLUMNS, "n_matches"])
    diagnostics = pd.DataFrame(rows)
    return diagnostics.sort_values(["world_cup_year", "log_loss", "alpha"], kind="stable").groupby("world_cup_year", as_index=False).head(1).reset_index(drop=True)


def _alpha_grid_metrics_by_stage(fold_results: list[dict[str, Any]], *, odds_method: str) -> pd.DataFrame:
    rows = []
    parts = _combined_market_parts(fold_results, odds_method=odds_method)
    if parts is None:
        return pd.DataFrame(columns=["stage", "alpha", *METRIC_COLUMNS, "n_matches"])
    y, model_prob, market_prob, stages = parts
    for stage in sorted(pd.Series(stages).dropna().unique()):
        stage_mask = np.asarray(stages) == stage
        if not bool(stage_mask.any()):
            continue
        for alpha in ALPHA_GRID:
            probabilities = blend_probabilities(model_prob[stage_mask], market_prob[stage_mask], alpha)
            rows.append({"stage": stage, "alpha": alpha, **evaluate_probabilities(y[stage_mask], probabilities), "n_matches": int(stage_mask.sum())})
    if not rows:
        return pd.DataFrame(columns=["stage", "alpha", *METRIC_COLUMNS, "n_matches"])
    diagnostics = pd.DataFrame(rows)
    return diagnostics.sort_values(["stage", "log_loss", "alpha"], kind="stable").groupby("stage", as_index=False).head(1).reset_index(drop=True)


def _confidence_weighted_blend_metrics(fold_results: list[dict[str, Any]], *, odds_method: str) -> pd.DataFrame:
    parts = _combined_market_parts(fold_results, odds_method=odds_method)
    if parts is None:
        return pd.DataFrame(columns=["blend_method", *METRIC_COLUMNS, "n_matches", "alpha_mean", "alpha_min", "alpha_max"])
    y, model_prob, market_prob, _ = parts
    model_conf = np.maximum(model_prob.max(axis=1) - (1.0 / 3.0), 0.0)
    market_conf = np.maximum(market_prob.max(axis=1) - (1.0 / 3.0), 0.0)
    denom = model_conf + market_conf
    alpha = np.where(denom > 0, model_conf / denom, 0.5)
    probabilities = _normalize_probabilities((alpha[:, None] * model_prob) + ((1.0 - alpha[:, None]) * market_prob))
    return pd.DataFrame(
        [
            {
                "blend_method": "confidence_weighted",
                **evaluate_probabilities(y, probabilities),
                "n_matches": len(y),
                "alpha_mean": float(alpha.mean()),
                "alpha_min": float(alpha.min()),
                "alpha_max": float(alpha.max()),
            }
        ]
    )


def _combined_market_parts(fold_results: list[dict[str, Any]], *, odds_method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    y_parts: list[np.ndarray] = []
    model_parts: list[np.ndarray] = []
    market_parts: list[np.ndarray] = []
    stage_parts: list[np.ndarray] = []
    for fold in fold_results:
        target_df = fold["target_df"]
        mask = _market_coverage_mask(target_df)
        if int(mask.sum()) == 0:
            continue
        covered = target_df[mask].copy()
        market_prob = _market_probabilities(covered, odds_method=odds_method)
        if market_prob is None:
            continue
        y_parts.append(covered["target_result_class"].astype(int).to_numpy())
        model_parts.append(fold["final_prob"][mask.to_numpy()])
        market_parts.append(market_prob)
        stage_parts.append(covered.get("_backtest_stage", pd.Series("unknown", index=covered.index)).astype(str).to_numpy())
    if not y_parts:
        return None
    return np.concatenate(y_parts), np.vstack(model_parts), np.vstack(market_parts), np.concatenate(stage_parts)


def _benchmark_significance(predictions: pd.DataFrame, *, n_bootstrap: int = 2000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_models = ["final_model", "bookmaker_odds_only", "market_blend"]
    if predictions.empty:
        return pd.DataFrame(columns=["model_name", "log_loss", "ci_lower", "ci_upper", "n_matches"]), pd.DataFrame(
            columns=["comparison", "log_loss_delta", "ci_lower", "ci_upper", "statistically_meaningful", "n_matches"]
        )
    scoped = predictions[predictions["evaluation_scope"].eq("odds_covered") & predictions["model_name"].isin(required_models)].copy()
    if scoped.empty:
        return pd.DataFrame(columns=["model_name", "log_loss", "ci_lower", "ci_upper", "n_matches"]), pd.DataFrame(
            columns=["comparison", "log_loss_delta", "ci_lower", "ci_upper", "statistically_meaningful", "n_matches"]
        )
    scoped["_row_key"] = scoped["world_cup_year"].astype(str) + "::" + scoped["match_id"].astype(str)
    scoped["row_log_loss"] = _row_log_loss(scoped)
    rng = np.random.default_rng(seed)
    summary_rows = []
    for model_name in required_models:
        values = scoped[scoped["model_name"].eq(model_name)]["row_log_loss"].to_numpy(dtype=float)
        if len(values) == 0:
            continue
        boot = _bootstrap_means(values, rng, n_bootstrap)
        summary_rows.append(
            {
                "model_name": model_name,
                "log_loss": float(values.mean()),
                "ci_lower": float(np.quantile(boot, 0.025)),
                "ci_upper": float(np.quantile(boot, 0.975)),
                "n_matches": len(values),
            }
        )

    paired_rows = []
    for candidate, baseline in [("market_blend", "bookmaker_odds_only"), ("market_blend", "final_model"), ("final_model", "bookmaker_odds_only")]:
        left = scoped[scoped["model_name"].eq(candidate)][["_row_key", "row_log_loss"]].rename(columns={"row_log_loss": "candidate"})
        right = scoped[scoped["model_name"].eq(baseline)][["_row_key", "row_log_loss"]].rename(columns={"row_log_loss": "baseline"})
        paired = left.merge(right, on="_row_key", how="inner")
        if paired.empty:
            continue
        delta = (paired["candidate"] - paired["baseline"]).to_numpy(dtype=float)
        boot = _bootstrap_means(delta, rng, n_bootstrap)
        ci_lower = float(np.quantile(boot, 0.025))
        ci_upper = float(np.quantile(boot, 0.975))
        paired_rows.append(
            {
                "comparison": f"{candidate}_minus_{baseline}",
                "log_loss_delta": float(delta.mean()),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "statistically_meaningful": bool(ci_upper < 0 or ci_lower > 0),
                "n_matches": len(delta),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def _bootstrap_means(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.array([np.nan])
    indexes = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    return values[indexes].mean(axis=1)


def _row_log_loss(predictions: pd.DataFrame) -> np.ndarray:
    probabilities = predictions[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    actual = predictions["actual_class"].astype(int).to_numpy()
    selected = np.clip(probabilities[np.arange(len(probabilities)), actual], 1e-12, 1.0)
    return -np.log(selected)


def _infer_worldcup_stage_labels(target_df: pd.DataFrame) -> list[str]:
    if "stage" in target_df.columns and target_df["stage"].notna().any():
        return target_df["stage"].fillna("unknown").astype(str).tolist()
    labels = []
    for idx in range(len(target_df)):
        if idx < 48:
            labels.append("group_stage")
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


def _prepare_backtest_frame(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    if "target_result_class" not in output.columns and {"home_score", "away_score"}.issubset(output.columns):
        output = _add_flags_and_targets(output)
    if "match_id" not in output.columns:
        output["match_id"] = [f"backtest_{i}" for i in range(len(output))]
    output["tournament"] = output.get("tournament", pd.Series("Other", index=output.index)).fillna("Other")
    return output.dropna(subset=["date", "home_team", "away_team"]).reset_index(drop=True)


def _target_tournament_mask(df: pd.DataFrame, year: int) -> pd.Series:
    dates = pd.to_datetime(df["date"], errors="coerce")
    tournament = df["tournament"].astype("string").str.strip().str.casefold()
    return dates.dt.year.eq(year) & tournament.eq("fifa world cup")


def _safe_feature_columns(feature_set: str, available_columns: pd.Index | list[str]) -> list[str]:
    requested = columns_for_feature_set(feature_set, available_columns)
    filtered = [column for column in requested if column not in CURRENT_ONLY_EXTERNAL_COLUMNS]
    dropped = sorted(set(requested) - set(filtered))
    if dropped:
        logger.warning("Dropping current-only external columns from historical backtest: %s", dropped)
    return filtered


def _fit_predict_feature_model(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    feature_columns: list[str],
    model_name: str,
    random_seed: int,
    modeling_config: dict[str, Any] | None = None,
    fallback_name: str | None = None,
) -> tuple[np.ndarray, str, list[str]]:
    usable_columns = [column for column in feature_columns if column in train_df.columns and train_df[column].notna().any()]
    train_target = train_df["target_result_class"].astype(int)
    if not usable_columns or train_target.nunique() < 2:
        return _majority_probabilities(train_df, len(target_df)), fallback_name or "majority_class_baseline", []

    estimator, resolved_name = _make_estimator(model_name, random_seed=random_seed, modeling_config=modeling_config)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("estimator", estimator),
        ]
    )
    model.fit(_feature_matrix(train_df, usable_columns), train_target)
    probabilities = _predict_proba_3(model, _feature_matrix(target_df, usable_columns))
    return probabilities, (fallback_name or resolved_name), usable_columns


def _benchmark_feature_sets() -> dict[str, list[str]]:
    return {
        "elo_only": BASIC_ELO_FEATURES,
        "fifa_ranking_only": FIFA_RANKING_MODEL_COLUMNS,
        "rolling_form_only": ROLLING_FEATURE_COLUMNS,
    }


def _majority_probabilities(train_df: pd.DataFrame, n_rows: int) -> np.ndarray:
    counts = train_df["target_result_class"].dropna().astype(int).value_counts(normalize=True)
    probabilities = np.array([float(counts.get(0, 0.0)), float(counts.get(1, 0.0)), float(counts.get(2, 0.0))], dtype=float)
    if probabilities.sum() <= 0:
        probabilities[:] = 1.0 / 3.0
    probabilities = probabilities / probabilities.sum()
    return np.tile(probabilities, (n_rows, 1))


def _market_probabilities(target_df: pd.DataFrame, odds_method: str | None = None) -> np.ndarray | None:
    if odds_method is not None and {"home_odds", "draw_odds", "away_odds"}.issubset(target_df.columns):
        converted = decimal_odds_to_probabilities(target_df[["home_odds", "draw_odds", "away_odds"]], method=odds_method)
        market = converted[["market_away_prob", "market_draw_prob", "market_home_prob"]].apply(pd.to_numeric, errors="coerce")
        if market.isna().to_numpy().any():
            return None
        return _normalize_probabilities(market.to_numpy(dtype=float))
    if not set(MARKET_PROBABILITY_COLUMNS).issubset(target_df.columns):
        return None
    market = target_df[MARKET_PROBABILITY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if market.isna().to_numpy().any():
        return None
    return _normalize_probabilities(market.to_numpy(dtype=float))


def blend_probabilities(model_probabilities: np.ndarray, market_probabilities: np.ndarray, alpha: float) -> np.ndarray:
    return _normalize_probabilities((float(alpha) * model_probabilities) + ((1.0 - float(alpha)) * market_probabilities))


def _market_coverage_mask(target_df: pd.DataFrame) -> pd.Series:
    if not set(MARKET_PROBABILITY_COLUMNS).issubset(target_df.columns):
        return pd.Series(False, index=target_df.index)
    market = target_df[MARKET_PROBABILITY_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return market.notna().all(axis=1)


def _alpha_grid_metrics(fold_results: list[dict[str, Any]], *, odds_method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y_parts: list[np.ndarray] = []
    model_parts: list[np.ndarray] = []
    market_parts: list[np.ndarray] = []
    for fold in fold_results:
        target_df = fold["target_df"]
        mask = _market_coverage_mask(target_df)
        if int(mask.sum()) == 0:
            continue
        covered = target_df[mask].copy()
        market_prob = _market_probabilities(covered, odds_method=odds_method)
        if market_prob is None:
            continue
        y_parts.append(covered["target_result_class"].astype(int).to_numpy())
        model_parts.append(fold["final_prob"][mask.to_numpy()])
        market_parts.append(market_prob)
    if not y_parts:
        return pd.DataFrame(columns=["alpha", *METRIC_COLUMNS, "n_matches"])
    y = np.concatenate(y_parts)
    model_prob = np.vstack(model_parts)
    market_prob = np.vstack(market_parts)
    for alpha in ALPHA_GRID:
        probabilities = blend_probabilities(model_prob, market_prob, alpha)
        rows.append({"alpha": alpha, **evaluate_probabilities(y, probabilities), "n_matches": len(y)})
    return pd.DataFrame(rows).sort_values(["log_loss", "alpha"], kind="stable").reset_index(drop=True)


def _best_alpha(alpha_grid: pd.DataFrame) -> float:
    if alpha_grid.empty or "alpha" not in alpha_grid.columns:
        return 0.5
    return float(alpha_grid.iloc[0]["alpha"])


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return clipped / row_sums


def _prediction_frame(
    target_df: pd.DataFrame,
    *,
    probabilities: np.ndarray,
    world_cup_year: int,
    model_name: str,
    feature_set: str,
    benchmark: str,
    tournament_start: pd.Timestamp,
    train_df: pd.DataFrame,
    feature_count: int,
    estimator_name: str | None = None,
    evaluation_scope: str = "all_matches",
    alpha: float | None = None,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "world_cup_year": world_cup_year,
            "match_id": target_df["match_id"].values,
            "date": target_df["date"].values,
            "home_team": target_df["home_team"].values,
            "away_team": target_df["away_team"].values,
            "actual_class": target_df["target_result_class"].astype(int).values,
            "p_home_loss": probabilities[:, 0],
            "p_draw": probabilities[:, 1],
            "p_home_win": probabilities[:, 2],
            "predicted_class": probabilities.argmax(axis=1),
            "model_name": model_name,
            "feature_set": feature_set,
            "benchmark": benchmark,
            "estimator_name": estimator_name or model_name,
            "evaluation_scope": evaluation_scope,
            "alpha": alpha,
            "tournament_start": tournament_start.date().isoformat(),
            "n_train": len(train_df),
            "train_max_date": _max_date_text(train_df),
            "feature_count": feature_count,
        }
    )
    if "home_score" in target_df.columns:
        output["home_score"] = target_df["home_score"].values
    if "away_score" in target_df.columns:
        output["away_score"] = target_df["away_score"].values
    if "_backtest_stage" in target_df.columns:
        output["stage"] = target_df["_backtest_stage"].values
    return output


def _metric_row(
    predictions: pd.DataFrame,
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    tournament_start: pd.Timestamp,
    feature_count: int,
    evaluation_scope: str,
    odds_covered_matches: int | None = None,
    odds_total_matches: int | None = None,
    alpha: float | None = None,
) -> dict[str, Any]:
    probabilities = predictions[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    metrics = evaluate_probabilities(predictions["actual_class"], probabilities)
    return {
        "world_cup_year": int(predictions["world_cup_year"].iloc[0]),
        "model_name": str(predictions["model_name"].iloc[0]),
        "feature_set": str(predictions["feature_set"].iloc[0]),
        "benchmark": str(predictions["benchmark"].iloc[0]),
        "evaluation_scope": evaluation_scope,
        **metrics,
        "n_matches": len(target_df),
        "n_train": len(train_df),
        "train_max_date": _max_date_text(train_df),
        "tournament_start": tournament_start.date().isoformat(),
        "feature_count": feature_count,
        "odds_covered_matches": odds_covered_matches,
        "odds_total_matches": odds_total_matches,
        "odds_coverage_rate": np.nan if odds_covered_matches is None or not odds_total_matches else odds_covered_matches / odds_total_matches,
        "alpha": alpha,
    }


def _average_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["evaluation_scope", "model_name"], as_index=False)
    averages = grouped[METRIC_COLUMNS].mean()
    counts = grouped.agg(tournaments=("world_cup_year", "nunique"), total_matches=("n_matches", "sum"))
    return averages.merge(counts, on=["evaluation_scope", "model_name"], how="left").sort_values(["evaluation_scope", "log_loss", "model_name"], kind="stable")


def _comparison_text(averages: pd.DataFrame, baseline: str, *, scope: str, candidate: str = "final_model") -> str:
    left = averages[averages["model_name"].eq(candidate) & averages["evaluation_scope"].eq(scope)]
    right = averages[averages["model_name"].eq(baseline) & averages["evaluation_scope"].eq(scope)]
    if left.empty or right.empty:
        return "not available"
    delta = float(left.iloc[0]["log_loss"] - right.iloc[0]["log_loss"])
    return f"{delta < 0} (log_loss_delta={delta:.6f})"


def _odds_coverage_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=["world_cup_year", "odds_covered_matches", "odds_total_matches", "odds_missing_matches", "odds_coverage_rate"])
    all_rows = metrics[metrics["model_name"].eq("final_model") & metrics["evaluation_scope"].eq("all_matches")]
    covered_rows = metrics[metrics["model_name"].eq("bookmaker_odds_only") & metrics["evaluation_scope"].eq("odds_covered")]
    records: list[dict[str, Any]] = []
    for year in sorted(all_rows["world_cup_year"].dropna().astype(int).unique()):
        total = int(all_rows[all_rows["world_cup_year"].eq(year)].iloc[0]["n_matches"])
        covered = covered_rows[covered_rows["world_cup_year"].eq(year)]
        covered_count = 0 if covered.empty else int(covered.iloc[0]["n_matches"])
        records.append(
            {
                "world_cup_year": year,
                "odds_covered_matches": covered_count,
                "odds_total_matches": total,
                "odds_missing_matches": total - covered_count,
                "odds_coverage_rate": 0.0 if total == 0 else covered_count / total,
            }
        )
    return pd.DataFrame(records)


def _missing_odds_matches(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["world_cup_year", "date", "home_team", "away_team"])
    final_rows = predictions[predictions["model_name"].eq("final_model") & predictions["evaluation_scope"].eq("all_matches")].copy()
    covered_ids = set(predictions[predictions["model_name"].eq("bookmaker_odds_only")]["match_id"].astype(str))
    missing = final_rows[~final_rows["match_id"].astype(str).isin(covered_ids)].copy()
    if missing.empty:
        return pd.DataFrame(columns=["world_cup_year", "date", "home_team", "away_team"])
    missing["date"] = pd.to_datetime(missing["date"], errors="coerce").dt.date.astype(str)
    return missing[["world_cup_year", "date", "home_team", "away_team"]].reset_index(drop=True)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in df.columns]
    if not available or df.empty:
        return "_No rows._"
    rows = ["| " + " | ".join(available) + " |", "| " + " | ".join(["---"] * len(available)) + " |"]
    for record in df[available].to_dict("records"):
        values = [_format_markdown_value(record.get(column)) for column in available]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return _format_float(value)
    return str(value)


def _format_float(value: Any) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.6f}"


def _max_date_text(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    value = pd.to_datetime(df["date"], errors="coerce").max()
    if pd.isna(value):
        return ""
    return value.date().isoformat()


def _path_or_default(config: dict[str, Any], key: str, default: str) -> Path:
    try:
        return config_path(config, key)
    except KeyError:
        return Path(default)
