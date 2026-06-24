from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtesting.worldcup_backtest import run_worldcup_backtests
from src.backtesting.stage5_benchmark import run_stage5_benchmark
from src.backtesting.stage5_ablation_reconciliation import run_stage5_ablation_reconciliation
from src.backtesting.stage5_calibrated_market import run_stage5_calibrated_market
from src.backtesting.stage5_draw_calibration import run_stage5_draw_calibration
from src.backtesting.stage5_failure_diagnosis import run_stage5_failure_diagnosis
from src.backtesting.stage5_signal_discovery import run_stage5_signal_discovery
from src.backtesting.worldcup_tournament_simulation import run_worldcup_tournament_simulation_backtest
from src.config import config_path, ensure_configured_directories, load_config
from src.config import resolve_project_path
from src.data_sources.api_football_client import ApiFootballClient
from src.data_sources.config import (
    API_FOOTBALL_INJURIES_LIVE_PATH,
    API_FOOTBALL_ODDS_2010_PROBE_PATH,
    API_FOOTBALL_ODDS_2026_PATH,
    SPORTMONKS_INJURIES_LIVE_PATH,
    SPORTMONKS_ODDS_2026_PATH,
    THE_ODDS_API_ODDS_2026_PATH,
)
from src.data_sources.coverage import check_2010_odds_coverage, run_api_coverage_check, validate_staged_api_data
from src.data_sources.injury_transform import write_injury_staging
from src.data_sources.merge_market_odds import merge_staged_market_odds
from src.data_sources.market_odds_workflow import (
    discover_odds_competitions,
    generate_missing_odds_template,
    refresh_market_odds,
    validate_manual_market_odds,
)
from src.data_sources.odds_transform import write_market_staging
from src.data_sources.sportmonks_client import SportmonksClient
from src.data_sources.the_odds_api_client import TheOddsApiClient
from src.dedupe import deduplicate_matches
from src.experiments.ablation_runner import run_ablation
from src.features.external_feature_joiner import build_advanced_features
from src.features.build_training_dataset import build_training_dataset
from src.features.internal_historical_elo import build_internal_historical_elo_artifacts, write_internal_elo_reconstruction_report
from src.features.null_rate_report import generate_feature_null_rate_report
from src.io_utils import read_dataframe, write_dataframe
from src.logging_utils import setup_logging
from src.models.calibration import calibrate_model
from src.models.ensemble import build_ensemble_predictions
from src.models.prediction_outputs import split_prediction_outputs
from src.models.poisson_model import train_poisson_model
from src.models.train_match_model import train_match_model
from src.normalize import add_match_outcome_columns, canonicalize_match_teams, classify_competition_type
from src.reports.final_model_recommendation import write_final_model_recommendation
from src.simulation.worldcup_simulator import simulate_worldcup
from src.sources.football_data_api import (
    MissingFootballDataToken,
    fetch_matches,
    fetch_recent_matches,
    parse_matches_response,
    save_clean_matches,
    save_raw_response,
)
from src.sources.external_rating_acquisition import (
    acquire_external_ratings,
    acquire_fifa_rankings,
    acquire_world_football_elo,
    generate_external_rating_activation_report,
    write_current_external_rating_acquisition_report,
)
from src.sources.fifa_rankings import load_fifa_rankings
from src.sources.historical_rating_snapshots import (
    import_fifa_ranking_snapshots,
    import_historical_rating_snapshots,
    import_world_football_elo_snapshots,
)
from src.sources.historical_results import clean_historical_results, download_historical_results
from src.sources.external_data_validation import prepare_external_templates, validate_external_data
from src.sources.world_football_elo import load_world_football_elo
from src.sources.worldcup_json import download_worldcup_2026_json, normalize_worldcup_matches, parse_worldcup_json
from src.validation import CORE_COLUMNS, add_missing_columns
from dashboard.live_api import generate_live_predictions as generate_dashboard_live_predictions
from dashboard.live_api import refresh_live_data as refresh_dashboard_live_data


logger = logging.getLogger(__name__)


def collect_historical(args: argparse.Namespace, config: dict[str, Any]) -> pd.DataFrame:
    download_historical_results(config, force=args.force)
    return clean_historical_results(config=config)


def collect_recent(args: argparse.Namespace, config: dict[str, Any]) -> pd.DataFrame:
    try:
        if args.date_from and args.date_to:
            payload = fetch_matches(args.date_from, args.date_to, only_finished=args.only_finished, config=config)
        else:
            payload = fetch_recent_matches(
                days_back=args.days_back,
                days_forward=args.days_forward,
                only_finished=args.only_finished,
                config=config,
            )
    except MissingFootballDataToken:
        logger.warning("FOOTBALL_DATA_TOKEN is not set; writing an empty recent matches file")
        payload = {"matches": []}
    save_raw_response(payload, config=config)
    df = parse_matches_response(payload, mapping_path=config_path(config, "team_name_mapping"))
    save_clean_matches(df, config=config)
    return df


def collect_worldcup(args: argparse.Namespace, config: dict[str, Any]) -> pd.DataFrame:
    raw_path = download_worldcup_2026_json(config, force=args.force)
    df = parse_worldcup_json(raw_path)
    output_path = config_path(config, "worldcup_2026_fixtures_clean")
    write_dataframe(df, output_path)
    write_dataframe(df, output_path.with_suffix(".csv"))
    return df


def build_clean_matches(config: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for path_key, name in [
        ("historical_matches_clean", "historical"),
        ("recent_matches_clean", "recent"),
        ("worldcup_2026_fixtures_clean", "worldcup"),
    ]:
        path = config_path(config, path_key)
        if not path.exists():
            logger.warning("Skipping missing %s processed file: %s", name, path)
            continue
        frame = read_dataframe(path)
        if frame.empty:
            logger.info("Skipping empty %s processed file: %s", name, path)
            continue
        frames.append(frame)

    if not frames:
        raise ValueError("No processed match files are available. Run collection commands first.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date", "home_team", "away_team"])
    combined = canonicalize_match_teams(combined, mapping_path=config_path(config, "team_name_mapping"))
    combined["tournament"] = combined.get("tournament", pd.Series("Other", index=combined.index)).fillna("Other")
    combined["competition_type"] = combined.get("competition_type", pd.Series(index=combined.index)).fillna(
        combined["tournament"].map(classify_competition_type)
    )
    combined = add_match_outcome_columns(combined)
    combined = add_missing_columns(combined, CORE_COLUMNS)
    cleaned = deduplicate_matches(combined)
    output_path = config_path(config, "all_matches_clean")
    write_dataframe(cleaned, output_path)
    write_dataframe(cleaned, output_path.with_suffix(".csv"))
    missing_scores = cleaned["home_score"].isna().sum()
    logger.info("Merged clean dataset contains %s rows; %s rows have missing scores", len(cleaned), missing_scores)
    return cleaned


def run_collect_all(args: argparse.Namespace, config: dict[str, Any]) -> None:
    collect_historical(args, config)
    try:
        collect_recent(args, config)
    except Exception as exc:
        logger.warning("Recent match collection failed gracefully: %s", exc)
    collect_worldcup(args, config)
    build_clean_matches(config)
    build_training_dataset(start_date=args.start_date, config=config)


def run_update(args: argparse.Namespace, config: dict[str, Any]) -> None:
    collect_recent(args, config)
    collect_worldcup(args, config)
    build_clean_matches(config)
    build_training_dataset(start_date=args.start_date, config=config)


def run_full_sota_pipeline(args: argparse.Namespace, config: dict[str, Any]) -> None:
    build_advanced_features(config=config)
    run_ablation(config=config)
    write_internal_elo_reconstruction_report(config=config)
    train_match_model(model_name=args.model, feature_set=args.feature_set, config=config)
    train_poisson_model(config=config)
    calibrate_model(config=config, model_name=args.model, feature_set=args.feature_set)
    build_ensemble_predictions(config=config)
    split_prediction_outputs(config=config)
    generate_feature_null_rate_report(config=config)
    generate_external_rating_activation_report(config=config)
    write_current_external_rating_acquisition_report(config=config)
    simulate_worldcup(n_sims=args.n_sims, config=config)
    split_prediction_outputs(config=config)
    write_final_model_recommendation(config=config, selected_model=args.model, selected_feature_set=args.feature_set)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="World Cup predictor data pipeline")
    parser.add_argument("--config", default=None, help="Path to sources.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    historical = subparsers.add_parser("collect-historical", help="Download and clean historical results")
    historical.add_argument("--force", action="store_true", help="Redownload existing raw files")

    recent = subparsers.add_parser("collect-recent", help="Collect football-data.org matches")
    _add_recent_args(recent)

    worldcup = subparsers.add_parser("collect-worldcup", help="Collect OpenFootball World Cup 2026 JSON")
    worldcup.add_argument("--force", action="store_true", help="Redownload existing raw JSON")

    subparsers.add_parser("build-clean", help="Merge and deduplicate clean source datasets")

    features = subparsers.add_parser("build-features", help="Build final training and prediction datasets")
    features.add_argument("--start-date", default="2010-01-01")

    collect_all = subparsers.add_parser("collect-all", help="Run all collectors and rebuild datasets")
    collect_all.add_argument("--force", action="store_true", help="Redownload existing raw files")
    collect_all.add_argument("--start-date", default="2010-01-01")
    _add_recent_args(collect_all)

    update = subparsers.add_parser("update", help="Collect recent/live sources and rebuild datasets")
    update.add_argument("--force", action="store_true", help="Redownload existing raw World Cup JSON")
    update.add_argument("--start-date", default="2010-01-01")
    _add_recent_args(update)

    subparsers.add_parser("collect-rankings", help="Validate/load local FIFA rankings from data/external/fifa_rankings.csv")
    subparsers.add_parser("collect-elo", help="Validate/load local World Football Elo from data/external/world_football_elo.csv")
    subparsers.add_parser("prepare-external-templates", help="Create template CSVs for optional external data")
    prepare_fifa = subparsers.add_parser("prepare-fifa-rankings", help="Prepare FIFA rankings from a raw CSV/XLSX/JSON file or configured source_url")
    prepare_fifa.add_argument("--input", default=None, help="Optional raw rankings file to normalize")
    prepare_elo = subparsers.add_parser("prepare-world-football-elo", help="Prepare World Football Elo from a raw CSV/XLSX/JSON file or configured source_url")
    prepare_elo.add_argument("--input", default=None, help="Optional raw Elo file to normalize")
    acquire_fifa = subparsers.add_parser("acquire-fifa-rankings", help="Acquire, normalize, validate, and save real FIFA rankings")
    _add_acquisition_args(acquire_fifa)
    acquire_elo = subparsers.add_parser("acquire-world-football-elo", help="Acquire, normalize, validate, and save real World Football Elo")
    _add_acquisition_args(acquire_elo)
    acquire_all = subparsers.add_parser("acquire-external-ratings", help="Acquire both FIFA rankings and World Football Elo")
    acquire_all.add_argument("--force", action="store_true", help="Allow replacing existing valid external files with newly validated data")
    acquire_all.add_argument("--skip-post-checks", action="store_true", help="Skip validation, advanced feature rebuild, null report, and ablation after acquisition")
    import_fifa_snapshots = subparsers.add_parser("import-fifa-ranking-snapshots", help="Import multiple dated FIFA ranking snapshots")
    import_fifa_snapshots.add_argument("--input-dir", default="data/external/fifa_rankings_snapshots")
    import_fifa_snapshots.add_argument("--output", default="data/external/fifa_rankings.csv")
    _add_snapshot_import_args(import_fifa_snapshots)
    import_elo_snapshots = subparsers.add_parser("import-world-football-elo-snapshots", help="Import multiple dated World Football Elo snapshots")
    import_elo_snapshots.add_argument("--input-dir", default="data/external/world_football_elo_snapshots")
    import_elo_snapshots.add_argument("--output", default="data/external/world_football_elo.csv")
    _add_snapshot_import_args(import_elo_snapshots)
    import_all_snapshots = subparsers.add_parser("import-historical-rating-snapshots", help="Import historical FIFA and Elo snapshot folders")
    import_all_snapshots.add_argument("--input-dir", default=None, help="Optional parent directory containing both snapshot subfolders")
    import_all_snapshots.add_argument("--fifa-input-dir", default="data/external/fifa_rankings_snapshots")
    import_all_snapshots.add_argument("--elo-input-dir", default="data/external/world_football_elo_snapshots")
    import_all_snapshots.add_argument("--output", default=None, help="Optional parent directory for both combined output CSV files")
    import_all_snapshots.add_argument("--fifa-output", default="data/external/fifa_rankings.csv")
    import_all_snapshots.add_argument("--elo-output", default="data/external/world_football_elo.csv")
    _add_snapshot_import_args(import_all_snapshots, include_rebuild=True)
    subparsers.add_parser("validate-external-data", help="Validate optional external ranking/Elo files")
    subparsers.add_parser("feature-null-report", help="Write advanced feature null-rate reports")
    subparsers.add_parser("split-prediction-outputs", help="Split prediction files into actual, group-stage, placeholder, and dynamic outputs")

    subparsers.add_parser("build-advanced-features", help="Build SOTA-style external and dynamic feature files")
    subparsers.add_parser("build-internal-historical-elo", help="Reconstruct internal historical Elo ratings and write coverage report")
    subparsers.add_parser("api-coverage-check", help="Probe configured paid APIs and write API coverage report")
    api_football_odds = subparsers.add_parser("fetch-api-football-odds", help="Fetch API-Football World Cup market odds into staging")
    api_football_odds.add_argument("--season", type=int, default=2026)
    api_football_odds.add_argument("--competition", default="World Cup")
    api_football_injuries = subparsers.add_parser("fetch-api-football-injuries", help="Fetch API-Football injury data into scenario staging")
    api_football_injuries.add_argument("--season", type=int, default=2026)
    api_football_injuries.add_argument("--competition", default="World Cup")
    subparsers.add_parser("fetch-the-odds-api-odds", help="Fetch The Odds API World Cup market odds into staging when available")
    subparsers.add_parser("fetch-sportmonks-football-data", help="Fetch Sportmonks World Cup odds and injuries into staging when available")
    subparsers.add_parser("validate-staged-api-data", help="Validate staged API odds and injury files")
    subparsers.add_parser("discover-odds-competitions", help="Discover provider competition keys/league IDs for World Cup odds")
    refresh_market = subparsers.add_parser("refresh-market-odds", help="Fetch, validate, merge, and report market odds from configured providers")
    refresh_market.add_argument("--provider", default="all", choices=["all", "the_odds_api", "api_football", "sportmonks"])
    refresh_market.add_argument("--season", type=int, default=2026)
    refresh_market.add_argument("--prefer-api", action="store_true")
    refresh_market.add_argument("--dry-run", action="store_true")
    refresh_market.add_argument("--skip-merge", action="store_true")
    refresh_market.add_argument("--rerun-reports", action=argparse.BooleanOptionalAction, default=True)
    refresh_market.add_argument("--rerun-live-predictions", action=argparse.BooleanOptionalAction, default=True)
    subparsers.add_parser("generate-missing-odds-template", help="Create a manual odds template for active fixtures missing market odds")
    validate_manual = subparsers.add_parser("validate-manual-market-odds", help="Validate a manually filled odds template")
    validate_manual.add_argument("--input", required=True)
    merge_api_odds = subparsers.add_parser("merge-staged-market-odds", help="Validate and merge staged market odds into production odds file")
    merge_api_odds.add_argument("--input", required=True, help="Staged market odds CSV to merge")
    merge_api_odds.add_argument("--prefer-api", action="store_true", help="Replace existing curated duplicate rows with staged API rows")
    subparsers.add_parser("check-2010-odds-coverage", help="Check market_odds.csv coverage against all 64 2010 World Cup matches")
    refresh_live = subparsers.add_parser("refresh-live-worldcup-data", help="Fetch live World Cup data into data/live and regenerate live predictions")
    refresh_live.add_argument("--fetch-lineups", action="store_true", help="Fetch lineups too; may use additional API quota")
    subparsers.add_parser("generate-live-predictions", help="Generate live/scenario predictions from local live CSVs")

    train = subparsers.add_parser("train-model", help="Train a match WDL model")
    train.add_argument("--model", default="catboost", choices=["catboost", "xgboost", "hist_gradient_boosting", "logistic"])
    train.add_argument("--feature-set", default="full_football_only")

    backtest = subparsers.add_parser("backtest-world-cups", help="Backtest match predictions on historical World Cups")
    backtest.add_argument("--model", default="catboost", choices=["catboost", "xgboost", "hist_gradient_boosting", "logistic"])
    backtest.add_argument("--feature-set", default="core_football_only")

    tournament_backtest = subparsers.add_parser("backtest-worldcup-tournaments", help="Backtest full historical World Cup tournament simulations")
    tournament_backtest.add_argument("--model", default="catboost", choices=["catboost", "xgboost", "hist_gradient_boosting", "logistic"])
    tournament_backtest.add_argument("--feature-set", default="core_football_only")
    tournament_backtest.add_argument("--n-sims", type=int, default=1000)

    stage5 = subparsers.add_parser("stage5-benchmark", help="Run Stage 5 historical benchmark, calibration, and significance suite")
    stage5.add_argument("--years", default="2014,2018,2022", help="Comma-separated World Cup years to evaluate")
    stage5.add_argument("--n-bootstrap", type=int, default=1000)
    stage5.add_argument("--include-market", action="store_true")
    stage5.add_argument("--include-poisson", action="store_true")
    stage5.add_argument("--include-dixon-coles", action="store_true")
    stage5.add_argument("--output-dir", default="data/backtests")
    stage5.add_argument("--model", default="catboost", choices=["catboost", "xgboost", "hist_gradient_boosting", "logistic"])
    stage5.add_argument("--feature-set", default="core_football_only")

    stage5_diagnose = subparsers.add_parser("stage5-diagnose-failures", help="Diagnose why Stage 5 benchmark evidence did not pass")
    stage5_diagnose.add_argument("--output-dir", default="data/backtests")

    stage5_reconcile = subparsers.add_parser("stage5-reconcile-ablation", help="Explain why ablation validation and Stage 5 benchmark metrics differ")
    stage5_reconcile.add_argument("--output-dir", default="data/backtests")

    stage5_calibrated = subparsers.add_parser("stage5-calibrated-market", help="Run time-safe calibration, tuned market blend, and model+market stacker benchmarks")
    stage5_calibrated.add_argument("--years", default="2014,2018,2022", help="Comma-separated World Cup years to evaluate")
    stage5_calibrated.add_argument("--n-bootstrap", type=int, default=1000)
    stage5_calibrated.add_argument("--alpha-grid-step", type=float, default=0.05)
    stage5_calibrated.add_argument("--include-stacker", action="store_true")
    stage5_calibrated.add_argument("--include-draw-adjustment", action="store_true")
    stage5_calibrated.add_argument("--output-dir", default="data/backtests")

    stage5_signal = subparsers.add_parser("stage5-signal-discovery", help="Discover where the official model adds or loses signal against bookmaker odds")
    stage5_signal.add_argument("--output-dir", default="data/backtests")
    stage5_signal.add_argument("--n-bootstrap", type=int, default=1000)

    stage5_draw = subparsers.add_parser("stage5-draw-calibration", help="Run time-safe draw and confidence calibration diagnostics")
    stage5_draw.add_argument("--years", default="2014,2018,2022", help="Comma-separated World Cup years to evaluate")
    stage5_draw.add_argument("--n-bootstrap", type=int, default=1000)
    stage5_draw.add_argument("--output-dir", default="data/backtests")

    subparsers.add_parser("train-poisson", help="Train Poisson scoreline model and create scoreline predictions")
    subparsers.add_parser("run-ablation", help="Run time-aware feature ablation")
    calibrate = subparsers.add_parser("calibrate-model", help="Fit probability calibration and save calibrated predictions")
    calibrate.add_argument("--model", default="catboost", choices=["catboost", "xgboost", "hist_gradient_boosting", "logistic"])
    calibrate.add_argument("--feature-set", default="full_football_only")
    subparsers.add_parser("build-ensemble", help="Build default football-only ensemble predictions")

    sim = subparsers.add_parser("simulate-worldcup", help="Run Monte Carlo World Cup simulation")
    sim.add_argument("--n-sims", type=int, default=100000)

    full = subparsers.add_parser("full-sota-pipeline", help="Run advanced features, ablation, training, calibration, ensemble, and simulation")
    full.add_argument("--model", default="catboost", choices=["catboost", "xgboost", "hist_gradient_boosting", "logistic"])
    full.add_argument("--feature-set", default="full_football_only")
    full.add_argument("--n-sims", type=int, default=100000)
    return parser


def _add_recent_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--days-back", type=int, default=14)
    parser.add_argument("--days-forward", type=int, default=2)
    parser.add_argument("--only-finished", action="store_true")


def _add_acquisition_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", default=None, help="Optional local CSV/XLSX/JSON/HTML file to acquire")
    parser.add_argument("--url", default=None, help="Optional URL override for this acquisition run")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing valid external file with newly validated data")
    parser.add_argument("--skip-post-checks", action="store_true", help="Skip validation, advanced feature rebuild, null report, and ablation after acquisition")


def _add_snapshot_import_args(parser: argparse.ArgumentParser, *, include_rebuild: bool = False) -> None:
    parser.add_argument("--min-date", default=None, help="Optional minimum snapshot date, YYYY-MM-DD")
    parser.add_argument("--max-date", default=None, help="Optional maximum snapshot date, YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="Allow replacing output with a non-empty failed candidate")
    if include_rebuild:
        parser.add_argument("--rebuild", action="store_true", help="Run validation, advanced features, null report, and ablation after import")


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    ensure_configured_directories(config)

    if args.command == "collect-historical":
        collect_historical(args, config)
    elif args.command == "collect-recent":
        collect_recent(args, config)
    elif args.command == "collect-worldcup":
        collect_worldcup(args, config)
    elif args.command == "build-clean":
        build_clean_matches(config)
    elif args.command == "build-features":
        build_training_dataset(start_date=args.start_date, config=config)
    elif args.command == "collect-all":
        run_collect_all(args, config)
    elif args.command == "update":
        run_update(args, config)
    elif args.command == "collect-rankings":
        rankings = load_fifa_rankings(config_path(config, "fifa_rankings"))
        logger.info("Loaded %s FIFA ranking rows", len(rankings))
    elif args.command == "collect-elo":
        elo = load_world_football_elo(config_path(config, "world_football_elo"))
        logger.info("Loaded %s World Football Elo rows", len(elo))
    elif args.command == "prepare-external-templates":
        paths = prepare_external_templates(config=config)
        for name, path in paths.items():
            logger.info("Wrote %s to %s", name, path)
    elif args.command == "prepare-fifa-rankings":
        from scripts.download_or_prepare_fifa_rankings import prepare_from_config as prepare_fifa_rankings_from_config

        prepare_fifa_rankings_from_config(input_path=args.input, config=config)
    elif args.command == "prepare-world-football-elo":
        from scripts.download_or_prepare_world_football_elo import prepare_from_config as prepare_world_football_elo_from_config

        prepare_world_football_elo_from_config(input_path=args.input, config=config)
    elif args.command == "acquire-fifa-rankings":
        acquire_fifa_rankings(
            input_path=args.input,
            url=args.url,
            force=args.force,
            config=config,
            run_post_checks=not args.skip_post_checks,
        )
    elif args.command == "acquire-world-football-elo":
        acquire_world_football_elo(
            input_path=args.input,
            url=args.url,
            force=args.force,
            config=config,
            run_post_checks=not args.skip_post_checks,
        )
    elif args.command == "acquire-external-ratings":
        acquire_external_ratings(force=args.force, config=config, run_post_checks=not args.skip_post_checks)
    elif args.command == "import-fifa-ranking-snapshots":
        import_fifa_ranking_snapshots(
            input_dir=Path(args.input_dir),
            output_path=Path(args.output),
            min_date=args.min_date,
            max_date=args.max_date,
            force=args.force,
            config=config,
        )
    elif args.command == "import-world-football-elo-snapshots":
        import_world_football_elo_snapshots(
            input_dir=Path(args.input_dir),
            output_path=Path(args.output),
            min_date=args.min_date,
            max_date=args.max_date,
            force=args.force,
            config=config,
        )
    elif args.command == "import-historical-rating-snapshots":
        fifa_input_dir = Path(args.fifa_input_dir)
        elo_input_dir = Path(args.elo_input_dir)
        if args.input_dir:
            parent = Path(args.input_dir)
            fifa_input_dir = parent / "fifa_rankings_snapshots"
            elo_input_dir = parent / "world_football_elo_snapshots"
        fifa_output = Path(args.fifa_output)
        elo_output = Path(args.elo_output)
        if args.output:
            parent_output = Path(args.output)
            fifa_output = parent_output / "fifa_rankings.csv"
            elo_output = parent_output / "world_football_elo.csv"
        config.setdefault("paths", {})["fifa_rankings"] = str(fifa_output)
        config.setdefault("paths", {})["world_football_elo"] = str(elo_output)
        config.setdefault("external_sources", {}).setdefault("fifa_rankings", {})["output_path"] = str(fifa_output)
        config.setdefault("external_sources", {}).setdefault("world_football_elo", {})["output_path"] = str(elo_output)
        import_historical_rating_snapshots(
            fifa_input_dir=fifa_input_dir,
            elo_input_dir=elo_input_dir,
            min_date=args.min_date,
            max_date=args.max_date,
            force=args.force,
            rebuild=args.rebuild,
            config=config,
        )
    elif args.command == "validate-external-data":
        validate_external_data(config=config)
    elif args.command == "feature-null-report":
        generate_feature_null_rate_report(config=config)
    elif args.command == "split-prediction-outputs":
        split_prediction_outputs(config=config)
    elif args.command == "build-advanced-features":
        build_advanced_features(config=config)
    elif args.command == "build-internal-historical-elo":
        build_internal_historical_elo_artifacts(config=config)
    elif args.command == "api-coverage-check":
        report_path = run_api_coverage_check()
        logger.info("Wrote API coverage check report to %s", report_path)
    elif args.command == "fetch-api-football-odds":
        client = ApiFootballClient()
        odds, raw_path, candidates = client.fetch_worldcup_odds(args.season)
        output_path = API_FOOTBALL_ODDS_2010_PROBE_PATH if args.season == 2010 else API_FOOTBALL_ODDS_2026_PATH if args.season == 2026 else resolve_project_path(f"data/staging/api_football_market_odds_{args.season}.csv")
        write_market_staging(odds, output_path)
        validate_staged_api_data(config=config)
        logger.info("Wrote %s API-Football odds rows to %s from raw payload %s", len(odds), output_path, raw_path)
        logger.info("World Cup league candidates discovered: %s", len(candidates))
    elif args.command == "fetch-api-football-injuries":
        client = ApiFootballClient()
        injuries, raw_path, candidates = client.fetch_live_injuries_for_worldcup_teams(season=args.season)
        write_injury_staging(injuries, API_FOOTBALL_INJURIES_LIVE_PATH)
        validate_staged_api_data(config=config)
        logger.info("Wrote %s API-Football injury rows to %s from raw payload %s", len(injuries), API_FOOTBALL_INJURIES_LIVE_PATH, raw_path)
        logger.info("World Cup league candidates discovered: %s", len(candidates))
    elif args.command == "fetch-the-odds-api-odds":
        client = TheOddsApiClient()
        odds, raw_path, sport_key = client.fetch_worldcup_2026_odds_if_available()
        write_market_staging(odds, THE_ODDS_API_ODDS_2026_PATH)
        validate_staged_api_data(config=config)
        logger.info("Wrote %s The Odds API rows to %s from raw payload %s; sport_key=%s", len(odds), THE_ODDS_API_ODDS_2026_PATH, raw_path, sport_key)
    elif args.command == "fetch-sportmonks-football-data":
        client = SportmonksClient()
        odds, odds_raw_path = client.fetch_worldcup_2026_odds_if_available()
        injuries, injuries_raw_path = client.fetch_worldcup_injuries_if_available()
        write_market_staging(odds, SPORTMONKS_ODDS_2026_PATH)
        write_injury_staging(injuries, SPORTMONKS_INJURIES_LIVE_PATH)
        validate_staged_api_data(config=config)
        logger.info("Wrote %s Sportmonks odds rows to %s from raw payload %s", len(odds), SPORTMONKS_ODDS_2026_PATH, odds_raw_path)
        logger.info("Wrote %s Sportmonks injury rows to %s from raw payload %s", len(injuries), SPORTMONKS_INJURIES_LIVE_PATH, injuries_raw_path)
    elif args.command == "validate-staged-api-data":
        market_report, injury_report = validate_staged_api_data(config=config)
        logger.info("Wrote staged market validation report to %s", market_report)
        logger.info("Wrote staged injury validation report to %s", injury_report)
    elif args.command == "discover-odds-competitions":
        report = discover_odds_competitions()
        print(f"ODDS_PROVIDER_DISCOVERY_REPORT={report}")
    elif args.command == "refresh-market-odds":
        summary = refresh_market_odds(
            provider=args.provider,
            season=args.season,
            prefer_api=args.prefer_api,
            dry_run=args.dry_run,
            skip_merge=args.skip_merge,
            rerun_reports=args.rerun_reports,
            rerun_live_predictions=args.rerun_live_predictions,
        )
        for key, value in summary.items():
            print(f"{key}={value}")
    elif args.command == "generate-missing-odds-template":
        path, summary = generate_missing_odds_template()
        for key, value in summary.items():
            print(f"{key.upper()}={value}")
        print(f"MISSING_ODDS_TEMPLATE={path}")
    elif args.command == "validate-manual-market-odds":
        summary = validate_manual_market_odds(args.input)
        for key, value in summary.items():
            print(f"{key}={value}")
    elif args.command == "merge-staged-market-odds":
        summary = merge_staged_market_odds(args.input, prefer_api=args.prefer_api)
        for key, value in summary.items():
            print(f"{key}={value}")
    elif args.command == "check-2010-odds-coverage":
        report_path, summary = check_2010_odds_coverage()
        print(f"WORLD_CUP_2010_ODDS_COVERAGE={summary['coverage']:.3f}")
        print(f"WORLD_CUP_2010_ROWS_PRESENT={summary['rows_present']}")
        print(f"WORLD_CUP_2010_ROWS_MISSING={summary['rows_missing']}")
        logger.info("Wrote 2010 odds coverage report to %s", report_path)
    elif args.command == "refresh-live-worldcup-data":
        summary = refresh_dashboard_live_data(fetch_lineups=args.fetch_lineups)
        _print_live_summary(summary)
    elif args.command == "generate-live-predictions":
        summary = generate_dashboard_live_predictions()
        _print_live_summary(summary)
    elif args.command == "train-model":
        train_match_model(model_name=args.model, feature_set=args.feature_set, config=config)
    elif args.command == "backtest-world-cups":
        run_worldcup_backtests(model_name=args.model, feature_set=args.feature_set, config=config)
    elif args.command == "backtest-worldcup-tournaments":
        run_worldcup_tournament_simulation_backtest(model_name=args.model, feature_set=args.feature_set, n_sims=args.n_sims, config=config)
    elif args.command == "stage5-benchmark":
        years = _parse_years(args.years)
        result = run_stage5_benchmark(
            years=years,
            n_bootstrap=args.n_bootstrap,
            include_market=args.include_market,
            include_poisson=args.include_poisson,
            include_dixon_coles=args.include_dixon_coles,
            output_dir=args.output_dir,
            model_name=args.model,
            feature_set=args.feature_set,
            config=config,
        )
        print(f"STAGE5_ACHIEVED={result.stage5_achieved}")
        print(f"STAGE5_METRICS_ROWS={len(result.metrics)}")
        print(f"STAGE5_PREDICTION_ROWS={len(result.predictions)}")
        print(f"STAGE5_REPORT={result.report_path}")
        print(f"STAGE5_DECISION={result.decision_path}")
    elif args.command == "stage5-diagnose-failures":
        result = run_stage5_failure_diagnosis(output_dir=args.output_dir, config=config)
        print(f"STAGE5_FAILURE_REPORT={result.report_path}")
        print(f"STAGE5_FAILURE_BY_YEAR={result.by_year_path}")
        print(f"STAGE5_FAILURE_BY_STAGE={result.by_stage_path}")
        print(f"STAGE5_FAILURE_BY_OUTCOME={result.by_outcome_path}")
        print(f"STAGE5_WORST_MATCHES={result.worst_matches_path}")
        print(f"STAGE5_ABLATION_RECONCILIATION={result.ablation_reconciliation_path}")
        print(f"STAGE5_ACHIEVED={result.stage5_achieved}")
    elif args.command == "stage5-reconcile-ablation":
        result = run_stage5_ablation_reconciliation(output_dir=args.output_dir)
        print(f"STAGE5_ABLATION_RECONCILIATION={result.csv_path}")
        print(f"STAGE5_ABLATION_RECONCILIATION_REPORT={result.report_path}")
        print(f"STAGE5_ACHIEVED={result.stage5_achieved}")
    elif args.command == "stage5-calibrated-market":
        years = _parse_years(args.years)
        result = run_stage5_calibrated_market(
            years=years,
            n_bootstrap=args.n_bootstrap,
            alpha_grid_step=args.alpha_grid_step,
            include_stacker=args.include_stacker,
            include_draw_adjustment=args.include_draw_adjustment,
            output_dir=args.output_dir,
            config=config,
        )
        print(f"STAGE5_CALIBRATED_MARKET_METRICS_ROWS={len(result.metrics)}")
        print(f"STAGE5_CALIBRATED_MARKET_PREDICTION_ROWS={len(result.predictions)}")
        print(f"STAGE5_MARKET_VALUE_SIGNIFICANCE_ROWS={len(result.significance)}")
        print(f"STAGE5_ALPHA_TUNING_ROWS={len(result.alpha_tuning)}")
        print(f"STAGE5_STACKER_COEFFICIENT_ROWS={len(result.stacker_coefficients)}")
        print(f"STAGE5_CALIBRATED_MARKET_REPORT={result.report_path}")
        print(f"STAGE5_CALIBRATED_MARKET_DECISION={result.decision_path}")
        print(f"STAGE5_ACHIEVED={result.stage5_achieved}")
    elif args.command == "stage5-signal-discovery":
        result = run_stage5_signal_discovery(
            output_dir=args.output_dir,
            n_bootstrap=args.n_bootstrap,
            config=config,
        )
        print(f"STAGE5_SIGNAL_BY_SEGMENT={resolve_project_path(args.output_dir) / 'stage5_signal_by_segment.csv'}")
        print(f"STAGE5_SIGNAL_WINNING_SEGMENTS={resolve_project_path(args.output_dir) / 'stage5_signal_winning_segments.csv'}")
        print(f"STAGE5_SIGNAL_LOSING_SEGMENTS={resolve_project_path(args.output_dir) / 'stage5_signal_losing_segments.csv'}")
        print(f"STAGE5_SIGNAL_WORST_MATCHES={resolve_project_path(args.output_dir) / 'stage5_signal_worst_matches.csv'}")
        print(f"STAGE5_SIGNAL_FEATURE_HYPOTHESES={resolve_project_path(args.output_dir) / 'stage5_signal_feature_hypotheses.csv'}")
        print(f"STAGE5_SIGNAL_DISCOVERY_REPORT={result.report_path}")
        print(f"STAGE5_ACHIEVED={result.stage5_achieved}")
    elif args.command == "stage5-draw-calibration":
        years = _parse_years(args.years)
        result = run_stage5_draw_calibration(
            years=years,
            n_bootstrap=args.n_bootstrap,
            output_dir=args.output_dir,
            config=config,
        )
        output_root = resolve_project_path(args.output_dir)
        print(f"STAGE5_DRAW_CALIBRATION_METRICS={output_root / 'stage5_draw_calibration_metrics.csv'}")
        print(f"STAGE5_DRAW_CALIBRATION_PREDICTIONS={output_root / 'stage5_draw_calibration_predictions.csv'}")
        print(f"STAGE5_DRAW_CALIBRATION_SIGNIFICANCE={output_root / 'stage5_draw_calibration_significance.csv'}")
        print(f"STAGE5_DRAW_CALIBRATION_TUNING={output_root / 'stage5_draw_calibration_tuning.csv'}")
        print(f"STAGE5_DRAW_CALIBRATION_REPORT={result.report_path}")
        print(f"STAGE5_DRAW_CALIBRATION_DECISION={result.decision_path}")
        print(f"STAGE5_ACHIEVED={result.stage5_achieved}")
    elif args.command == "train-poisson":
        train_poisson_model(config=config)
    elif args.command == "run-ablation":
        run_ablation(config=config)
    elif args.command == "calibrate-model":
        calibrate_model(config=config, model_name=args.model, feature_set=args.feature_set)
    elif args.command == "build-ensemble":
        build_ensemble_predictions(config=config)
    elif args.command == "simulate-worldcup":
        simulate_worldcup(n_sims=args.n_sims, config=config)
    elif args.command == "full-sota-pipeline":
        run_full_sota_pipeline(args, config)
    else:
        parser.error(f"Unknown command: {args.command}")
    return 0


def _print_live_summary(summary: dict[str, Any]) -> None:
    for key in [
        "LIVE_FIXTURES",
        "LIVE_ODDS_ROWS",
        "LIVE_INJURY_ROWS",
        "LIVE_LINEUP_ROWS",
        "LIVE_PREDICTION_ROWS",
        "INVALID_ODDS",
        "UNMATCHED_TEAMS",
        "LAST_UPDATED",
        "SUCCESS",
    ]:
        print(f"{key}={summary.get(key)}")


def _parse_years(value: str) -> list[int]:
    years: list[int] = []
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        years.append(int(part))
    return years


if __name__ == "__main__":
    raise SystemExit(main())
