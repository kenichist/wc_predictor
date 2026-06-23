from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtesting.worldcup_backtest import (
    BASIC_ELO_FEATURES,
    PROBABILITY_COLUMNS,
    WORLD_CUP_START_DATES,
    _fit_predict_feature_model,
    _market_coverage_mask,
    _market_probabilities,
    _safe_feature_columns,
    build_worldcup_backtest_feature_frame,
    blend_probabilities,
    split_worldcup_backtest_fold,
)
from src.config import config_path, load_config, resolve_project_path
from src.io_utils import read_dataframe, read_json, write_dataframe
from src.models.evaluate import evaluate_probabilities
from src.normalize import normalize_team_name


logger = logging.getLogger(__name__)

GROUP_MATCH_COUNT = 48
ROUND_OF_16_MATCH_COUNT = 8
QUARTERFINAL_MATCH_COUNT = 4
SEMIFINAL_MATCH_COUNT = 2
STAGE_COLUMNS = [
    "round_of_16_probability",
    "quarterfinal_probability",
    "semifinal_probability",
    "final_probability",
    "champion_probability",
]


def run_worldcup_tournament_simulation_backtest(
    *,
    model_name: str = "catboost",
    feature_set: str = "core_football_only",
    n_sims: int = 1_000,
    config: dict[str, Any] | None = None,
    feature_frame: pd.DataFrame | None = None,
    years: list[int] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, Path]:
    cfg = config or load_config()
    tournament_years = years or list(WORLD_CUP_START_DATES)
    df = feature_frame.copy() if feature_frame is not None else build_worldcup_backtest_feature_frame(config=cfg, feature_set=feature_set)
    shootout_lookup = _load_shootout_lookup(cfg)
    rows: list[dict[str, Any]] = []
    for offset, year in enumerate(tournament_years):
        train_df, target_df = split_worldcup_backtest_fold(df, year)
        if train_df.empty or target_df.empty:
            logger.warning("Skipping %s tournament simulation backtest due to empty train/target data", year)
            continue
        target_df = target_df.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
        if len(target_df) < 64:
            logger.warning("Skipping %s tournament simulation backtest because only %s World Cup rows are available", year, len(target_df))
            continue

        final_probabilities, resolved_model_name, feature_columns = _fit_predict_feature_model(
            train_df,
            target_df,
            feature_columns=_safe_feature_columns(feature_set, df.columns),
            model_name=model_name,
            random_seed=int(cfg.get("modeling", {}).get("random_seed", 42)),
            modeling_config=cfg.get("modeling", {}),
        )
        elo_probabilities, _, elo_columns = _fit_predict_feature_model(
            train_df,
            target_df,
            feature_columns=[column for column in BASIC_ELO_FEATURES if column in df.columns],
            model_name="hist_gradient_boosting",
            random_seed=int(cfg.get("modeling", {}).get("random_seed", 42)),
            modeling_config=cfg.get("modeling", {}),
            fallback_name="elo_only",
        )
        structure = _build_tournament_structure(target_df, shootout_lookup)
        actual = _actual_outcomes(structure, shootout_lookup)
        market_params = _market_params(cfg)
        model_specs = [
            {
                "model_name": "final_model",
                "feature_set": feature_set,
                "probabilities": final_probabilities,
                "feature_count": len(feature_columns),
                "odds_coverage_rate": np.nan,
                "missing_probability_fallback": "",
            },
            {
                "model_name": "elo_only",
                "feature_set": "elo_only",
                "probabilities": elo_probabilities,
                "feature_count": len(elo_columns),
                "odds_coverage_rate": np.nan,
                "missing_probability_fallback": "",
            },
        ]
        market_spec = _market_tournament_spec(target_df, final_probabilities, elo_probabilities, market_params)
        if market_spec is not None:
            model_specs.extend(market_spec)

        for model_index, spec in enumerate(model_specs):
            predictions = _prediction_frame(target_df, spec["probabilities"])
            stage_probabilities = _simulate_tournament(
                structure=structure,
                predictions=predictions,
                train_df=train_df,
                n_sims=n_sims,
                seed=seed + offset + (model_index * 10_000),
            )
            rows.append(
                _metrics_row(
                year=year,
                model_name=spec["model_name"],
                feature_set=spec["feature_set"],
                n_sims=n_sims,
                n_train=len(train_df),
                train_max_date=_max_date_text(train_df),
                feature_count=spec["feature_count"],
                target_df=target_df,
                predictions=predictions,
                structure=structure,
                stage_probabilities=stage_probabilities,
                actual=actual,
                odds_coverage_rate=spec["odds_coverage_rate"],
                missing_probability_fallback=spec["missing_probability_fallback"],
                alpha=spec.get("alpha", np.nan),
            )
            )

    results = pd.DataFrame(rows)
    output_path = _path_or_default(cfg, "worldcup_tournament_simulation_backtest_csv", "data/backtests/worldcup_tournament_simulation_backtest.csv")
    report_path = _path_or_default(
        cfg,
        "worldcup_tournament_simulation_backtest_report_md",
        "data/reports/worldcup_tournament_simulation_backtest_report.md",
    )
    write_dataframe(results, output_path)
    write_tournament_simulation_backtest_report(results=results, path=report_path)
    logger.info("Wrote World Cup tournament simulation backtest to %s", output_path)
    logger.info("Wrote World Cup tournament simulation backtest report to %s", report_path)
    return results, report_path


def write_tournament_simulation_backtest_report(*, results: pd.DataFrame, path: str | Path) -> Path:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# World Cup Tournament Simulation Backtest",
        "",
        "- Tournaments: 2010, 2014, 2018, 2022 FIFA World Cups",
        "- Each tournament fold trains only on matches before that tournament's opening match.",
        "- Group membership and knockout bracket slots are inferred from the historical fixture order.",
        "- Group-stage matches are simulated from model WDL probabilities; knockout winners use decisive win/loss probability after removing draw mass.",
        "",
    ]
    if results.empty:
        lines.extend(["No tournament simulation backtest rows were produced.", ""])
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    metric_columns = [
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
    lines.extend(["## Metrics", "", _markdown_table(results, metric_columns), ""])
    averages = (
        results.groupby("model_name", as_index=False)
        .agg(
            tournaments=("world_cup_year", "nunique"),
            champion_probability_assigned=("champion_probability_assigned", "mean"),
            actual_finalists_probability=("actual_finalists_probability", "mean"),
            semifinalist_probability_recall=("semifinalist_probability_recall", "mean"),
            round_of_16_brier_score=("round_of_16_brier_score", "mean"),
            group_qualification_accuracy=("group_qualification_accuracy", "mean"),
            group_match_log_loss=("group_match_log_loss", "mean"),
            odds_coverage_rate=("odds_coverage_rate", "mean"),
        )
        .sort_values(["group_match_log_loss", "model_name"], kind="stable")
    )
    lines.extend(["## Averages", "", _markdown_table(averages, list(averages.columns)), ""])
    lines.extend(
        [
            "## Known Limitations",
            "",
            "- Historical group labels are inferred from the first 48 fixtures rather than sourced from an official tournament fixture table.",
            "- Penalty shootout winners are read from the historical shootouts source for actual knockout outcomes.",
            "- Simulated knockout pairings not seen in the actual fixture list use a rating-based probability fallback.",
            "- Bookmaker-only tournament simulations use market probabilities where odds exist and Elo fallback where odds are missing, so odds coverage is reported for those rows.",
            "- Market-blend tournament simulations use blend probabilities where odds exist and final-model fallback where odds are missing.",
            "- The bracket backtest measures stage probabilities, not exact scoreline or exact bracket-path accuracy.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _prediction_frame(target_df: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    output = target_df[["match_id", "date", "home_team", "away_team", "target_result_class"]].copy()
    output[PROBABILITY_COLUMNS] = probabilities
    return output


def _market_params(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        path = config_path(cfg, "market_blend_params")
    except KeyError:
        return {"alpha": 0.1, "odds_conversion_method": "multiplicative"}
    if not path.exists():
        return {"alpha": 0.1, "odds_conversion_method": "multiplicative"}
    try:
        params = read_json(path)
    except Exception:
        return {"alpha": 0.1, "odds_conversion_method": "multiplicative"}
    return {
        "alpha": float(params.get("alpha", 0.1)),
        "odds_conversion_method": str(params.get("odds_conversion_method", "multiplicative")),
    }


def _market_tournament_spec(
    target_df: pd.DataFrame,
    final_probabilities: np.ndarray,
    elo_probabilities: np.ndarray,
    market_params: dict[str, Any],
) -> list[dict[str, Any]] | None:
    market_mask = _market_coverage_mask(target_df)
    covered_count = int(market_mask.sum())
    if covered_count == 0:
        return None
    market_prob = _market_probabilities(target_df[market_mask].copy(), odds_method=str(market_params.get("odds_conversion_method", "multiplicative")))
    if market_prob is None:
        return None
    full_market = np.array(elo_probabilities, copy=True)
    full_market[market_mask.to_numpy()] = market_prob
    alpha = float(market_params.get("alpha", 0.1))
    full_blend = np.array(final_probabilities, copy=True)
    full_blend[market_mask.to_numpy()] = blend_probabilities(final_probabilities[market_mask.to_numpy()], market_prob, alpha)
    coverage = covered_count / len(target_df) if len(target_df) else 0.0
    return [
        {
            "model_name": "bookmaker_odds_only",
            "feature_set": "market",
            "probabilities": full_market,
            "feature_count": 3,
            "odds_coverage_rate": coverage,
            "missing_probability_fallback": "elo_only",
        },
        {
            "model_name": "market_blend",
            "feature_set": "core_football_only+market",
            "probabilities": full_blend,
            "feature_count": np.nan,
            "odds_coverage_rate": coverage,
            "missing_probability_fallback": "final_model",
            "alpha": alpha,
        },
    ]


def _build_tournament_structure(target_df: pd.DataFrame, shootout_lookup: dict[tuple[str, str, str], str]) -> dict[str, Any]:
    ordered = target_df.sort_values(["date", "match_id"], kind="stable").reset_index(drop=True).copy()
    group_matches = ordered.iloc[:GROUP_MATCH_COUNT].copy().reset_index(drop=True)
    knockout = ordered.iloc[GROUP_MATCH_COUNT:].copy().reset_index(drop=True)
    if len(knockout) < 16:
        raise ValueError("Expected 16 knockout matches after the first 48 group matches")
    groups = _infer_groups(group_matches)
    group_matches = group_matches.drop(columns=["group"], errors="ignore")
    group_matches = group_matches.merge(groups[["team", "group"]], left_on="home_team", right_on="team", how="left").drop(columns=["team"])
    group_matches = group_matches.rename(columns={"group": "home_group"})
    group_matches = group_matches.merge(groups[["team", "group"]], left_on="away_team", right_on="team", how="left").drop(columns=["team"])
    group_matches = group_matches.rename(columns={"group": "away_group"})
    if not group_matches["home_group"].eq(group_matches["away_group"]).all():
        raise ValueError("Could not infer consistent historical World Cup groups")
    group_matches["group"] = group_matches["home_group"]
    group_matches = group_matches.drop(columns=["home_group", "away_group"])

    actual_rankings = _rank_groups(group_matches, rng=random.Random(0), random_ties=False)
    team_slots = _actual_team_slots(actual_rankings)
    r16 = knockout.iloc[:ROUND_OF_16_MATCH_COUNT].copy().reset_index(drop=True)
    qf = knockout.iloc[ROUND_OF_16_MATCH_COUNT : ROUND_OF_16_MATCH_COUNT + QUARTERFINAL_MATCH_COUNT].copy().reset_index(drop=True)
    sf_start = ROUND_OF_16_MATCH_COUNT + QUARTERFINAL_MATCH_COUNT
    sf = knockout.iloc[sf_start : sf_start + SEMIFINAL_MATCH_COUNT].copy().reset_index(drop=True)
    third = knockout.iloc[sf_start + SEMIFINAL_MATCH_COUNT : sf_start + SEMIFINAL_MATCH_COUNT + 1].copy().reset_index(drop=True)
    final = knockout.iloc[sf_start + SEMIFINAL_MATCH_COUNT + 1 : sf_start + SEMIFINAL_MATCH_COUNT + 2].copy().reset_index(drop=True)

    r16_specs = [{"home": team_slots[row.home_team], "away": team_slots[row.away_team]} for row in r16.itertuples(index=False)]
    r16_winners = [_actual_match_winner(row._asdict(), shootout_lookup) for row in r16.itertuples(index=False)]
    qf_specs = _winner_source_specs(qf, r16_winners, "r16")
    qf_winners = [_actual_match_winner(row._asdict(), shootout_lookup) for row in qf.itertuples(index=False)]
    sf_specs = _winner_source_specs(sf, qf_winners, "qf")
    sf_winners = [_actual_match_winner(row._asdict(), shootout_lookup) for row in sf.itertuples(index=False)]
    sf_losers = [_actual_match_loser(row._asdict(), shootout_lookup) for row in sf.itertuples(index=False)]
    final_specs = _winner_source_specs(final, sf_winners, "sf")
    third_specs = _winner_source_specs(third, sf_losers, "sf_loser") if not third.empty else []

    return {
        "group_matches": group_matches,
        "groups": groups,
        "actual_rankings": actual_rankings,
        "r16": r16,
        "qf": qf,
        "sf": sf,
        "third": third,
        "final": final,
        "r16_specs": r16_specs,
        "qf_specs": qf_specs,
        "sf_specs": sf_specs,
        "final_specs": final_specs,
        "third_specs": third_specs,
    }


def _infer_groups(group_matches: pd.DataFrame) -> pd.DataFrame:
    graph: dict[str, set[str]] = defaultdict(set)
    first_seen: dict[str, int] = {}
    for idx, row in enumerate(group_matches.itertuples(index=False)):
        home = str(row.home_team)
        away = str(row.away_team)
        graph[home].add(away)
        graph[away].add(home)
        first_seen.setdefault(home, idx)
        first_seen.setdefault(away, idx)
    components: list[list[str]] = []
    remaining = set(graph)
    while remaining:
        start = min(remaining, key=lambda team: first_seen[team])
        queue: deque[str] = deque([start])
        component: list[str] = []
        remaining.remove(start)
        while queue:
            team = queue.popleft()
            component.append(team)
            for neighbor in sorted(graph[team], key=lambda item: first_seen[item]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
    components.sort(key=lambda teams: min(first_seen[team] for team in teams))
    rows = []
    for index, teams in enumerate(components):
        group = chr(ord("A") + index)
        for team in teams:
            rows.append({"team": team, "group": group})
    output = pd.DataFrame(rows)
    if len(output) != 32 or output["group"].nunique() != 8:
        raise ValueError(f"Expected 8 groups and 32 teams, found {output['group'].nunique()} groups and {len(output)} teams")
    return output


def _simulate_tournament(
    *,
    structure: dict[str, Any],
    predictions: pd.DataFrame,
    train_df: pd.DataFrame,
    n_sims: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    group_matches = structure["group_matches"]
    teams = sorted(structure["groups"]["team"].unique())
    prediction_by_match, prediction_by_pair = _prediction_maps(predictions)
    strengths = _team_strengths(train_df, group_matches)
    group_records = _group_simulation_records(group_matches, prediction_by_match, prediction_by_pair, strengths)
    teams_by_group = {str(group): sorted(set(rows["home_team"]) | set(rows["away_team"])) for group, rows in group_matches.groupby("group", sort=True)}
    stage_counts: dict[str, Counter[str]] = {column: Counter() for column in STAGE_COLUMNS}

    for _ in range(n_sims):
        rankings = _simulate_group_rankings_fast(group_records, teams_by_group, rng)
        r16_teams = _resolve_r16_teams_fast(structure["r16_specs"], rankings)
        _increment(stage_counts["round_of_16_probability"], r16_teams)
        r16_winners, r16_losers = _simulate_stage_fast(structure["r16_specs"], rankings, {}, {}, prediction_by_pair, strengths, rng)
        _increment(stage_counts["quarterfinal_probability"], r16_winners)
        qf_winners, qf_losers = _simulate_stage_fast(structure["qf_specs"], rankings, {"r16": r16_winners}, {"r16": r16_losers}, prediction_by_pair, strengths, rng)
        _increment(stage_counts["semifinal_probability"], qf_winners)
        sf_winners, sf_losers = _simulate_stage_fast(structure["sf_specs"], rankings, {"qf": qf_winners}, {"qf": qf_losers}, prediction_by_pair, strengths, rng)
        _increment(stage_counts["final_probability"], sf_winners)
        final_winners, _ = _simulate_stage_fast(structure["final_specs"], rankings, {"sf": sf_winners}, {"sf": sf_losers}, prediction_by_pair, strengths, rng)
        _increment(stage_counts["champion_probability"], final_winners)

    rows = []
    for team in teams:
        row = {"team": team}
        for column in STAGE_COLUMNS:
            row[column] = stage_counts[column][team] / n_sims
        rows.append(row)
    return pd.DataFrame(rows).sort_values("champion_probability", ascending=False, kind="stable").reset_index(drop=True)


def _group_simulation_records(
    group_matches: pd.DataFrame,
    prediction_by_match: dict[str, tuple[float, float, float]],
    prediction_by_pair: dict[tuple[str, str], tuple[float, float, float]],
    strengths: dict[str, float],
) -> list[tuple[str, str, str, tuple[float, float, float]]]:
    records = []
    for row in group_matches.itertuples(index=False):
        probs = prediction_by_match.get(row.match_id) or _prediction_for_pair(row.home_team, row.away_team, prediction_by_pair, strengths)
        records.append((str(row.group), str(row.home_team), str(row.away_team), probs))
    return records


def _simulate_group_rankings_fast(
    records: list[tuple[str, str, str, tuple[float, float, float]]],
    teams_by_group: dict[str, list[str]],
    rng: random.Random,
) -> dict[str, list[str]]:
    tables = {
        group: {team: {"points": 0, "goal_diff": 0, "goals_for": 0, "wins": 0} for team in teams}
        for group, teams in teams_by_group.items()
    }
    for group, home, away, probs in records:
        home_goals, away_goals = _sample_score(probs, rng)
        home_row = tables[group][home]
        away_row = tables[group][away]
        home_row["goals_for"] += home_goals
        home_row["goal_diff"] += home_goals - away_goals
        away_row["goals_for"] += away_goals
        away_row["goal_diff"] += away_goals - home_goals
        if home_goals > away_goals:
            home_row["points"] += 3
            home_row["wins"] += 1
        elif away_goals > home_goals:
            away_row["points"] += 3
            away_row["wins"] += 1
        else:
            home_row["points"] += 1
            away_row["points"] += 1
    rankings: dict[str, list[str]] = {}
    for group, table in tables.items():
        rankings[group] = sorted(
            table,
            key=lambda team: (
                -table[team]["points"],
                -table[team]["goal_diff"],
                -table[team]["goals_for"],
                -table[team]["wins"],
                rng.random(),
            ),
        )
    return rankings


def _rank_groups(
    group_matches: pd.DataFrame,
    scores: dict[str, tuple[int, int]] | None = None,
    *,
    rng: random.Random,
    random_ties: bool,
) -> dict[str, pd.DataFrame]:
    rankings = {}
    for group, matches in group_matches.groupby("group", sort=True):
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        table = {team: {"team": team, "points": 0, "goal_diff": 0, "goals_for": 0, "goals_against": 0, "wins": 0} for team in teams}
        for row in matches.itertuples(index=False):
            if scores is None:
                home_goals = int(row.home_score)
                away_goals = int(row.away_score)
            else:
                home_goals, away_goals = scores[row.match_id]
            _apply_score(table[row.home_team], home_goals, away_goals)
            _apply_score(table[row.away_team], away_goals, home_goals)
            if home_goals > away_goals:
                table[row.home_team]["points"] += 3
                table[row.home_team]["wins"] += 1
            elif away_goals > home_goals:
                table[row.away_team]["points"] += 3
                table[row.away_team]["wins"] += 1
            else:
                table[row.home_team]["points"] += 1
                table[row.away_team]["points"] += 1
        ranked = pd.DataFrame(table.values())
        ranked["_draw"] = [rng.random() if random_ties else team for team in ranked["team"]]
        ranked = ranked.sort_values(
            ["points", "goal_diff", "goals_for", "wins", "_draw"],
            ascending=[False, False, False, False, True],
            kind="stable",
        ).drop(columns=["_draw"])
        rankings[str(group)] = ranked.reset_index(drop=True)
    return rankings


def _actual_team_slots(actual_rankings: dict[str, pd.DataFrame]) -> dict[str, str]:
    slots = {}
    for group, table in actual_rankings.items():
        for idx, team in enumerate(table["team"].head(2), start=1):
            slots[str(team)] = f"{idx}{group}"
    return slots


def _winner_source_specs(matches: pd.DataFrame, previous_winners: list[str], stage_name: str) -> list[dict[str, tuple[str, int]]]:
    specs = []
    for row in matches.itertuples(index=False):
        specs.append({"home": (stage_name, previous_winners.index(row.home_team)), "away": (stage_name, previous_winners.index(row.away_team))})
    return specs


def _resolve_r16_teams(specs: list[dict[str, str]], rankings: dict[str, pd.DataFrame]) -> list[str]:
    teams = []
    for spec in specs:
        teams.append(_resolve_group_slot(spec["home"], rankings))
        teams.append(_resolve_group_slot(spec["away"], rankings))
    return teams


def _resolve_r16_teams_fast(specs: list[dict[str, str]], rankings: dict[str, list[str]]) -> list[str]:
    teams = []
    for spec in specs:
        teams.append(_resolve_group_slot_fast(spec["home"], rankings))
        teams.append(_resolve_group_slot_fast(spec["away"], rankings))
    return teams


def _simulate_stage(
    specs: list[dict[str, Any]],
    rankings: dict[str, pd.DataFrame],
    winners_by_stage: dict[str, list[str]],
    losers_by_stage: dict[str, list[str]],
    prediction_by_pair: dict[tuple[str, str], tuple[float, float, float]],
    strengths: dict[str, float],
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    winners: list[str] = []
    losers: list[str] = []
    for spec in specs:
        home = _resolve_source(spec["home"], rankings, winners_by_stage, losers_by_stage)
        away = _resolve_source(spec["away"], rankings, winners_by_stage, losers_by_stage)
        probs = _prediction_for_pair(home, away, prediction_by_pair, strengths)
        winner = _knockout_winner(home, away, probs, rng)
        loser = away if winner == home else home
        winners.append(winner)
        losers.append(loser)
    return winners, losers


def _simulate_stage_fast(
    specs: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    winners_by_stage: dict[str, list[str]],
    losers_by_stage: dict[str, list[str]],
    prediction_by_pair: dict[tuple[str, str], tuple[float, float, float]],
    strengths: dict[str, float],
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    winners: list[str] = []
    losers: list[str] = []
    for spec in specs:
        home = _resolve_source_fast(spec["home"], rankings, winners_by_stage, losers_by_stage)
        away = _resolve_source_fast(spec["away"], rankings, winners_by_stage, losers_by_stage)
        probs = _prediction_for_pair(home, away, prediction_by_pair, strengths)
        winner = _knockout_winner(home, away, probs, rng)
        loser = away if winner == home else home
        winners.append(winner)
        losers.append(loser)
    return winners, losers


def _resolve_source(source: Any, rankings: dict[str, pd.DataFrame], winners_by_stage: dict[str, list[str]], losers_by_stage: dict[str, list[str]]) -> str:
    if isinstance(source, str):
        return _resolve_group_slot(source, rankings)
    stage, idx = source
    if stage.endswith("_loser"):
        base_stage = stage.replace("_loser", "")
        return losers_by_stage[base_stage][idx]
    return winners_by_stage[stage][idx]


def _resolve_source_fast(source: Any, rankings: dict[str, list[str]], winners_by_stage: dict[str, list[str]], losers_by_stage: dict[str, list[str]]) -> str:
    if isinstance(source, str):
        return _resolve_group_slot_fast(source, rankings)
    stage, idx = source
    if stage.endswith("_loser"):
        base_stage = stage.replace("_loser", "")
        return losers_by_stage[base_stage][idx]
    return winners_by_stage[stage][idx]


def _resolve_group_slot(slot: str, rankings: dict[str, pd.DataFrame]) -> str:
    rank = int(slot[0]) - 1
    group = slot[1]
    return str(rankings[group].iloc[rank]["team"])


def _resolve_group_slot_fast(slot: str, rankings: dict[str, list[str]]) -> str:
    rank = int(slot[0]) - 1
    group = slot[1]
    return rankings[group][rank]


def _actual_outcomes(structure: dict[str, Any], shootout_lookup: dict[tuple[str, str, str], str]) -> dict[str, Any]:
    final = structure["final"].iloc[0].to_dict()
    champion = _actual_match_winner(final, shootout_lookup)
    finalists = {final["home_team"], final["away_team"]}
    semifinalists = set(structure["sf"]["home_team"]) | set(structure["sf"]["away_team"])
    r16_qualifiers = set()
    for table in structure["actual_rankings"].values():
        r16_qualifiers.update(table["team"].head(2).tolist())
    return {"champion": champion, "finalists": finalists, "semifinalists": semifinalists, "r16_qualifiers": r16_qualifiers}


def _metrics_row(
    *,
    year: int,
    model_name: str,
    feature_set: str,
    n_sims: int,
    n_train: int,
    train_max_date: str,
    feature_count: int,
    target_df: pd.DataFrame,
    predictions: pd.DataFrame,
    structure: dict[str, Any],
    stage_probabilities: pd.DataFrame,
    actual: dict[str, Any],
    odds_coverage_rate: float,
    missing_probability_fallback: str,
    alpha: float,
) -> dict[str, Any]:
    lookup = stage_probabilities.set_index("team")
    champion_probability = float(lookup.loc[actual["champion"], "champion_probability"])
    finalist_values = [float(lookup.loc[team, "final_probability"]) for team in actual["finalists"]]
    semifinal_top4 = set(stage_probabilities.sort_values("semifinal_probability", ascending=False, kind="stable")["team"].head(4))
    semifinal_recall = len(semifinal_top4 & actual["semifinalists"]) / max(1, len(actual["semifinalists"]))
    groups = structure["groups"]
    team_group = dict(zip(groups["team"], groups["group"]))
    qualified_by_group = {}
    for group, teams in groups.groupby("group"):
        group_probs = stage_probabilities[stage_probabilities["team"].isin(teams["team"])]
        predicted = set(group_probs.sort_values("round_of_16_probability", ascending=False, kind="stable")["team"].head(2))
        for team in teams["team"]:
            qualified_by_group[team] = team in predicted
    actual_flags = {team: team in actual["r16_qualifiers"] for team in groups["team"]}
    group_accuracy = float(np.mean([qualified_by_group[team] == actual_flags[team] for team in groups["team"]]))
    r16_brier = float(
        np.mean(
            [
                (float(lookup.loc[team, "round_of_16_probability"]) - float(actual_flags[team])) ** 2
                for team in groups["team"]
            ]
        )
    )
    group_ids = set(structure["group_matches"]["match_id"])
    group_predictions = predictions[predictions["match_id"].isin(group_ids)].copy()
    match_metrics = evaluate_probabilities(group_predictions["target_result_class"], group_predictions[PROBABILITY_COLUMNS].to_numpy(dtype=float))
    return {
        "world_cup_year": year,
        "model_name": model_name,
        "feature_set": feature_set,
        "n_sims": n_sims,
        "n_train": n_train,
        "train_max_date": train_max_date,
        "feature_count": feature_count,
        "odds_coverage_rate": odds_coverage_rate,
        "missing_probability_fallback": missing_probability_fallback,
        "alpha": alpha,
        "actual_champion": actual["champion"],
        "actual_finalists": ", ".join(sorted(actual["finalists"])),
        "actual_semifinalists": ", ".join(sorted(actual["semifinalists"])),
        "champion_probability_assigned": champion_probability,
        "actual_finalists_probability": float(np.mean(finalist_values)),
        "actual_finalists_probability_sum": float(np.sum(finalist_values)),
        "semifinalist_probability_recall": semifinal_recall,
        "round_of_16_brier_score": r16_brier,
        "group_qualification_accuracy": group_accuracy,
        "group_match_log_loss": match_metrics["log_loss"],
        "group_match_accuracy": match_metrics["accuracy"],
        "teams": len(groups),
        "group_matches": len(structure["group_matches"]),
        "tournament_matches": len(target_df),
    }


def _prediction_maps(predictions: pd.DataFrame) -> tuple[dict[str, tuple[float, float, float]], dict[tuple[str, str], tuple[float, float, float]]]:
    by_match = {}
    by_pair = {}
    for row in predictions.itertuples(index=False):
        probs = _normalize_probs((row.p_home_loss, row.p_draw, row.p_home_win))
        by_match[row.match_id] = probs
        by_pair[(row.home_team, row.away_team)] = probs
    return by_match, by_pair


def _prediction_for_pair(
    home: str,
    away: str,
    prediction_by_pair: dict[tuple[str, str], tuple[float, float, float]],
    strengths: dict[str, float],
) -> tuple[float, float, float]:
    direct = prediction_by_pair.get((home, away))
    if direct is not None:
        return direct
    reverse = prediction_by_pair.get((away, home))
    if reverse is not None:
        return reverse[2], reverse[1], reverse[0]
    home_strength = strengths.get(home, 1500.0)
    away_strength = strengths.get(away, 1500.0)
    draw = 0.24
    home_decisive = 1.0 / (1.0 + 10 ** ((away_strength - home_strength) / 400.0))
    return _normalize_probs(((1.0 - draw) * (1.0 - home_decisive), draw, (1.0 - draw) * home_decisive))


def _team_strengths(train_df: pd.DataFrame, group_matches: pd.DataFrame) -> dict[str, float]:
    teams = set(group_matches["home_team"]) | set(group_matches["away_team"])
    strengths = {team: 1500.0 for team in teams}
    candidates = ["dynamic_rating_pre", "elo_pre_match", "internal_elo"]
    ordered = train_df.sort_values("date", kind="stable")
    for side in ("home", "away"):
        team_col = f"{side}_team"
        usable = [f"{side}_{column}" for column in candidates if f"{side}_{column}" in ordered.columns]
        if not usable:
            usable = [f"{side}_elo_pre_match"] if f"{side}_elo_pre_match" in ordered.columns else []
        for team, rows in ordered[ordered[team_col].isin(teams)].groupby(team_col):
            values = []
            for column in usable:
                values.extend(pd.to_numeric(rows[column], errors="coerce").dropna().tail(5).tolist())
            if values:
                strengths[team] = float(np.mean(values))
    return strengths


def _sample_score(probs: tuple[float, float, float], rng: random.Random) -> tuple[int, int]:
    outcome = rng.choices([0, 1, 2], weights=probs, k=1)[0]
    if outcome == 2:
        return rng.choice([(1, 0), (2, 0), (2, 1), (3, 1), (3, 2)])
    if outcome == 1:
        goals = rng.choice([0, 1, 1, 2])
        return goals, goals
    return rng.choice([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])


def _knockout_winner(home: str, away: str, probs: tuple[float, float, float], rng: random.Random) -> str:
    p_loss, _, p_win = probs
    decisive = p_loss + p_win
    if decisive <= 0:
        return home if rng.random() < 0.5 else away
    return home if rng.random() < (p_win / decisive) else away


def _actual_match_winner(row: dict[str, Any], shootout_lookup: dict[tuple[str, str, str], str]) -> str:
    home = row["home_team"]
    away = row["away_team"]
    if float(row["home_score"]) > float(row["away_score"]):
        return home
    if float(row["away_score"]) > float(row["home_score"]):
        return away
    key = _shootout_key(row["date"], home, away)
    return shootout_lookup.get(key, home)


def _actual_match_loser(row: dict[str, Any], shootout_lookup: dict[tuple[str, str, str], str]) -> str:
    winner = _actual_match_winner(row, shootout_lookup)
    return row["away_team"] if winner == row["home_team"] else row["home_team"]


def _load_shootout_lookup(cfg: dict[str, Any]) -> dict[tuple[str, str, str], str]:
    try:
        path = config_path(cfg, "historical_shootouts_raw")
    except KeyError:
        path = resolve_project_path("data/raw/historical_shootouts_raw.csv")
    if not path.exists():
        return {}
    shootouts = read_dataframe(path)
    lookup = {}
    for row in shootouts.itertuples(index=False):
        home = normalize_team_name(row.home_team)
        away = normalize_team_name(row.away_team)
        winner = normalize_team_name(row.winner)
        if home and away and winner:
            lookup[_shootout_key(row.date, home, away)] = winner
    return lookup


def _shootout_key(date: Any, home: str, away: str) -> tuple[str, str, str]:
    date_key = pd.Timestamp(date).strftime("%Y-%m-%d") if pd.notna(date) else ""
    return date_key, *sorted([normalize_team_name(home) or str(home), normalize_team_name(away) or str(away)])


def _apply_score(row: dict[str, Any], goals_for: int, goals_against: int) -> None:
    row["goals_for"] += goals_for
    row["goals_against"] += goals_against
    row["goal_diff"] += goals_for - goals_against


def _increment(counter: Counter[str], teams: list[str]) -> None:
    for team in teams:
        counter[team] += 1


def _normalize_probs(values: tuple[Any, Any, Any]) -> tuple[float, float, float]:
    probs = np.array([float(value) if pd.notna(value) else 1 / 3 for value in values], dtype=float)
    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum()
    return float(probs[0]), float(probs[1]), float(probs[2])


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in df.columns]
    if df.empty or not available:
        return "_No rows._"
    rows = ["| " + " | ".join(available) + " |", "| " + " | ".join(["---"] * len(available)) + " |"]
    for record in df[available].to_dict("records"):
        rows.append("| " + " | ".join(_format_value(record[column]) for column in available) + " |")
    return "\n".join(rows)


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _max_date_text(df: pd.DataFrame) -> str:
    value = pd.to_datetime(df["date"], errors="coerce").max()
    if pd.isna(value):
        return ""
    return value.date().isoformat()


def _path_or_default(config: dict[str, Any], key: str, default: str) -> Path:
    try:
        return config_path(config, key)
    except KeyError:
        return resolve_project_path(default)
