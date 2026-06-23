from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from src.config import config_path, load_config
from src.io_utils import read_dataframe
from src.models.prediction_outputs import is_placeholder_team_name
from src.normalize import normalize_team_name
from src.simulation.knockout_stage import knockout_winner


logger = logging.getLogger(__name__)

STAGE_COLUMNS = [
    "round_of_32_probability",
    "round_of_16_probability",
    "quarterfinal_probability",
    "semifinal_probability",
    "final_probability",
    "champion_probability",
]

GROUP_LETTERS = list("ABCDEFGHIJKL")


def simulate_worldcup(n_sims: int = 100_000, *, config: dict[str, Any] | None = None, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or load_config()
    schedule = _load_worldcup_schedule(cfg)
    group_matches = _overlay_completed_group_scores(schedule["group_matches"], cfg)
    bracket_matches = schedule["bracket_matches"]
    tournament_teams = _tournament_teams(group_matches)
    if len(tournament_teams) != 48:
        raise ValueError(f"Expected 48 World Cup 2026 group-stage teams, found {len(tournament_teams)}")

    predictions = _load_predictions(cfg)
    prediction_metadata = _prediction_metadata(predictions, n_sims=n_sims, seed=seed)
    prediction_by_match, prediction_by_pair = _prediction_maps(predictions)
    team_strengths = _load_team_strengths(cfg, tournament_teams)
    rng = random.Random(seed)
    stage_counts: dict[str, Counter[str]] = {stage: Counter() for stage in STAGE_COLUMNS}
    sample_bracket: dict[str, Any] | None = None
    dynamic_prediction_rows: dict[tuple[str, str], dict[str, Any]] = {}

    for _ in range(n_sims):
        simulated_scores: dict[str, tuple[int, int]] = {}
        for match in group_matches:
            simulated_scores[match["match_id"]] = _score_for_actual_match(match, prediction_by_match, prediction_by_pair, team_strengths, rng)

        group_rankings = _compute_group_rankings(group_matches, simulated_scores, rng)
        bracket_context = _build_group_context(group_rankings)
        round_of_32 = _resolve_round_of_32(bracket_matches["round_of_32"], bracket_context)
        if len(round_of_32) != 32:
            raise RuntimeError(f"Round of 32 resolution produced {len(round_of_32)} teams instead of 32")
        _increment(stage_counts["round_of_32_probability"], round_of_32)
        bracket_context["used_third_groups"] = set()

        winners: dict[int, str] = {}
        losers: dict[int, str] = {}
        bracket_log: dict[str, Any] = {
            "groups": group_rankings,
            "round_of_32": round_of_32.copy(),
            "matches": [],
        }

        for stage_key, probability_column in [
            ("round_of_32", "round_of_16_probability"),
            ("round_of_16", "quarterfinal_probability"),
            ("quarterfinal", "semifinal_probability"),
            ("semifinal", "final_probability"),
            ("final", "champion_probability"),
        ]:
            stage_winners: list[str] = []
            for match in bracket_matches[stage_key]:
                home = _resolve_slot(match["home_team"], bracket_context, winners, losers)
                away = _resolve_slot(match["away_team"], bracket_context, winners, losers)
                if _is_placeholder_team(home) or _is_placeholder_team(away):
                    raise RuntimeError(f"Unresolved knockout placeholder: {home} vs {away}")
                probs = _prediction_for_actual_pair(home, away, prediction_by_pair, team_strengths, dynamic_prediction_rows)
                winner = knockout_winner(home, away, probs[0], probs[1], probs[2], rng=rng)
                loser = away if winner == home else home
                winners[match["match_number"]] = winner
                losers[match["match_number"]] = loser
                stage_winners.append(winner)
                bracket_log["matches"].append(
                    {
                        "match_number": match["match_number"],
                        "stage": stage_key,
                        "home_team": home,
                        "away_team": away,
                        "winner": winner,
                    }
                )
            _increment(stage_counts[probability_column], stage_winners)
            bracket_log[probability_column.replace("_probability", "")] = stage_winners.copy()

        if bracket_matches["third_place"]:
            for match in bracket_matches["third_place"]:
                home = _resolve_slot(match["home_team"], bracket_context, winners, losers)
                away = _resolve_slot(match["away_team"], bracket_context, winners, losers)
                probs = _prediction_for_actual_pair(home, away, prediction_by_pair, team_strengths, dynamic_prediction_rows)
                winner = knockout_winner(home, away, probs[0], probs[1], probs[2], rng=rng)
                bracket_log["matches"].append(
                    {
                        "match_number": match["match_number"],
                        "stage": "third_place",
                        "home_team": home,
                        "away_team": away,
                        "winner": winner,
                    }
                )

        bracket_log["champion"] = winners.get(104)
        if sample_bracket is None:
            sample_bracket = bracket_log

    stage_rows = []
    for team in tournament_teams:
        row = {"team": team}
        for column, counter in stage_counts.items():
            row[column] = counter[team] / n_sims
        stage_rows.append(row)
    stage_df = pd.DataFrame(stage_rows).sort_values("champion_probability", ascending=False, kind="stable")
    result_df = stage_df[
        [
            "team",
            "champion_probability",
            "final_probability",
            "semifinal_probability",
            "quarterfinal_probability",
            "round_of_16_probability",
            "round_of_32_probability",
        ]
    ].copy()
    _validate_stage_sums(result_df, tolerance=1e-9)
    _write_dynamic_predictions(cfg, dynamic_prediction_rows)
    _write_outputs(cfg, result_df, stage_df, sample_bracket or {}, prediction_metadata)
    return result_df, stage_df


def _load_worldcup_schedule(cfg: dict[str, Any]) -> dict[str, Any]:
    path = config_path(cfg, "worldcup_2026_fixtures_clean")
    if not path.exists():
        raise FileNotFoundError(f"World Cup 2026 fixture file is missing: {path}")
    fixtures = read_dataframe(path)
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    fixtures["home_team"] = fixtures["home_team"].map(normalize_team_name)
    fixtures["away_team"] = fixtures["away_team"].map(normalize_team_name)
    group_matches = _group_stage_fixtures(fixtures)
    bracket_matches = _bracket_fixtures(fixtures)
    return {"group_matches": group_matches, "bracket_matches": bracket_matches}


def _group_stage_fixtures(fixtures: pd.DataFrame) -> list[dict[str, Any]]:
    group_rows = fixtures[fixtures["group"].notna()].copy()
    group_rows = group_rows[~group_rows["home_team"].map(_is_placeholder_team) & ~group_rows["away_team"].map(_is_placeholder_team)]
    group_rows["group_letter"] = group_rows["group"].map(_group_letter)
    group_rows = group_rows[group_rows["group_letter"].isin(GROUP_LETTERS)]
    group_rows = group_rows.sort_values(["date", "group_letter", "match_id"], kind="stable")
    records = group_rows.to_dict("records")
    if len(records) != 72:
        raise ValueError(f"Expected 72 World Cup 2026 group fixtures, found {len(records)}")
    return records


def _bracket_fixtures(fixtures: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    bracket_rows = fixtures[fixtures["group"].isna()].copy().sort_values(["date", "match_id"], kind="stable")
    stage_map = {
        "round_of_32": bracket_rows[bracket_rows["stage"].astype(str).str.lower().eq("round of 32")],
        "round_of_16": bracket_rows[bracket_rows["stage"].astype(str).str.lower().eq("round of 16")],
        "quarterfinal": bracket_rows[bracket_rows["stage"].astype(str).str.lower().str.contains("quarter", na=False)],
        "semifinal": bracket_rows[bracket_rows["stage"].astype(str).str.lower().str.contains("semi", na=False)],
        "third_place": bracket_rows[bracket_rows["stage"].astype(str).str.lower().str.contains("third", na=False)],
        "final": bracket_rows[bracket_rows["stage"].astype(str).str.lower().eq("final")],
    }
    starts = {
        "round_of_32": 73,
        "round_of_16": 89,
        "quarterfinal": 97,
        "semifinal": 101,
        "third_place": 103,
        "final": 104,
    }
    output: dict[str, list[dict[str, Any]]] = {}
    for stage, rows in stage_map.items():
        records = []
        for offset, (_, row) in enumerate(rows.iterrows()):
            record = row.to_dict()
            record["match_number"] = starts[stage] + offset
            records.append(record)
        output[stage] = records
    expected = {"round_of_32": 16, "round_of_16": 8, "quarterfinal": 4, "semifinal": 2, "final": 1}
    for stage, count in expected.items():
        if len(output[stage]) != count:
            raise ValueError(f"Expected {count} {stage} fixtures, found {len(output[stage])}")
    return output


def _overlay_completed_group_scores(group_matches: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    all_path = config_path(cfg, "all_matches_clean")
    if not all_path.exists():
        return group_matches
    all_matches = read_dataframe(all_path)
    all_matches["date"] = pd.to_datetime(all_matches["date"], errors="coerce")
    all_matches["home_team"] = all_matches["home_team"].map(normalize_team_name)
    all_matches["away_team"] = all_matches["away_team"].map(normalize_team_name)
    all_matches["source_priority_for_scores"] = all_matches["source"].map({"football-data": 1, "worldcup_json": 2, "historical_results": 3}).fillna(99)
    scored = all_matches[
        all_matches["competition_type"].eq("world_cup")
        & all_matches["home_score"].notna()
        & all_matches["away_score"].notna()
        & all_matches["date"].dt.year.eq(2026)
    ].copy()
    scored["_score_key"] = scored.apply(lambda row: _fixture_key(row["date"], row["home_team"], row["away_team"]), axis=1)
    scored = scored.sort_values(["source_priority_for_scores", "date"], kind="stable").drop_duplicates("_score_key", keep="first")
    score_lookup = {row["_score_key"]: row for _, row in scored.iterrows()}
    output = []
    for match in group_matches:
        record = match.copy()
        key = _fixture_key(record["date"], record["home_team"], record["away_team"])
        score = score_lookup.get(key)
        if score is not None:
            record["home_score"] = float(score["home_score"]) if score["home_team"] == record["home_team"] else float(score["away_score"])
            record["away_score"] = float(score["away_score"]) if score["away_team"] == record["away_team"] else float(score["home_score"])
            record["status"] = "FINISHED"
            record["score_source"] = score["source"]
        output.append(record)
    return output


def _tournament_teams(group_matches: list[dict[str, Any]]) -> list[str]:
    teams = sorted({team for match in group_matches for team in (match["home_team"], match["away_team"]) if team and not _is_placeholder_team(team)})
    return teams


def _load_predictions(cfg: dict[str, Any]) -> pd.DataFrame:
    for key in ("ensemble_predictions", "calibrated_predictions", "match_predictions", "poisson_predictions"):
        path = config_path(cfg, key)
        if path.exists():
            predictions = read_dataframe(path)
            predictions["home_team"] = predictions["home_team"].map(normalize_team_name)
            predictions["away_team"] = predictions["away_team"].map(normalize_team_name)
            predictions.attrs["source_key"] = key
            predictions.attrs["source_path"] = str(path)
            return predictions
    logger.warning("No prediction file found. Simulator will use dynamic rating fallback probabilities.")
    return pd.DataFrame(columns=["match_id", "home_team", "away_team", "p_home_loss", "p_draw", "p_home_win"])


def _prediction_metadata(predictions: pd.DataFrame, *, n_sims: int, seed: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "prediction_source_key": predictions.attrs.get("source_key", "none"),
        "prediction_source_path": predictions.attrs.get("source_path", ""),
        "prediction_rows": int(len(predictions)),
        "n_sims": int(n_sims),
        "seed": int(seed),
    }
    if "model_name" in predictions.columns:
        metadata["model_name_counts"] = {str(key): int(value) for key, value in predictions["model_name"].value_counts(dropna=False).items()}
    if "feature_set" in predictions.columns:
        metadata["feature_set_counts"] = {str(key): int(value) for key, value in predictions["feature_set"].value_counts(dropna=False).items()}
    return metadata


def _load_team_strengths(cfg: dict[str, Any], teams: list[str]) -> dict[str, float]:
    strengths = {team: 1500.0 for team in teams}
    for key in ("worldcup_2026_prediction_input_advanced", "match_training_dataset_advanced_parquet", "match_training_dataset_parquet"):
        path = config_path(cfg, key)
        if not path.exists():
            continue
        df = read_dataframe(path)
        for side in ("home", "away"):
            team_col = f"{side}_team"
            candidate_cols = [
                f"{side}_dynamic_rating_pre",
                f"{side}_elo_pre_match",
                f"{side}_external_elo",
            ]
            usable = [column for column in candidate_cols if column in df.columns]
            if not usable:
                continue
            for team, group in df.dropna(subset=[team_col]).groupby(team_col):
                team = normalize_team_name(team)
                if team not in strengths:
                    continue
                values = []
                for column in usable:
                    values.extend(pd.to_numeric(group[column], errors="coerce").dropna().tail(5).tolist())
                if values:
                    strengths[team] = float(np.mean(values))
        break
    return strengths


def _prediction_maps(predictions: pd.DataFrame) -> tuple[dict[str, tuple[float, float, float]], dict[tuple[str, str], tuple[float, float, float]]]:
    by_match: dict[str, tuple[float, float, float]] = {}
    by_pair: dict[tuple[str, str], tuple[float, float, float]] = {}
    for row in predictions.itertuples(index=False):
        rec = row._asdict()
        home = rec.get("home_team")
        away = rec.get("away_team")
        if _is_placeholder_team(home) or _is_placeholder_team(away):
            continue
        probs = _normalize_probs(
            (
                rec.get("p_home_loss", 1 / 3),
                rec.get("p_draw", 1 / 3),
                rec.get("p_home_win", 1 / 3),
            )
        )
        match_id = rec.get("match_id")
        if match_id:
            by_match[match_id] = probs
        if home and away:
            by_pair[(home, away)] = probs
    return by_match, by_pair


def _score_for_match(match: dict[str, Any], predictions: pd.DataFrame, rng: random.Random) -> tuple[int, int]:
    """Backward-compatible helper used by tests."""
    if pd.notna(match.get("home_score")) and pd.notna(match.get("away_score")):
        return int(match["home_score"]), int(match["away_score"])
    prediction_by_match, prediction_by_pair = _prediction_maps(predictions if isinstance(predictions, pd.DataFrame) else pd.DataFrame())
    return _score_for_actual_match(match, prediction_by_match, prediction_by_pair, {}, rng)


def _score_for_actual_match(
    match: dict[str, Any],
    prediction_by_match: dict[str, tuple[float, float, float]],
    prediction_by_pair: dict[tuple[str, str], tuple[float, float, float]],
    team_strengths: dict[str, float],
    rng: random.Random,
) -> tuple[int, int]:
    if pd.notna(match.get("home_score")) and pd.notna(match.get("away_score")):
        return int(match["home_score"]), int(match["away_score"])
    probs = prediction_by_match.get(match.get("match_id"))
    if probs is None:
        probs = _prediction_for_actual_pair(match["home_team"], match["away_team"], prediction_by_pair, team_strengths)
    outcome = rng.choices([0, 1, 2], weights=probs, k=1)[0]
    if outcome == 2:
        return rng.choice([(1, 0), (2, 0), (2, 1), (3, 1)])
    if outcome == 1:
        goals = rng.choice([0, 1, 1, 2])
        return goals, goals
    return rng.choice([(0, 1), (0, 2), (1, 2), (1, 3)])


def _prediction_for_actual_pair(
    home: str,
    away: str,
    prediction_by_pair: dict[tuple[str, str], tuple[float, float, float]],
    team_strengths: dict[str, float],
    dynamic_prediction_rows: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> tuple[float, float, float]:
    direct = prediction_by_pair.get((home, away))
    if direct is not None:
        return direct
    reverse = prediction_by_pair.get((away, home))
    if reverse is not None:
        return reverse[2], reverse[1], reverse[0]
    home_strength = team_strengths.get(home, 1500.0)
    away_strength = team_strengths.get(away, 1500.0)
    draw = 0.24
    home_decisive = 1.0 / (1.0 + 10 ** ((away_strength - home_strength) / 400.0))
    probs = _normalize_probs(((1 - draw) * (1 - home_decisive), draw, (1 - draw) * home_decisive))
    if dynamic_prediction_rows is not None:
        dynamic_prediction_rows.setdefault(
            (home, away),
            {
                "home_team": home,
                "away_team": away,
                "p_home_loss": probs[0],
                "p_draw": probs[1],
                "p_home_win": probs[2],
                "source": "rating_fallback_dynamic",
            },
        )
    return probs


def _compute_group_rankings(
    group_matches: list[dict[str, Any]],
    simulated_scores: dict[str, tuple[int, int]],
    rng: random.Random,
) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, dict[str, dict[str, Any]]] = {letter: {} for letter in GROUP_LETTERS}
    for match in group_matches:
        group = match["group_letter"]
        table = tables[group]
        home = match["home_team"]
        away = match["away_team"]
        table.setdefault(home, _standing_row(home, group))
        table.setdefault(away, _standing_row(away, group))
        home_goals, away_goals = simulated_scores[match["match_id"]]
        _apply_score(table[home], home_goals, away_goals)
        _apply_score(table[away], away_goals, home_goals)
        if home_goals > away_goals:
            table[home]["points"] += 3
            table[home]["wins"] += 1
        elif away_goals > home_goals:
            table[away]["points"] += 3
            table[away]["wins"] += 1
        else:
            table[home]["points"] += 1
            table[away]["points"] += 1
    rankings = {}
    for group, table in tables.items():
        rows = list(table.values())
        for row in rows:
            row["_draw"] = rng.random()
        rows.sort(key=lambda row: (-row["points"], -row["goal_diff"], -row["goals_for"], -row["wins"], row["_draw"]))
        rankings[group] = [{key: value for key, value in row.items() if key != "_draw"} for row in rows]
    return rankings


def _build_group_context(group_rankings: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    winners: dict[str, str] = {}
    runners_up: dict[str, str] = {}
    third_rows: list[dict[str, Any]] = []
    for group, table in sorted(group_rankings.items()):
        winners[group] = table[0]["team"]
        runners_up[group] = table[1]["team"]
        third = table[2].copy()
        third["group"] = group
        third_rows.append(third)
    third_rows.sort(key=lambda row: (-row["points"], -row["goal_diff"], -row["goals_for"], -row["wins"], row["group"]))
    best_thirds = third_rows[:8]
    return {
        "winners": winners,
        "runners_up": runners_up,
        "best_thirds": best_thirds,
        "best_third_by_group": {row["group"]: row["team"] for row in best_thirds},
        "used_third_groups": set(),
    }


def _resolve_round_of_32(matches: list[dict[str, Any]], context: dict[str, Any]) -> list[str]:
    teams: list[str] = []
    context["used_third_groups"] = set()
    for match in matches:
        home = _resolve_group_slot(match["home_team"], context)
        away = _resolve_group_slot(match["away_team"], context)
        teams.extend([home, away])
    if len(teams) != 32 or len(set(teams)) != 32:
        raise RuntimeError(f"Round of 32 bracket has {len(teams)} slots and {len(set(teams))} unique teams")
    return teams


def _resolve_slot(slot: str, context: dict[str, Any], winners: dict[int, str], losers: dict[int, str]) -> str:
    slot = str(slot)
    if slot.startswith("W") and slot[1:].isdigit():
        return winners[int(slot[1:])]
    if slot.startswith("L") and slot[1:].isdigit():
        return losers[int(slot[1:])]
    return _resolve_group_slot(slot, context)


def _resolve_group_slot(slot: str, context: dict[str, Any]) -> str:
    slot = str(slot)
    match = re.fullmatch(r"([12])([A-L])", slot)
    if match:
        rank, group = match.groups()
        return context["winners"][group] if rank == "1" else context["runners_up"][group]
    third_match = re.fullmatch(r"3([A-L](?:/[A-L])*)", slot)
    if third_match:
        allowed = third_match.group(1).split("/")
        for group in allowed:
            if group in context["best_third_by_group"] and group not in context["used_third_groups"]:
                context["used_third_groups"].add(group)
                return context["best_third_by_group"][group]
        for row in context["best_thirds"]:
            group = row["group"]
            if group not in context["used_third_groups"]:
                context["used_third_groups"].add(group)
                return row["team"]
    return slot


def _standing_row(team: str, group: str) -> dict[str, Any]:
    return {"team": team, "group": group, "points": 0, "goal_diff": 0, "goals_for": 0, "wins": 0, "goals_against": 0}


def _apply_score(row: dict[str, Any], goals_for: int, goals_against: int) -> None:
    row["goals_for"] += goals_for
    row["goals_against"] += goals_against
    row["goal_diff"] += goals_for - goals_against


def _increment(counter: Counter[str], teams: list[str]) -> None:
    for team in teams:
        if _is_placeholder_team(team):
            raise RuntimeError(f"Placeholder team reached stage counter: {team}")
        counter[team] += 1


def _validate_stage_sums(result_df: pd.DataFrame, *, tolerance: float = 1e-6) -> None:
    expected = {
        "champion_probability": 1,
        "final_probability": 2,
        "semifinal_probability": 4,
        "quarterfinal_probability": 8,
        "round_of_16_probability": 16,
        "round_of_32_probability": 32,
    }
    for column, target in expected.items():
        actual = float(result_df[column].sum())
        if abs(actual - target) > tolerance:
            raise RuntimeError(f"{column} sums to {actual}, expected {target}")


def _write_outputs(cfg: dict[str, Any], result_df: pd.DataFrame, stage_df: pd.DataFrame, sample_bracket: dict[str, Any], metadata: dict[str, Any]) -> None:
    result_path = config_path(cfg, "simulation_results")
    stage_path = config_path(cfg, "simulation_stage_probabilities")
    bracket_path = config_path(cfg, "simulation_bracket_sample")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    bracket_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(result_path, index=False)
    stage_df.to_csv(stage_path, index=False)
    bracket_path.write_text(json.dumps(sample_bracket, indent=2, default=str), encoding="utf-8")
    try:
        metadata_path = config_path(cfg, "simulation_metadata")
    except KeyError:
        return
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def _write_dynamic_predictions(cfg: dict[str, Any], rows: dict[tuple[str, str], dict[str, Any]]) -> None:
    df = pd.DataFrame(list(rows.values()))
    parquet_path = config_path(cfg, "simulation_dynamic_predictions_parquet")
    csv_path = config_path(cfg, "simulation_dynamic_predictions_csv")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)


def _fixture_key(match_date: Any, home: str, away: str) -> tuple[str, tuple[str, str]]:
    date_key = pd.Timestamp(match_date).strftime("%Y-%m-%d") if pd.notna(match_date) else ""
    return date_key, tuple(sorted([str(home), str(away)]))


def _group_letter(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).upper().replace("_", " ")
    match = re.search(r"GROUP\s+([A-L])", text)
    if match:
        return match.group(1)
    match = re.fullmatch(r"([A-L])", text.strip())
    return match.group(1) if match else None


def _is_placeholder_team(team: Any) -> bool:
    return is_placeholder_team_name(team)


def _normalize_probs(values: tuple[Any, Any, Any]) -> tuple[float, float, float]:
    probs = np.array([float(value) if pd.notna(value) else 1 / 3 for value in values], dtype=float)
    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum()
    return float(probs[0]), float(probs[1]), float(probs[2])
