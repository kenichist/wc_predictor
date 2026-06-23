from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import pandas as pd

from src.simulation.tiebreakers import rank_group


def initialize_standings(teams: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team": team, "points": 0, "goal_diff": 0, "goals_for": 0, "wins": 0, "goals_against": 0}
            for team in sorted(set(teams))
            if team and not pd.isna(team)
        ]
    )


def apply_match_to_standings(standings: pd.DataFrame, home: str, away: str, home_goals: int, away_goals: int) -> pd.DataFrame:
    output = standings.copy()
    for team in (home, away):
        if team not in set(output["team"]):
            output.loc[len(output)] = {"team": team, "points": 0, "goal_diff": 0, "goals_for": 0, "wins": 0, "goals_against": 0}
    _apply_team(output, home, home_goals, away_goals)
    _apply_team(output, away, away_goals, home_goals)
    if home_goals > away_goals:
        output.loc[output["team"].eq(home), "points"] += 3
        output.loc[output["team"].eq(home), "wins"] += 1
    elif away_goals > home_goals:
        output.loc[output["team"].eq(away), "points"] += 3
        output.loc[output["team"].eq(away), "wins"] += 1
    else:
        output.loc[output["team"].isin([home, away]), "points"] += 1
    return output


def _apply_team(standings: pd.DataFrame, team: str, goals_for: int, goals_against: int) -> None:
    mask = standings["team"].eq(team)
    standings.loc[mask, "goals_for"] += goals_for
    standings.loc[mask, "goals_against"] += goals_against
    standings.loc[mask, "goal_diff"] += goals_for - goals_against


def group_qualifiers(group_matches: pd.DataFrame, simulated_scores: dict[str, tuple[int, int]], *, rng: random.Random) -> tuple[list[str], list[str]]:
    teams = list(group_matches["home_team"].dropna()) + list(group_matches["away_team"].dropna())
    standings = initialize_standings(teams)
    for match in group_matches.itertuples(index=False):
        score = simulated_scores.get(match.match_id)
        if score is None:
            continue
        standings = apply_match_to_standings(standings, match.home_team, match.away_team, score[0], score[1])
    ranked = rank_group(standings, rng=rng)
    top_two = ranked["team"].head(2).tolist()
    third = ranked["team"].iloc[2:3].tolist()
    return top_two, third


def compute_group_rankings(matches: pd.DataFrame, simulated_scores: dict[str, tuple[int, int]], *, rng: random.Random) -> dict[str, pd.DataFrame]:
    rankings: dict[str, pd.DataFrame] = {}
    for group, group_matches in matches.dropna(subset=["group"]).groupby("group"):
        teams = list(group_matches["home_team"].dropna()) + list(group_matches["away_team"].dropna())
        standings = initialize_standings(teams)
        for match in group_matches.itertuples(index=False):
            score = simulated_scores.get(match.match_id)
            if score is not None:
                standings = apply_match_to_standings(standings, match.home_team, match.away_team, score[0], score[1])
        rankings[str(group)] = rank_group(standings, rng=rng)
    return rankings
