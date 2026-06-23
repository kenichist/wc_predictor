from __future__ import annotations

import random


def knockout_winner(home_team: str, away_team: str, p_home_loss: float, p_draw: float, p_home_win: float, *, rng: random.Random) -> str:
    """Return a knockout winner by removing draw mass and using penalty fallback for residual draws."""
    decisive_total = p_home_loss + p_home_win
    if decisive_total <= 0:
        return home_team if rng.random() < 0.5 else away_team
    p_home_advances = p_home_win / decisive_total
    return home_team if rng.random() < p_home_advances else away_team


def pair_bracket(teams: list[str]) -> list[tuple[str, str]]:
    ordered = [team for team in teams if team]
    return [(ordered[i], ordered[-(i + 1)]) for i in range(len(ordered) // 2)]
