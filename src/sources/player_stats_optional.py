from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def soccerdata_available() -> bool:
    try:
        import soccerdata  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class PlayerStatsRequest:
    team: str
    season: str
    competition: str | None = None


class OptionalPlayerStatsCollector:
    """Interface stub for future player, squad, and availability features."""

    def __init__(self) -> None:
        self.available = soccerdata_available()

    def collect_player_minutes(self, request: PlayerStatsRequest) -> pd.DataFrame:
        if not self.available:
            return _empty_player_stats("player_minutes")
        return _not_implemented_stub(request, "player_minutes")

    def collect_attacking_stats(self, request: PlayerStatsRequest) -> pd.DataFrame:
        if not self.available:
            return _empty_player_stats("attacking_stats")
        return _not_implemented_stub(request, "attacking_stats")

    def collect_squad_market_values(self, request: PlayerStatsRequest) -> pd.DataFrame:
        if not self.available:
            return _empty_player_stats("squad_market_value")
        return _not_implemented_stub(request, "squad_market_value")

    def collect_availability_placeholders(self, team: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "team": team,
                    "player": pd.NA,
                    "injury_status": pd.NA,
                    "suspension_status": pd.NA,
                    "source": "placeholder",
                }
            ]
        )


def _empty_player_stats(feature_family: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["team", "player", "season", "competition", "feature_family", "value"]).assign(feature_family=feature_family)


def _not_implemented_stub(request: PlayerStatsRequest, feature_family: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": request.team,
                "player": pd.NA,
                "season": request.season,
                "competition": request.competition,
                "feature_family": feature_family,
                "value": pd.NA,
                "source": "soccerdata_stub",
            }
        ]
    )
