import pandas as pd

from src.data_sources.injury_transform import api_football_injuries_to_rows
from src.data_sources.odds_transform import api_football_odds_to_market_rows, the_odds_api_to_market_rows
from src.data_sources.validators import validate_injury_schema


def test_api_football_match_winner_odds_average_to_market_schema() -> None:
    payload = {
        "response": [
            {
                "fixture": {"id": 10, "date": "2026-06-11T19:00:00+00:00"},
                "teams": {"home": {"name": "USA"}, "away": {"name": "South Korea"}},
                "bookmakers": [
                    {
                        "name": "Book A",
                        "bets": [
                            {
                                "name": "Match Winner",
                                "values": [
                                    {"value": "Home", "odd": "2.00"},
                                    {"value": "Draw", "odd": "3.20"},
                                    {"value": "Away", "odd": "4.00"},
                                ],
                            }
                        ],
                    },
                    {
                        "name": "Book B",
                        "bets": [
                            {
                                "name": "1X2",
                                "values": [
                                    {"value": "Home", "odd": "2.20"},
                                    {"value": "Draw", "odd": "3.40"},
                                    {"value": "Away", "odd": "3.80"},
                                ],
                            }
                        ],
                    },
                ],
            }
        ]
    }

    df = api_football_odds_to_market_rows(payload, raw_payload_file="raw.json")

    assert len(df) == 1
    assert df.loc[0, "home_team"] == "United States"
    assert df.loc[0, "away_team"] == "South Korea"
    assert df.loc[0, "home_odds"] == 2.1
    assert df.loc[0, "bookmaker"] == "api_football_average"
    assert df.loc[0, "api_provider"] == "api_football"


def test_the_odds_api_h2h_requires_draw_and_extracts_prices() -> None:
    payload = [
        {
            "id": "evt1",
            "commence_time": "2026-06-12T20:00:00Z",
            "home_team": "Netherlands",
            "away_team": "Cabo Verde",
            "bookmakers": [
                {
                    "title": "Book",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Netherlands", "price": 1.7},
                                {"name": "Draw", "price": 3.6},
                                {"name": "Cape Verde", "price": 5.8},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    df = the_odds_api_to_market_rows(payload)

    assert len(df) == 1
    assert df.loc[0, "away_team"] == "Cape Verde"
    assert df.loc[0, "draw_odds"] == 3.6


def test_the_odds_api_h2h_without_draw_is_skipped() -> None:
    payload = [
        {
            "id": "evt1",
            "commence_time": "2026-06-12T20:00:00Z",
            "home_team": "Netherlands",
            "away_team": "Cape Verde",
            "bookmakers": [
                {
                    "title": "Book",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Netherlands", "price": 1.7},
                                {"name": "Cape Verde", "price": 5.8},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    df = the_odds_api_to_market_rows(payload)

    assert df.empty


def test_api_football_injuries_are_scenario_only() -> None:
    payload = {
        "response": [
            {
                "fixture": {"id": 1, "date": "2026-06-11"},
                "team": {"name": "USA"},
                "player": {"name": "Example Player"},
                "type": "Suspended",
                "reason": "Cards",
            }
        ]
    }

    df = api_football_injuries_to_rows(payload)

    assert validate_injury_schema(df) == []
    assert df.loc[0, "team"] == "United States"
    assert df.loc[0, "status"] == "suspended"
    assert bool(df.loc[0, "scenario_only"]) is True
