import pandas as pd

from dashboard.live_api import combine_fixtures_with_scores, enrich_live_odds_from_fixtures, parse_api_football_fixtures
from dashboard.parsers import markdown_table_after_heading


def test_markdown_parser_missing_section_returns_empty_frame() -> None:
    table = markdown_table_after_heading("# Report\n\nNo table here.", "Missing Section")

    assert table.empty


def test_parse_api_football_fixtures_normalizes_live_rows() -> None:
    payload = {
        "response": [
            {
                "fixture": {
                    "id": 123,
                    "date": "2026-06-11T19:00:00+00:00",
                    "status": {"long": "First Half", "elapsed": 31},
                    "venue": {"name": "Test Stadium"},
                },
                "league": {"round": "Group A"},
                "teams": {"home": {"name": "USA"}, "away": {"name": "Korea Republic"}},
                "goals": {"home": 1, "away": 0},
            }
        ]
    }

    fixtures, scores = parse_api_football_fixtures(payload)

    assert fixtures.loc[0, "home_team"] == "United States"
    assert fixtures.loc[0, "away_team"] == "South Korea"
    assert fixtures.loc[0, "minute"] == 31
    assert scores.loc[0, "home_score"] == 1
    assert scores.loc[0, "away_score"] == 0


def test_enrich_live_odds_from_fixtures_fills_missing_teams() -> None:
    odds = pd.DataFrame(
        [
            {
                "fixture_id": 123,
                "date": "2026-06-11",
                "home_team": pd.NA,
                "away_team": pd.NA,
                "home_odds": 2.0,
                "draw_odds": 3.5,
                "away_odds": 4.0,
                "bookmaker": "api_football_average",
                "source": "unit",
                "updated_at": "2026-06-24",
                "api_provider": "api_football",
            }
        ]
    )
    fixtures = pd.DataFrame(
        [{"fixture_id": 123, "date": "2026-06-11", "home_team": "Argentina", "away_team": "France"}]
    )

    enriched = enrich_live_odds_from_fixtures(odds, fixtures)

    assert enriched.loc[0, "home_team"] == "Argentina"
    assert enriched.loc[0, "away_team"] == "France"


def test_combine_fixtures_with_scores_includes_finished_score_rows() -> None:
    fixtures = pd.DataFrame(
        [{"fixture_id": "upcoming", "date": "2026-06-12", "home_team": "Brazil", "away_team": "Spain", "status": "Not Started"}]
    )
    scores = pd.DataFrame(
        [{"fixture_id": "finished", "date": "2026-06-11", "home_team": "Argentina", "away_team": "France", "status": "Match Finished", "home_score": 2, "away_score": 1}]
    )

    combined = combine_fixtures_with_scores(fixtures, scores)

    assert set(combined["fixture_id"]) == {"upcoming", "finished"}
    finished = combined[combined["fixture_id"].eq("finished")].iloc[0]
    assert finished["status"] == "Match Finished"
    assert finished["home_team"] == "Argentina"
