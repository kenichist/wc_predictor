import pandas as pd

from src.data_sources import market_odds_workflow as workflow
from src.data_sources.merge_market_odds import FINAL_MARKET_ODDS_SCHEMA, merge_staged_market_odds


def test_refresh_market_odds_dry_run_does_not_merge(monkeypatch, tmp_path) -> None:
    staged = pd.DataFrame(
        [
            {
                "date": "2026-06-24",
                "home_team": "Argentina",
                "away_team": "France",
                "home_odds": 2.0,
                "draw_odds": 3.2,
                "away_odds": 4.0,
                "bookmaker": "api",
                "source": "unit",
                "updated_at": "2026-06-24",
                "api_provider": "api_football",
                "api_fixture_id": "1",
                "raw_home_team": "Argentina",
                "raw_away_team": "France",
                "raw_payload_file": "raw.json",
            }
        ]
    )

    def fake_fetch(provider: str, season: int):
        result = workflow.ProviderFetchResult(provider, tmp_path / f"{provider}.csv", 1, 1, 0, True)
        return result, staged

    def fail_merge(*args, **kwargs):
        raise AssertionError("dry-run should not merge")

    monkeypatch.setattr(workflow, "_fetch_provider_odds", fake_fetch)
    monkeypatch.setattr(workflow, "COMBINED_MARKET_ODDS_2026_PATH", tmp_path / "combined.csv")
    monkeypatch.setattr(workflow, "merge_staged_market_odds", fail_merge)
    monkeypatch.setattr(workflow, "active_fixture_coverage", lambda: {"total": 1, "matched": 0, "coverage": 0.0})
    monkeypatch.setattr(workflow, "generate_missing_odds_template", lambda: (tmp_path / "missing.csv", {"missing_fixtures": 1}))
    monkeypatch.setattr(workflow, "generate_feature_null_rate_report", lambda: None)
    monkeypatch.setattr(workflow, "generate_live_predictions", lambda: None)
    monkeypatch.setattr(workflow, "_read_market_odds", lambda: pd.DataFrame(columns=FINAL_MARKET_ODDS_SCHEMA))
    monkeypatch.setattr(workflow, "REFRESH_MARKET_ODDS_REPORT", tmp_path / "report.md")

    summary = workflow.refresh_market_odds(provider="api_football", dry_run=True)

    assert summary["STAGED_ROWS"] == 1
    assert summary["VALID_ROWS"] == 1
    assert summary["ROWS_ADDED"] == 0


def test_merge_preserves_unrelated_years_and_final_schema(tmp_path) -> None:
    market_path = tmp_path / "market_odds.csv"
    backup_path = tmp_path / "backup.csv"
    staged_path = tmp_path / "staged.csv"
    pd.DataFrame(
        [
            {
                "date": "2018-06-14",
                "home_team": "Russia",
                "away_team": "Saudi Arabia",
                "home_odds": 1.5,
                "draw_odds": 4.0,
                "away_odds": 7.0,
                "bookmaker": "existing",
                "source": "manual_verified",
                "updated_at": "2026-01-01",
            }
        ]
    ).to_csv(market_path, index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-06-24",
                "home_team": "Argentina",
                "away_team": "France",
                "home_odds": 2.0,
                "draw_odds": 3.2,
                "away_odds": 4.0,
                "bookmaker": "api",
                "source": "api_football:/odds",
                "updated_at": "2026-06-24",
                "api_provider": "api_football",
                "api_fixture_id": "1",
                "raw_home_team": "Argentina",
                "raw_away_team": "France",
                "raw_payload_file": "raw.json",
            }
        ]
    ).to_csv(staged_path, index=False)

    summary = merge_staged_market_odds(staged_path, market_odds_path=market_path, backup_path=backup_path)
    merged = pd.read_csv(market_path)

    assert summary["VALIDATION_PASSED"] is True
    assert len(merged) == 2
    assert "2018-06-14" in set(merged["date"])
    assert merged.columns.tolist() == FINAL_MARKET_ODDS_SCHEMA
    assert backup_path.exists()


def test_duplicate_fixture_prefer_api_replaces_existing(tmp_path) -> None:
    market_path = tmp_path / "market_odds.csv"
    backup_path = tmp_path / "backup.csv"
    staged_path = tmp_path / "staged.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-24",
                "home_team": "Argentina",
                "away_team": "France",
                "home_odds": 1.9,
                "draw_odds": 3.1,
                "away_odds": 4.1,
                "bookmaker": "existing",
                "source": "manual_verified",
                "updated_at": "2026-01-01",
            }
        ]
    ).to_csv(market_path, index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-06-24",
                "home_team": "Argentina",
                "away_team": "France",
                "home_odds": 2.0,
                "draw_odds": 3.2,
                "away_odds": 4.0,
                "bookmaker": "api",
                "source": "api_football:/odds",
                "updated_at": "2026-06-24",
                "api_provider": "api_football",
                "api_fixture_id": "1",
                "raw_home_team": "Argentina",
                "raw_away_team": "France",
                "raw_payload_file": "raw.json",
            }
        ]
    ).to_csv(staged_path, index=False)

    summary = merge_staged_market_odds(staged_path, market_odds_path=market_path, backup_path=backup_path, prefer_api=True)
    merged = pd.read_csv(market_path)

    assert summary["ROWS_REPLACED"] == 1
    assert merged.loc[0, "home_odds"] == 2.0


def test_api_football_refresh_uses_configured_league_ids(tmp_path) -> None:
    class FakeApiFootballClient:
        raw_dir = tmp_path

        def __init__(self) -> None:
            self.calls = []

        def get_odds(self, league: int | None = None, season: int | None = None):
            self.calls.append((league, season))
            return {
                "response": [
                    {
                        "fixture": {"id": 123, "date": "2026-06-24T19:00:00+00:00"},
                        "teams": {"home": {"name": "Argentina"}, "away": {"name": "France"}},
                        "bookmakers": [
                            {
                                "name": "api_football_test",
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
                            }
                        ],
                    }
                ]
            }

    client = FakeApiFootballClient()
    odds, _, candidates = workflow._fetch_api_football_odds_with_config(
        client,
        {"api_football": {"league_ids": [1, "2"]}},
        2026,
    )

    assert client.calls == [(1, 2026), (2, 2026)]
    assert len(odds) == 2
    assert candidates == [
        {"league_id": 1, "source": "config/odds_provider_keys.json"},
        {"league_id": 2, "source": "config/odds_provider_keys.json"},
    ]
