import pandas as pd

from src.data_sources import market_odds_workflow as workflow


def test_invalid_odds_are_rejected(monkeypatch, tmp_path) -> None:
    active = pd.DataFrame([{"date": "2026-06-24", "home_team": "Argentina", "away_team": "France"}])
    monkeypatch.setattr(workflow, "load_active_live_fixtures", lambda: active)
    monkeypatch.setattr(workflow, "MANUAL_MARKET_ODDS_VALIDATED_PATH", tmp_path / "validated.csv")
    monkeypatch.setattr(workflow, "MANUAL_MARKET_ODDS_VALIDATION_ERRORS", tmp_path / "errors.md")
    input_path = tmp_path / "manual.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-24",
                "home_team": "Argentina",
                "away_team": "France",
                "home_odds": 1.0,
                "draw_odds": 3.2,
                "away_odds": 4.0,
                "bookmaker": "manual_market_average",
                "source": "manual_verified",
                "updated_at": "2026-06-24",
                "notes": "",
            }
        ]
    ).to_csv(input_path, index=False)

    summary = workflow.validate_manual_market_odds(input_path)

    assert summary["SUCCESS"] is False
    assert summary["ERRORS"] > 0
    assert (tmp_path / "errors.md").exists()


def test_rows_without_draw_odds_are_ignored_not_validated(monkeypatch, tmp_path) -> None:
    active = pd.DataFrame([{"date": "2026-06-24", "home_team": "Argentina", "away_team": "France"}])
    monkeypatch.setattr(workflow, "load_active_live_fixtures", lambda: active)
    monkeypatch.setattr(workflow, "MANUAL_MARKET_ODDS_VALIDATED_PATH", tmp_path / "validated.csv")
    monkeypatch.setattr(workflow, "MANUAL_MARKET_ODDS_VALIDATION_ERRORS", tmp_path / "errors.md")
    input_path = tmp_path / "manual.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-24",
                "home_team": "Argentina",
                "away_team": "France",
                "home_odds": 2.0,
                "draw_odds": "",
                "away_odds": 4.0,
                "bookmaker": "manual_market_average",
                "source": "manual_verified",
                "updated_at": "2026-06-24",
                "notes": "",
            }
        ]
    ).to_csv(input_path, index=False)

    summary = workflow.validate_manual_market_odds(input_path)

    assert summary["SUCCESS"] is True
    assert summary["FILLED_ROWS"] == 0
    assert summary["VALID_ROWS"] == 0


def test_valid_manual_rows_are_written(monkeypatch, tmp_path) -> None:
    active = pd.DataFrame([{"date": "2026-06-24", "home_team": "Argentina", "away_team": "France"}])
    output = tmp_path / "validated.csv"
    monkeypatch.setattr(workflow, "load_active_live_fixtures", lambda: active)
    monkeypatch.setattr(workflow, "MANUAL_MARKET_ODDS_VALIDATED_PATH", output)
    monkeypatch.setattr(workflow, "MANUAL_MARKET_ODDS_VALIDATION_ERRORS", tmp_path / "errors.md")
    input_path = tmp_path / "manual.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-24",
                "home_team": "Argentina",
                "away_team": "France",
                "home_odds": 2.0,
                "draw_odds": 3.2,
                "away_odds": 4.0,
                "bookmaker": "manual_market_average",
                "source": "manual_verified",
                "updated_at": "2026-06-24",
                "notes": "",
            }
        ]
    ).to_csv(input_path, index=False)

    summary = workflow.validate_manual_market_odds(input_path)

    assert summary["SUCCESS"] is True
    assert summary["VALID_ROWS"] == 1
    assert output.exists()
