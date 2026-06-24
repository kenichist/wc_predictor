import pandas as pd

from src.data_sources import market_odds_workflow as workflow


def test_missing_odds_template_includes_only_uncovered_active_fixtures(monkeypatch, tmp_path) -> None:
    active = pd.DataFrame(
        [
            {"date": "2026-06-24", "home_team": "Argentina", "away_team": "France"},
            {"date": "2026-06-25", "home_team": "Brazil", "away_team": "Spain"},
        ]
    )
    market = pd.DataFrame(
        [
            {
                "date": "2026-06-24",
                "home_team": "Argentina",
                "away_team": "France",
                "home_odds": 2.0,
                "draw_odds": 3.2,
                "away_odds": 3.8,
                "bookmaker": "verified",
                "source": "manual_verified",
                "updated_at": "2026-06-24",
            }
        ]
    )
    monkeypatch.setattr(workflow, "load_active_live_fixtures", lambda: active)
    monkeypatch.setattr(workflow, "_read_market_odds", lambda: market)

    output_path, summary = workflow.generate_missing_odds_template(
        output_path=tmp_path / "missing.csv",
        report_path=tmp_path / "report.md",
    )
    template = pd.read_csv(output_path)

    assert summary["total_active_fixtures"] == 2
    assert summary["odds_covered_fixtures"] == 1
    assert summary["missing_fixtures"] == 1
    assert template.loc[0, "home_team"] == "Brazil"
    assert template.loc[0, "away_team"] == "Spain"
    assert template.loc[0, "bookmaker"] == "manual_market_average"
