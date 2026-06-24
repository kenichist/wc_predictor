import pandas as pd

from src.data_sources.coverage import check_2010_odds_coverage, required_2010_matches_frame
from src.data_sources.validators import (
    summarize_market_odds_validation,
    validate_coverage_against_active_fixtures,
    validate_market_odds_values,
    validate_no_duplicate_match_rows,
)


def _staged_row(date="2026-06-11", home="Mexico", away="South Africa", home_odds=2.1):
    return {
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_odds": home_odds,
        "draw_odds": 3.1,
        "away_odds": 3.6,
        "bookmaker": "api_football_average",
        "source": "api_football:/odds",
        "updated_at": "2026-06-23T00:00:00+00:00",
        "api_provider": "api_football",
        "api_fixture_id": "1",
        "raw_home_team": home,
        "raw_away_team": away,
        "raw_payload_file": "raw.json",
    }


def test_market_odds_validation_detects_bad_odds() -> None:
    df = pd.DataFrame([_staged_row(home_odds=1.0)])

    values = validate_market_odds_values(df)
    summary = summarize_market_odds_validation(df)

    assert values["bad_odds"] == 1
    assert summary.safe_to_merge is False


def test_duplicate_date_home_away_detection_uses_normalized_names() -> None:
    df = pd.DataFrame([_staged_row(home="United States"), _staged_row(home="USA")])

    assert validate_no_duplicate_match_rows(df) == 2


def test_coverage_against_active_fixtures_reports_missing_rows() -> None:
    staged = pd.DataFrame([_staged_row()])
    fixtures = pd.DataFrame(
        [
            {"date": "2026-06-11", "home_team": "Mexico", "away_team": "South Africa"},
            {"date": "2026-06-12", "home_team": "United States", "away_team": "South Korea"},
        ]
    )

    coverage = validate_coverage_against_active_fixtures(staged, fixtures)

    assert coverage["matched_fixtures"] == 1
    assert coverage["total_fixtures"] == 2
    assert coverage["coverage"] == 0.5
    assert len(coverage["missing_fixtures"]) == 1


def test_2010_coverage_checker_counts_required_matches(tmp_path) -> None:
    required = required_2010_matches_frame()
    odds = required.head(2).assign(
        home_odds=2.0,
        draw_odds=3.0,
        away_odds=4.0,
        bookmaker="manual",
        source="unit",
        updated_at="2026-06-23",
    )
    odds_path = tmp_path / "market_odds.csv"
    report_path = tmp_path / "market_odds_2010_coverage_report.md"
    odds.to_csv(odds_path, index=False)

    _, summary = check_2010_odds_coverage(odds_path, report_path=report_path)

    assert summary["rows_present"] == 2
    assert summary["rows_missing"] == 62
    assert report_path.exists()
