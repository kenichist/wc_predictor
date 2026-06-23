import pandas as pd

from src.sources.fifa_rankings import (
    FIFA_RANKINGS_SCHEMA,
    join_fifa_rankings_asof,
    prepare_fifa_rankings,
    validate_fifa_rankings,
)


def test_fifa_rankings_prepare_parses_and_normalizes(tmp_path) -> None:
    raw_path = tmp_path / "raw_fifa.csv"
    output_path = tmp_path / "fifa_rankings.csv"
    pd.DataFrame(
        [
            {"date": "2024-12-19", "team": "USA", "rank": "18", "points": "1630.5", "source": "test"},
            {"date": "2024-11-28", "team": "Korea Republic", "rank": 23, "points": 1585.2, "source": "test"},
        ]
    ).to_csv(raw_path, index=False)

    prepared = prepare_fifa_rankings(raw_path, output_path)

    assert list(prepared.columns) == FIFA_RANKINGS_SCHEMA
    assert prepared.loc[prepared["team"].eq("United States"), "rank"].iloc[0] == 18
    assert "South Korea" in set(prepared["team"])
    assert prepared["retrieved_at"].notna().all()
    assert output_path.exists()


def test_fifa_rankings_validation_errors_are_useful() -> None:
    missing_errors = validate_fifa_rankings(pd.DataFrame([{"date": "2024-01-01", "team": "USA", "points": 1800}]))
    duplicate_errors = validate_fifa_rankings(
        pd.DataFrame(
            [
                {"date": "2024-01-01", "team": "USA", "rank": 1, "points": 1800},
                {"date": "2024-01-01", "team": "United States", "rank": 1, "points": 1800},
            ]
        )
    )

    assert "missing required columns" in missing_errors[0]
    assert any("duplicate date/team" in error for error in duplicate_errors)


def test_fifa_rankings_points_are_optional() -> None:
    errors = validate_fifa_rankings(pd.DataFrame([{"date": "2024-01-01", "team": "USA", "rank": 1}]))

    assert errors == []


def test_fifa_join_uses_only_rows_before_match_date_and_computes_changes() -> None:
    rankings = pd.DataFrame(
        [
            {"date": "2024-11-28", "team": "USA", "rank": 20, "points": 1500, "source": "test"},
            {"date": "2024-12-19", "team": "United States", "rank": 18, "points": 1530, "source": "test"},
            {"date": "2024-11-28", "team": "IR Iran", "rank": 21, "points": 1490, "source": "test"},
        ]
    )
    matches = pd.DataFrame(
        [
            {"date": "2024-12-19", "home_team": "United States", "away_team": "Iran"},
            {"date": "2024-12-20", "home_team": "United States", "away_team": "Iran"},
        ]
    )

    joined = join_fifa_rankings_asof(matches, rankings)

    assert joined.loc[0, "home_fifa_rank"] == 20
    assert joined.loc[0, "home_fifa_ranking_date"] == pd.Timestamp("2024-11-28")
    assert joined.loc[1, "home_fifa_rank"] == 18
    assert joined.loc[1, "home_fifa_rank_change"] == -2
    assert joined.loc[1, "home_fifa_points_change"] == 30
