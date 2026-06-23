import pandas as pd

from src.sources.world_football_elo import (
    WORLD_FOOTBALL_ELO_SCHEMA,
    join_world_football_elo_asof,
    prepare_world_football_elo,
    validate_world_football_elo,
)


def test_world_football_elo_prepare_parses_and_normalizes(tmp_path) -> None:
    raw_path = tmp_path / "raw_elo.csv"
    output_path = tmp_path / "world_football_elo.csv"
    pd.DataFrame(
        [
            {"date": "2024-12-31", "team": "Congo DR", "elo": "1720", "source": "test"},
            {"date": "2024-12-31", "team": "Islamic Republic of Iran", "elo": 1810, "source": "test"},
        ]
    ).to_csv(raw_path, index=False)

    prepared = prepare_world_football_elo(raw_path, output_path)

    assert list(prepared.columns) == WORLD_FOOTBALL_ELO_SCHEMA
    assert "DR Congo" in set(prepared["team"])
    assert "Iran" in set(prepared["team"])
    assert prepared["retrieved_at"].notna().all()
    assert output_path.exists()


def test_world_football_elo_validation_errors_are_useful() -> None:
    bad_schema = validate_world_football_elo(pd.DataFrame([{"date": "2024-01-01", "team": "France"}]))
    duplicates = validate_world_football_elo(
        pd.DataFrame(
            [
                {"date": "2024-01-01", "team": "Korea Republic", "elo": 1800},
                {"date": "2024-01-01", "team": "South Korea", "elo": 1801},
            ]
        )
    )

    assert "missing required columns" in bad_schema[0]
    assert any("duplicate date/team" in error for error in duplicates)


def test_external_elo_join_uses_only_rows_before_match_date_and_computes_changes() -> None:
    elo = pd.DataFrame(
        [
            {"date": "2024-01-01", "team": "Korea Republic", "elo": 1800, "source": "test"},
            {"date": "2024-12-31", "team": "South Korea", "elo": 1825, "source": "test"},
            {"date": "2024-01-01", "team": "France", "elo": 1990, "source": "test"},
        ]
    )
    matches = pd.DataFrame(
        [
            {"date": "2024-12-31", "home_team": "South Korea", "away_team": "France"},
            {"date": "2025-01-01", "home_team": "South Korea", "away_team": "France"},
        ]
    )

    joined = join_world_football_elo_asof(matches, elo)

    assert joined.loc[0, "home_external_elo"] == 1800
    assert joined.loc[0, "home_external_elo_date"] == pd.Timestamp("2024-01-01")
    assert joined.loc[1, "home_external_elo"] == 1825
    assert joined.loc[1, "home_external_elo_change"] == 25
