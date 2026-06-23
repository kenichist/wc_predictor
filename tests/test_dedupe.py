import pandas as pd

from src.dedupe import deduplicate_matches


def test_deduplicate_matches_respects_source_priority() -> None:
    df = pd.DataFrame(
        [
            {
                "match_id": "json_1",
                "source": "worldcup_json",
                "date": "2026-06-11",
                "home_team": "Mexico",
                "away_team": "Canada",
                "home_score": 2,
                "away_score": 1,
                "tournament": "FIFA World Cup",
                "competition_type": "world_cup",
            },
            {
                "match_id": "api_1",
                "source": "football-data",
                "date": "2026-06-11",
                "home_team": "Mexico",
                "away_team": "Canada",
                "home_score": 2,
                "away_score": 1,
                "tournament": "FIFA World Cup",
                "competition_type": "world_cup",
            },
        ]
    )

    result = deduplicate_matches(df)

    assert len(result) == 1
    assert result.loc[0, "source"] == "football-data"
    assert result.loc[0, "source_priority"] == 1
    assert result.loc[0, "duplicate_group_id"].startswith("dup_")
