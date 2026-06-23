import pandas as pd

from src.features.external_feature_joiner import add_market_features
from src.reports.external_feature_activation_debug import write_external_feature_activation_debug_report
from src.sources.injury_interface import add_injury_features
from src.sources.player_stats_interface import add_squad_player_features


def test_squad_player_features_with_only_team_and_market_value_activate(tmp_path) -> None:
    path = tmp_path / "squad_player_features.csv"
    pd.DataFrame(
        [
            {"team": "USA", "squad_market_value": 250.0},
            {"team": "England", "squad_market_value": 1360.0},
        ]
    ).to_csv(path, index=False)
    matches = pd.DataFrame(
        [{"match_id": "m1", "date": "2026-06-11", "home_team": "United States", "away_team": "England"}]
    )

    result = add_squad_player_features(matches, path)

    assert result.loc[0, "home_squad_market_value"] == 250.0
    assert result.loc[0, "away_squad_market_value"] == 1360.0
    assert result.loc[0, "squad_market_value_diff"] == -1110.0


def test_blank_optional_squad_columns_do_not_discard_group(tmp_path) -> None:
    path = tmp_path / "squad_player_features.csv"
    pd.DataFrame(
        [
            {
                "team": "France",
                "squad_market_value": 1520.0,
                "top_5_player_rating_avg": "",
                "club_minutes_recent_90_days": "",
            }
        ]
    ).to_csv(path, index=False)
    matches = pd.DataFrame([{"match_id": "m1", "date": "2026-06-11", "home_team": "France", "away_team": "France"}])

    result = add_squad_player_features(matches, path)

    assert result.loc[0, "home_squad_market_value"] == 1520.0
    assert pd.isna(result.loc[0, "home_top_5_player_rating_avg"])


def test_aggregate_injuries_activate_worldcup_prediction_rows(tmp_path) -> None:
    path = tmp_path / "injuries_suspensions.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-22",
                "team": "France",
                "missing_starters_count": 1,
                "missing_key_players_count": 1,
                "injury_impact_score": 0.75,
                "suspension_impact_score": 0.0,
                "goalkeeper_missing": 0,
                "captain_missing": 1,
                "minutes_lost_from_expected_xi": "",
            },
            {
                "date": "2026-06-22",
                "team": "England",
                "missing_starters_count": 0,
                "missing_key_players_count": 0,
                "injury_impact_score": 0.25,
                "suspension_impact_score": 0.0,
                "goalkeeper_missing": 0,
                "captain_missing": 0,
                "minutes_lost_from_expected_xi": "",
            },
        ]
    ).to_csv(path, index=False)
    matches = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2026-06-11",
                "home_team": "France",
                "away_team": "England",
                "_advanced_split": "prediction",
            }
        ]
    )

    result = add_injury_features(matches, path)

    assert result.loc[0, "home_missing_starters_count"] == 1
    assert result.loc[0, "home_missing_key_players_count"] == 1
    assert result.loc[0, "home_injury_impact_score"] == 0.75
    assert result.loc[0, "home_suspension_impact_score"] == 0.0
    assert result.loc[0, "home_goalkeeper_missing"] == 0
    assert result.loc[0, "home_captain_missing"] == 1


def test_injury_team_names_are_normalized_before_joining(tmp_path) -> None:
    path = tmp_path / "injuries_suspensions.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-22",
                "team": "USA",
                "missing_starters_count": 2,
                "missing_key_players_count": 1,
                "injury_impact_score": 1.5,
                "suspension_impact_score": 0.5,
                "goalkeeper_missing": 1,
                "captain_missing": 0,
            }
        ]
    ).to_csv(path, index=False)
    matches = pd.DataFrame(
        [{"match_id": "m1", "date": "2026-06-11", "home_team": "United States", "away_team": "United States", "_advanced_split": "prediction"}]
    )

    result = add_injury_features(matches, path)

    assert result.loc[0, "home_missing_starters_count"] == 2
    assert result.loc[0, "away_goalkeeper_missing"] == 1


def test_empty_market_odds_does_not_crash_pipeline(tmp_path) -> None:
    path = tmp_path / "market_odds.csv"
    path.write_text("date,home_team,away_team,home_odds,draw_odds,away_odds,bookmaker,source,updated_at\n", encoding="utf-8")
    matches = pd.DataFrame([{"match_id": "m1", "date": "2026-06-11", "home_team": "A", "away_team": "B"}])

    result = add_market_features(matches, path)

    assert {"home_odds", "draw_odds", "away_odds", "market_home_prob", "market_draw_prob", "market_away_prob"} <= set(result.columns)
    assert result["home_odds"].isna().all()


def test_external_feature_activation_debug_report_is_created(tmp_path) -> None:
    squad_path = tmp_path / "squad_player_features.csv"
    injury_path = tmp_path / "injuries_suspensions.csv"
    market_path = tmp_path / "market_odds.csv"
    report_path = tmp_path / "external_feature_activation_debug.md"
    pd.DataFrame([{"team": "France", "squad_market_value": 1520.0}]).to_csv(squad_path, index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-06-22",
                "team": "France",
                "missing_starters_count": 0,
                "missing_key_players_count": 0,
                "injury_impact_score": 0.25,
                "suspension_impact_score": 0.0,
                "goalkeeper_missing": 0,
                "captain_missing": 0,
            }
        ]
    ).to_csv(injury_path, index=False)
    market_path.write_text("date,home_team,away_team,home_odds,draw_odds,away_odds,bookmaker,source,updated_at\n", encoding="utf-8")
    prediction = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "date": "2026-06-11",
                "home_team": "France",
                "away_team": "France",
                "home_squad_market_value": 1520.0,
                "away_squad_market_value": 1520.0,
                "home_missing_starters_count": 0.0,
                "away_missing_starters_count": 0.0,
            }
        ]
    )
    cfg = {
        "paths": {
            "squad_player_features": str(squad_path),
            "injuries_suspensions": str(injury_path),
            "betting_odds": str(market_path),
            "external_feature_activation_debug_md": str(report_path),
        }
    }

    path = write_external_feature_activation_debug_report(config=cfg, advanced_training=pd.DataFrame(), advanced_prediction=prediction)

    assert path == report_path
    text = path.read_text(encoding="utf-8")
    assert "External Feature Activation Debug" in text
    assert "Matched teams" in text
    assert "home_squad_market_value" in text
