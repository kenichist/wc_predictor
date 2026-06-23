import pandas as pd

from src.sources.external_rating_acquisition import (
    acquire_fifa_rankings,
    parse_ranking_table,
    validate_rating_quality,
)
from src.sources.fifa_rankings import add_fifa_ranking_features


def _rating_rows(n: int = 32, **overrides):
    rows = []
    for idx in range(n):
        row = {
            "release_date": "2024-12-19",
            "country": f"Team {idx:02d}",
            "position": idx + 1,
            "total_points": 1800 - idx,
        }
        row.update(overrides)
        rows.append(row)
    return rows


def _cfg(tmp_path, fifa_path=None):
    return {
        "external_sources": {
            "fifa_rankings": {
                "output_path": str(fifa_path or tmp_path / "fifa_rankings.csv"),
                "raw_dir": str(tmp_path / "raw" / "fifa_rankings"),
                "source_type": "auto",
            }
        },
        "paths": {
            "reports_dir": str(tmp_path / "reports"),
            "feature_null_rate_report_csv": str(tmp_path / "feature_null_rate_report.csv"),
            "ablation_results_csv": str(tmp_path / "ablation_results.csv"),
            "external_rating_acquisition_report_csv": str(tmp_path / "acquisition.csv"),
            "external_rating_acquisition_report_md": str(tmp_path / "acquisition.md"),
            "external_rating_activation_report_csv": str(tmp_path / "activation.csv"),
            "external_rating_activation_report_md": str(tmp_path / "activation.md"),
        },
    }


def test_bad_schema_does_not_overwrite_existing_valid_file(tmp_path) -> None:
    output_path = tmp_path / "fifa_rankings.csv"
    existing = pd.DataFrame(_rating_rows()).rename(
        columns={"release_date": "date", "country": "team", "position": "rank", "total_points": "points"}
    )
    existing["source"] = "existing"
    existing["retrieved_at"] = "2026-01-01T00:00:00Z"
    existing.to_csv(output_path, index=False)
    bad_input = tmp_path / "bad.csv"
    pd.DataFrame([{"team": "United States", "rank": 1}]).to_csv(bad_input, index=False)

    result = acquire_fifa_rankings(input_path=bad_input, config=_cfg(tmp_path, output_path), run_post_checks=False)

    assert result.status == "failed"
    assert result.preserved_existing is True
    preserved = pd.read_csv(output_path)
    assert set(preserved["source"]) == {"existing"}


def test_quality_gate_fails_if_fewer_than_30_teams() -> None:
    parsed = parse_ranking_table(pd.DataFrame(_rating_rows(12)), "test_source")

    quality = validate_rating_quality(parsed, dataset="fifa_rankings", config={"paths": {}})

    assert not quality.ok
    assert any("at least 30 teams" in error for error in quality.errors)


def test_quality_gate_detects_low_worldcup_coverage(tmp_path) -> None:
    wc_path = tmp_path / "worldcup.parquet"
    pd.DataFrame(
        [
            {"home_team": "United States", "away_team": "France"},
            {"home_team": "Spain", "away_team": "Argentina"},
            {"home_team": "Brazil", "away_team": "England"},
        ]
    ).to_parquet(wc_path, index=False)
    rows = _rating_rows()
    rows[0]["country"] = "United States"
    parsed = parse_ranking_table(pd.DataFrame(rows), "test_source")

    quality = validate_rating_quality(
        parsed,
        dataset="fifa_rankings",
        config={"paths": {"worldcup_2026_prediction_input_advanced": str(wc_path)}},
    )

    assert not quality.ok
    assert any("World Cup 2026 team coverage" in error for error in quality.errors)


def test_duplicate_date_team_rows_are_detected() -> None:
    rows = [{"date": "2024-12-19", "team": "USA", "rank": 1, "points": 1800}]
    rows.extend(
        {"date": "2024-12-19", "team": f"Team {idx:02d}", "rank": idx + 2, "points": 1700 - idx}
        for idx in range(30)
    )
    rows.append({"date": "2024-12-19", "team": "United States", "rank": 1, "points": 1800})
    parsed = parse_ranking_table(
        pd.DataFrame(rows),
        "test_source",
    )
    quality = validate_rating_quality(parsed, dataset="fifa_rankings", config={"paths": {}})

    assert quality.ok
    assert any("duplicate date/team" in warning for warning in quality.warnings)


def test_existing_valid_file_is_preserved_when_download_fails(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "fifa_rankings.csv"
    existing = pd.DataFrame(_rating_rows()).rename(
        columns={"release_date": "date", "country": "team", "position": "rank", "total_points": "points"}
    )
    existing["source"] = "existing"
    existing["retrieved_at"] = "2026-01-01T00:00:00Z"
    existing.to_csv(output_path, index=False)

    def fail_download(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("src.sources.external_rating_acquisition.download_file", fail_download)

    result = acquire_fifa_rankings(url="https://example.invalid/fifa.csv", config=_cfg(tmp_path, output_path), run_post_checks=False)

    assert result.status == "failed"
    assert result.preserved_existing is True
    assert set(pd.read_csv(output_path)["source"]) == {"existing"}


def test_asof_join_still_prevents_future_leakage_after_acquisition_module_changes(tmp_path) -> None:
    path = tmp_path / "fifa_rankings.csv"
    pd.DataFrame(
        [
            {"date": "2024-01-01", "team": "USA", "rank": 20, "points": 1500},
            {"date": "2024-02-01", "team": "USA", "rank": 1, "points": 1900},
            {"date": "2024-01-01", "team": "Mexico", "rank": 15, "points": 1550},
        ]
    ).to_csv(path, index=False)

    result = add_fifa_ranking_features(
        pd.DataFrame([{"date": "2024-02-01", "home_team": "United States", "away_team": "Mexico"}]),
        path,
    )

    assert result.loc[0, "home_fifa_rank"] == 20
