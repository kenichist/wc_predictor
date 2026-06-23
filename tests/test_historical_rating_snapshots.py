from pathlib import Path

import pandas as pd

import src.sources.historical_rating_snapshots as snapshots
from src.normalize import normalize_team_name
from src.sources.historical_rating_snapshots import (
    import_fifa_ranking_snapshots,
    import_historical_rating_snapshots,
    import_world_football_elo_snapshots,
)
from src.sources.fifa_rankings import join_fifa_rankings_asof


def _cfg(tmp_path: Path, fifa_output: Path | None = None, elo_output: Path | None = None) -> dict:
    return {
        "paths": {
            "fifa_rankings": str(fifa_output or tmp_path / "fifa_rankings.csv"),
            "world_football_elo": str(elo_output or tmp_path / "world_football_elo.csv"),
            "historical_rating_coverage_report_csv": str(tmp_path / "reports" / "historical_rating_coverage_report.csv"),
            "historical_rating_coverage_report_md": str(tmp_path / "reports" / "historical_rating_coverage_report.md"),
        }
    }


def _teams(n: int = 32) -> list[str]:
    base = [
        "USA",
        "Korea Republic",
        "IR Iran",
        "Congo DR",
        "Cabo Verde",
        "Curacao",
        "Czech Republic",
        "Cote d'Ivoire",
        "Turkiye",
        "Bosnia and Herzegovina",
        "South Africa",
        "Saudi Arabia",
        "New Zealand",
    ]
    base.extend(f"Team {idx:02d}" for idx in range(len(base), n))
    return base[:n]


def _fifa_rows(n: int = 32, date: str | None = None) -> list[dict]:
    rows = []
    for idx, team in enumerate(_teams(n), start=1):
        row = {"team": team, "rank": idx, "points": 1900 - idx}
        if date is not None:
            row["date"] = date
        rows.append(row)
    return rows


def _elo_rows(n: int = 32, date: str | None = None) -> list[dict]:
    rows = []
    for idx, team in enumerate(_teams(n), start=1):
        row = {"team": team, "elo": 2100 - idx}
        if date is not None:
            row["date"] = date
        rows.append(row)
    return rows


def _write_existing_fifa(path: Path) -> None:
    existing = pd.DataFrame(_fifa_rows(date="2026-06-11"))
    existing["source"] = "existing"
    existing["retrieved_at"] = "2026-06-23T00:00:00Z"
    existing.to_csv(path, index=False)


def _write_existing_elo(path: Path) -> None:
    existing = pd.DataFrame(_elo_rows(date="2026-06-22"))
    existing["source"] = "existing"
    existing["retrieved_at"] = "2026-06-23T00:00:00Z"
    existing.to_csv(path, index=False)


def test_import_multiple_fifa_snapshot_csv_files(tmp_path) -> None:
    input_dir = tmp_path / "fifa_snapshots"
    input_dir.mkdir()
    pd.DataFrame(_fifa_rows()).to_csv(input_dir / "fifa_2010-05-26.csv", index=False)
    pd.DataFrame(_fifa_rows()).to_csv(input_dir / "fifa_2014-06-05.csv", index=False)
    output = tmp_path / "fifa_rankings.csv"

    imported = import_fifa_ranking_snapshots(input_dir, output, config=_cfg(tmp_path, fifa_output=output))

    assert output.exists()
    assert imported["date"].nunique() == 2
    assert len(imported) == 64
    assert set(imported["source"]) == {"snapshot:fifa_2010-05-26.csv", "snapshot:fifa_2014-06-05.csv"}


def test_import_multiple_elo_snapshot_csv_files(tmp_path) -> None:
    input_dir = tmp_path / "elo_snapshots"
    input_dir.mkdir()
    pd.DataFrame(_elo_rows()).to_csv(input_dir / "elo_2010-06-10.csv", index=False)
    pd.DataFrame(_elo_rows()).to_csv(input_dir / "elo_2014-06-12.csv", index=False)
    output = tmp_path / "world_football_elo.csv"

    imported = import_world_football_elo_snapshots(input_dir, output, config=_cfg(tmp_path, elo_output=output))

    assert output.exists()
    assert imported["date"].nunique() == 2
    assert len(imported) == 64


def test_infers_date_from_supported_filename(tmp_path) -> None:
    input_dir = tmp_path / "fifa_snapshots"
    input_dir.mkdir()
    pd.DataFrame(_fifa_rows()).to_csv(input_dir / "fifa_2018_06_07.csv", index=False)
    output = tmp_path / "fifa_rankings.csv"

    imported = import_fifa_ranking_snapshots(input_dir, output, config=_cfg(tmp_path, fifa_output=output))

    assert imported["date"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2018-06-07"]


def test_prefers_date_column_over_filename_date(tmp_path) -> None:
    input_dir = tmp_path / "fifa_snapshots"
    input_dir.mkdir()
    pd.DataFrame(_fifa_rows(date="2019-01-01")).to_csv(input_dir / "fifa_2018-06-07.csv", index=False)
    output = tmp_path / "fifa_rankings.csv"

    imported = import_fifa_ranking_snapshots(input_dir, output, config=_cfg(tmp_path, fifa_output=output))

    assert imported["date"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2019-01-01"]


def test_parse_markdown_table_snapshot(tmp_path) -> None:
    input_dir = tmp_path / "fifa_snapshots"
    input_dir.mkdir()
    lines = ["| rank | team | points |", "| --- | --- | --- |"]
    lines.extend(f"| {idx} | {team} | {1900 - idx} |" for idx, team in enumerate(_teams(), start=1))
    (input_dir / "fifa-ranking-2022-10-06.md").write_text("\n".join(lines), encoding="utf-8")
    output = tmp_path / "fifa_rankings.csv"

    imported = import_fifa_ranking_snapshots(input_dir, output, config=_cfg(tmp_path, fifa_output=output))

    assert len(imported) == 32
    assert "Bosnia and Herzegovina" in set(imported["team"])


def test_parse_txt_whitespace_snapshot_and_team_names_with_spaces(tmp_path) -> None:
    input_dir = tmp_path / "fifa_snapshots"
    input_dir.mkdir()
    lines = ["rank team points"]
    lines.extend(f"{idx} {team} {1900 - idx}" for idx, team in enumerate(_teams(), start=1))
    (input_dir / "fifa_2022-10-06.txt").write_text("\n".join(lines), encoding="utf-8")
    output = tmp_path / "fifa_rankings.csv"

    imported = import_fifa_ranking_snapshots(input_dir, output, config=_cfg(tmp_path, fifa_output=output))

    assert len(imported) == 32
    assert {"South Africa", "Saudi Arabia", "New Zealand"} <= set(imported["team"])


def test_parse_html_table_snapshot(tmp_path) -> None:
    input_dir = tmp_path / "elo_snapshots"
    input_dir.mkdir()
    rows = "\n".join(f"<tr><td>{team}</td><td>{2100 - idx}</td></tr>" for idx, team in enumerate(_teams(), start=1))
    html = f"<table><tr><th>nation</th><th>rating</th></tr>{rows}</table>"
    (input_dir / "world-football-elo-2022-11-20.html").write_text(html, encoding="utf-8")
    output = tmp_path / "world_football_elo.csv"

    imported = import_world_football_elo_snapshots(input_dir, output, config=_cfg(tmp_path, elo_output=output))

    assert len(imported) == 32
    assert imported["date"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2022-11-20"]


def test_normalizes_common_team_aliases(tmp_path) -> None:
    input_dir = tmp_path / "fifa_snapshots"
    input_dir.mkdir()
    pd.DataFrame(_fifa_rows()).to_csv(input_dir / "fifa_2026-06-11.csv", index=False)
    output = tmp_path / "fifa_rankings.csv"

    imported = import_fifa_ranking_snapshots(input_dir, output, config=_cfg(tmp_path, fifa_output=output))
    teams = set(imported["team"])

    assert "United States" in teams
    assert "South Korea" in teams
    assert "Iran" in teams
    assert "DR Congo" in teams
    assert "Cape Verde" in teams
    assert normalize_team_name("Curacao") in teams
    assert "Czechia" in teams


def test_fifa_snapshot_with_blank_points_imports_as_rank_only(tmp_path) -> None:
    input_dir = tmp_path / "fifa_snapshots"
    input_dir.mkdir()
    rows = _fifa_rows(date="2018-05-17")
    for row in rows:
        row["points"] = ""
    output = tmp_path / "fifa_rankings.csv"
    cfg = _cfg(tmp_path, fifa_output=output)
    pd.DataFrame(rows).to_csv(input_dir / "fifa_2018-05-17.csv", index=False)

    imported = import_fifa_ranking_snapshots(input_dir, output, config=cfg)
    report = pd.read_csv(cfg["paths"]["historical_rating_coverage_report_csv"])

    assert len(imported) == 32
    assert imported["points"].isna().all()
    assert "imported_with_missing_points" in report.loc[0, "snapshot_statuses"]
    assert int(report.loc[0, "snapshots_imported_with_missing_points"]) == 1


def test_missing_points_snapshot_creates_rank_features_and_null_point_features(tmp_path) -> None:
    input_dir = tmp_path / "fifa_snapshots"
    input_dir.mkdir()
    rows = _fifa_rows(date="2018-05-17")
    for row in rows:
        row["points"] = "-"
    output = tmp_path / "fifa_rankings.csv"
    pd.DataFrame(rows).to_csv(input_dir / "fifa_2018-05-17.csv", index=False)

    imported = import_fifa_ranking_snapshots(input_dir, output, config=_cfg(tmp_path, fifa_output=output))
    joined = join_fifa_rankings_asof(
        pd.DataFrame([{"date": "2018-05-18", "home_team": "United States", "away_team": "South Korea"}]),
        imported,
    )

    assert joined.loc[0, "home_fifa_rank"] == 1
    assert joined.loc[0, "away_fifa_rank"] == 2
    assert joined.loc[0, "fifa_rank_diff"] == -1
    assert pd.isna(joined.loc[0, "home_fifa_points"])
    assert pd.isna(joined.loc[0, "away_fifa_points"])
    assert pd.isna(joined.loc[0, "fifa_points_diff"])


def test_rejects_snapshot_with_fewer_than_30_teams_and_preserves_existing(tmp_path) -> None:
    input_dir = tmp_path / "fifa_snapshots"
    input_dir.mkdir()
    pd.DataFrame(_fifa_rows(12)).to_csv(input_dir / "fifa_2010-05-26.csv", index=False)
    output = tmp_path / "fifa_rankings.csv"
    _write_existing_fifa(output)

    import_fifa_ranking_snapshots(input_dir, output, config=_cfg(tmp_path, fifa_output=output))

    preserved = pd.read_csv(output)
    assert set(preserved["source"]) == {"existing"}


def test_rejects_duplicate_date_team_rows(tmp_path) -> None:
    input_dir = tmp_path / "fifa_snapshots"
    input_dir.mkdir()
    rows = _fifa_rows(31, date="2022-10-06")
    rows.append({"date": "2022-10-06", "team": "United States", "rank": 99, "points": 1200})
    pd.DataFrame(rows).to_csv(input_dir / "fifa_2022-10-06.csv", index=False)
    output = tmp_path / "fifa_rankings.csv"
    _write_existing_fifa(output)

    import_fifa_ranking_snapshots(input_dir, output, config=_cfg(tmp_path, fifa_output=output))

    assert set(pd.read_csv(output)["source"]) == {"existing"}


def test_preserves_current_snapshot_if_no_historical_files_exist(tmp_path) -> None:
    input_dir = tmp_path / "elo_snapshots"
    input_dir.mkdir()
    output = tmp_path / "world_football_elo.csv"
    _write_existing_elo(output)

    imported = import_world_football_elo_snapshots(input_dir, output, config=_cfg(tmp_path, elo_output=output))

    assert set(pd.read_csv(output)["source"]) == {"existing"}
    assert set(imported["source"]) == {"existing"}


def test_coverage_report_is_created(tmp_path) -> None:
    fifa_dir = tmp_path / "fifa_snapshots"
    elo_dir = tmp_path / "elo_snapshots"
    fifa_dir.mkdir()
    elo_dir.mkdir()
    fifa_output = tmp_path / "fifa_rankings.csv"
    elo_output = tmp_path / "world_football_elo.csv"
    pd.DataFrame(_fifa_rows()).to_csv(fifa_dir / "fifa_2014-06-05.csv", index=False)
    pd.DataFrame(_elo_rows()).to_csv(elo_dir / "elo_2014-06-12.csv", index=False)
    cfg = _cfg(tmp_path, fifa_output=fifa_output, elo_output=elo_output)

    import_historical_rating_snapshots(fifa_dir, elo_dir, config=cfg)

    report_csv = Path(cfg["paths"]["historical_rating_coverage_report_csv"])
    report_md = Path(cfg["paths"]["historical_rating_coverage_report_md"])
    assert report_csv.exists()
    assert report_md.exists()
    report = pd.read_csv(report_csv)
    assert {"historical_training_coverage", "worldcup_2026_coverage"} <= set(report.columns)


def test_rebuild_flag_triggers_validation_feature_report_and_ablation(tmp_path, monkeypatch) -> None:
    fifa_dir = tmp_path / "fifa_snapshots"
    elo_dir = tmp_path / "elo_snapshots"
    fifa_dir.mkdir()
    elo_dir.mkdir()
    fifa_output = tmp_path / "fifa_rankings.csv"
    elo_output = tmp_path / "world_football_elo.csv"
    _write_existing_fifa(fifa_output)
    _write_existing_elo(elo_output)
    cfg = _cfg(tmp_path, fifa_output=fifa_output, elo_output=elo_output)
    calls: list[str] = []

    monkeypatch.setattr(snapshots, "validate_external_data", lambda config: calls.append("validate") or pd.DataFrame())
    monkeypatch.setattr(snapshots, "build_advanced_features", lambda config: calls.append("features") or (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(snapshots, "generate_feature_null_rate_report", lambda config: calls.append("null_report") or (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(snapshots, "run_ablation", lambda config: calls.append("ablation") or pd.DataFrame())

    result = import_historical_rating_snapshots(fifa_dir, elo_dir, rebuild=True, config=cfg)

    assert result["rebuild_status"] == "completed"
    assert calls == ["validate", "features", "null_report", "ablation"]
