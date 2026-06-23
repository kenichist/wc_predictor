import pandas as pd

from src.sources.external_rating_acquisition import acquire_world_football_elo


def _elo_rows(n: int = 32):
    rows = []
    for idx in range(n):
        team = "Korea Republic" if idx == 0 else f"Elo Nation {idx:02d}"
        rows.append({"rating_date": "2024-12-31", "nation": team, "rating": 2100 - idx})
    return rows


def _cfg(tmp_path):
    return {
        "external_sources": {
            "world_football_elo": {
                "output_path": str(tmp_path / "world_football_elo.csv"),
                "raw_dir": str(tmp_path / "raw" / "world_football_elo"),
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


def test_csv_elo_input_normalizes_correctly(tmp_path) -> None:
    input_path = tmp_path / "elo_source.csv"
    pd.DataFrame(_elo_rows()).to_csv(input_path, index=False)

    result = acquire_world_football_elo(input_path=input_path, config=_cfg(tmp_path), run_post_checks=False)

    output = pd.read_csv(tmp_path / "world_football_elo.csv")
    assert result.status == "saved"
    assert list(output.columns) == ["date", "team", "elo", "source", "retrieved_at"]
    assert "South Korea" in set(output["team"])
    assert output["elo"].notna().all()


def test_html_elo_table_input_selects_correct_table(tmp_path) -> None:
    input_path = tmp_path / "elo_source.html"
    bad = pd.DataFrame([{"foo": 1, "bar": 2}]).to_html(index=False)
    good = pd.DataFrame(_elo_rows()).to_html(index=False)
    input_path.write_text(f"<html><body>{bad}{good}</body></html>", encoding="utf-8")

    result = acquire_world_football_elo(input_path=input_path, config=_cfg(tmp_path), run_post_checks=False)

    output = pd.read_csv(tmp_path / "world_football_elo.csv")
    assert result.status == "saved"
    assert len(output) == 32
    assert "South Korea" in set(output["team"])
