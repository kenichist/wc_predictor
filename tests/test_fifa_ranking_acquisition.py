import importlib.util

import pandas as pd
import pytest

from src.sources.external_rating_acquisition import acquire_fifa_rankings


def _fifa_rows(n: int = 32):
    rows = []
    for idx in range(n):
        team = "USA" if idx == 0 else f"Nation {idx:02d}"
        rows.append({"release_date": "2024-12-19", "country": team, "position": idx + 1, "total_points": 1900 - idx})
    return rows


def _cfg(tmp_path):
    return {
        "external_sources": {
            "fifa_rankings": {
                "output_path": str(tmp_path / "fifa_rankings.csv"),
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


def test_csv_fifa_ranking_input_normalizes_correctly(tmp_path) -> None:
    input_path = tmp_path / "fifa_source.csv"
    pd.DataFrame(_fifa_rows()).to_csv(input_path, index=False)

    result = acquire_fifa_rankings(input_path=input_path, config=_cfg(tmp_path), run_post_checks=False)

    output = pd.read_csv(tmp_path / "fifa_rankings.csv")
    assert result.status == "saved"
    assert list(output.columns) == ["date", "team", "rank", "points", "source", "retrieved_at"]
    assert "United States" in set(output["team"])
    assert output["rank"].notna().all()
    assert output["points"].notna().all()


def test_excel_fifa_ranking_input_normalizes_correctly_if_openpyxl_installed(tmp_path) -> None:
    if importlib.util.find_spec("openpyxl") is None:
        pytest.skip("openpyxl is not installed")
    input_path = tmp_path / "fifa_source.xlsx"
    pd.DataFrame(_fifa_rows()).to_excel(input_path, index=False)

    result = acquire_fifa_rankings(input_path=input_path, config=_cfg(tmp_path), run_post_checks=False)

    assert result.status == "saved"
    assert (tmp_path / "fifa_rankings.csv").exists()


def test_html_fifa_ranking_table_input_selects_correct_table(tmp_path) -> None:
    input_path = tmp_path / "fifa_source.html"
    bad = pd.DataFrame([{"foo": 1, "bar": 2}]).to_html(index=False)
    good = pd.DataFrame(_fifa_rows()).to_html(index=False)
    input_path.write_text(f"<html><body>{bad}{good}</body></html>", encoding="utf-8")

    result = acquire_fifa_rankings(input_path=input_path, config=_cfg(tmp_path), run_post_checks=False)

    output = pd.read_csv(tmp_path / "fifa_rankings.csv")
    assert result.status == "saved"
    assert len(output) == 32
    assert "United States" in set(output["team"])
