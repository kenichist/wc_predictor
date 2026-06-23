from src.experiments.ablation_runner import _ablation_warning


def test_ablation_warning_for_all_null_new_feature_group() -> None:
    warning = _ablation_warning(
        "v3_plus_external_elo",
        ["home_external_elo", "away_external_elo"],
        ["home_external_elo", "away_external_elo"],
        [],
        [],
        True,
        True,
    )

    assert "fully null" in warning
    assert "world_football_elo.csv" in warning
