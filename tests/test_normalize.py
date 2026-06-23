from src.normalize import normalize_team_name


def test_team_normalization_examples() -> None:
    assert normalize_team_name("USA") == "United States"
    assert normalize_team_name("United States of America") == "United States"
    assert normalize_team_name("Korea Republic") == "South Korea"
    assert normalize_team_name("Republic of Korea") == "South Korea"
    assert normalize_team_name("IR Iran") == "Iran"
    assert normalize_team_name("Islamic Republic of Iran") == "Iran"
    assert normalize_team_name("Türkiye") == "Turkey"
    assert normalize_team_name("Turkiye") == "Turkey"
    assert normalize_team_name("Côte d'Ivoire") == "Ivory Coast"
    assert normalize_team_name("Cote d'Ivoire") == "Ivory Coast"
    assert normalize_team_name("Congo DR") == "DR Congo"
    assert normalize_team_name("Democratic Republic of Congo") == "DR Congo"
    assert normalize_team_name("DR Congo") == "DR Congo"
    assert normalize_team_name("Curaçao") == "Curaçao"
    assert normalize_team_name("Curacao") == "Curaçao"
