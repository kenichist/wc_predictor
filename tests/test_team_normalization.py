from src.data_sources.team_normalization import normalize_api_team_name


def test_api_team_aliases_normalize_to_project_names() -> None:
    assert normalize_api_team_name("USA") == "United States"
    assert normalize_api_team_name("Korea Republic") == "South Korea"
    assert normalize_api_team_name("Korea DPR") == "North Korea"
    assert normalize_api_team_name("Cote d'Ivoire") == "Ivory Coast"
    assert normalize_api_team_name("D.R. Congo") == "DR Congo"
    assert normalize_api_team_name("Czech Republic") == "Czechia"
    assert normalize_api_team_name("Bosnia-Herzegovina") == "Bosnia and Herzegovina"
    assert normalize_api_team_name("KSA") == "Saudi Arabia"
    assert normalize_api_team_name("Cabo Verde") == "Cape Verde"
    assert normalize_api_team_name("Holland") == "Netherlands"

