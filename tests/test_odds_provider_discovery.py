from src.data_sources.market_odds_workflow import looks_like_club_world_cup, looks_like_national_world_cup


def test_club_world_cup_key_is_flagged_wrong() -> None:
    assert looks_like_club_world_cup("soccer_fifa_club_world_cup", "FIFA Club World Cup")
    assert not looks_like_national_world_cup("soccer_fifa_club_world_cup", "FIFA Club World Cup")


def test_national_world_cup_key_is_likely_valid() -> None:
    assert looks_like_national_world_cup("soccer_fifa_world_cup", "FIFA World Cup")
    assert not looks_like_club_world_cup("soccer_fifa_world_cup", "FIFA World Cup")
