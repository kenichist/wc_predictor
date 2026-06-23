# External Feature Activation Debug

## Files

- `squad_player_features` path: `C:\Users\kenic\OneDrive\Desktop\wc_predictor\data\external\squad_player_features.csv`; exists: `True`; rows: `48`
- `injuries_suspensions` path: `C:\Users\kenic\OneDrive\Desktop\wc_predictor\data\external\injuries_suspensions.csv`; exists: `True`; rows: `48`
- `market_odds` path: `C:\Users\kenic\OneDrive\Desktop\wc_predictor\data\external\market_odds.csv`; exists: `True`; rows: `216`

## Squad Player Features

- Detected columns: `team, squad_market_value, starting_xi_market_value, top_5_player_rating_avg, top_11_player_rating_avg, goalkeeper_strength, defense_strength, midfield_strength, attack_strength, bench_strength, avg_age, total_caps, international_goals, club_minutes_last_season, club_minutes_recent_90_days, source, updated_at`
- Required columns missing: `none`
- External teams: `48`
- World Cup prediction teams: `48`
- Matched teams: `48`
- Matched team names: `Algeria, Argentina, Australia, Austria, Belgium, Bosnia and Herzegovina, Brazil, Canada, Cape Verde, Colombia, Croatia, Curaçao, Czechia, DR Congo, Ecuador, Egypt, England, France, Germany, Ghana, Haiti, Iran, Iraq, Ivory Coast, Japan, Jordan, Mexico, Morocco, Netherlands, New Zealand, Norway, Panama, Paraguay, Portugal, Qatar, Saudi Arabia, Scotland, Senegal, South Africa, South Korea, Spain, Sweden, Switzerland, Tunisia, Turkey, United States, Uruguay, Uzbekistan`
- Unmatched external teams: `none`
- Unmatched prediction teams: `none`
- Generated non-null counts:
  - `home_squad_market_value`: `5106`
  - `home_starting_xi_market_value`: `0`
  - `home_top_5_player_rating_avg`: `0`
  - `home_top_11_player_rating_avg`: `0`
  - `home_goalkeeper_strength`: `0`
  - `home_defense_strength`: `0`
  - `home_midfield_strength`: `0`
  - `home_attack_strength`: `0`
  - `home_bench_strength`: `0`
  - `home_avg_age`: `0`
  - `home_total_caps`: `0`
  - `home_international_goals`: `0`
  - `home_club_minutes_last_season`: `0`
  - `home_club_minutes_recent_90_days`: `0`
  - `away_squad_market_value`: `4441`
  - `away_starting_xi_market_value`: `0`
  - `away_top_5_player_rating_avg`: `0`
  - `away_top_11_player_rating_avg`: `0`
  - `away_goalkeeper_strength`: `0`
  - `away_defense_strength`: `0`
  - `away_midfield_strength`: `0`
  - `away_attack_strength`: `0`
  - `away_bench_strength`: `0`
  - `away_avg_age`: `0`
  - `away_total_caps`: `0`
  - `away_international_goals`: `0`
  - `away_club_minutes_last_season`: `0`
  - `away_club_minutes_recent_90_days`: `0`
  - `squad_market_value_diff`: `2058`
  - `starting_xi_market_value_diff`: `0`
  - `top_11_player_rating_diff`: `0`
  - `goalkeeper_strength_diff`: `0`
  - `defense_strength_diff`: `0`
  - `midfield_strength_diff`: `0`
  - `attack_strength_diff`: `0`
  - `bench_strength_diff`: `0`
  - `avg_age_diff`: `0`
  - `total_caps_diff`: `0`
  - `club_minutes_recent_90_days_diff`: `0`
- Fully null reason: `active`

## Injuries And Suspensions

- Detected columns: `date, team, missing_starters_count, missing_key_players_count, injury_impact_score, suspension_impact_score, goalkeeper_missing, captain_missing, minutes_lost_from_expected_xi, source, updated_at`
- Required columns missing: `none`
- External teams: `48`
- World Cup prediction teams: `48`
- Matched teams: `48`
- Matched team names: `Algeria, Argentina, Australia, Austria, Belgium, Bosnia and Herzegovina, Brazil, Canada, Cape Verde, Colombia, Croatia, Curaçao, Czechia, DR Congo, Ecuador, Egypt, England, France, Germany, Ghana, Haiti, Iran, Iraq, Ivory Coast, Japan, Jordan, Mexico, Morocco, Netherlands, New Zealand, Norway, Panama, Paraguay, Portugal, Qatar, Saudi Arabia, Scotland, Senegal, South Africa, South Korea, Spain, Sweden, Switzerland, Tunisia, Turkey, United States, Uruguay, Uzbekistan`
- Unmatched external teams: `none`
- Unmatched prediction teams: `none`
- Generated non-null counts:
  - `home_missing_starters_count`: `72`
  - `away_missing_starters_count`: `72`
  - `home_missing_key_players_count`: `72`
  - `away_missing_key_players_count`: `72`
  - `home_injury_impact_score`: `72`
  - `away_injury_impact_score`: `72`
  - `home_suspension_impact_score`: `72`
  - `away_suspension_impact_score`: `72`
  - `home_goalkeeper_missing`: `72`
  - `away_goalkeeper_missing`: `72`
  - `home_captain_missing`: `72`
  - `away_captain_missing`: `72`
  - `home_minutes_lost_from_expected_xi`: `0`
  - `away_minutes_lost_from_expected_xi`: `0`
  - `missing_starters_count_diff`: `72`
  - `missing_key_players_count_diff`: `72`
  - `injury_impact_score_diff`: `72`
  - `suspension_impact_score_diff`: `72`
  - `minutes_lost_from_expected_xi_diff`: `0`
- Fully null reason: `active`

## Market Odds

- Configured path: `C:\Users\kenic\OneDrive\Desktop\wc_predictor\data\external\market_odds.csv`
- Rows: `216`
- Detected columns: `date, home_team, away_team, home_odds, draw_odds, away_odds, bookmaker, source, updated_at`
- Status: `available`
