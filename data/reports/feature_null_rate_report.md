# Feature Null Rate Report

## Summary By Feature Group

| feature_group | average_null_rate | fully_null_columns | usable_columns | total_columns | historical_training_coverage | worldcup_2026_coverage | warning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.000 | 0 | 5 | 5 | 1.000 | 1.000 |  |
| basic_elo | 0.000 | 0 | 3 | 3 | 1.000 | 1.000 |  |
| dynamic_ratings | 0.000 | 0 | 3 | 3 | 1.000 | 1.000 |  |
| external_elo | 0.999 | 2 | 5 | 7 | 0.000 | 0.311 | average null rate above 50%; World Cup 2026 coverage below 90%; historical training coverage below 50% |
| fifa_rankings | 0.680 | 0 | 12 | 12 | 0.649 | 0.689 | average null rate above 50%; World Cup 2026 coverage below 90% |
| injuries | 0.996 | 3 | 16 | 19 | 0.000 | 0.689 | average null rate above 50% |
| internal_historical_elo | 0.000 | 0 | 6 | 6 | 1.000 | 1.000 |  |
| market | 0.985 | 0 | 6 | 6 | 0.014 | 0.223 | average null rate above 50% |
| metadata | 0.162 | 1 | 24 | 25 | 1.000 | 1.000 |  |
| pi_ratings | 0.000 | 0 | 11 | 11 | 1.000 | 1.000 |  |
| rolling_form | 0.003 | 0 | 23 | 23 | 1.000 | 1.000 |  |
| squad | 0.981 | 36 | 3 | 39 | 0.467 | 0.689 | average null rate above 50% |
| target | 0.006 | 0 | 3 | 3 | 1.000 | 0.000 |  |
| xg | 1.000 | 16 | 0 | 16 | 0.000 | 0.000 | average null rate above 50% |

## Fully Null Columns

- `venue` (metadata)
- `home_external_elo_change` (external_elo)
- `away_external_elo_change` (external_elo)
- `home_xg_for_avg_last_5` (xg)
- `away_xg_for_avg_last_5` (xg)
- `home_xg_against_avg_last_5` (xg)
- `away_xg_against_avg_last_5` (xg)
- `home_xg_diff_last_5` (xg)
- `away_xg_diff_last_5` (xg)
- `home_xg_for_avg_last_10` (xg)
- `away_xg_for_avg_last_10` (xg)
- `home_xg_against_avg_last_10` (xg)
- `away_xg_against_avg_last_10` (xg)
- `home_xg_diff_last_10` (xg)
- `away_xg_diff_last_10` (xg)
- `home_shots_for_avg_last_5` (xg)
- `away_shots_for_avg_last_5` (xg)
- `home_shots_against_avg_last_5` (xg)
- `away_shots_against_avg_last_5` (xg)
- `home_starting_xi_market_value` (squad)
- `home_top_5_player_rating_avg` (squad)
- `home_top_11_player_rating_avg` (squad)
- `home_goalkeeper_strength` (squad)
- `home_defense_strength` (squad)
- `home_midfield_strength` (squad)
- `home_attack_strength` (squad)
- `home_bench_strength` (squad)
- `home_avg_age` (squad)
- `home_total_caps` (squad)
- `home_international_goals` (squad)
- `home_club_minutes_last_season` (squad)
- `home_club_minutes_recent_90_days` (squad)
- `away_starting_xi_market_value` (squad)
- `away_top_5_player_rating_avg` (squad)
- `away_top_11_player_rating_avg` (squad)
- `away_goalkeeper_strength` (squad)
- `away_defense_strength` (squad)
- `away_midfield_strength` (squad)
- `away_attack_strength` (squad)
- `away_bench_strength` (squad)
- `away_avg_age` (squad)
- `away_total_caps` (squad)
- `away_international_goals` (squad)
- `away_club_minutes_last_season` (squad)
- `away_club_minutes_recent_90_days` (squad)
- `starting_xi_market_value_diff` (squad)
- `top_11_player_rating_diff` (squad)
- `goalkeeper_strength_diff` (squad)
- `defense_strength_diff` (squad)
- `midfield_strength_diff` (squad)
- `attack_strength_diff` (squad)
- `bench_strength_diff` (squad)
- `avg_age_diff` (squad)
- `total_caps_diff` (squad)
- `club_minutes_recent_90_days_diff` (squad)
- `home_minutes_lost_from_expected_xi` (injuries)
- `away_minutes_lost_from_expected_xi` (injuries)
- `minutes_lost_from_expected_xi_diff` (injuries)
