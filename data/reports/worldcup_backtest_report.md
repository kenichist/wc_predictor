# Historical World Cup Backtest Report

- Final model feature set: `core_football_only`
- Tournaments: 2010, 2014, 2018, 2022 FIFA World Cups
- Leakage rule: each fold trains only on matches dated before that tournament's opening match.
- Current 2026 squad/player, injury, suspension, and xG interfaces are excluded from historical backtests unless proper historical as-of data is available.

## Metrics By Tournament

| world_cup_year | evaluation_scope | model_name | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | n_matches | n_train | odds_coverage_rate | alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | all_matches | final_model | 0.546875 | 0.945475 | 0.555694 | 0.194898 | 0.137268 | 64 | 37861 |  |  |
| 2014 | all_matches | rolling_form_only | 0.625000 | 0.990914 | 0.589019 | 0.209534 | 0.176224 | 64 | 37861 |  |  |
| 2014 | all_matches | elo_only | 0.593750 | 1.005079 | 0.595746 | 0.212972 | 0.148541 | 64 | 37861 |  |  |
| 2014 | all_matches | majority_class_baseline | 0.453125 | 1.059328 | 0.641554 | 0.239570 | 0.040628 | 64 | 37861 |  |  |
| 2014 | all_matches | fifa_ranking_only | 0.437500 | 1.069958 | 0.646672 | 0.229912 | 0.182763 | 64 | 37861 |  |  |
| 2014 | odds_covered | bookmaker_odds_only | 0.571429 | 0.952307 | 0.565444 | 0.198209 | 0.091628 | 63 | 37861 | 0.984375 |  |
| 2014 | odds_covered | market_blend | 0.571429 | 0.952307 | 0.565444 | 0.198209 | 0.091628 | 63 | 37861 | 0.984375 | 0.000000 |
| 2014 | odds_covered | final_model | 0.539683 | 0.955988 | 0.562923 | 0.197484 | 0.135532 | 63 | 37861 | 0.984375 |  |
| 2018 | all_matches | final_model | 0.500000 | 0.987186 | 0.592829 | 0.208069 | 0.071885 | 64 | 41639 |  |  |
| 2018 | all_matches | elo_only | 0.531250 | 1.024678 | 0.615187 | 0.223880 | 0.080164 | 64 | 41639 |  |  |
| 2018 | all_matches | rolling_form_only | 0.421875 | 1.049011 | 0.627449 | 0.231522 | 0.091060 | 64 | 41639 |  |  |
| 2018 | all_matches | fifa_ranking_only | 0.562500 | 1.079270 | 0.636370 | 0.231709 | 0.159331 | 64 | 41639 |  |  |
| 2018 | all_matches | majority_class_baseline | 0.390625 | 1.094111 | 0.667756 | 0.252659 | 0.101582 | 64 | 41639 |  |  |
| 2018 | odds_covered | bookmaker_odds_only | 0.564516 | 0.935959 | 0.551525 | 0.196510 | 0.068938 | 62 | 41639 | 0.968750 |  |
| 2018 | odds_covered | market_blend | 0.564516 | 0.935959 | 0.551525 | 0.196510 | 0.068938 | 62 | 41639 | 0.968750 | 0.000000 |
| 2018 | odds_covered | final_model | 0.516129 | 0.971863 | 0.583042 | 0.207106 | 0.059067 | 62 | 41639 | 0.968750 |  |
| 2022 | all_matches | elo_only | 0.515625 | 1.037723 | 0.607103 | 0.212762 | 0.091100 | 64 | 45700 |  |  |
| 2022 | all_matches | majority_class_baseline | 0.437500 | 1.074251 | 0.651156 | 0.235831 | 0.054010 | 64 | 45700 |  |  |
| 2022 | all_matches | rolling_form_only | 0.437500 | 1.079741 | 0.654020 | 0.236285 | 0.089200 | 64 | 45700 |  |  |
| 2022 | all_matches | final_model | 0.484375 | 1.079986 | 0.635455 | 0.231486 | 0.123212 | 64 | 45700 |  |  |
| 2022 | all_matches | fifa_ranking_only | 0.515625 | 1.084651 | 0.619255 | 0.220330 | 0.146350 | 64 | 45700 |  |  |
| 2022 | odds_covered | bookmaker_odds_only | 0.523810 | 1.010652 | 0.591800 | 0.210385 | 0.112230 | 63 | 45700 | 0.984375 |  |
| 2022 | odds_covered | market_blend | 0.523810 | 1.010652 | 0.591800 | 0.210385 | 0.112230 | 63 | 45700 | 0.984375 | 0.000000 |
| 2022 | odds_covered | final_model | 0.476190 | 1.087986 | 0.640953 | 0.233349 | 0.132118 | 63 | 45700 | 0.984375 |  |

## Average Metrics

| evaluation_scope | model_name | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | tournaments | total_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_matches | final_model | 0.510417 | 1.004215 | 0.594659 | 0.211484 | 0.110788 | 3 | 192 |
| all_matches | elo_only | 0.546875 | 1.022493 | 0.606012 | 0.216538 | 0.106602 | 3 | 192 |
| all_matches | rolling_form_only | 0.494792 | 1.039889 | 0.623496 | 0.225780 | 0.118828 | 3 | 192 |
| all_matches | majority_class_baseline | 0.427083 | 1.075897 | 0.653489 | 0.242687 | 0.065407 | 3 | 192 |
| all_matches | fifa_ranking_only | 0.505208 | 1.077960 | 0.634099 | 0.227317 | 0.162815 | 3 | 192 |
| odds_covered | bookmaker_odds_only | 0.553251 | 0.966306 | 0.569590 | 0.201701 | 0.090932 | 3 | 188 |
| odds_covered | market_blend | 0.553251 | 0.966306 | 0.569590 | 0.201701 | 0.090932 | 3 | 188 |
| odds_covered | final_model | 0.510667 | 1.005279 | 0.595639 | 0.212647 | 0.108906 | 3 | 188 |

## Model Ranking By Log Loss

| evaluation_scope | model_name | log_loss | accuracy | calibration_error | tournaments |
| --- | --- | --- | --- | --- | --- |
| all_matches | final_model | 1.004215 | 0.510417 | 0.110788 | 3 |
| all_matches | elo_only | 1.022493 | 0.546875 | 0.106602 | 3 |
| all_matches | rolling_form_only | 1.039889 | 0.494792 | 0.118828 | 3 |
| all_matches | majority_class_baseline | 1.075897 | 0.427083 | 0.065407 | 3 |
| all_matches | fifa_ranking_only | 1.077960 | 0.505208 | 0.162815 | 3 |
| odds_covered | bookmaker_odds_only | 0.966306 | 0.553251 | 0.090932 | 3 |
| odds_covered | market_blend | 0.966306 | 0.553251 | 0.090932 | 3 |
| odds_covered | final_model | 1.005279 | 0.510667 | 0.108906 | 3 |

## Benchmark Comparisons

- Final model beats Elo-only: `True (log_loss_delta=-0.018278)`
- Final model beats FIFA-only: `True (log_loss_delta=-0.073744)`
- Final model beats bookmaker odds on odds-covered matches: `False (log_loss_delta=0.038973)`
- Market blend beats final model on odds-covered matches: `True (log_loss_delta=-0.038973)`
- Market blend beats bookmaker odds on odds-covered matches: `False (log_loss_delta=0.000000)`

## Odds Coverage

| world_cup_year | odds_covered_matches | odds_total_matches | odds_missing_matches | odds_coverage_rate |
| --- | --- | --- | --- | --- |
| 2014 | 63 | 64 | 1 | 0.984375 |
| 2018 | 62 | 64 | 2 | 0.968750 |
| 2022 | 63 | 64 | 1 | 0.984375 |

## Missing Odds Matches

| world_cup_year | date | home_team | away_team |
| --- | --- | --- | --- |
| 2014 | 2014-06-23 | Brazil | Cameroon |
| 2018 | 2018-06-25 | Russia | Uruguay |
| 2018 | 2018-07-01 | Russia | Spain |
| 2022 | 2022-11-29 | Qatar | Netherlands |

## Calibration Summary

Final model average calibration_error is 0.110788. Best average calibration_error is `majority_class_baseline` at 0.065407.

## Known Limitations

- This is a match-level backtest, not a full historical tournament simulation with group tables and knockout brackets.
- Historical FIFA and external rating features are joined strictly as-of the match date; sparse snapshots can reduce benchmark strength.
- Historical squad/player, injury, suspension, and xG features are excluded because the available project files are current snapshots, not historical as-of records.
- Bookmaker odds benchmarks are included where complete odds were available.
- Knockout matches are evaluated using the stored match outcome class in the training data.
