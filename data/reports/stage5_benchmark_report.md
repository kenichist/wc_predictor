# Stage 5 Benchmark Report

## Executive Summary

- Stage 5 status: `Stage 5 not yet achieved`
- Official production model remains `football_only_ensemble`.
- Official production feature set remains `core_football_only`.
- Market blend remains benchmark-only unless significance results support promotion.
- This report is research/evaluation output, not a betting tool.

## Data Coverage

- Backtest years requested: `2014, 2018, 2022`
| test_year | total_matches | market_covered_matches | market_coverage_rate |
| --- | --- | --- | --- |
| 2014 | 64 | 63 | 0.984375 |
| 2018 | 64 | 62 | 0.968750 |
| 2022 | 64 | 63 | 0.984375 |

## Models Compared

| model_name | baseline_type | feature_set | years | total_matches | total_skipped |
| --- | --- | --- | --- | --- | --- |
| bookmaker_calibrated | market_calibration_baseline | market_odds_multiplicative_temperature | 2 | 125 | 3 |
| bookmaker_odds_only | market_baseline | best_market_odds_conversion | 3 | 188 | 4 |
| bookmaker_raw_implied | market_baseline | market_odds_multiplicative | 3 | 188 | 4 |
| dixon_coles | skipped_or_unavailable | not_fitted | 3 | 0 | 192 |
| elo_only | elo_baseline | elo_rating_features | 3 | 192 | 0 |
| fifa_ranking_only | ranking_baseline | fifa_ranking_features | 3 | 192 | 0 |
| historical_class_frequency | historical_frequency | historical_outcome_frequency | 3 | 192 | 0 |
| market_blend | market_blend_benchmark | football_only_ensemble+market | 3 | 188 | 4 |
| official_model | official_model | football_only_ensemble/core_football_only | 3 | 192 | 0 |
| poisson_goal_model | goal_model_baseline | attack_defense_smoothing | 3 | 192 | 0 |
| random_uniform | random_baseline | none | 3 | 192 | 0 |
| rolling_form_only | form_baseline | rolling_form_features | 3 | 192 | 0 |

## Metrics Table

| model_name | feature_set | baseline_type | test_year | n_matches | n_skipped | accuracy | log_loss | brier | rps | ece | mean_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_model | football_only_ensemble/core_football_only | official_model | 2014 | 64 | 0 | 0.546875 | 0.945475 | 0.555694 | 0.194898 | 0.137268 | 0.521105 |
| bookmaker_odds_only | best_market_odds_conversion | market_baseline | 2014 | 63 | 1 | 0.571429 | 0.952307 | 0.565444 | 0.198209 | 0.091628 | 0.536091 |
| bookmaker_raw_implied | market_odds_multiplicative | market_baseline | 2014 | 63 | 1 | 0.571429 | 0.952307 | 0.565444 | 0.198209 | 0.091628 | 0.536091 |
| market_blend | football_only_ensemble+market | market_blend_benchmark | 2014 | 63 | 1 | 0.571429 | 0.952307 | 0.565444 | 0.198209 | 0.091628 | 0.536091 |
| rolling_form_only | rolling_form_features | form_baseline | 2014 | 64 | 0 | 0.625000 | 0.990914 | 0.589019 | 0.209534 | 0.176224 | 0.515329 |
| elo_only | elo_rating_features | elo_baseline | 2014 | 64 | 0 | 0.593750 | 1.005079 | 0.595746 | 0.212972 | 0.148541 | 0.508351 |
| historical_class_frequency | historical_outcome_frequency | historical_frequency | 2014 | 64 | 0 | 0.453125 | 1.059328 | 0.641554 | 0.239570 | 0.040628 | 0.493753 |
| fifa_ranking_only | fifa_ranking_features | ranking_baseline | 2014 | 64 | 0 | 0.437500 | 1.069958 | 0.646672 | 0.229912 | 0.182763 | 0.551995 |
| random_uniform | none | random_baseline | 2014 | 64 | 0 | 0.343750 | 1.098612 | 0.666667 | 0.243924 | 0.010417 | 0.333333 |
| poisson_goal_model | attack_defense_smoothing | goal_model_baseline | 2014 | 64 | 0 | 0.390625 | 1.147116 | 0.694521 | 0.265430 | 0.144358 | 0.534983 |
| bookmaker_calibrated | market_odds_multiplicative_temperature | market_calibration_baseline | 2018 | 62 | 2 | 0.564516 | 0.935285 | 0.550559 | 0.196330 | 0.064565 | 0.565552 |
| bookmaker_odds_only | best_market_odds_conversion | market_baseline | 2018 | 62 | 2 | 0.564516 | 0.935959 | 0.551525 | 0.196510 | 0.068938 | 0.547106 |
| bookmaker_raw_implied | market_odds_multiplicative | market_baseline | 2018 | 62 | 2 | 0.564516 | 0.935959 | 0.551525 | 0.196510 | 0.068938 | 0.547106 |
| market_blend | football_only_ensemble+market | market_blend_benchmark | 2018 | 62 | 2 | 0.564516 | 0.935959 | 0.551525 | 0.196510 | 0.068938 | 0.547106 |
| official_model | football_only_ensemble/core_football_only | official_model | 2018 | 64 | 0 | 0.500000 | 0.987186 | 0.592829 | 0.208069 | 0.071885 | 0.522398 |
| elo_only | elo_rating_features | elo_baseline | 2018 | 64 | 0 | 0.531250 | 1.024678 | 0.615187 | 0.223880 | 0.080164 | 0.512061 |
| rolling_form_only | rolling_form_features | form_baseline | 2018 | 64 | 0 | 0.421875 | 1.049011 | 0.627449 | 0.231522 | 0.091060 | 0.492867 |
| fifa_ranking_only | fifa_ranking_features | ranking_baseline | 2018 | 64 | 0 | 0.562500 | 1.079270 | 0.636370 | 0.231709 | 0.159331 | 0.572296 |
| historical_class_frequency | historical_outcome_frequency | historical_frequency | 2018 | 64 | 0 | 0.390625 | 1.094111 | 0.667756 | 0.252659 | 0.101582 | 0.492207 |
| random_uniform | none | random_baseline | 2018 | 64 | 0 | 0.406250 | 1.098612 | 0.666667 | 0.243924 | 0.072917 | 0.333333 |
| poisson_goal_model | attack_defense_smoothing | goal_model_baseline | 2018 | 64 | 0 | 0.406250 | 1.124609 | 0.685038 | 0.259310 | 0.114406 | 0.513201 |
| bookmaker_odds_only | best_market_odds_conversion | market_baseline | 2022 | 63 | 1 | 0.523810 | 1.010652 | 0.591800 | 0.210385 | 0.112230 | 0.563178 |
| bookmaker_raw_implied | market_odds_multiplicative | market_baseline | 2022 | 63 | 1 | 0.523810 | 1.010652 | 0.591800 | 0.210385 | 0.112230 | 0.563178 |
| market_blend | football_only_ensemble+market | market_blend_benchmark | 2022 | 63 | 1 | 0.523810 | 1.010652 | 0.591800 | 0.210385 | 0.112230 | 0.563178 |
| bookmaker_calibrated | market_odds_multiplicative_temperature | market_calibration_baseline | 2022 | 63 | 1 | 0.523810 | 1.018546 | 0.594215 | 0.212011 | 0.131456 | 0.583200 |
| elo_only | elo_rating_features | elo_baseline | 2022 | 64 | 0 | 0.515625 | 1.037723 | 0.607103 | 0.212762 | 0.091100 | 0.536443 |
| poisson_goal_model | attack_defense_smoothing | goal_model_baseline | 2022 | 64 | 0 | 0.453125 | 1.065477 | 0.642020 | 0.230201 | 0.067814 | 0.507490 |
| historical_class_frequency | historical_outcome_frequency | historical_frequency | 2022 | 64 | 0 | 0.437500 | 1.074251 | 0.651156 | 0.235831 | 0.054010 | 0.491510 |
| rolling_form_only | rolling_form_features | form_baseline | 2022 | 64 | 0 | 0.437500 | 1.079741 | 0.654020 | 0.236285 | 0.089200 | 0.487789 |
| official_model | football_only_ensemble/core_football_only | official_model | 2022 | 64 | 0 | 0.484375 | 1.079986 | 0.635455 | 0.231486 | 0.123212 | 0.554533 |
| fifa_ranking_only | fifa_ranking_features | ranking_baseline | 2022 | 64 | 0 | 0.515625 | 1.084651 | 0.619255 | 0.220330 | 0.146350 | 0.601465 |
| random_uniform | none | random_baseline | 2022 | 64 | 0 | 0.328125 | 1.098612 | 0.666667 | 0.238715 | 0.005208 | 0.333333 |
| dixon_coles | not_fitted | skipped_or_unavailable | 2014 | 0 | 64 |  |  |  |  |  |  |
| dixon_coles | not_fitted | skipped_or_unavailable | 2018 | 0 | 64 |  |  |  |  |  |  |
| dixon_coles | not_fitted | skipped_or_unavailable | 2022 | 0 | 64 |  |  |  |  |  |  |

## Average Metrics

| model_name | baseline_type | years | total_matches | log_loss | brier | rps | ece | accuracy | mean_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bookmaker_odds_only | market_baseline | 3 | 188 | 0.966306 | 0.569590 | 0.201701 | 0.090932 | 0.553251 | 0.548792 |
| bookmaker_raw_implied | market_baseline | 3 | 188 | 0.966306 | 0.569590 | 0.201701 | 0.090932 | 0.553251 | 0.548792 |
| market_blend | market_blend_benchmark | 3 | 188 | 0.966306 | 0.569590 | 0.201701 | 0.090932 | 0.553251 | 0.548792 |
| bookmaker_calibrated | market_calibration_baseline | 2 | 125 | 0.976915 | 0.572387 | 0.204171 | 0.098011 | 0.544163 | 0.574376 |
| official_model | official_model | 3 | 192 | 1.004215 | 0.594659 | 0.211484 | 0.110788 | 0.510417 | 0.532679 |
| elo_only | elo_baseline | 3 | 192 | 1.022493 | 0.606012 | 0.216538 | 0.106602 | 0.546875 | 0.518952 |
| rolling_form_only | form_baseline | 3 | 192 | 1.039889 | 0.623496 | 0.225780 | 0.118828 | 0.494792 | 0.498662 |
| historical_class_frequency | historical_frequency | 3 | 192 | 1.075897 | 0.653489 | 0.242687 | 0.065407 | 0.427083 | 0.492490 |
| fifa_ranking_only | ranking_baseline | 3 | 192 | 1.077960 | 0.634099 | 0.227317 | 0.162815 | 0.505208 | 0.575252 |
| random_uniform | random_baseline | 3 | 192 | 1.098612 | 0.666667 | 0.242188 | 0.029514 | 0.359375 | 0.333333 |
| poisson_goal_model | goal_model_baseline | 3 | 192 | 1.112401 | 0.673860 | 0.251647 | 0.108860 | 0.416667 | 0.518558 |

## Best Models By Metric

- Best log loss: `bookmaker_odds_only (log_loss=0.966306)`
- Best Brier: `bookmaker_odds_only (brier=0.569590)`
- Best RPS: `bookmaker_odds_only (rps=0.201701)`
- Best ECE: `random_uniform (ece=0.029514)`

## Significance Results

| candidate_model | baseline_model | metric | mean_delta | ci_lower | ci_upper | statistically_meaningful | n_matches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| official_model | bookmaker_odds_only | log_loss | 0.038990 | -0.004686 | 0.076844 | False | 188 |
| official_model | bookmaker_odds_only | brier | 0.026020 | -0.001464 | 0.050292 | False | 188 |
| official_model | bookmaker_calibrated | log_loss | 0.053141 | -0.002539 | 0.111680 | False | 125 |
| official_model | bookmaker_calibrated | brier | 0.039667 | 0.006141 | 0.076086 | False | 125 |
| official_model | elo_only | log_loss | -0.018278 | -0.057390 | 0.020500 | False | 192 |
| official_model | elo_only | brier | -0.011353 | -0.036342 | 0.015591 | False | 192 |
| official_model | poisson_goal_model | log_loss | -0.108185 | -0.185538 | -0.031376 | True | 192 |
| official_model | poisson_goal_model | brier | -0.079200 | -0.131963 | -0.030664 | True | 192 |
| market_blend | bookmaker_odds_only | log_loss | -0.000000 | -0.000000 | 0.000000 | False | 188 |
| market_blend | bookmaker_odds_only | brier | -0.000000 | -0.000000 | 0.000000 | False | 188 |
| market_blend | official_model | log_loss | -0.038990 | -0.076844 | 0.004686 | False | 188 |
| market_blend | official_model | brier | -0.026020 | -0.050292 | 0.001464 | False | 188 |

## Calibration Summary

| model_name | average_ece | years |
| --- | --- | --- |
| random_uniform | 0.029514 | 3 |
| historical_class_frequency | 0.065407 | 3 |
| bookmaker_odds_only | 0.090932 | 3 |
| bookmaker_raw_implied | 0.090932 | 3 |
| market_blend | 0.090932 | 3 |
| bookmaker_calibrated | 0.098011 | 2 |
| elo_only | 0.106602 | 3 |
| poisson_goal_model | 0.108860 | 3 |
| official_model | 0.110788 | 3 |
| rolling_form_only | 0.118828 | 3 |
| fifa_ranking_only | 0.162815 | 3 |

## Reliability Table Preview

| model_name | test_year | bucket | mean_confidence | empirical_accuracy | count | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| bookmaker_calibrated | 2018 | (0.3,0.4] | 0.373769 | 0.333333 | 9 | -0.040436 |
| bookmaker_calibrated | 2018 | (0.4,0.5] | 0.453200 | 0.411765 | 17 | -0.041435 |
| bookmaker_calibrated | 2018 | (0.5,0.6] | 0.548230 | 0.666667 | 12 | 0.118436 |
| bookmaker_calibrated | 2018 | (0.6,0.7] | 0.664651 | 0.692308 | 13 | 0.027657 |
| bookmaker_calibrated | 2018 | (0.7,0.8] | 0.741329 | 0.500000 | 4 | -0.241329 |
| bookmaker_calibrated | 2018 | (0.8,0.9] | 0.830195 | 0.857143 | 7 | 0.026948 |
| bookmaker_calibrated | 2022 | (0.3,0.4] | 0.377147 | 0.125000 | 8 | -0.252147 |
| bookmaker_calibrated | 2022 | (0.4,0.5] | 0.446575 | 0.400000 | 10 | -0.046575 |
| bookmaker_calibrated | 2022 | (0.5,0.6] | 0.545621 | 0.687500 | 16 | 0.141879 |
| bookmaker_calibrated | 2022 | (0.6,0.7] | 0.647071 | 0.533333 | 15 | -0.113738 |
| bookmaker_calibrated | 2022 | (0.7,0.8] | 0.727519 | 0.555556 | 9 | -0.171963 |
| bookmaker_calibrated | 2022 | (0.8,0.9] | 0.855002 | 0.800000 | 5 | -0.055002 |
| bookmaker_odds_only | 2014 | (0.3,0.4] | 0.373320 | 0.333333 | 12 | -0.039986 |
| bookmaker_odds_only | 2014 | (0.4,0.5] | 0.454883 | 0.687500 | 16 | 0.232617 |
| bookmaker_odds_only | 2014 | (0.5,0.6] | 0.540251 | 0.500000 | 14 | -0.040251 |
| bookmaker_odds_only | 2014 | (0.6,0.7] | 0.644150 | 0.583333 | 12 | -0.060816 |
| bookmaker_odds_only | 2014 | (0.7,0.8] | 0.735371 | 0.750000 | 8 | 0.014629 |
| bookmaker_odds_only | 2014 | (0.8,0.9] | 0.839498 | 1.000000 | 1 | 0.160502 |
| bookmaker_odds_only | 2018 | (0.3,0.4] | 0.374871 | 0.363636 | 11 | -0.011235 |
| bookmaker_odds_only | 2018 | (0.4,0.5] | 0.453298 | 0.411765 | 17 | -0.041533 |
| bookmaker_odds_only | 2018 | (0.5,0.6] | 0.544826 | 0.750000 | 12 | 0.205174 |
| bookmaker_odds_only | 2018 | (0.6,0.7] | 0.648775 | 0.666667 | 12 | 0.017892 |
| bookmaker_odds_only | 2018 | (0.7,0.8] | 0.754354 | 0.666667 | 6 | -0.087688 |
| bookmaker_odds_only | 2018 | (0.8,0.9] | 0.810397 | 0.750000 | 4 | -0.060397 |
| bookmaker_odds_only | 2022 | (0.3,0.4] | 0.375836 | 0.111111 | 9 | -0.264725 |
| bookmaker_odds_only | 2022 | (0.4,0.5] | 0.456984 | 0.615385 | 13 | 0.158401 |
| bookmaker_odds_only | 2022 | (0.5,0.6] | 0.553173 | 0.529412 | 17 | -0.023761 |
| bookmaker_odds_only | 2022 | (0.6,0.7] | 0.651746 | 0.562500 | 16 | -0.089246 |
| bookmaker_odds_only | 2022 | (0.7,0.8] | 0.752819 | 0.800000 | 5 | 0.047181 |
| bookmaker_odds_only | 2022 | (0.8,0.9] | 0.853648 | 0.666667 | 3 | -0.186982 |
| bookmaker_raw_implied | 2014 | (0.3,0.4] | 0.373320 | 0.333333 | 12 | -0.039986 |
| bookmaker_raw_implied | 2014 | (0.4,0.5] | 0.454883 | 0.687500 | 16 | 0.232617 |
| bookmaker_raw_implied | 2014 | (0.5,0.6] | 0.540251 | 0.500000 | 14 | -0.040251 |
| bookmaker_raw_implied | 2014 | (0.6,0.7] | 0.644150 | 0.583333 | 12 | -0.060816 |
| bookmaker_raw_implied | 2014 | (0.7,0.8] | 0.735371 | 0.750000 | 8 | 0.014629 |
| bookmaker_raw_implied | 2014 | (0.8,0.9] | 0.839498 | 1.000000 | 1 | 0.160502 |
| bookmaker_raw_implied | 2018 | (0.3,0.4] | 0.374871 | 0.363636 | 11 | -0.011235 |
| bookmaker_raw_implied | 2018 | (0.4,0.5] | 0.453298 | 0.411765 | 17 | -0.041533 |
| bookmaker_raw_implied | 2018 | (0.5,0.6] | 0.544826 | 0.750000 | 12 | 0.205174 |
| bookmaker_raw_implied | 2018 | (0.6,0.7] | 0.648775 | 0.666667 | 12 | 0.017892 |

## Stage Calibration Preview

| model_name | stage | bucket | mean_confidence | empirical_accuracy | count | calibration_gap |
| --- | --- | --- | --- | --- | --- | --- |
| bookmaker_calibrated | final | (0.3,0.4] | 0.363949 | 0.000000 | 1 | -0.363949 |
| bookmaker_calibrated | final | (0.4,0.5] | 0.466357 | 1.000000 | 1 | 0.533643 |
| bookmaker_calibrated | group_stage | (0.3,0.4] | 0.378422 | 0.166667 | 12 | -0.211755 |
| bookmaker_calibrated | group_stage | (0.4,0.5] | 0.446191 | 0.333333 | 15 | -0.112858 |
| bookmaker_calibrated | group_stage | (0.5,0.6] | 0.551497 | 0.666667 | 24 | 0.115170 |
| bookmaker_calibrated | group_stage | (0.6,0.7] | 0.660174 | 0.608696 | 23 | -0.051479 |
| bookmaker_calibrated | group_stage | (0.7,0.8] | 0.732753 | 0.444444 | 9 | -0.288309 |
| bookmaker_calibrated | group_stage | (0.8,0.9] | 0.842843 | 0.818182 | 11 | -0.024661 |
| bookmaker_calibrated | quarterfinal | (0.3,0.4] | 0.354097 | 1.000000 | 1 | 0.645903 |
| bookmaker_calibrated | quarterfinal | (0.4,0.5] | 0.466604 | 0.250000 | 4 | -0.216604 |
| bookmaker_calibrated | quarterfinal | (0.5,0.6] | 0.540453 | 1.000000 | 1 | 0.459547 |
| bookmaker_calibrated | quarterfinal | (0.6,0.7] | 0.622887 | 0.000000 | 1 | -0.622887 |
| bookmaker_calibrated | quarterfinal | (0.7,0.8] | 0.711266 | 0.000000 | 1 | -0.711266 |
| bookmaker_calibrated | round_of_16 | (0.3,0.4] | 0.372780 | 0.000000 | 2 | -0.372780 |
| bookmaker_calibrated | round_of_16 | (0.4,0.5] | 0.445342 | 0.333333 | 3 | -0.112008 |
| bookmaker_calibrated | round_of_16 | (0.5,0.6] | 0.510773 | 0.666667 | 3 | 0.155894 |
| bookmaker_calibrated | round_of_16 | (0.6,0.7] | 0.638081 | 0.666667 | 3 | 0.028586 |
| bookmaker_calibrated | round_of_16 | (0.7,0.8] | 0.735647 | 1.000000 | 3 | 0.264353 |
| bookmaker_calibrated | round_of_16 | (0.8,0.9] | 0.815104 | 1.000000 | 1 | 0.184896 |
| bookmaker_calibrated | semifinal | (0.3,0.4] | 0.376436 | 1.000000 | 1 | 0.623564 |
| bookmaker_calibrated | semifinal | (0.4,0.5] | 0.462988 | 0.500000 | 2 | 0.037012 |
| bookmaker_calibrated | semifinal | (0.6,0.7] | 0.625397 | 1.000000 | 1 | 0.374603 |
| bookmaker_calibrated | third_place | (0.4,0.5] | 0.441253 | 1.000000 | 2 | 0.558747 |
| bookmaker_odds_only | final | (0.3,0.4] | 0.361141 | 0.000000 | 1 | -0.361141 |
| bookmaker_odds_only | final | (0.4,0.5] | 0.431952 | 1.000000 | 2 | 0.568048 |
| bookmaker_odds_only | group_stage | (0.3,0.4] | 0.377613 | 0.227273 | 22 | -0.150340 |
| bookmaker_odds_only | group_stage | (0.4,0.5] | 0.453287 | 0.538462 | 26 | 0.085174 |
| bookmaker_odds_only | group_stage | (0.5,0.6] | 0.543609 | 0.578947 | 38 | 0.035338 |
| bookmaker_odds_only | group_stage | (0.6,0.7] | 0.647936 | 0.580645 | 31 | -0.067291 |
| bookmaker_odds_only | group_stage | (0.7,0.8] | 0.744340 | 0.687500 | 16 | -0.056840 |
| bookmaker_odds_only | group_stage | (0.8,0.9] | 0.830254 | 0.750000 | 8 | -0.080254 |
| bookmaker_odds_only | quarterfinal | (0.3,0.4] | 0.349502 | 1.000000 | 2 | 0.650498 |
| bookmaker_odds_only | quarterfinal | (0.4,0.5] | 0.460143 | 0.400000 | 5 | -0.060143 |
| bookmaker_odds_only | quarterfinal | (0.5,0.6] | 0.554871 | 0.666667 | 3 | 0.111795 |
| bookmaker_odds_only | quarterfinal | (0.6,0.7] | 0.655809 | 0.000000 | 2 | -0.655809 |
| bookmaker_odds_only | round_of_16 | (0.3,0.4] | 0.374124 | 0.250000 | 4 | -0.124124 |
| bookmaker_odds_only | round_of_16 | (0.4,0.5] | 0.470632 | 0.625000 | 8 | 0.154368 |
| bookmaker_odds_only | round_of_16 | (0.5,0.6] | 0.591795 | 0.500000 | 2 | -0.091795 |
| bookmaker_odds_only | round_of_16 | (0.6,0.7] | 0.657549 | 0.833333 | 6 | 0.175785 |
| bookmaker_odds_only | round_of_16 | (0.7,0.8] | 0.754584 | 1.000000 | 3 | 0.245416 |

## Ablation Summary

| feature_set | log_loss | brier_score | ranked_probability_score | calibration_error |
| --- | --- | --- | --- | --- |
| core_football_only | 0.894662 | 0.525878 | 0.175830 | 0.035662 |
| core_plus_internal_elo | 0.898702 | 0.528615 | 0.176726 | 0.034603 |
| v5_plus_pi_ratings | 0.904905 | 0.531451 | 0.177276 | 0.041453 |
| v6_full_rating_stack | 0.904905 | 0.531451 | 0.177276 | 0.041453 |
| rating_stack_experimental | 0.904905 | 0.531451 | 0.177276 | 0.041453 |
| v1_baseline | 0.904491 | 0.531969 | 0.177840 | 0.028365 |
| v2_baseline_plus_fifa_rankings | 0.912637 | 0.536159 | 0.179116 | 0.032421 |
| v3_plus_external_elo | 0.912637 | 0.536159 | 0.179116 | 0.032421 |
| v4_plus_dynamic_ratings | 0.912219 | 0.536890 | 0.179271 | 0.038784 |

## Skipped Or Limited Baselines

| model_name | test_year | reason | n_skipped |
| --- | --- | --- | --- |
| bookmaker_calibrated | 2014 | insufficient historical odds-covered training rows for time-safe calibration | 63 |
| dixon_coles | 2014 | skipped: no validated time-decayed Dixon-Coles parameter fitting exists in project yet | 64 |
| dixon_coles | 2018 | skipped: no validated time-decayed Dixon-Coles parameter fitting exists in project yet | 64 |
| dixon_coles | 2022 | skipped: no validated time-decayed Dixon-Coles parameter fitting exists in project yet | 64 |

## Limitations

- Historical squad/player, injury, lineup, and xG data remain scenario/live-only unless historically validated as-of each match date.
- Bookmaker benchmarks are evaluated only where pre-match 1X2 market odds are available.
- Calibrated bookmaker output uses only odds-covered matches before each target tournament; sparse historical odds can limit calibration.
- Dixon-Coles is skipped unless a validated time-decayed scoreline fitting routine is added.
- Accuracy is secondary; log loss, Brier, RPS, and calibration drive the benchmark decision.

## Final Recommendation

Stage 5 is not achieved yet because the official model did not show a statistically meaningful log-loss improvement over bookmaker odds-only. Keep the project SOTA-inspired and research-backtested, not true SOTA.
