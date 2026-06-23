# Historical World Cup Backtest Report

- Final model feature set: `core_football_only`
- Tournaments: 2010, 2014, 2018, 2022 FIFA World Cups
- Leakage rule: each fold trains only on matches dated before that tournament's opening match.
- Current 2026 squad/player, injury, suspension, and xG interfaces are excluded from historical backtests unless proper historical as-of data is available.

## Metrics By Tournament

| world_cup_year | evaluation_scope | model_name | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | n_matches | n_train | odds_coverage_rate | alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2010 | all_matches | final_model | 0.562500 | 0.944394 | 0.558949 | 0.188222 | 0.143559 | 64 | 33864 |  |  |
| 2010 | all_matches | elo_only | 0.531250 | 1.054211 | 0.619297 | 0.212548 | 0.117637 | 64 | 33864 |  |  |
| 2010 | all_matches | rolling_form_only | 0.453125 | 1.058554 | 0.637372 | 0.223442 | 0.109094 | 64 | 33864 |  |  |
| 2010 | all_matches | majority_class_baseline | 0.375000 | 1.114277 | 0.680523 | 0.246199 | 0.120511 | 64 | 33864 |  |  |
| 2010 | all_matches | fifa_ranking_only | 0.375000 | 1.114284 | 0.680528 | 0.246202 | 0.120521 | 64 | 33864 |  |  |
| 2014 | all_matches | final_model | 0.546875 | 0.902930 | 0.532650 | 0.184741 | 0.236816 | 64 | 37861 |  |  |
| 2014 | all_matches | rolling_form_only | 0.625000 | 0.982520 | 0.584660 | 0.206694 | 0.156258 | 64 | 37861 |  |  |
| 2014 | all_matches | elo_only | 0.609375 | 1.007321 | 0.596176 | 0.212633 | 0.156123 | 64 | 37861 |  |  |
| 2014 | all_matches | majority_class_baseline | 0.453125 | 1.059328 | 0.641554 | 0.239570 | 0.040628 | 64 | 37861 |  |  |
| 2014 | all_matches | fifa_ranking_only | 0.437500 | 1.069958 | 0.646672 | 0.229912 | 0.182763 | 64 | 37861 |  |  |
| 2014 | odds_covered | final_model | 0.539683 | 0.913596 | 0.540007 | 0.187315 | 0.243848 | 63 | 37861 | 0.984375 |  |
| 2014 | odds_covered | market_blend | 0.603175 | 0.934863 | 0.553836 | 0.193115 | 0.073277 | 63 | 37861 | 0.984375 | 0.300000 |
| 2014 | odds_covered | bookmaker_odds_only | 0.571429 | 0.952307 | 0.565444 | 0.198209 | 0.091628 | 63 | 37861 | 0.984375 |  |
| 2018 | all_matches | final_model | 0.546875 | 0.964451 | 0.572869 | 0.204241 | 0.051124 | 64 | 41639 |  |  |
| 2018 | all_matches | elo_only | 0.546875 | 1.033521 | 0.620381 | 0.226063 | 0.117459 | 64 | 41639 |  |  |
| 2018 | all_matches | rolling_form_only | 0.468750 | 1.036880 | 0.620524 | 0.229467 | 0.061064 | 64 | 41639 |  |  |
| 2018 | all_matches | fifa_ranking_only | 0.546875 | 1.076106 | 0.632840 | 0.231070 | 0.145611 | 64 | 41639 |  |  |
| 2018 | all_matches | majority_class_baseline | 0.390625 | 1.094111 | 0.667756 | 0.252659 | 0.101582 | 64 | 41639 |  |  |
| 2018 | odds_covered | bookmaker_odds_only | 0.564516 | 0.935959 | 0.551525 | 0.196510 | 0.068938 | 62 | 41639 | 0.968750 |  |
| 2018 | odds_covered | market_blend | 0.548387 | 0.936431 | 0.553247 | 0.197478 | 0.068308 | 62 | 41639 | 0.968750 | 0.300000 |
| 2018 | odds_covered | final_model | 0.548387 | 0.957752 | 0.567746 | 0.204262 | 0.034232 | 62 | 41639 | 0.968750 |  |
| 2022 | all_matches | elo_only | 0.484375 | 1.027267 | 0.602523 | 0.211792 | 0.171117 | 64 | 45700 |  |  |
| 2022 | all_matches | final_model | 0.500000 | 1.060925 | 0.622404 | 0.224403 | 0.148191 | 64 | 45700 |  |  |
| 2022 | all_matches | majority_class_baseline | 0.437500 | 1.074251 | 0.651156 | 0.235831 | 0.054010 | 64 | 45700 |  |  |
| 2022 | all_matches | rolling_form_only | 0.437500 | 1.079741 | 0.654020 | 0.236285 | 0.089200 | 64 | 45700 |  |  |
| 2022 | all_matches | fifa_ranking_only | 0.515625 | 1.084651 | 0.619255 | 0.220330 | 0.146350 | 64 | 45700 |  |  |
| 2022 | odds_covered | bookmaker_odds_only | 0.523810 | 1.010652 | 0.591800 | 0.210385 | 0.112230 | 63 | 45700 | 0.984375 |  |
| 2022 | odds_covered | market_blend | 0.523810 | 1.016749 | 0.597852 | 0.213006 | 0.087274 | 63 | 45700 | 0.984375 | 0.300000 |
| 2022 | odds_covered | final_model | 0.492063 | 1.070416 | 0.628940 | 0.226726 | 0.156425 | 63 | 45700 | 0.984375 |  |

## Average Metrics

| evaluation_scope | model_name | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | tournaments | total_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_matches | final_model | 0.539062 | 0.968175 | 0.571718 | 0.200402 | 0.144923 | 4 | 256 |
| all_matches | elo_only | 0.542969 | 1.030580 | 0.609594 | 0.215759 | 0.140584 | 4 | 256 |
| all_matches | rolling_form_only | 0.496094 | 1.039424 | 0.624144 | 0.223972 | 0.103904 | 4 | 256 |
| all_matches | majority_class_baseline | 0.414062 | 1.085492 | 0.660247 | 0.243565 | 0.079183 | 4 | 256 |
| all_matches | fifa_ranking_only | 0.468750 | 1.086250 | 0.644824 | 0.231878 | 0.148811 | 4 | 256 |
| odds_covered | market_blend | 0.558457 | 0.962681 | 0.568312 | 0.201200 | 0.076287 | 3 | 188 |
| odds_covered | bookmaker_odds_only | 0.553251 | 0.966306 | 0.569590 | 0.201701 | 0.090932 | 3 | 188 |
| odds_covered | final_model | 0.526711 | 0.980588 | 0.578898 | 0.206101 | 0.144835 | 3 | 188 |

## Model Ranking By Log Loss

| evaluation_scope | model_name | log_loss | accuracy | calibration_error | tournaments |
| --- | --- | --- | --- | --- | --- |
| all_matches | final_model | 0.968175 | 0.539062 | 0.144923 | 4 |
| all_matches | elo_only | 1.030580 | 0.542969 | 0.140584 | 4 |
| all_matches | rolling_form_only | 1.039424 | 0.496094 | 0.103904 | 4 |
| all_matches | majority_class_baseline | 1.085492 | 0.414062 | 0.079183 | 4 |
| all_matches | fifa_ranking_only | 1.086250 | 0.468750 | 0.148811 | 4 |
| odds_covered | market_blend | 0.962681 | 0.558457 | 0.076287 | 3 |
| odds_covered | bookmaker_odds_only | 0.966306 | 0.553251 | 0.090932 | 3 |
| odds_covered | final_model | 0.980588 | 0.526711 | 0.144835 | 3 |

## Benchmark Comparisons

- Final model beats Elo-only: `True (log_loss_delta=-0.062405)`
- Final model beats FIFA-only: `True (log_loss_delta=-0.118075)`
- Final model beats bookmaker odds on odds-covered matches: `False (log_loss_delta=0.014283)`
- Market blend beats final model on odds-covered matches: `True (log_loss_delta=-0.017907)`
- Market blend beats bookmaker odds on odds-covered matches: `True (log_loss_delta=-0.003625)`

## Odds Coverage

| world_cup_year | odds_covered_matches | odds_total_matches | odds_missing_matches | odds_coverage_rate |
| --- | --- | --- | --- | --- |
| 2010 | 0 | 64 | 64 | 0.000000 |
| 2014 | 63 | 64 | 1 | 0.984375 |
| 2018 | 62 | 64 | 2 | 0.968750 |
| 2022 | 63 | 64 | 1 | 0.984375 |

## Missing Odds Matches

| world_cup_year | date | home_team | away_team |
| --- | --- | --- | --- |
| 2010 | 2010-06-11 | South Africa | Mexico |
| 2010 | 2010-06-11 | Uruguay | France |
| 2010 | 2010-06-12 | England | United States |
| 2010 | 2010-06-12 | South Korea | Greece |
| 2010 | 2010-06-12 | Argentina | Nigeria |
| 2010 | 2010-06-13 | Serbia | Ghana |
| 2010 | 2010-06-13 | Algeria | Slovenia |
| 2010 | 2010-06-13 | Germany | Australia |
| 2010 | 2010-06-14 | Netherlands | Denmark |
| 2010 | 2010-06-14 | Italy | Paraguay |
| 2010 | 2010-06-14 | Japan | Cameroon |
| 2010 | 2010-06-15 | Ivory Coast | Portugal |
| 2010 | 2010-06-15 | Brazil | North Korea |
| 2010 | 2010-06-15 | New Zealand | Slovakia |
| 2010 | 2010-06-16 | Honduras | Chile |
| 2010 | 2010-06-16 | Spain | Switzerland |
| 2010 | 2010-06-16 | South Africa | Uruguay |
| 2010 | 2010-06-17 | France | Mexico |
| 2010 | 2010-06-17 | Argentina | South Korea |
| 2010 | 2010-06-17 | Greece | Nigeria |
| 2010 | 2010-06-18 | England | Algeria |
| 2010 | 2010-06-18 | Slovenia | United States |
| 2010 | 2010-06-18 | Germany | Serbia |
| 2010 | 2010-06-19 | Netherlands | Japan |
| 2010 | 2010-06-19 | Ghana | Australia |
| 2010 | 2010-06-19 | Cameroon | Denmark |
| 2010 | 2010-06-20 | Slovakia | Paraguay |
| 2010 | 2010-06-20 | Italy | New Zealand |
| 2010 | 2010-06-20 | Brazil | Ivory Coast |
| 2010 | 2010-06-21 | Spain | Honduras |
| 2010 | 2010-06-21 | Portugal | North Korea |
| 2010 | 2010-06-21 | Chile | Switzerland |
| 2010 | 2010-06-22 | South Africa | France |
| 2010 | 2010-06-22 | Nigeria | South Korea |
| 2010 | 2010-06-22 | Mexico | Uruguay |
| 2010 | 2010-06-22 | Greece | Argentina |
| 2010 | 2010-06-23 | Ghana | Germany |
| 2010 | 2010-06-23 | Australia | Serbia |
| 2010 | 2010-06-23 | Slovenia | England |
| 2010 | 2010-06-23 | United States | Algeria |
| 2010 | 2010-06-24 | Denmark | Japan |
| 2010 | 2010-06-24 | Paraguay | New Zealand |
| 2010 | 2010-06-24 | Slovakia | Italy |
| 2010 | 2010-06-24 | Cameroon | Netherlands |
| 2010 | 2010-06-25 | Switzerland | Honduras |
| 2010 | 2010-06-25 | Portugal | Brazil |
| 2010 | 2010-06-25 | North Korea | Ivory Coast |
| 2010 | 2010-06-25 | Chile | Spain |
| 2010 | 2010-06-26 | United States | Ghana |
| 2010 | 2010-06-26 | Uruguay | South Korea |
| 2010 | 2010-06-27 | Argentina | Mexico |
| 2010 | 2010-06-27 | Germany | England |
| 2010 | 2010-06-28 | Brazil | Chile |
| 2010 | 2010-06-28 | Netherlands | Slovakia |
| 2010 | 2010-06-29 | Spain | Portugal |
| 2010 | 2010-06-29 | Paraguay | Japan |
| 2010 | 2010-07-02 | Netherlands | Brazil |
| 2010 | 2010-07-02 | Uruguay | Ghana |
| 2010 | 2010-07-03 | Paraguay | Spain |
| 2010 | 2010-07-03 | Argentina | Germany |
| 2010 | 2010-07-06 | Uruguay | Netherlands |
| 2010 | 2010-07-07 | Germany | Spain |
| 2010 | 2010-07-10 | Uruguay | Germany |
| 2010 | 2010-07-11 | Netherlands | Spain |
| 2014 | 2014-06-23 | Brazil | Cameroon |
| 2018 | 2018-06-25 | Russia | Uruguay |
| 2018 | 2018-07-01 | Russia | Spain |
| 2022 | 2022-11-29 | Qatar | Netherlands |

## Calibration Summary

Final model average calibration_error is 0.144923. Best average calibration_error is `market_blend` at 0.076287.

## Known Limitations

- This is a match-level backtest, not a full historical tournament simulation with group tables and knockout brackets.
- Historical FIFA and external rating features are joined strictly as-of the match date; sparse snapshots can reduce benchmark strength.
- Historical squad/player, injury, suspension, and xG features are excluded because the available project files are current snapshots, not historical as-of records.
- Bookmaker odds benchmarks are included where complete odds were available.
- Knockout matches are evaluated using the stored match outcome class in the training data.
