# Stage 5 Failure Diagnosis Report

## Executive Summary

- Stage 5 status remains: `Stage 5 not yet achieved`.
- This diagnosis does not change the official production model.
- Official production model remains `football_only_ensemble` with `core_football_only`.
- All official-vs-bookmaker comparisons below use paired matches where both prediction rows exist.

## Main Evidence

| candidate_model | baseline_model | metric | mean_delta | ci_lower | ci_upper | statistically_meaningful | n_matches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| official_model | bookmaker_odds_only | log_loss | 0.038990 | -0.004686 | 0.076844 | False | 188 |

Negative delta means the candidate is better. The Stage 5 benchmark requires the upper confidence bound to be below 0.

## Official Model vs Bookmaker Odds By Year

| tournament_year | n_matches | official_log_loss | bookmaker_log_loss | delta_log_loss | official_brier | bookmaker_brier | delta_brier | official_rps | bookmaker_rps | delta_rps | official_accuracy | bookmaker_accuracy | winner_by_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | 63 | 0.955988 | 0.952307 | 0.003681 | 0.562923 | 0.565444 | -0.002522 | 0.197484 | 0.198209 | -0.000724 | 0.539683 | 0.571429 | bookmaker_odds_only |
| 2018 | 62 | 0.971863 | 0.935959 | 0.035905 | 0.583042 | 0.551525 | 0.031517 | 0.207106 | 0.196510 | 0.010597 | 0.516129 | 0.564516 | bookmaker_odds_only |
| 2022 | 63 | 1.087986 | 1.010652 | 0.077334 | 0.640953 | 0.591800 | 0.049153 | 0.233349 | 0.210385 | 0.022964 | 0.476190 | 0.523810 | bookmaker_odds_only |

## Official Model vs Bookmaker Odds By Stage

| stage | n_matches | official_log_loss | bookmaker_log_loss | delta_log_loss | official_brier | bookmaker_brier | delta_brier | official_rps | bookmaker_rps | delta_rps | official_accuracy | bookmaker_accuracy | winner_by_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final | 3 | 1.231681 | 0.956943 | 0.274738 | 0.756407 | 0.566269 | 0.190138 | 0.241478 | 0.171350 | 0.070128 | 0.000000 | 0.666667 | bookmaker_odds_only |
| group_stage | 141 | 1.017844 | 0.981888 | 0.035955 | 0.601396 | 0.577486 | 0.023910 | 0.219004 | 0.209284 | 0.009720 | 0.517730 | 0.539007 | bookmaker_odds_only |
| quarterfinal | 12 | 1.049727 | 1.118702 | -0.068975 | 0.643406 | 0.691241 | -0.047835 | 0.204110 | 0.227382 | -0.023272 | 0.500000 | 0.500000 | official_model |
| round_of_16 | 23 | 0.853336 | 0.787686 | 0.065650 | 0.492000 | 0.453788 | 0.038212 | 0.152632 | 0.135857 | 0.016776 | 0.608696 | 0.652174 | bookmaker_odds_only |
| semifinal | 6 | 1.119769 | 0.986769 | 0.133000 | 0.688715 | 0.583583 | 0.105132 | 0.271814 | 0.213607 | 0.058207 | 0.333333 | 0.500000 | bookmaker_odds_only |
| third_place | 3 | 0.957598 | 0.972296 | -0.014698 | 0.585843 | 0.580998 | 0.004845 | 0.262782 | 0.255649 | 0.007132 | 0.333333 | 0.666667 | official_model |

## Official Model vs Bookmaker Odds By Actual Result

| actual_result | n_matches | official_log_loss | bookmaker_log_loss | delta_log_loss | official_brier | bookmaker_brier | delta_brier | official_rps | bookmaker_rps | delta_rps | official_accuracy | bookmaker_accuracy | winner_by_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| away_win | 67 | 0.997916 | 0.985082 | 0.012833 | 0.579586 | 0.565873 | 0.013713 | 0.255135 | 0.247716 | 0.007418 | 0.582090 | 0.582090 | bookmaker_odds_only |
| draw | 40 | 1.399248 | 1.337922 | 0.061326 | 0.892826 | 0.876475 | 0.016351 | 0.165730 | 0.167992 | -0.002262 | 0.050000 | 0.000000 | bookmaker_odds_only |
| home_win | 81 | 0.817229 | 0.767635 | 0.049595 | 0.462314 | 0.421339 | 0.040976 | 0.200739 | 0.180349 | 0.020390 | 0.679012 | 0.802469 | bookmaker_odds_only |

## Worst Official-Model Misses Relative To Bookmaker Odds

| date | match_id | tournament_year | stage | home_team | away_team | actual_result | official_predicted_result | bookmaker_predicted_result | official_log_loss | bookmaker_log_loss | delta_log_loss | official_home_prob | official_draw_prob | official_away_prob | bookmaker_home_prob | bookmaker_draw_prob | bookmaker_away_prob |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022-11-28 | historical_41a08824e106c23b | 2022 | group_stage | South Korea | Ghana | away_win | home_win | home_win | 2.207921 | 1.079133 | 1.128788 | 0.677052 | 0.213019 | 0.109929 | 0.351220 | 0.308890 | 0.339890 |
| 2018-06-17 | historical_c01e81b0eb561061 | 2018 | group_stage | Costa Rica | Serbia | away_win | draw | away_win | 1.322009 | 0.636049 | 0.685960 | 0.328492 | 0.404909 | 0.266599 | 0.184132 | 0.286488 | 0.529380 |
| 2022-11-25 | historical_5f148dad567ebddb | 2022 | group_stage | Qatar | Senegal | away_win | home_win | away_win | 1.255806 | 0.582982 | 0.672823 | 0.421151 | 0.294002 | 0.284846 | 0.174447 | 0.267322 | 0.558231 |
| 2022-12-02 | historical_ffc2f7f054e75464 | 2022 | group_stage | Cameroon | Brazil | home_win | away_win | away_win | 2.868255 | 2.246416 | 0.621839 | 0.056798 | 0.066667 | 0.876535 | 0.105778 | 0.182410 | 0.711812 |
| 2022-11-30 | historical_0f4bd52b72301d86 | 2022 | group_stage | Tunisia | France | home_win | away_win | away_win | 2.366697 | 1.818099 | 0.548598 | 0.093790 | 0.211279 | 0.694931 | 0.162334 | 0.248827 | 0.588839 |
| 2018-07-10 | historical_e110976fb6338b7b | 2018 | semifinal | France | Belgium | home_win | away_win | home_win | 1.495201 | 0.987715 | 0.507486 | 0.224204 | 0.260047 | 0.515749 | 0.372427 | 0.308174 | 0.319399 |
| 2014-07-01 | historical_3513de9543161c04 | 2014 | round_of_16 | Belgium | United States | home_win | away_win | home_win | 1.230311 | 0.736456 | 0.493855 | 0.292202 | 0.290950 | 0.416848 | 0.478808 | 0.284159 | 0.237033 |
| 2022-12-18 | historical_90fa2d826fe342e0 | 2022 | final | Argentina | France | draw | away_win | home_win | 1.679004 | 1.189337 | 0.489667 | 0.313910 | 0.186560 | 0.499530 | 0.361141 | 0.304423 | 0.334436 |
| 2018-06-26 | historical_41887b6dd6e01905 | 2018 | group_stage | Denmark | France | draw | away_win | away_win | 1.535925 | 1.052238 | 0.483687 | 0.221227 | 0.215257 | 0.563516 | 0.184185 | 0.349156 | 0.466660 |
| 2018-06-15 | historical_3f2d2e78a580b1d6 | 2018 | group_stage | Portugal | Spain | draw | away_win | away_win | 1.656697 | 1.213146 | 0.443551 | 0.152989 | 0.190768 | 0.656243 | 0.225365 | 0.297261 | 0.477374 |
| 2018-06-16 | historical_aeab86a03dc9c3d9 | 2018 | group_stage | Peru | Denmark | away_win | home_win | away_win | 1.305497 | 0.865120 | 0.440377 | 0.399558 | 0.329404 | 0.271038 | 0.264270 | 0.314729 | 0.421001 |
| 2022-11-29 | historical_433d870297c7ca84 | 2022 | group_stage | Ecuador | Senegal | away_win | home_win | home_win | 1.650003 | 1.211869 | 0.438134 | 0.600275 | 0.207676 | 0.192049 | 0.387909 | 0.314451 | 0.297641 |
| 2018-06-14 | historical_e21f91d5c8997faa | 2018 | group_stage | Russia | Saudi Arabia | home_win | draw | home_win | 0.838607 | 0.406558 | 0.432049 | 0.432312 | 0.441344 | 0.126344 | 0.665939 | 0.225063 | 0.108999 |
| 2022-12-01 | historical_70c40eac2d3a18d9 | 2022 | group_stage | Costa Rica | Germany | away_win | away_win | away_win | 0.563598 | 0.137588 | 0.426011 | 0.237667 | 0.193176 | 0.569157 | 0.038930 | 0.089612 | 0.871458 |
| 2018-06-21 | historical_f2cf6cfbdc7e317c | 2018 | group_stage | Denmark | Australia | draw | home_win | home_win | 1.662753 | 1.258890 | 0.403863 | 0.597905 | 0.189616 | 0.212479 | 0.505820 | 0.283969 | 0.210211 |
| 2022-11-27 | historical_145dfa46f0fe6a6f | 2022 | group_stage | Croatia | Canada | home_win | away_win | home_win | 1.234596 | 0.839462 | 0.395134 | 0.290952 | 0.261230 | 0.447818 | 0.431943 | 0.294203 | 0.273854 |
| 2014-06-25 | historical_d79643fb4a9d4b24 | 2014 | group_stage | Bosnia and Herzegovina | Iran | home_win | away_win | home_win | 1.205085 | 0.821528 | 0.383557 | 0.299666 | 0.279638 | 0.420695 | 0.439759 | 0.272351 | 0.287890 |
| 2022-11-27 | historical_ddb7b2cf8119d2e2 | 2022 | group_stage | Belgium | Morocco | away_win | home_win | home_win | 1.912721 | 1.539479 | 0.373242 | 0.691300 | 0.161022 | 0.147678 | 0.503145 | 0.282362 | 0.214493 |
| 2022-12-02 | historical_db2cc8f055377b92 | 2022 | group_stage | South Korea | Portugal | home_win | away_win | away_win | 1.844461 | 1.498411 | 0.346049 | 0.158111 | 0.194570 | 0.647319 | 0.223485 | 0.254641 | 0.521874 |
| 2022-11-27 | historical_b9ace2b40b17c60a | 2022 | group_stage | Spain | Germany | draw | home_win | home_win | 1.610628 | 1.270735 | 0.339893 | 0.581516 | 0.199762 | 0.218722 | 0.379406 | 0.280625 | 0.339969 |
| 2018-06-21 | historical_3444ffdebc66d575 | 2018 | group_stage | France | Peru | home_win | draw | home_win | 0.848807 | 0.511671 | 0.337137 | 0.427925 | 0.428648 | 0.143427 | 0.599493 | 0.252911 | 0.147596 |
| 2014-06-15 | historical_c6f31ccc429d2c69 | 2014 | group_stage | Switzerland | Ecuador | home_win | away_win | home_win | 1.255502 | 0.922049 | 0.333453 | 0.284933 | 0.283619 | 0.431449 | 0.397703 | 0.297348 | 0.304948 |
| 2018-07-15 | historical_22eca05f3197aafd | 2018 | final | France | Croatia | home_win | draw | home_win | 1.110900 | 0.789626 | 0.321274 | 0.329262 | 0.372964 | 0.297774 | 0.454014 | 0.312961 | 0.233024 |
| 2014-07-05 | historical_9083848259398f0e | 2014 | quarterfinal | Netherlands | Costa Rica | draw | home_win | home_win | 1.771420 | 1.456737 | 0.314684 | 0.691870 | 0.170091 | 0.138038 | 0.630007 | 0.232995 | 0.136997 |
| 2018-06-22 | historical_c7b2d0524018f32a | 2018 | group_stage | Nigeria | Iceland | home_win | away_win | away_win | 1.428551 | 1.121780 | 0.306771 | 0.239656 | 0.328942 | 0.431402 | 0.325699 | 0.316151 | 0.358149 |
| 2018-06-16 | historical_43f34ab319babc05 | 2018 | group_stage | France | Australia | home_win | home_win | home_win | 0.560055 | 0.261691 | 0.298364 | 0.571178 | 0.299273 | 0.129550 | 0.769749 | 0.138753 | 0.091498 |
| 2014-06-29 | historical_79e2778394feef76 | 2014 | round_of_16 | Costa Rica | Greece | draw | away_win | home_win | 1.465580 | 1.169370 | 0.296210 | 0.379179 | 0.230944 | 0.389877 | 0.359599 | 0.310563 | 0.329839 |
| 2014-07-09 | historical_139c424f41341d1f | 2014 | semifinal | Netherlands | Argentina | draw | away_win | away_win | 1.492921 | 1.201573 | 0.291348 | 0.370575 | 0.224715 | 0.404709 | 0.315559 | 0.300721 | 0.383720 |
| 2014-06-17 | historical_0b0c31de1a58c00c | 2014 | group_stage | Belgium | Algeria | home_win | home_win | home_win | 0.620943 | 0.333748 | 0.287195 | 0.537437 | 0.270672 | 0.191891 | 0.716234 | 0.186723 | 0.097043 |
| 2022-11-21 | historical_5d4d1b65bd378920 | 2022 | group_stage | United States | Wales | draw | home_win | home_win | 1.455725 | 1.173048 | 0.282676 | 0.457751 | 0.233231 | 0.309017 | 0.390916 | 0.309422 | 0.299661 |

## Why Market Blend Equals Bookmaker Odds

- Prediction rows identical: `True`.
- Metric rows identical: `True`.
- Stored blend alpha: `0.0`.
- Stored odds conversion method: `multiplicative`.
- Explanation: The stored market-blend alpha is 0.0, so blend = alpha * model + (1 - alpha) * market collapses to pure bookmaker market probabilities.

## Ablation vs Stage 5 Reconciliation

| source | evaluation_scope | model_name | feature_set | n_matches | log_loss | brier_score | ranked_probability_score | calibration_error | explanation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ablation_results.csv | project time-aware validation split | ablation_estimator | core_football_only |  | 0.894662 | 0.525878 | 0.175830 | 0.035662 | Ablation ranking uses the full historical match validation split, not held-out World Cup-only tournaments. |
| stage5_benchmark_metrics.csv | held-out World Cup tournaments, all matches | official_model | football_only_ensemble/core_football_only | 192.000000 | 1.004215 | 0.594659 | 0.211484 | 0.110788 | Stage 5 official-model score is tournament-only and no-leakage by World Cup year. |
| stage5_match_predictions.csv | paired odds-covered World Cup matches | official_model | football_only_ensemble/core_football_only | 188.000000 | 1.005457 | 0.595706 | 0.212676 |  | This is the exact official-model subset used against bookmaker odds-only. |
| stage5_match_predictions.csv | paired odds-covered World Cup matches | bookmaker_odds_only | best_market_odds_conversion | 188.000000 | 0.966467 | 0.569686 | 0.201729 |  | Bookmaker baseline is evaluated only where pre-match 1X2 odds are available. |

The ablation log loss and Stage 5 official-model log loss are not measuring the same experiment. Ablation uses the project's broad time-aware validation split across the full historical match dataset, while Stage 5 uses held-out World Cup tournaments only with no-leakage tournament cutoffs. World Cup-only matches are a smaller, harder, and differently distributed sample.

## CatBoost Status

| catboost_installed | catboost_being_used_if_rerun_now | saved_model_name | expected_training_backend | catboost_gpu_enabled | stage5_outputs_store_estimator_backend | notes |
| --- | --- | --- | --- | --- | --- | --- |
| True | True | present_not_inspected | catboost | True | False | CatBoost is importable in the current environment. Stage 5 reruns that request --model catboost should use CatBoost. Existing Stage 5 CSVs do not store the fold estimator backend. |

## Calibration Notes

| model_name | rows | mean_abs_calibration_gap | max_abs_calibration_gap |
| --- | --- | --- | --- |
| random_uniform | 4 | 0.028646 | 0.072917 |
| historical_class_frequency | 4 | 0.065407 | 0.101582 |
| bookmaker_odds_only | 24 | 0.088307 | 0.264725 |
| bookmaker_raw_implied | 24 | 0.088307 | 0.264725 |
| market_blend | 24 | 0.088307 | 0.264725 |
| bookmaker_calibrated | 18 | 0.102299 | 0.252147 |
| rolling_form_only | 20 | 0.110445 | 0.260899 |
| elo_only | 22 | 0.118912 | 0.317927 |
| poisson_goal_model | 20 | 0.149082 | 0.374506 |
| official_model | 22 | 0.166824 | 0.876535 |
| fifa_ranking_only | 27 | 0.180056 | 0.800208 |

## Next Recommended Improvement

- Keep the production model unchanged until a rerun shows statistically meaningful improvement over bookmaker odds-only.
- Focus on the largest positive official-minus-bookmaker log-loss slices in the by-year, by-stage, and by-outcome CSVs.
- Retune market blend only after the model side improves; alpha 0.0 currently means the blend is just bookmaker odds.
- First diagnostic slice to inspect: stage `final` with delta_log_loss `0.274738`.
- Outcome slice to inspect: `draw` with delta_log_loss `0.061326`.

## Input Reports Used

- Stage 5 decision report present: `True`.
- Stage 5 benchmark report present: `True`.
