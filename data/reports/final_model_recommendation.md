# Final Model Recommendation

This project is SOTA-inspired, historically backtested, and benchmarked against bookmaker odds. It is not a true SOTA claim.

## Recommendation

- Safe production model: `football_only_ensemble`
- Safe production feature set: `core_football_only`
- Training estimator: `catboost`
- Current pipeline feature set: `core_football_only`
- Current World Cup simulation used: `safe` features
- Best feature set by validation log_loss: `core_football_only`
- Best safe ablation feature set: `core_football_only`
- Safe feature-set policy: `core_plus_internal_elo` worsens log_loss by 0.004040; keeping the simpler current production path.

## Market-Assisted Benchmark

- Market-assisted status: `benchmark-only`
- Best market blend alpha: `0.300000`
- Best odds conversion method: `multiplicative`
- Market blend formula: `alpha * model_prob + (1 - alpha) * market_prob`
- World Cup 2026 market feature coverage: `0.323944`
- Policy: `market_blend` is benchmark-only until 2026 market odds successfully join to all active fixtures.
- Market blend vs bookmaker odds log_loss delta CI: `delta=-0.003646, 95% CI [-0.015687, 0.007937]`
- Market blend improvement statistically meaningful: `False`

## Validation Metrics

- `accuracy`: 0.593856
- `log_loss`: 0.885778
- `brier_score`: 0.521142
- `ranked_probability_score`: 0.174057
- `calibration_error`: 0.022588

## Ablation Summary

| feature_set | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | warning |
| --- | --- | --- | --- | --- | --- | --- |
| core_football_only | 0.584946 | 0.894662 | 0.525878 | 0.175830 | 0.035662 | unknown |
| core_plus_internal_elo | 0.586482 | 0.898702 | 0.528615 | 0.176726 | 0.034603 | internal historical Elo worsened validation log_loss by 0.004040 |
| v1_baseline | 0.589247 | 0.904491 | 0.531969 | 0.177840 | 0.028365 | unknown |
| v5_plus_pi_ratings | 0.583410 | 0.904905 | 0.531451 | 0.177276 | 0.041453 | unknown |
| v6_full_rating_stack | 0.583410 | 0.904905 | 0.531451 | 0.177276 | 0.041453 | feature set has same requested columns as previous step |
| rating_stack_experimental | 0.583410 | 0.904905 | 0.531451 | 0.177276 | 0.041453 | feature set has same requested columns as previous step. experimental rating stack worsened validation log_loss by 0.010244 |
| v4_plus_dynamic_ratings | 0.581260 | 0.912219 | 0.536890 | 0.179271 | 0.038784 | unknown |
| v2_baseline_plus_fifa_rankings | 0.585561 | 0.912637 | 0.536159 | 0.179116 | 0.032421 | FIFA rankings worsened validation log_loss by 0.008145 |
| v3_plus_external_elo | 0.585561 | 0.912637 | 0.536159 | 0.179116 | 0.032421 | v3_plus_external_elo added 5 columns, but none add usable variation. 3 newly added columns are constant. feature set has same usable columns as previous set. metrics are identical to previous feature set. Add data/external/world_football_elo.csv to enable external Elo features |

## Match-Level World Cup Backtests

| evaluation_scope | model_name | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error |
| --- | --- | --- | --- | --- | --- | --- |
| all_matches | final_model | 0.539062 | 0.968175 | 0.571718 | 0.200402 | 0.144923 |
| all_matches | elo_only | 0.542969 | 1.030580 | 0.609594 | 0.215759 | 0.140584 |
| all_matches | rolling_form_only | 0.496094 | 1.039424 | 0.624144 | 0.223972 | 0.103904 |
| all_matches | majority_class_baseline | 0.414062 | 1.085492 | 0.660247 | 0.243565 | 0.079183 |
| all_matches | fifa_ranking_only | 0.468750 | 1.086250 | 0.644824 | 0.231878 | 0.148811 |
| odds_covered | market_blend | 0.558457 | 0.962681 | 0.568312 | 0.201200 | 0.076287 |
| odds_covered | bookmaker_odds_only | 0.553251 | 0.966306 | 0.569590 | 0.201701 | 0.090932 |
| odds_covered | final_model | 0.526711 | 0.980588 | 0.578898 | 0.206101 | 0.144835 |

- Final model beats Elo-only on all-match log_loss: `True (log_loss_delta=-0.062405)`
- Final model beats FIFA-only on all-match log_loss: `True (log_loss_delta=-0.118075)`
- Final model beats rolling-form-only on all-match log_loss: `True (log_loss_delta=-0.071249)`
- Final model beats majority-class on all-match log_loss: `True (log_loss_delta=-0.117317)`
- Market blend beats final model on odds-covered log_loss: `True (log_loss_delta=-0.017907)`
- Market blend beats bookmaker odds on odds-covered log_loss: `True (log_loss_delta=-0.003625)`

## Tournament Simulation Backtest

| world_cup_year | model_name | actual_champion | champion_probability_assigned | actual_finalists_probability | semifinalist_probability_recall | round_of_16_brier_score | group_qualification_accuracy | group_match_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2010 | final_model | Spain | 0.196000 | 0.245000 | 0.500000 | 0.160176 | 0.812500 | 0.981597 |
| 2010 | elo_only | Spain | 0.239000 | 0.272500 | 0.500000 | 0.177531 | 0.750000 | 1.103924 |
| 2014 | final_model | Germany | 0.122000 | 0.203000 | 0.750000 | 0.184356 | 0.625000 | 0.901995 |
| 2014 | elo_only | Germany | 0.127000 | 0.204000 | 0.750000 | 0.232563 | 0.687500 | 1.045095 |
| 2014 | bookmaker_odds_only | Germany | 0.129000 | 0.213500 | 0.750000 | 0.238097 | 0.500000 | 0.952901 |
| 2014 | market_blend | Germany | 0.134000 | 0.240500 | 0.750000 | 0.222206 | 0.625000 | 0.930878 |
| 2018 | final_model | France | 0.070000 | 0.090500 | 0.000000 | 0.131538 | 0.812500 | 0.942663 |
| 2018 | elo_only | France | 0.080000 | 0.100000 | 0.000000 | 0.156700 | 0.812500 | 1.035081 |
| 2018 | bookmaker_odds_only | France | 0.095000 | 0.104500 | 0.250000 | 0.123914 | 0.875000 | 0.929146 |
| 2018 | market_blend | France | 0.095000 | 0.108500 | 0.250000 | 0.127810 | 0.875000 | 0.926595 |
| 2022 | final_model | Argentina | 0.150000 | 0.217500 | 0.500000 | 0.203976 | 0.625000 | 1.100672 |
| 2022 | elo_only | Argentina | 0.112000 | 0.203000 | 0.500000 | 0.193620 | 0.562500 | 1.076344 |
| 2022 | bookmaker_odds_only | Argentina | 0.166000 | 0.230000 | 0.500000 | 0.221797 | 0.625000 | 1.040835 |
| 2022 | market_blend | Argentina | 0.144000 | 0.229500 | 0.500000 | 0.208195 | 0.625000 | 1.041137 |

Average:

| model_name | world_cups | champion_probability_assigned | actual_finalists_probability | semifinalist_probability_recall | round_of_16_brier_score | group_qualification_accuracy | group_match_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| market_blend | 3 | 0.124333 | 0.192833 | 0.500000 | 0.186070 | 0.708333 | 0.966203 |
| bookmaker_odds_only | 3 | 0.130000 | 0.182667 | 0.500000 | 0.194603 | 0.666667 | 0.974294 |
| final_model | 4 | 0.134500 | 0.189000 | 0.437500 | 0.170011 | 0.718750 | 0.981732 |
| elo_only | 4 | 0.139500 | 0.194875 | 0.437500 | 0.190103 | 0.703125 | 1.065111 |

## Prediction And Simulation Metadata

- Prediction file matches safe recommendation: `True`
- Simulation metadata matches safe recommendation: `True`
- Prediction model counts: `football_only_ensemble:71`
- Prediction feature-set counts: `core_football_only:71`
- Simulation prediction source: `C:\Users\kenic\OneDrive\Desktop\wc_predictor\data\predictions\ensemble_predictions.parquet`
- Simulation model counts: `football_only_ensemble:103`
- Simulation feature-set counts: `core_football_only:103`

## External Rating Checks

- FIFA helped validation: `False` (log_loss_delta=0.008145)
- FIFA historical training coverage: `0.408255`
- External Elo helped validation: `False` (log_loss_delta=0.000000)
- External Elo historical training coverage: `0.000000`

## Feature Policy

- `core_football_only` groups: baseline, basic_elo, rolling_form, dynamic_ratings, pi_ratings
- Experimental and excluded from the safe production path: sparse FIFA rankings, external Elo snapshots, xG, squad/player snapshots, injury/suspension snapshots, and market odds unless coverage is complete and no-leakage validation supports promotion.
- Historical FIFA/Elo ingestion is implemented, but sparse coverage means these sources are not selected for the safe final model.
- Squad/player, injury/suspension, and xG interfaces are not safe historical training features with the currently available data.

## Warnings

- Do not use `market_blend` for 2026 production predictions until market odds join to all active 2026 fixtures.
- Current 2026 market odds rows exist, but the latest feature-null report shows market World Cup 2026 coverage below production readiness.
- The final simulation should be interpreted as the safe football-only path unless simulation metadata shows complete market blending.
- `v1_baseline` is not recommended because current ablation results place stronger safe feature sets ahead of it.
- This project is SOTA-inspired, historically backtested, and benchmarked against bookmaker odds.
