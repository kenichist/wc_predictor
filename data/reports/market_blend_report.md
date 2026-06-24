# Market Blend Report

- Market odds join key: `date`, normalized `home_team`, normalized `away_team`
- Best odds conversion method: `multiplicative`
- Alpha formula: `alpha * model_prob + (1 - alpha) * market_prob`
- Best alpha from odds-covered historical backtests: `0.0`

## Alpha Search

| alpha | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | n_matches |
| --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 0.553191 | 0.966467 | 0.569686 | 0.201729 | 0.065499 | 188 |
| 0.100000 | 0.563830 | 0.966883 | 0.570149 | 0.201958 | 0.039771 | 188 |
| 0.200000 | 0.563830 | 0.968109 | 0.571088 | 0.202380 | 0.038704 | 188 |
| 0.300000 | 0.563830 | 0.970104 | 0.572503 | 0.202994 | 0.045319 | 188 |
| 0.400000 | 0.558511 | 0.972845 | 0.574392 | 0.203801 | 0.038633 | 188 |
| 0.500000 | 0.563830 | 0.976326 | 0.576756 | 0.204800 | 0.037641 | 188 |
| 0.600000 | 0.553191 | 0.980548 | 0.579596 | 0.205990 | 0.051141 | 188 |
| 0.700000 | 0.531915 | 0.985531 | 0.582911 | 0.207373 | 0.052035 | 188 |
| 0.800000 | 0.521277 | 0.991305 | 0.586701 | 0.208949 | 0.035485 | 188 |
| 0.900000 | 0.526596 | 0.997921 | 0.590966 | 0.210716 | 0.039236 | 188 |
| 1.000000 | 0.510638 | 1.005457 | 0.595706 | 0.212676 | 0.038339 | 188 |

## Tournament-Year Alpha Diagnostics

| world_cup_year | alpha | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | n_matches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | 0.400000 | 0.603175 | 0.948635 | 0.559039 | 0.195761 | 0.099949 | 63 |
| 2018 | 0.100000 | 0.580645 | 0.935459 | 0.552722 | 0.196841 | 0.096067 | 62 |
| 2022 | 0.000000 | 0.523810 | 1.010652 | 0.591800 | 0.210385 | 0.112230 | 63 |

## Stage-Specific Alpha Diagnostics

| stage | alpha | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | n_matches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final | 0.000000 | 0.666667 | 0.956943 | 0.566269 | 0.171350 | 0.499079 | 3 |
| group_stage | 0.100000 | 0.553191 | 0.981577 | 0.577505 | 0.209284 | 0.025982 | 141 |
| quarterfinal | 1.000000 | 0.500000 | 1.049727 | 0.643406 | 0.204110 | 0.144012 | 12 |
| round_of_16 | 0.000000 | 0.652174 | 0.787686 | 0.453788 | 0.135857 | 0.161130 | 23 |
| semifinal | 0.000000 | 0.500000 | 0.986769 | 0.583583 | 0.213607 | 0.103279 | 6 |
| third_place | 0.800000 | 0.666667 | 0.957069 | 0.582938 | 0.260713 | 0.555454 | 3 |

## Confidence-Weighted Blend Diagnostic

| blend_method | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | n_matches | alpha_mean | alpha_min | alpha_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| confidence_weighted | 0.563830 | 0.980572 | 0.578819 | 0.206161 | 0.049671 | 188 | 0.493202 | 0.158969 | 0.950536 |

## Odds-Covered Metrics

| world_cup_year | model_name | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | n_matches | odds_coverage_rate | alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | bookmaker_odds_only | 0.571429 | 0.952307 | 0.565444 | 0.198209 | 0.091628 | 63 | 0.984375 |  |
| 2014 | market_blend | 0.571429 | 0.952307 | 0.565444 | 0.198209 | 0.091628 | 63 | 0.984375 | 0.000000 |
| 2014 | final_model | 0.539683 | 0.955988 | 0.562923 | 0.197484 | 0.135532 | 63 | 0.984375 |  |
| 2018 | bookmaker_odds_only | 0.564516 | 0.935959 | 0.551525 | 0.196510 | 0.068938 | 62 | 0.968750 |  |
| 2018 | market_blend | 0.564516 | 0.935959 | 0.551525 | 0.196510 | 0.068938 | 62 | 0.968750 | 0.000000 |
| 2018 | final_model | 0.516129 | 0.971863 | 0.583042 | 0.207106 | 0.059067 | 62 | 0.968750 |  |
| 2022 | bookmaker_odds_only | 0.523810 | 1.010652 | 0.591800 | 0.210385 | 0.112230 | 63 | 0.984375 |  |
| 2022 | market_blend | 0.523810 | 1.010652 | 0.591800 | 0.210385 | 0.112230 | 63 | 0.984375 | 0.000000 |
| 2022 | final_model | 0.476190 | 1.087986 | 0.640953 | 0.233349 | 0.132118 | 63 | 0.984375 |  |

## Average Odds-Covered Metrics

| model_name | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | tournaments | total_matches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bookmaker_odds_only | 0.553251 | 0.966306 | 0.569590 | 0.201701 | 0.090932 | 3 | 188 |
| market_blend | 0.553251 | 0.966306 | 0.569590 | 0.201701 | 0.090932 | 3 | 188 |
| final_model | 0.510667 | 1.005279 | 0.595639 | 0.212647 | 0.108906 | 3 | 188 |

## Comparisons

- Final model beats bookmaker_odds_only: `False (log_loss_delta=0.038973)`
- Market blend beats final_model: `True (log_loss_delta=-0.038973)`
- Market blend beats bookmaker_odds_only: `False (log_loss_delta=0.000000)`

## Statistical Significance Summary

| comparison | log_loss_delta | ci_lower | ci_upper | statistically_meaningful |
| --- | --- | --- | --- | --- |
| market_blend_minus_bookmaker_odds_only | 0.000000 | -0.000000 | 0.000000 | False |
| market_blend_minus_final_model | -0.038990 | -0.079331 | -0.002517 | True |
| final_model_minus_bookmaker_odds_only | 0.038990 | 0.001700 | 0.076875 | True |

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

## Notes

- Bookmaker and blend metrics are evaluated only on matches with complete odds.
- Coverage-adjusted final_model rows are included on the same odds-covered match set.
- If odds coverage is incomplete, full-tournament final_model metrics and odds-covered metrics should not be compared as if they used the same rows.
