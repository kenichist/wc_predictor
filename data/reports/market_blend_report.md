# Market Blend Report

- Market odds join key: `date`, normalized `home_team`, normalized `away_team`
- Best odds conversion method: `multiplicative`
- Alpha formula: `alpha * model_prob + (1 - alpha) * market_prob`
- Best alpha from odds-covered historical backtests: `0.3`

## Alpha Search

| alpha | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | n_matches |
| --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 0.553191 | 0.966467 | 0.569686 | 0.201729 | 0.065499 | 188 |
| 0.100000 | 0.558511 | 0.964449 | 0.568866 | 0.201385 | 0.037685 | 188 |
| 0.200000 | 0.574468 | 0.963252 | 0.568435 | 0.201216 | 0.043319 | 188 |
| 0.300000 | 0.558511 | 0.962821 | 0.568392 | 0.201220 | 0.026706 | 188 |
| 0.400000 | 0.547872 | 0.963121 | 0.568737 | 0.201397 | 0.043608 | 188 |
| 0.500000 | 0.547872 | 0.964137 | 0.569470 | 0.201749 | 0.049218 | 188 |
| 0.600000 | 0.547872 | 0.965871 | 0.570591 | 0.202274 | 0.045462 | 188 |
| 0.700000 | 0.547872 | 0.968342 | 0.572100 | 0.202973 | 0.047640 | 188 |
| 0.800000 | 0.531915 | 0.971589 | 0.573998 | 0.203845 | 0.054021 | 188 |
| 0.900000 | 0.526596 | 0.975676 | 0.576283 | 0.204891 | 0.068321 | 188 |
| 1.000000 | 0.526596 | 0.980710 | 0.578957 | 0.206111 | 0.068720 | 188 |

## Tournament-Year Alpha Diagnostics

| world_cup_year | alpha | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | n_matches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | 1.000000 | 0.539683 | 0.913596 | 0.540007 | 0.187315 | 0.243848 | 63 |
| 2018 | 0.100000 | 0.580645 | 0.935511 | 0.551799 | 0.196703 | 0.101079 | 62 |
| 2022 | 0.000000 | 0.523810 | 1.010652 | 0.591800 | 0.210385 | 0.112230 | 63 |

## Stage-Specific Alpha Diagnostics

| stage | alpha | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | n_matches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| final | 1.000000 | 0.666667 | 0.894712 | 0.519407 | 0.146731 | 0.193067 | 3 |
| group_stage | 0.400000 | 0.546099 | 0.974438 | 0.574073 | 0.207595 | 0.053691 | 141 |
| quarterfinal | 0.000000 | 0.500000 | 1.118702 | 0.691241 | 0.227382 | 0.270726 | 12 |
| round_of_16 | 0.000000 | 0.652174 | 0.787686 | 0.453788 | 0.135857 | 0.161130 | 23 |
| semifinal | 0.000000 | 0.500000 | 0.986769 | 0.583583 | 0.213607 | 0.103279 | 6 |
| third_place | 0.000000 | 0.666667 | 0.972296 | 0.580998 | 0.255649 | 0.230734 | 3 |

## Confidence-Weighted Blend Diagnostic

| blend_method | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | n_matches | alpha_mean | alpha_min | alpha_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| confidence_weighted | 0.547872 | 0.969586 | 0.571982 | 0.203435 | 0.031070 | 188 | 0.503081 | 0.111849 | 0.944940 |

## Odds-Covered Metrics

| world_cup_year | model_name | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | n_matches | odds_coverage_rate | alpha |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014 | final_model | 0.539683 | 0.913596 | 0.540007 | 0.187315 | 0.243848 | 63 | 0.984375 |  |
| 2014 | market_blend | 0.603175 | 0.934863 | 0.553836 | 0.193115 | 0.073277 | 63 | 0.984375 | 0.300000 |
| 2014 | bookmaker_odds_only | 0.571429 | 0.952307 | 0.565444 | 0.198209 | 0.091628 | 63 | 0.984375 |  |
| 2018 | bookmaker_odds_only | 0.564516 | 0.935959 | 0.551525 | 0.196510 | 0.068938 | 62 | 0.968750 |  |
| 2018 | market_blend | 0.548387 | 0.936431 | 0.553247 | 0.197478 | 0.068308 | 62 | 0.968750 | 0.300000 |
| 2018 | final_model | 0.548387 | 0.957752 | 0.567746 | 0.204262 | 0.034232 | 62 | 0.968750 |  |
| 2022 | bookmaker_odds_only | 0.523810 | 1.010652 | 0.591800 | 0.210385 | 0.112230 | 63 | 0.984375 |  |
| 2022 | market_blend | 0.523810 | 1.016749 | 0.597852 | 0.213006 | 0.087274 | 63 | 0.984375 | 0.300000 |
| 2022 | final_model | 0.492063 | 1.070416 | 0.628940 | 0.226726 | 0.156425 | 63 | 0.984375 |  |

## Average Odds-Covered Metrics

| model_name | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | tournaments | total_matches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| market_blend | 0.558457 | 0.962681 | 0.568312 | 0.201200 | 0.076287 | 3 | 188 |
| bookmaker_odds_only | 0.553251 | 0.966306 | 0.569590 | 0.201701 | 0.090932 | 3 | 188 |
| final_model | 0.526711 | 0.980588 | 0.578898 | 0.206101 | 0.144835 | 3 | 188 |

## Comparisons

- Final model beats bookmaker_odds_only: `False (log_loss_delta=0.014283)`
- Market blend beats final_model: `True (log_loss_delta=-0.017907)`
- Market blend beats bookmaker_odds_only: `True (log_loss_delta=-0.003625)`

## Statistical Significance Summary

| comparison | log_loss_delta | ci_lower | ci_upper | statistically_meaningful |
| --- | --- | --- | --- | --- |
| market_blend_minus_bookmaker_odds_only | -0.003646 | -0.015687 | 0.007937 | False |
| market_blend_minus_final_model | -0.017889 | -0.045609 | 0.008059 | False |
| final_model_minus_bookmaker_odds_only | 0.014243 | -0.025572 | 0.055032 | False |

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

## Notes

- Bookmaker and blend metrics are evaluated only on matches with complete odds.
- Coverage-adjusted final_model rows are included on the same odds-covered match set.
- If odds coverage is incomplete, full-tournament final_model metrics and odds-covered metrics should not be compared as if they used the same rows.
