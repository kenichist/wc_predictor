# Stage 5 Draw & Overconfidence Calibration Report

## Executive Summary

- Stage 5 status after this attempt: `Stage 5 not yet achieved`.
- Official production model logic was not changed.
- This is historical no-leakage research output only.
- Negative deltas mean the candidate is better than the baseline.

## Tuning Protocol

- 2014 uses identity calibration and is marked limited because no prior tournament is available.
- 2018 tunes only on 2014.
- 2022 tunes only on 2014 and 2018.
- No 2026 live data is used.

## Selected Tuning Parameters

| model_name | test_year | parameter | value | tuning_source_years | tuning_log_loss | selected | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| market_blend_draw_calibrated | 2014 | alpha | 0.000000 | none |  | True | limited: no prior calibrated rows; safe default alpha=0.0 |
| official_model_confidence_calibrated | 2014 | confidence_gamma:0.00-0.40 | 1.000000 | none |  | True | limited: no prior rows; identity confidence calibration |
| official_model_confidence_calibrated | 2014 | confidence_gamma:0.40-0.50 | 1.000000 | none |  | True | limited: no prior rows; identity confidence calibration |
| official_model_confidence_calibrated | 2014 | confidence_gamma:0.50-0.60 | 1.000000 | none |  | True | limited: no prior rows; identity confidence calibration |
| official_model_confidence_calibrated | 2014 | confidence_gamma:0.60-0.70 | 1.000000 | none |  | True | limited: no prior rows; identity confidence calibration |
| official_model_confidence_calibrated | 2014 | confidence_gamma:0.70-1.00 | 1.000000 | none |  | True | limited: no prior rows; identity confidence calibration |
| official_model_draw_calibrated | 2014 | draw_multiplier | 1.000000 | none |  | True | limited: no prior tournament rows; identity draw multiplier |
| official_model_draw_confidence_calibrated | 2014 | confidence_gamma:0.00-0.40 | 1.000000 | none |  | True | limited: no prior rows; identity confidence calibration |
| official_model_draw_confidence_calibrated | 2014 | confidence_gamma:0.40-0.50 | 1.000000 | none |  | True | limited: no prior rows; identity confidence calibration |
| official_model_draw_confidence_calibrated | 2014 | confidence_gamma:0.50-0.60 | 1.000000 | none |  | True | limited: no prior rows; identity confidence calibration |
| official_model_draw_confidence_calibrated | 2014 | confidence_gamma:0.60-0.70 | 1.000000 | none |  | True | limited: no prior rows; identity confidence calibration |
| official_model_draw_confidence_calibrated | 2014 | confidence_gamma:0.70-1.00 | 1.000000 | none |  | True | limited: no prior rows; identity confidence calibration |
| market_blend_draw_calibrated | 2018 | alpha | 0.400000 | 2014 | 0.948635 | True | alpha selected using prior tournament rows only |
| official_model_confidence_calibrated | 2018 | confidence_gamma:0.00-0.40 | 1.000000 | 2014 |  | True | identity: insufficient rows/classes in bucket |
| official_model_confidence_calibrated | 2018 | confidence_gamma:0.40-0.50 | 0.600000 | 2014 | 1.083062 | True | bucket gamma selected using prior tournament rows only |
| official_model_confidence_calibrated | 2018 | confidence_gamma:0.50-0.60 | 1.350000 | 2014 | 0.711440 | True | bucket gamma selected using prior tournament rows only |
| official_model_confidence_calibrated | 2018 | confidence_gamma:0.60-0.70 | 0.800000 | 2014 | 0.950015 | True | bucket gamma selected using prior tournament rows only |
| official_model_confidence_calibrated | 2018 | confidence_gamma:0.70-1.00 | 1.000000 | 2014 |  | True | identity: insufficient rows/classes in bucket |
| official_model_draw_calibrated | 2018 | draw_multiplier | 1.800000 | 2014 | 1.018091 | True | draw-specific log loss selected using prior tournament draw rows only |
| official_model_draw_confidence_calibrated | 2018 | confidence_gamma:0.00-0.40 | 1.000000 | 2014 |  | True | identity: insufficient rows/classes in bucket |
| official_model_draw_confidence_calibrated | 2018 | confidence_gamma:0.40-0.50 | 0.550000 | 2014 | 1.132284 | True | bucket gamma selected using prior tournament rows only |
| official_model_draw_confidence_calibrated | 2018 | confidence_gamma:0.50-0.60 | 1.350000 | 2014 | 0.891962 | True | bucket gamma selected using prior tournament rows only |
| official_model_draw_confidence_calibrated | 2018 | confidence_gamma:0.60-0.70 | 1.300000 | 2014 | 0.849602 | True | bucket gamma selected using prior tournament rows only |
| official_model_draw_confidence_calibrated | 2018 | confidence_gamma:0.70-1.00 | 1.000000 | 2014 |  | True | identity: insufficient rows/classes in bucket |
| market_blend_draw_calibrated | 2022 | alpha | 0.000000 | 2014,2018 | 0.944198 | True | alpha selected using prior tournament rows only |
| official_model_confidence_calibrated | 2022 | confidence_gamma:0.00-0.40 | 0.550000 | 2014,2018 | 1.100135 | True | bucket gamma selected using prior tournament rows only |
| official_model_confidence_calibrated | 2022 | confidence_gamma:0.40-0.50 | 0.650000 | 2014,2018 | 1.079768 | True | bucket gamma selected using prior tournament rows only |
| official_model_confidence_calibrated | 2022 | confidence_gamma:0.50-0.60 | 1.350000 | 2014,2018 | 0.840632 | True | bucket gamma selected using prior tournament rows only |
| official_model_confidence_calibrated | 2022 | confidence_gamma:0.60-0.70 | 0.800000 | 2014,2018 | 0.944907 | True | bucket gamma selected using prior tournament rows only |
| official_model_confidence_calibrated | 2022 | confidence_gamma:0.70-1.00 | 1.000000 | 2014,2018 |  | True | identity: insufficient rows/classes in bucket |
| official_model_draw_calibrated | 2022 | draw_multiplier | 1.800000 | 2014,2018 | 0.982649 | True | draw-specific log loss selected using prior tournament draw rows only |
| official_model_draw_confidence_calibrated | 2022 | confidence_gamma:0.00-0.40 | 0.550000 | 2014,2018 | 1.130465 | True | bucket gamma selected using prior tournament rows only |
| official_model_draw_confidence_calibrated | 2022 | confidence_gamma:0.40-0.50 | 0.550000 | 2014,2018 | 1.136219 | True | bucket gamma selected using prior tournament rows only |
| official_model_draw_confidence_calibrated | 2022 | confidence_gamma:0.50-0.60 | 1.350000 | 2014,2018 | 0.942190 | True | bucket gamma selected using prior tournament rows only |
| official_model_draw_confidence_calibrated | 2022 | confidence_gamma:0.60-0.70 | 1.250000 | 2014,2018 | 0.875689 | True | bucket gamma selected using prior tournament rows only |
| official_model_draw_confidence_calibrated | 2022 | confidence_gamma:0.70-1.00 | 1.000000 | 2014,2018 |  | True | identity: insufficient rows/classes in bucket |

## Metrics

| model_name | test_year | n_matches | accuracy | log_loss | brier | rps | ece | draw_log_loss | draw_recall | draw_prediction_rate | confidence_060_070_log_loss | tuning_source_years | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bookmaker_odds_only | 2014 | 63 | 0.571429 | 0.952307 | 0.565444 | 0.198209 | 0.091628 | 1.406684 | 0.000000 | 0.000000 | 0.991003 | none | bookmaker odds-only from Stage 5 benchmark |
| market_blend_draw_calibrated | 2014 | 63 | 0.571429 | 0.952307 | 0.565444 | 0.198209 | 0.091628 | 1.406684 | 0.000000 | 0.000000 | 0.991003 | none | time-safe blend of bookmaker and draw/confidence calibrated official alpha=0.00 |
| market_blend_tuned | 2014 | 63 | 0.571429 | 0.952307 | 0.565444 | 0.198209 | 0.091628 | 1.406684 | 0.000000 | 0.000000 | 0.991003 | none | time-safe alpha tuned blend alpha=0.00; p=alpha*model+(1-alpha)*market |
| official_model | 2014 | 63 | 0.539683 | 0.955988 | 0.562923 | 0.197484 | 0.135532 | 1.424825 | 0.076923 | 0.031746 | 0.959347 | none | official model from Stage 5 benchmark |
| official_model_confidence_calibrated | 2014 | 63 | 0.539683 | 0.955988 | 0.562923 | 0.197484 | 0.135532 | 1.424825 | 0.076923 | 0.031746 | 0.959347 | none | time-safe confidence-bucket exponent calibration |
| official_model_draw_calibrated | 2014 | 63 | 0.539683 | 0.955988 | 0.562923 | 0.197484 | 0.135532 | 1.424825 | 0.076923 | 0.031746 | 0.959347 | none | time-safe draw multiplier=1.000 |
| official_model_draw_confidence_calibrated | 2014 | 63 | 0.539683 | 0.955988 | 0.562923 | 0.197484 | 0.135532 | 1.424825 | 0.076923 | 0.031746 | 0.959347 | none | time-safe draw multiplier plus confidence-bucket calibration |
| bookmaker_odds_only | 2018 | 62 | 0.564516 | 0.935959 | 0.551525 | 0.196510 | 0.068938 | 1.317070 | 0.000000 | 0.000000 | 0.802333 | none | bookmaker odds-only from Stage 5 benchmark |
| market_blend_draw_calibrated | 2018 | 62 | 0.532258 | 0.957210 | 0.571476 | 0.201117 | 0.102167 | 1.159590 | 0.083333 | 0.112903 | 0.901149 | 2014 | time-safe blend of bookmaker and draw/confidence calibrated official alpha=0.40 |
| market_blend_tuned | 2018 | 62 | 0.548387 | 0.939849 | 0.558918 | 0.198807 | 0.091079 | 1.314536 | 0.000000 | 0.000000 | 0.861885 | 2014 | time-safe alpha tuned blend alpha=0.40; p=alpha*model+(1-alpha)*market |
| official_model | 2018 | 62 | 0.516129 | 0.971863 | 0.583042 | 0.207106 | 0.059067 | 1.338032 | 0.083333 | 0.096774 | 0.942494 | none | official model from Stage 5 benchmark |
| official_model_confidence_calibrated | 2018 | 62 | 0.516129 | 0.971612 | 0.583424 | 0.207234 | 0.029193 | 1.367149 | 0.083333 | 0.096774 | 0.931000 | 2014 | time-safe confidence-bucket exponent calibration |
| official_model_draw_calibrated | 2018 | 62 | 0.370968 | 1.053711 | 0.643472 | 0.218446 | 0.168555 | 0.944254 | 0.333333 | 0.483871 | 0.687610 | 2014 | time-safe draw multiplier=1.800 |
| official_model_draw_confidence_calibrated | 2018 | 62 | 0.370968 | 1.033765 | 0.630143 | 0.215935 | 0.166083 | 0.985566 | 0.333333 | 0.483871 | 0.717634 | 2014 | time-safe draw multiplier plus confidence-bucket calibration |
| bookmaker_odds_only | 2022 | 63 | 0.523810 | 1.010652 | 0.591800 | 0.210385 | 0.112230 | 1.295011 | 0.000000 | 0.000000 | 1.030488 | none | bookmaker odds-only from Stage 5 benchmark |
| market_blend_draw_calibrated | 2022 | 63 | 0.523810 | 1.010652 | 0.591800 | 0.210385 | 0.112230 | 1.295011 | 0.000000 | 0.000000 | 1.030488 | 2014,2018 | time-safe blend of bookmaker and draw/confidence calibrated official alpha=0.00 |
| market_blend_tuned | 2022 | 63 | 0.523810 | 1.018212 | 0.597305 | 0.213101 | 0.053517 | 1.317903 | 0.000000 | 0.000000 | 1.027726 | 2014,2018 | time-safe alpha tuned blend alpha=0.20; p=alpha*model+(1-alpha)*market |
| official_model | 2022 | 63 | 0.476190 | 1.087986 | 0.640953 | 0.233349 | 0.132118 | 1.426054 | 0.000000 | 0.000000 | 0.865642 | none | official model from Stage 5 benchmark |
| official_model_confidence_calibrated | 2022 | 63 | 0.476190 | 1.119131 | 0.665974 | 0.242722 | 0.137651 | 1.448858 | 0.000000 | 0.000000 | 1.357168 | 2014,2018 | time-safe confidence-bucket exponent calibration |
| official_model_draw_calibrated | 2022 | 63 | 0.428571 | 1.111886 | 0.646750 | 0.231991 | 0.159982 | 1.015537 | 0.333333 | 0.269841 | 1.156891 | 2014,2018 | time-safe draw multiplier=1.800 |
| official_model_draw_confidence_calibrated | 2022 | 63 | 0.428571 | 1.137983 | 0.653144 | 0.237215 | 0.130823 | 1.047175 | 0.333333 | 0.269841 | 1.105960 | 2014,2018 | time-safe draw multiplier plus confidence-bucket calibration |

## Average Metrics

| model_name | years | total_matches | log_loss | brier | rps | ece | draw_log_loss | draw_recall | draw_prediction_rate | confidence_060_070_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bookmaker_odds_only | 3 | 188 | 0.966467 | 0.569686 | 0.201729 | 0.091049 | 1.339708 | 0.000000 | 0.000000 | 0.942014 |
| market_blend_tuned | 3 | 188 | 0.970284 | 0.573969 | 0.203397 | 0.078676 | 1.346544 | 0.000000 | 0.000000 | 0.960728 |
| market_blend_draw_calibrated | 3 | 188 | 0.973476 | 0.576265 | 0.203248 | 0.102007 | 1.287773 | 0.027482 | 0.037234 | 0.974602 |
| official_model | 3 | 188 | 1.005457 | 0.595706 | 0.212676 | 0.109171 | 1.396614 | 0.053260 | 0.042553 | 0.922388 |
| official_model_confidence_calibrated | 3 | 188 | 1.015811 | 0.604217 | 0.215859 | 0.101173 | 1.413858 | 0.053260 | 0.042553 | 1.083311 |
| official_model_draw_calibrated | 3 | 188 | 1.040458 | 0.617578 | 0.215961 | 0.154616 | 1.129184 | 0.247409 | 0.260638 | 0.935930 |
| official_model_draw_confidence_calibrated | 3 | 188 | 1.042625 | 0.615325 | 0.216883 | 0.144029 | 1.153410 | 0.247409 | 0.260638 | 0.928764 |

## Significance

| candidate_model | baseline_model | metric | mean_delta | ci_lower | ci_upper | statistically_meaningful | n_matches | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_model_draw_calibrated | official_model | log_loss | 0.035002 | 0.005793 | 0.060670 | False | 188 | candidate is worse on mean metric |
| official_model_draw_calibrated | official_model | brier | 0.021872 | 0.000928 | 0.040121 | False | 188 | candidate is worse on mean metric |
| official_model_draw_calibrated | official_model | rps | 0.003285 | -0.001207 | 0.007474 | False | 188 | candidate is worse on mean metric |
| official_model_draw_confidence_calibrated | official_model | log_loss | 0.037169 | 0.006629 | 0.064761 | False | 188 | candidate is worse on mean metric |
| official_model_draw_confidence_calibrated | official_model | brier | 0.019619 | 0.001587 | 0.035481 | False | 188 | candidate is worse on mean metric |
| official_model_draw_confidence_calibrated | official_model | rps | 0.004207 | 0.000196 | 0.007707 | False | 188 | candidate is worse on mean metric |
| official_model_draw_confidence_calibrated | bookmaker_odds_only | log_loss | 0.076158 | 0.024930 | 0.125982 | False | 188 | candidate is worse on mean metric |
| official_model_draw_confidence_calibrated | bookmaker_odds_only | brier | 0.045639 | 0.013724 | 0.076200 | False | 188 | candidate is worse on mean metric |
| official_model_draw_confidence_calibrated | bookmaker_odds_only | rps | 0.015155 | 0.002416 | 0.027381 | False | 188 | candidate is worse on mean metric |
| market_blend_draw_calibrated | bookmaker_odds_only | log_loss | 0.007008 | -0.005920 | 0.019238 | False | 188 | candidate is worse on mean metric |
| market_blend_draw_calibrated | bookmaker_odds_only | brier | 0.006580 | -0.001601 | 0.014388 | False | 188 | candidate is worse on mean metric |
| market_blend_draw_calibrated | bookmaker_odds_only | rps | 0.001520 | -0.001449 | 0.004147 | False | 188 | candidate is worse on mean metric |
| market_blend_draw_calibrated | market_blend_tuned | log_loss | 0.003192 | -0.006008 | 0.012823 | False | 188 | candidate is worse on mean metric |
| market_blend_draw_calibrated | market_blend_tuned | brier | 0.002297 | -0.003547 | 0.008029 | False | 188 | candidate is worse on mean metric |
| market_blend_draw_calibrated | market_blend_tuned | rps | -0.000148 | -0.001886 | 0.001815 | False | 188 | candidate improves mean metric, but CI crosses 0 |

## Draw And Overconfidence Summary

- Official draw log loss: `1.396614`.
- Draw+confidence calibrated draw log loss: `1.153410`.
- Official confidence 0.60-0.70 log loss: `0.922388`.
- Confidence-calibrated confidence 0.60-0.70 log loss: `1.083311`.

## Final Recommendation

Stage 5 is still not achieved. Keep production unchanged and continue testing draw-aware scoreline/Dixon-Coles and stronger pre-match football signals through no-leakage folds.
