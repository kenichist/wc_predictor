# Stage 5 Calibrated Market Report

## Executive Summary

- Stage 5 status after this attempt: `Stage 5 not yet achieved`.
- This is research/evaluation output only.
- Official production model logic was not changed.
- Official production model remains `football_only_ensemble` with `core_football_only`.
- Market blend remains benchmark-only unless statistically validated.

## Tuning Protocol

- All calibration, alpha tuning, and stacking use only prior tournament years.
- 2014 has no prior Stage 5 paired market/model tournament rows, so tuned models use safe defaults or are skipped.
- 2018 tunes on 2014 only.
- 2022 tunes on 2014 and 2018 only.
- All fair market-value comparisons use only rows where both official_model and bookmaker_odds_only are present.

## Calibration Method

- `official_model_calibrated` uses temperature/exponent scaling over prior tournament probabilities.
- `bookmaker_calibrated` uses the same time-safe temperature method on bookmaker probabilities.
- The selected exponent minimizes prior-year log loss; no target tournament rows are used for tuning.

## Alpha Tuning Results

| test_year | alpha | tuning_years | tuning_log_loss | selected_alpha | selected | notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2014 | 0.000000 | none |  | 0.000000 | True | limited: no prior tournament with paired model and market rows; safe default alpha=0.0 |
| 2018 | 0.400000 | 2014 | 0.948635 | 0.400000 | True | alpha selected using only prior tournament rows |
| 2022 | 0.200000 | 2014,2018 | 0.942857 | 0.200000 | True | alpha selected using only prior tournament rows |

## Stacker Results

- Stacker requested: `True`.
- Stacker coefficient rows: `66`.
| test_year | class_label | feature_name | coefficient | intercept | tuning_source_years |
| --- | --- | --- | --- | --- | --- |
| 2018 | away_win | official_away_prob | 0.800795 | 0.083237 | 2014 |
| 2018 | away_win | official_draw_prob | 0.066832 | 0.083237 | 2014 |
| 2018 | away_win | official_home_prob | -0.866876 | 0.083237 | 2014 |
| 2018 | away_win | bookmaker_away_prob | 0.551193 | 0.083237 | 2014 |
| 2018 | away_win | bookmaker_draw_prob | 0.046201 | 0.083237 | 2014 |
| 2018 | away_win | bookmaker_home_prob | -0.596643 | 0.083237 | 2014 |
| 2018 | away_win | prob_diff_away | 0.249601 | 0.083237 | 2014 |
| 2018 | away_win | prob_diff_draw | 0.020631 | 0.083237 | 2014 |
| 2018 | away_win | prob_diff_home | -0.270232 | 0.083237 | 2014 |
| 2018 | away_win | official_confidence | 0.218991 | 0.083237 | 2014 |
| 2018 | away_win | bookmaker_confidence | 0.015534 | 0.083237 | 2014 |
| 2018 | draw | official_away_prob | -0.402175 | -0.412441 | 2014 |
| 2018 | draw | official_draw_prob | 0.026847 | -0.412441 | 2014 |
| 2018 | draw | official_home_prob | 0.375238 | -0.412441 | 2014 |
| 2018 | draw | bookmaker_away_prob | 0.176726 | -0.412441 | 2014 |
| 2018 | draw | bookmaker_draw_prob | -0.061838 | -0.412441 | 2014 |
| 2018 | draw | bookmaker_home_prob | -0.114979 | -0.412441 | 2014 |
| 2018 | draw | prob_diff_away | -0.578902 | -0.412441 | 2014 |
| 2018 | draw | prob_diff_draw | 0.088685 | -0.412441 | 2014 |
| 2018 | draw | prob_diff_home | 0.490217 | -0.412441 | 2014 |
| 2018 | draw | official_confidence | -0.111427 | -0.412441 | 2014 |
| 2018 | draw | bookmaker_confidence | 0.124624 | -0.412441 | 2014 |
| 2018 | home_win | official_away_prob | -0.398619 | 0.329203 | 2014 |
| 2018 | home_win | official_draw_prob | -0.093679 | 0.329203 | 2014 |
| 2018 | home_win | official_home_prob | 0.491638 | 0.329203 | 2014 |
| 2018 | home_win | bookmaker_away_prob | -0.727919 | 0.329203 | 2014 |
| 2018 | home_win | bookmaker_draw_prob | 0.015637 | 0.329203 | 2014 |
| 2018 | home_win | bookmaker_home_prob | 0.711622 | 0.329203 | 2014 |
| 2018 | home_win | prob_diff_away | 0.329300 | 0.329203 | 2014 |
| 2018 | home_win | prob_diff_draw | -0.109316 | 0.329203 | 2014 |
| 2018 | home_win | prob_diff_home | -0.219984 | 0.329203 | 2014 |
| 2018 | home_win | official_confidence | -0.107564 | 0.329203 | 2014 |
| 2018 | home_win | bookmaker_confidence | -0.140158 | 0.329203 | 2014 |
| 2022 | away_win | official_away_prob | 0.639035 | 0.425862 | 2014,2018 |
| 2022 | away_win | official_draw_prob | 0.335751 | 0.425862 | 2014,2018 |
| 2022 | away_win | official_home_prob | -0.972525 | 0.425862 | 2014,2018 |
| 2022 | away_win | bookmaker_away_prob | 0.694513 | 0.425862 | 2014,2018 |
| 2022 | away_win | bookmaker_draw_prob | 0.089209 | 0.425862 | 2014,2018 |
| 2022 | away_win | bookmaker_home_prob | -0.781461 | 0.425862 | 2014,2018 |
| 2022 | away_win | prob_diff_away | -0.055478 | 0.425862 | 2014,2018 |

## Metrics Table

| model_name | test_year | n_matches | n_skipped | accuracy | log_loss | brier | rps | ece | mean_confidence | tuning_source_years | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bookmaker_calibrated | 2014 | 63 | 0 | 0.571429 | 0.952307 | 0.565444 | 0.198209 | 0.091628 | 0.536091 | none | limited: no prior tournament rows; identity calibration used |
| bookmaker_odds_only | 2014 | 63 | 0 | 0.571429 | 0.952307 | 0.565444 | 0.198209 | 0.091628 | 0.536091 | none | bookmaker odds-only from Stage 5 benchmark |
| market_blend_tuned | 2014 | 63 | 0 | 0.571429 | 0.952307 | 0.565444 | 0.198209 | 0.091628 | 0.536091 | none | time-safe alpha tuned blend alpha=0.00; p=alpha*model+(1-alpha)*market |
| official_model | 2014 | 63 | 0 | 0.539683 | 0.955988 | 0.562923 | 0.197484 | 0.135532 | 0.517418 | none | official model from Stage 5 benchmark |
| official_model_calibrated | 2014 | 63 | 0 | 0.539683 | 0.955988 | 0.562923 | 0.197484 | 0.135532 | 0.517418 | none | limited: no prior tournament rows; identity calibration used |
| bookmaker_calibrated | 2018 | 62 | 0 | 0.564516 | 0.935285 | 0.550559 | 0.196330 | 0.064565 | 0.565552 | 2014 | temperature/exponent calibration gamma=1.100; train_log_loss=0.951360 |
| bookmaker_odds_only | 2018 | 62 | 0 | 0.564516 | 0.935959 | 0.551525 | 0.196510 | 0.068938 | 0.547106 | none | bookmaker odds-only from Stage 5 benchmark |
| market_blend_tuned | 2018 | 62 | 0 | 0.548387 | 0.939849 | 0.558918 | 0.198807 | 0.091079 | 0.532377 | 2014 | time-safe alpha tuned blend alpha=0.40; p=alpha*model+(1-alpha)*market |
| official_model | 2018 | 62 | 0 | 0.516129 | 0.971863 | 0.583042 | 0.207106 | 0.059067 | 0.524112 | none | official model from Stage 5 benchmark |
| official_model_calibrated | 2018 | 62 | 0 | 0.516129 | 0.973992 | 0.585634 | 0.208057 | 0.065094 | 0.558010 | 2014 | temperature/exponent calibration gamma=1.200; train_log_loss=0.951716 |
| stacked_model_market | 2018 | 62 | 0 | 0.532258 | 0.945884 | 0.558571 | 0.200169 | 0.089329 | 0.547192 | 2014 | time-safe logistic stacker trained on prior paired World Cup rows |
| bookmaker_calibrated | 2022 | 63 | 0 | 0.523810 | 1.018546 | 0.594215 | 0.212011 | 0.131456 | 0.583200 | 2014,2018 | temperature/exponent calibration gamma=1.100; train_log_loss=0.943387 |
| bookmaker_odds_only | 2022 | 63 | 0 | 0.523810 | 1.010652 | 0.591800 | 0.210385 | 0.112230 | 0.563178 | none | bookmaker odds-only from Stage 5 benchmark |
| market_blend_tuned | 2022 | 63 | 0 | 0.523810 | 1.018212 | 0.597305 | 0.213101 | 0.053517 | 0.559184 | 2014,2018 | time-safe alpha tuned blend alpha=0.20; p=alpha*model+(1-alpha)*market |
| official_model | 2022 | 63 | 0 | 0.476190 | 1.087986 | 0.640953 | 0.233349 | 0.132118 | 0.554412 | none | official model from Stage 5 benchmark |
| official_model_calibrated | 2022 | 63 | 0 | 0.476190 | 1.109564 | 0.649279 | 0.237612 | 0.132202 | 0.584259 | 2014,2018 | temperature/exponent calibration gamma=1.150; train_log_loss=0.962349 |
| stacked_model_market | 2022 | 63 | 0 | 0.539683 | 1.045503 | 0.619565 | 0.218029 | 0.091869 | 0.572605 | 2014,2018 | time-safe logistic stacker trained on prior paired World Cup rows |
| stacked_model_market | 2014 | 0 | 63 |  |  |  |  |  |  | none | skipped: insufficient prior paired tournament rows/classes for time-safe stacker |

## Average Metrics

| model_name | years | total_matches | log_loss | brier | rps | ece | accuracy | mean_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bookmaker_odds_only | 3 | 188 | 0.966467 | 0.569686 | 0.201729 | 0.091049 | 0.553191 | 0.548801 |
| bookmaker_calibrated | 3 | 188 | 0.968890 | 0.570177 | 0.202215 | 0.096050 | 0.553191 | 0.561594 |
| market_blend_tuned | 3 | 188 | 0.970284 | 0.573969 | 0.203397 | 0.078676 | 0.547872 | 0.542605 |
| stacked_model_market | 2 | 125 | 0.996092 | 0.589312 | 0.209170 | 0.090609 | 0.536000 | 0.560000 |
| official_model | 3 | 188 | 1.005457 | 0.595706 | 0.212676 | 0.109171 | 0.510638 | 0.532022 |
| official_model_calibrated | 3 | 188 | 1.013390 | 0.599351 | 0.214418 | 0.111187 | 0.510638 | 0.553204 |

## Best Models

- Best log loss: `bookmaker_odds_only (log_loss=0.966467)`
- Best Brier: `bookmaker_odds_only (brier=0.569686)`
- Best RPS: `bookmaker_odds_only (rps=0.201729)`
- Best ECE: `market_blend_tuned (ece=0.078676)`

## Significance Table

| candidate_model | baseline_model | metric | mean_delta | ci_lower | ci_upper | statistically_meaningful | n_matches | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_model_calibrated | bookmaker_odds_only | log_loss | 0.046923 | -0.000310 | 0.086722 | False | 188 | candidate is worse on mean metric |
| official_model_calibrated | bookmaker_odds_only | brier | 0.029666 | 0.000878 | 0.055071 | False | 188 | candidate is worse on mean metric |
| official_model_calibrated | bookmaker_odds_only | rps | 0.012690 | -0.000204 | 0.024200 | False | 188 | candidate is worse on mean metric |
| market_blend_tuned | bookmaker_odds_only | log_loss | 0.003817 | -0.009127 | 0.014546 | False | 188 | candidate is worse on mean metric |
| market_blend_tuned | bookmaker_odds_only | brier | 0.004283 | -0.003366 | 0.010917 | False | 188 | candidate is worse on mean metric |
| market_blend_tuned | bookmaker_odds_only | rps | 0.001668 | -0.001616 | 0.004510 | False | 188 | candidate is worse on mean metric |
| stacked_model_market | bookmaker_odds_only | log_loss | 0.022488 | -0.020251 | 0.070694 | False | 125 | candidate is worse on mean metric |
| stacked_model_market | bookmaker_odds_only | brier | 0.017488 | -0.003652 | 0.042104 | False | 125 | candidate is worse on mean metric |
| stacked_model_market | bookmaker_odds_only | rps | 0.005668 | -0.001196 | 0.013379 | False | 125 | candidate is worse on mean metric |
| market_blend_tuned | official_model | log_loss | -0.035173 | -0.064009 | -0.001276 | True | 188 | candidate significantly improves baseline |
| market_blend_tuned | official_model | brier | -0.021737 | -0.040661 | -0.000354 | True | 188 | candidate significantly improves baseline |
| market_blend_tuned | official_model | rps | -0.009280 | -0.017958 | 0.000527 | False | 188 | candidate improves mean metric, but CI crosses 0 |
| stacked_model_market | official_model | log_loss | -0.034297 | -0.080498 | 0.008757 | False | 125 | candidate improves mean metric, but CI crosses 0 |
| stacked_model_market | official_model | brier | -0.022917 | -0.048729 | 0.002538 | False | 125 | candidate improves mean metric, but CI crosses 0 |
| stacked_model_market | official_model | rps | -0.011162 | -0.021693 | -0.001864 | True | 125 | candidate significantly improves baseline |
| official_model_calibrated | official_model | log_loss | 0.007933 | -0.002436 | 0.017720 | False | 188 | candidate is worse on mean metric |
| official_model_calibrated | official_model | brier | 0.003645 | -0.002272 | 0.009288 | False | 188 | candidate is worse on mean metric |
| official_model_calibrated | official_model | rps | 0.001742 | -0.000816 | 0.004200 | False | 188 | candidate is worse on mean metric |

## Does The Model Add Value Over Bookmaker Odds?

- `official_model_calibrated` vs bookmaker log-loss delta `0.046923` with CI `[-0.000310, 0.086722]`; significant: `False`.
- `market_blend_tuned` vs bookmaker log-loss delta `0.003817` with CI `[-0.009127, 0.014546]`; significant: `False`.
- `stacked_model_market` vs bookmaker log-loss delta `0.022488` with CI `[-0.020251, 0.070694]`; significant: `False`.
- No candidate significantly beats bookmaker odds-only on log loss.

## Limitations

- 2014 tuned variants are limited because no earlier Stage 5 paired market/model tournament rows are available.
- Stacking has very small training samples: 2018 uses 2014 only, and 2022 uses 2014 plus 2018.
- This suite evaluates historical World Cup market-covered rows only; it does not promote market features into production.
- Accuracy is secondary; log loss, Brier, RPS, calibration, and paired significance drive conclusions.
- Draw adjustment included: `False`.

## Final Recommendation

Stage 5 is still not achieved. Keep the production model unchanged and continue treating market blend/stacking as benchmark-only research.
