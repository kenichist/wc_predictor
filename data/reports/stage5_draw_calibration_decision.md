# Stage 5 Draw Calibration Decision

Decision: **Stage 5 not yet achieved**

## Main Evidence

| candidate_model | baseline_model | metric | mean_delta | ci_lower | ci_upper | statistically_meaningful | n_matches | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_model_draw_confidence_calibrated | official_model | log_loss | 0.037169 | 0.006629 | 0.064761 | False | 188 | candidate is worse on mean metric |
| official_model_draw_confidence_calibrated | bookmaker_odds_only | log_loss | 0.076158 | 0.024930 | 0.125982 | False | 188 | candidate is worse on mean metric |
| market_blend_draw_calibrated | bookmaker_odds_only | log_loss | 0.007008 | -0.005920 | 0.019238 | False | 188 | candidate is worse on mean metric |

## Selected Draw Multipliers

| model_name | test_year | parameter | value | tuning_source_years | tuning_log_loss | notes |
| --- | --- | --- | --- | --- | --- | --- |
| official_model_draw_calibrated | 2014 | draw_multiplier | 1.000000 | none |  | limited: no prior tournament rows; identity draw multiplier |
| official_model_draw_calibrated | 2018 | draw_multiplier | 1.800000 | 2014 | 1.018091 | draw-specific log loss selected using prior tournament draw rows only |
| official_model_draw_calibrated | 2022 | draw_multiplier | 1.800000 | 2014,2018 | 0.982649 | draw-specific log loss selected using prior tournament draw rows only |

## Decision Rule

- Stage 5 can only be claimed if a no-leakage candidate significantly beats bookmaker odds-only on log loss.
- Statistically meaningful requires the 95% CI upper bound to be below 0.

## Recommendation

Stage 5 is still not achieved. Keep production unchanged and continue testing draw-aware scoreline/Dixon-Coles and stronger pre-match football signals through no-leakage folds.
