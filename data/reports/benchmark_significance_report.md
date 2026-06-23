# Benchmark Significance Report

- Bootstrap confidence intervals use odds-covered World Cup matches only.
- Log-loss deltas are paired by `world_cup_year` and `match_id`.
- Negative delta means the first model in the comparison has lower log loss.

## Log Loss Confidence Intervals

| model_name | log_loss | ci_lower | ci_upper | n_matches |
| --- | --- | --- | --- | --- |
| final_model | 0.980710 | 0.910057 | 1.054094 | 188 |
| bookmaker_odds_only | 0.966467 | 0.889407 | 1.047646 | 188 |
| market_blend | 0.962821 | 0.891187 | 1.040796 | 188 |

## Paired Delta Confidence Intervals

| comparison | log_loss_delta | ci_lower | ci_upper | statistically_meaningful | n_matches |
| --- | --- | --- | --- | --- | --- |
| market_blend_minus_bookmaker_odds_only | -0.003646 | -0.015687 | 0.007937 | False | 188 |
| market_blend_minus_final_model | -0.017889 | -0.045609 | 0.008059 | False | 188 |
| final_model_minus_bookmaker_odds_only | 0.014243 | -0.025572 | 0.055032 | False | 188 |
