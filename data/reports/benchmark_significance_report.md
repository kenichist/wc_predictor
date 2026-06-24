# Benchmark Significance Report

- Bootstrap confidence intervals use odds-covered World Cup matches only.
- Log-loss deltas are paired by `world_cup_year` and `match_id`.
- Negative delta means the first model in the comparison has lower log loss.

## Log Loss Confidence Intervals

| model_name | log_loss | ci_lower | ci_upper | n_matches |
| --- | --- | --- | --- | --- |
| final_model | 1.005457 | 0.934590 | 1.076891 | 188 |
| bookmaker_odds_only | 0.966467 | 0.889407 | 1.047646 | 188 |
| market_blend | 0.966467 | 0.888488 | 1.049264 | 188 |

## Paired Delta Confidence Intervals

| comparison | log_loss_delta | ci_lower | ci_upper | statistically_meaningful | n_matches |
| --- | --- | --- | --- | --- | --- |
| market_blend_minus_bookmaker_odds_only | 0.000000 | -0.000000 | 0.000000 | False | 188 |
| market_blend_minus_final_model | -0.038990 | -0.079331 | -0.002517 | True | 188 |
| final_model_minus_bookmaker_odds_only | 0.038990 | 0.001700 | 0.076875 | True | 188 |
