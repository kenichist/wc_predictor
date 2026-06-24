# Stage 5 Calibrated Market Decision

Decision: **Stage 5 not yet achieved**

## Main Evidence

| candidate_model | baseline_model | metric | mean_delta | ci_lower | ci_upper | statistically_meaningful | n_matches | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_model_calibrated | bookmaker_odds_only | log_loss | 0.046923 | -0.000310 | 0.086722 | False | 188 | candidate is worse on mean metric |
| market_blend_tuned | bookmaker_odds_only | log_loss | 0.003817 | -0.009127 | 0.014546 | False | 188 | candidate is worse on mean metric |
| stacked_model_market | bookmaker_odds_only | log_loss | 0.022488 | -0.020251 | 0.070694 | False | 125 | candidate is worse on mean metric |

## Best Average Models

- Best log loss: `bookmaker_odds_only (log_loss=0.966467)`
- Best Brier: `bookmaker_odds_only (brier=0.569686)`
- Best RPS: `bookmaker_odds_only (rps=0.201729)`
- Best ECE: `market_blend_tuned (ece=0.078676)`

## Decision Rule

- Negative delta means the candidate is better than the baseline.
- A candidate is statistically meaningful only when the 95% CI upper bound is below 0.
- Stage 5 is not promoted unless a time-safe calibrated/blended/stacked candidate significantly beats bookmaker odds-only.

## Next Step

Focus next on `market_blend_tuned`, which had the best mean log-loss delta but did not satisfy the confidence interval rule.
