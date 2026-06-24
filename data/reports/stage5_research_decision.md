# Stage 5 Research Decision

Decision: **Stage 5 not yet achieved**

## Reasons

- Stage 5 requires statistically meaningful benchmark evidence, not only dashboard functionality.
- The official model must be compared against strong non-market and market baselines using no-leakage historical World Cup folds.
- Negative log-loss delta means the candidate model is better than the baseline.

## Evidence

| candidate_model | baseline_model | metric | mean_delta | ci_lower | ci_upper | statistically_meaningful | n_matches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| official_model | bookmaker_odds_only | log_loss | 0.038990 | -0.004686 | 0.076844 | False | 188 |

## What To Improve Next

- Add more complete historical pre-match odds coverage.
- Add a validated Dixon-Coles fitting implementation.
- Validate any squad, injury, lineup, or xG data historically before treating it as production model evidence.
- Improve probability calibration if ECE remains weaker than market baselines.

## Skipped Or Limited Items

| model_name | test_year | reason | n_skipped |
| --- | --- | --- | --- |
| bookmaker_calibrated | 2014 | insufficient historical odds-covered training rows for time-safe calibration | 63 |
| dixon_coles | 2014 | skipped: no validated time-decayed Dixon-Coles parameter fitting exists in project yet | 64 |
| dixon_coles | 2018 | skipped: no validated time-decayed Dixon-Coles parameter fitting exists in project yet | 64 |
| dixon_coles | 2022 | skipped: no validated time-decayed Dixon-Coles parameter fitting exists in project yet | 64 |
