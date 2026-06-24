# Refresh Market Odds Report

This command stages provider data first, validates it, and merges only valid odds rows.

- dry_run: `False`
- skip_merge: `False`
- missing_odds_template: `C:\Users\kenic\OneDrive\Desktop\wc_predictor\data\staging\missing_market_odds_template_2026.csv`

## Summary

- PROVIDERS_CHECKED=the_odds_api
- STAGED_ROWS=25
- VALID_ROWS=25
- INVALID_ROWS=0
- ROWS_ADDED=25
- ROWS_REPLACED=0
- ROWS_AFTER=246
- ACTIVE_FIXTURE_COVERAGE_BEFORE=0.0
- ACTIVE_FIXTURE_COVERAGE_AFTER=0.9615384615384616
- MISSING_ACTIVE_FIXTURES=1
- SUCCESS=True

## Provider Fetch Results

| provider | rows | valid_rows | invalid_rows | safe_to_merge | path | error |
| --- | --- | --- | --- | --- | --- | --- |
| the_odds_api | 25 | 25 | 0 | True | C:\Users\kenic\OneDrive\Desktop\wc_predictor\data\staging\the_odds_api_market_odds_2026.csv |  |

## Merge Status

- MARKET_ODDS_ROWS_BEFORE=221
- STAGED_ROWS=25
- ROWS_ADDED=25
- ROWS_REPLACED=0
- ROWS_AFTER=246
- DUPLICATES=0
- BAD_ODDS=0
- VALIDATION_PASSED=True