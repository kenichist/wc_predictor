# Project Stage Audit

Audit date: 2026-06-23

Project root: `C:\Users\kenic\OneDrive\Desktop\wc_predictor`

## Current Stage

CURRENT_STAGE=Stage 6 - Market blend benchmark completed

The project is SOTA-inspired, historically backtested, and benchmarked against bookmaker odds on odds-covered World Cup matches. It is not a true SOTA claim and should not be described as Stage 7 until the market benchmark narrative, 2026 odds joins, and production policy are fully publishable.

## Completed Checklist

- [x] historical match dataset
- [x] World Cup 2026 fixture input
- [x] baseline model
- [x] CatBoost model
- [x] calibrated predictions
- [x] Monte Carlo simulation
- [x] ablation testing
- [x] feature null-rate report
- [x] FIFA snapshot ingestion
- [x] external Elo ingestion
- [x] internal historical Elo reconstruction
- [x] squad/injury external feature activation
- [x] historical World Cup backtesting
- [x] baseline comparisons: Elo-only, FIFA-only, rolling-form-only, majority-class
- [x] market odds data loaded
- [x] bookmaker_odds_only benchmark
- [x] market_blend benchmark
- [x] final report / project summary

## Current Final Recommended Model

| Item | Value |
| --- | --- |
| Safe production model | `football_only_ensemble` |
| Training estimator | `catboost` |
| Safe production feature set | `core_football_only` |
| Market-assisted benchmark | `market_blend`, alpha `0.1` |
| Market production readiness | Benchmark-only, not production-ready |
| Current stage | Stage 6 |

Validation metrics for the calibrated `catboost` / `core_football_only` path:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.579109 |
| Log loss | 0.902250 |
| Brier score | 0.531676 |
| Ranked probability score | 0.177199 |
| Calibration error | 0.018630 |

Recommendation match checks:

- Prediction file matches recommendation: true. `data/predictions/actual_team_match_predictions.csv` has 71 rows labeled `football_only_ensemble` and `core_football_only`.
- Simulation metadata matches recommendation: true. `data/simulation/worldcup_2026_simulation_metadata.json` records `football_only_ensemble`, `core_football_only`, 103 prediction rows, seed `42`, and `n_sims=100000`.
- `v1_baseline` is not recommended. Current ablation ranks `core_plus_internal_elo` and `core_football_only` ahead of it.
- `core_plus_internal_elo` is not promoted because its log-loss edge over `core_football_only` is only `0.000190`.

## Feature Group Status

### Active and Used in Final Safe Model

| Feature group | Usable columns | Historical coverage | 2026 coverage |
| --- | ---: | ---: | ---: |
| `baseline` | 5 | 1.000 | 1.000 |
| `basic_elo` | 3 | 1.000 | 1.000 |
| `rolling_form` | 23 | 1.000 | 1.000 |
| `dynamic_ratings` | 3 | 1.000 | 1.000 |
| `pi_ratings` | 11 | 1.000 | 1.000 |

### Active but Not Used in Final Safe Model

| Feature group | Usable columns | Historical coverage | 2026 coverage | Reason not selected |
| --- | ---: | ---: | ---: | --- |
| `internal_historical_elo` | 6 | 1.000 | 1.000 | Experimental; tiny validation edge only |
| `fifa_rankings` | 12 | 0.565 | 0.000 | Sparse coverage; worsened validation |
| `squad` | 3 | 0.467 | 0.689 | Current/snapshot-style, not safe historical as-of data |
| `injuries` | 16 | 0.000 | 0.689 | Current/snapshot-style, not safe historical as-of data |
| `market` | 6 | 0.012 | 0.000 | Benchmark-only; 2026 odds do not join active fixtures |

### Inactive

| Feature group | Usable columns | Historical coverage | 2026 coverage |
| --- | ---: | ---: | ---: |
| `external_elo` | 0 | 0.000 | 0.000 |
| `xg` | 0 | 0.000 | 0.000 |

## Backtesting Completed

Match-level World Cup backtesting is implemented for 2010, 2014, 2018, and 2022. Each tournament fold trains only on matches before that tournament's opening match.

Average log loss by model:

| Scope | Model | Average log loss |
| --- | --- | ---: |
| all matches | `final_model` | 1.024418 |
| all matches | `elo_only` | 1.028577 |
| all matches | `rolling_form_only` | 1.043198 |
| all matches | `majority_class_baseline` | 1.085492 |
| all matches | `fifa_ranking_only` | 1.087041 |
| odds-covered | `market_blend` | 0.965646 |
| odds-covered | `bookmaker_odds_only` | 0.965886 |
| odds-covered | `final_model` | 1.032708 |

Backtest conclusions:

- Final model beats Elo-only on all-match log loss: true.
- Final model beats FIFA-only on all-match log loss: true.
- Bookmaker odds are included where odds exist.
- Final model does not beat bookmaker odds on odds-covered matches.
- Market blend beats final model and bookmaker odds on odds-covered matches, but only as a benchmark.

Tournament simulation backtesting is implemented for 2010, 2014, 2018, and 2022.

Average tournament metrics:

| Metric | Value |
| --- | ---: |
| Champion probability assigned to actual champion | 0.125500 |
| Actual finalists probability | 0.193750 |
| Semifinalist probability recall | 0.375000 |
| Round-of-16 Brier score | 0.178967 |
| Group qualification accuracy | 0.703125 |
| Group match log loss | 1.052103 |

## Market Odds Status

`data/external/market_odds.csv` exists and has 216 rows.

| Year | Rows |
| ---: | ---: |
| 2014 | 64 |
| 2018 | 64 |
| 2022 | 64 |
| 2026 | 24 |

Coverage status:

- 2010 odds coverage: no.
- 2014 odds coverage: mostly complete in backtests, 62/64 matches.
- 2018 odds coverage: mostly complete in backtests, 62/64 matches.
- 2022 odds coverage: mostly complete in backtests, 63/64 matches.
- 2026 rows exist: yes, 24 rows.
- 2026 active feature coverage: 0.000 in the latest feature-null report.
- `bookmaker_odds_only` implemented: yes.
- `market_blend` implemented: yes.
- Best alpha: `0.1`.

Conclusion: Current stage remains Stage 6 - Market odds benchmark completed. `market_blend` is not production-ready for 2026 until odds join to active fixture rows.

## What To Do Next

Recommended next commands:

```powershell
python -m src.cli feature-null-report
python -m src.cli backtest-world-cups --model catboost --feature-set core_football_only
python -m src.cli full-sota-pipeline --model catboost --feature-set core_football_only --n-sims 100000
python -m pytest
```

Recommended next Codex task:

`Fix the 2026 market odds join so data/external/market_odds.csv rows match active World Cup 2026 fixture rows, then report whether market feature coverage reaches production readiness. Keep market_blend benchmark-only unless 2026 coverage is complete.`

## Files To Upload To ChatGPT Next

- `data/reports/final_model_recommendation.md`
- `data/reports/final_project_summary.md`
- `data/reports/feature_null_rate_report.md`
- `data/reports/worldcup_backtest_report.md`
- `data/reports/market_blend_report.md`
- `data/reports/worldcup_tournament_simulation_backtest_report.md`
- `data/experiments/ablation_results.csv`
- `data/backtests/worldcup_backtest_metrics.csv`
- `data/backtests/worldcup_tournament_simulation_backtest.csv`
- `data/predictions/actual_team_match_predictions.csv`
- `data/simulation/worldcup_2026_simulation_results.csv`
- `data/simulation/worldcup_2026_simulation_metadata.json`

## Warnings

- Data leakage risk: squad/player and injury/suspension files look like current snapshots and should not enter historical training without historical as-of dates.
- Market odds risk: 2026 market odds rows exist but do not join to active 2026 prediction fixtures.
- External Elo risk: external Elo remains historically uncovered and inactive.
- FIFA risk: FIFA rankings are sparse and did not improve validation.
- xG risk: xG features are fully null.
- Production policy risk: do not promote `market_blend` or snapshot features based only on current 2026 availability.
- Do not overclaim SOTA. The correct wording is: SOTA-inspired, historically backtested, and benchmarked against bookmaker odds on odds-covered World Cup matches.

## Terminal Summary

CURRENT_STAGE=Stage 6 - Market blend benchmark completed
FINAL_MODEL=football_only_ensemble
FINAL_FEATURE_SET=core_football_only
NEXT_TASK=Fix 2026 market odds joins; keep market_blend benchmark-only until coverage is complete
FILES_TO_UPLOAD=data/reports/final_model_recommendation.md,data/reports/final_project_summary.md,data/reports/feature_null_rate_report.md,data/reports/worldcup_backtest_report.md,data/reports/market_blend_report.md,data/reports/worldcup_tournament_simulation_backtest_report.md,data/experiments/ablation_results.csv,data/backtests/worldcup_backtest_metrics.csv,data/backtests/worldcup_tournament_simulation_backtest.csv,data/predictions/actual_team_match_predictions.csv,data/simulation/worldcup_2026_simulation_results.csv,data/simulation/worldcup_2026_simulation_metadata.json
