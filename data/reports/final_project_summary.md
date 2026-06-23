# Final Project Summary: World Cup 2026 Prediction Pipeline

This project is SOTA-inspired, historically backtested, and benchmarked against bookmaker odds. It is not a true SOTA claim.

## 1. Project Objective

Build a reproducible World Cup 2026 prediction pipeline that trains match-outcome models without target leakage, benchmarks them against football and market baselines, generates 2026 fixture probabilities, and simulates the tournament.

Current policy:

- Safe production path: `football_only_ensemble` using `core_football_only`.
- Market-assisted benchmark: `market_blend` using multiplicative odds conversion and alpha `0.3`.
- Production restriction: `market_blend` is not production-ready for 2026 because market odds cover only 23 of 71 active real-team fixtures.

## 2. Data Pipeline Overview

The full pipeline builds advanced features, runs ablation, trains CatBoost, calibrates probabilities, builds a football-only ensemble, blocks partial market blending in production, splits prediction outputs, runs a 100,000-simulation World Cup forecast, and writes final reports.

Current command:

```powershell
python -m src.cli full-sota-pipeline --model catboost --feature-set core_football_only --n-sims 100000
```

CatBoost is configured for GPU device `0` with bounded training settings.

## 3. Feature Groups Implemented

| Feature group | Status | Historical coverage | 2026 coverage | Production policy |
| --- | ---: | ---: | ---: | --- |
| `baseline` | Active | 1.000 | 1.000 | Used in safe model |
| `basic_elo` | Active | 1.000 | 1.000 | Used in safe model |
| `rolling_form` | Active | 1.000 | 1.000 | Used in safe model |
| `dynamic_ratings` | Active | 1.000 | 1.000 | Used in safe model |
| `pi_ratings` | Active | 1.000 | 1.000 | Used in safe model |
| `internal_historical_elo` | Active | 1.000 | 1.000 | Experimental, not promoted |
| `fifa_rankings` | Partially active | 0.649 | 0.689 | Excluded; worsened validation |
| `external_elo` | Mostly inactive | 0.000 | 0.311 | Excluded; sparse historical coverage |
| `squad` | Snapshot-style | 0.467 | 0.689 | Not safe historical training data |
| `injuries` | Snapshot-style | 0.000 | 0.689 | Not safe historical training data |
| `market` | Benchmark-only | 0.014 | 0.223 | Not production-ready |
| `xg` | Inactive | 0.000 | 0.000 | Interface only |

## 4. Final Selected Model and Feature Set

| Item | Current value |
| --- | --- |
| Safe production model | `football_only_ensemble` |
| Training estimator | `catboost` |
| Safe production feature set | `core_football_only` |
| Prediction rows | 71 |
| Prediction model/feature match | Yes |
| Simulation source | `data/predictions/ensemble_predictions.parquet` |
| Simulation metadata | `data/simulation/worldcup_2026_simulation_metadata.json` |

`core_football_only` is selected because it has the best current ablation log loss and uses complete football-derived features. `core_plus_internal_elo` now worsens validation log loss by `0.004040`, so it remains experimental.

## 5. Validation Metrics

Current calibrated validation metrics for `catboost` with `core_football_only`:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.593856 |
| Log loss | 0.885778 |
| Brier score | 0.521142 |
| Ranked probability score | 0.174057 |
| Calibration error | 0.022588 |

## 6. Ablation Results

Lower log loss is better.

| Feature set | Log loss | Accuracy | Notes |
| --- | ---: | ---: | --- |
| `core_football_only` | 0.894662 | 0.584946 | Selected safe production set |
| `core_plus_internal_elo` | 0.898702 | 0.586482 | Worsened log loss by 0.004040 |
| `v1_baseline` | 0.904491 | 0.589247 | Not recommended |
| `v5_plus_pi_ratings` | 0.904905 | 0.583410 | Below safe core |
| `v6_full_rating_stack` | 0.904905 | 0.583410 | Same requested columns as previous step |
| `rating_stack_experimental` | 0.904905 | 0.583410 | Experimental |
| `v4_plus_dynamic_ratings` | 0.912219 | 0.581260 | Below safe core |
| `v2_baseline_plus_fifa_rankings` | 0.912637 | 0.585561 | FIFA worsened validation |
| `v3_plus_external_elo` | 0.912637 | 0.585561 | External Elo added no usable variation |

## 7. Market Odds Join and Conversion

The 2026 market odds join was fixed by:

- preserving mixed date formats during advanced feature generation;
- adding a controlled one-day same-team fallback for market odds rows;
- adding the `D.R. Congo` alias to team normalization.

Coverage:

| Measure | Coverage |
| --- | ---: |
| Previous generated 2026 market coverage | 0/71 |
| Strict same-date coverage after date fix | 12/71 |
| Final tolerant join coverage | 23/71 |

One 2026 odds row, `Spain` vs `Cape Verde`, does not correspond to an active fixture.

Odds conversion benchmark:

| Method | Log loss |
| --- | ---: |
| `multiplicative` | 0.966467 |
| `favorite_longshot` | 0.967234 |
| `power` | 0.968515 |
| `shin` | 0.968631 |
| `additive` | 0.969928 |

The selected odds conversion method is `multiplicative`.

## 8. Backtesting and Significance

Match-level World Cup backtesting covers 2010, 2014, 2018, and 2022. Each fold trains only before the tournament start date.

Average log loss:

| Scope | Model | Log loss |
| --- | --- | ---: |
| all matches | `final_model` | 0.968175 |
| all matches | `elo_only` | 1.030580 |
| all matches | `rolling_form_only` | 1.039424 |
| all matches | `majority_class_baseline` | 1.085492 |
| all matches | `fifa_ranking_only` | 1.086250 |
| odds-covered | `market_blend` | 0.962681 |
| odds-covered | `bookmaker_odds_only` | 0.966306 |
| odds-covered | `final_model` | 0.980588 |

Market blend is numerically best on odds-covered matches, but the paired bootstrap result is not statistically meaningful:

| Comparison | Delta | 95% CI | Meaningful |
| --- | ---: | --- | --- |
| `market_blend - bookmaker_odds_only` | -0.003646 | [-0.015687, 0.007937] | False |

Therefore the project can claim a stronger near-SOTA-style benchmark, but not true SOTA.

## 9. Tournament Simulation Backtesting

Tournament simulation backtesting now compares final model, Elo-only, bookmaker odds where available, and market blend where available.

Average group-match log loss:

| Model | Tournaments | Group-match log loss |
| --- | ---: | ---: |
| `market_blend` | 3 | 0.966203 |
| `bookmaker_odds_only` | 3 | 0.974294 |
| `final_model` | 4 | 0.981732 |
| `elo_only` | 4 | 1.065111 |

Bookmaker-only tournament simulations use market probabilities where odds exist and Elo fallback where odds are missing. Market-blend tournament simulations use blend probabilities where odds exist and final-model fallback where odds are missing.

## 10. Top World Cup 2026 Champion Probabilities

The production simulation uses the safe `football_only_ensemble` / `core_football_only` path with 100,000 simulations.

| Rank | Team | Champion | Final | Semifinal |
| ---: | --- | ---: | ---: | ---: |
| 1 | Argentina | 17.852% | 26.232% | 36.359% |
| 2 | Spain | 15.199% | 23.089% | 32.558% |
| 3 | France | 13.167% | 22.326% | 37.387% |
| 4 | England | 10.035% | 19.795% | 33.546% |
| 5 | Brazil | 6.285% | 12.778% | 23.418% |
| 6 | Germany | 5.703% | 13.092% | 24.589% |
| 7 | Colombia | 4.169% | 9.063% | 17.396% |
| 8 | Netherlands | 4.026% | 8.753% | 18.712% |
| 9 | Portugal | 3.730% | 8.353% | 15.910% |
| 10 | Mexico | 2.983% | 6.651% | 17.421% |

## 11. Known Limitations and Future Work

- `market_blend` is benchmark-only until 2026 market odds cover all active fixtures.
- The market-blend improvement over bookmaker odds is small and not statistically meaningful.
- xG is inactive.
- Squad/player and injury/suspension data are current snapshots, not safe historical training features.
- FIFA and external Elo ingestion exist, but sparse coverage means they are not selected.
- Some parquet reads still fall back to CSV because of parquet reader warnings.

Next best task: complete 2026 market odds coverage for all active fixtures, then rerun the market join report, null report, backtests, significance report, and full production pipeline.

## Source Artifacts

- `data/reports/final_model_recommendation.md`
- `data/reports/market_odds_join_debug.md`
- `data/reports/odds_conversion_benchmark_report.md`
- `data/reports/benchmark_significance_report.md`
- `data/reports/market_blend_report.md`
- `data/reports/worldcup_backtest_report.md`
- `data/reports/worldcup_tournament_simulation_backtest_report.md`
- `data/experiments/ablation_results.csv`
- `data/backtests/worldcup_backtest_metrics.csv`
- `data/backtests/worldcup_tournament_simulation_backtest.csv`
- `data/predictions/actual_team_match_predictions.csv`
- `data/simulation/worldcup_2026_simulation_results.csv`
