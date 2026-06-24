# Stage 5 Ablation Reconciliation Report

## Executive Summary

- The ablation and Stage 5 log-loss values are not directly comparable.
- Ablation is feature-selection validation.
- Stage 5 is a held-out World Cup benchmark with tournament-year cutoffs.
- Stage 5 benchmark evidence is more trustworthy for near-SOTA or SOTA-style claims.
- The ablation score should not be used to claim Stage 5.
- Stage 5 status remains: `Stage 5 not achieved`.

## Metric Comparison

| metric_source | evaluation_protocol | years_used | row_count | feature_set | model_label | accuracy | log_loss | brier_score | ranked_probability_score | calibration_ece | comparable_to_stage5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| final_model_recommendation.md | feature-selection validation from Ablation Summary | not reported in final_model_recommendation.md |  | core_football_only | ablation_core_football_only | 0.584946 | 0.894662 | 0.525878 | 0.175830 | 0.035662 | No - different protocol, dataset, and purpose |
| stage5_benchmark_metrics.csv | held-out World Cup benchmark; train before each target tournament | 2014, 2018, 2022 | 192.000000 | football_only_ensemble/core_football_only | official_model_weighted_average | 0.510417 | 1.004215 | 0.594659 | 0.211484 | 0.110788 | Yes - this is the Stage 5 reference result |
| stage5_benchmark_metrics.csv | held-out World Cup fold; train before tournament start | 2014 | 64.000000 | football_only_ensemble/core_football_only | official_model_2014 | 0.546875 | 0.945475 | 0.555694 | 0.194898 | 0.137268 | Yes - this is a Stage 5 tournament fold |
| stage5_benchmark_metrics.csv | held-out World Cup fold; train before tournament start | 2018 | 64.000000 | football_only_ensemble/core_football_only | official_model_2018 | 0.500000 | 0.987186 | 0.592829 | 0.208069 | 0.071885 | Yes - this is a Stage 5 tournament fold |
| stage5_benchmark_metrics.csv | held-out World Cup fold; train before tournament start | 2022 | 64.000000 | football_only_ensemble/core_football_only | official_model_2022 | 0.484375 | 1.079986 | 0.635455 | 0.231486 | 0.123212 | Yes - this is a Stage 5 tournament fold |

## Why The Numbers Differ

- The ablation `core_football_only` row comes from the final recommendation's ablation table and is used to choose among feature sets.
- The Stage 5 `official_model` row is evaluated on held-out historical World Cups only: 2014, 2018, and 2022.
- Stage 5 trains before each target tournament and evaluates the tournament as a benchmark fold.
- World Cup matches are a smaller and harder distribution than the broad validation set used by ablation.
- The model labels differ because ablation reports a feature set, while Stage 5 reports the official prediction policy: `football_only_ensemble/core_football_only`.

## Log-Loss Gap

- Ablation `core_football_only` log_loss: `0.894662`.
- Stage 5 official_model weighted log_loss: `1.004215`.
- Raw difference: `0.109553`.
- This difference is expected because the two rows use different evaluation protocols.

## Stage 5 Claim Check

| candidate_model | baseline_model | metric | mean_delta | ci_lower | ci_upper | statistically_meaningful | n_matches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| official_model | bookmaker_odds_only | log_loss | 0.038990 | -0.004686 | 0.076844 | False | 188 |

The Stage 5 decision must come from held-out World Cup benchmark evidence and paired significance tests, not from ablation validation.

## Conclusion

- Use ablation to decide the safe production feature set.
- Use Stage 5 to judge whether the project has benchmark-leading evidence.
- The current evidence does not support claiming Stage 5 is achieved.
- Keep the project wording as SOTA-inspired and historically benchmarked, not true SOTA.

## Input File Check

- Stage 5 benchmark report present: `True`.
- Stage 5 decision report present: `True`.
