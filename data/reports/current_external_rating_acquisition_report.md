# Current External Rating Acquisition Report

## Acquisition Status

- FIFA rankings acquired automatically: unknown
- World Football Elo acquired automatically: no (failed)

## File Coverage

| dataset | file_exists | rows | date_min | date_max | unique_teams | worldcup_2026_team_coverage | historical_training_coverage | active |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fifa_rankings | True | 782 | 2010-05-26 | 2026-06-11 | 226 | 1.000 | 0.651 | True |
| world_football_elo | True | 242 | 2026-06-22 | 2026-06-22 | 242 | 1.000 | 0.731 | True |

## Feature Activation

| feature_group | null_rate_before | null_rate_after | usable_columns | ablation_feature_set | ablation_log_loss_delta | ablation_accuracy_delta | identical_to_previous |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fifa_rankings |  | 0.680 | 12 | v2_baseline_plus_fifa_rankings | 0.008 | -0.004 | False |
| external_elo |  | 0.999 | 5 | v3_plus_external_elo | 0.000 | 0.000 | True |

## Manual Steps Still Needed

No manual steps required by the latest acquisition report.

## Snapshot Warning

Current snapshots help the 2026 prediction input, but they do not provide strong historical training coverage. For stronger validation and SOTA-style modeling, historical ranking releases from 2010-2026 should be added later.
