# Ablation Results

| feature_set | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | num_features | new_columns_added | usable_new_columns | null_new_columns | constant_new_columns | identical_to_previous | warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1_baseline | 0.589247 | 0.904491 | 0.531969 | 0.177840 | 0.028365 | 32 | 32 | 32 | 0 | 0 | False |  |
| core_football_only | 0.584946 | 0.894662 | 0.525878 | 0.175830 | 0.035662 | 46 | 14 | 14 | 0 | 0 | False |  |
| core_plus_internal_elo | 0.586482 | 0.898702 | 0.528615 | 0.176726 | 0.034603 | 52 | 6 | 6 | 0 | 0 | False | internal historical Elo worsened validation log_loss by 0.004040 |
| v2_baseline_plus_fifa_rankings | 0.585561 | 0.912637 | 0.536159 | 0.179116 | 0.032421 | 42 | 10 | 10 | 0 | 0 | False | FIFA rankings worsened validation log_loss by 0.008145 |
| v3_plus_external_elo | 0.585561 | 0.912637 | 0.536159 | 0.179116 | 0.032421 | 45 | 5 | 0 | 2 | 3 | True | v3_plus_external_elo added 5 columns, but none add usable variation. 3 newly added columns are constant. feature set has same usable columns as previous set. metrics are identical to previous feature set. Add data/external/world_football_elo.csv to enable external Elo features |
| v4_plus_dynamic_ratings | 0.581260 | 0.912219 | 0.536890 | 0.179271 | 0.038784 | 48 | 3 | 3 | 0 | 0 | False |  |
| v5_plus_pi_ratings | 0.583410 | 0.904905 | 0.531451 | 0.177276 | 0.041453 | 59 | 11 | 11 | 0 | 0 | False |  |
| v6_full_rating_stack | 0.583410 | 0.904905 | 0.531451 | 0.177276 | 0.041453 | 59 | 0 | 0 | 0 | 0 | True | feature set has same requested columns as previous step |
| rating_stack_experimental | 0.583410 | 0.904905 | 0.531451 | 0.177276 | 0.041453 | 59 | 0 | 0 | 0 | 0 | True | feature set has same requested columns as previous step. experimental rating stack worsened validation log_loss by 0.010244 |
