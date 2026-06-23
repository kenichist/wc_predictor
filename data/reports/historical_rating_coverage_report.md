# Historical Rating Coverage Report

## Summary

| dataset | files_found | snapshots_imported | snapshots_imported_with_missing_points | snapshots_failed | date_min | date_max | total_rows | unique_teams | latest_snapshot_date | latest_snapshot_team_count | historical_training_coverage | worldcup_2026_coverage | ablation_can_meaningfully_test | output_written | preserved_existing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fifa_rankings | 8 | 8 | 2 | 0 | 2010-05-26 | 2026-06-11 | 782 | 226 | 2026-06-11 | 211 | 0.408 | 0.000 | False | True | False |
| world_football_elo | 1 | 1 | 0 | 0 | 2026-06-22 | 2026-06-22 | 242 | 242 | 2026-06-22 | 242 | 0.000 | 0.000 | False | True | False |

## Null Rate By Year

- `fifa_rankings`: 2010:0.918;2011:0.943;2012:0.937;2013:0.928;2014:0.871;2015:0.925;2016:0.842;2017:0.869;2018:0.292;2019:0.188;2020:0.037;2021:0.076;2022:0.115;2023:0.185;2024:0.122;2025:0.250
- `world_football_elo`: 2010:1.000;2011:1.000;2012:1.000;2013:1.000;2014:1.000;2015:1.000;2016:1.000;2017:1.000;2018:1.000;2019:1.000;2020:1.000;2021:1.000;2022:1.000;2023:1.000;2024:1.000;2025:1.000

## Failed Files

No failed snapshot files.

## Snapshot Files

- `fifa_rankings`: fifa_2010-05-26.csv:imported; fifa_2014-05-08.csv:imported; fifa_2016-06-02.csv:imported; fifa_2018-05-17.csv:imported_with_missing_points; fifa_2019-07-25.csv:imported; fifa_2022-10-06.csv:imported; fifa_2024-04-04.csv:imported_with_missing_points; fifa_2026-06-11.csv:imported
- `world_football_elo`: elo_2026-06-22.csv:imported

## Ablation Readiness

- `fifa_rankings` is not yet ready for meaningful ablation. Add more dated historical snapshots before validation matches.
- `world_football_elo` is not yet ready for meaningful ablation. Add more dated historical snapshots before validation matches.
