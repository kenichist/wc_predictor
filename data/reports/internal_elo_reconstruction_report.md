# Internal Historical Elo Reconstruction Report

This report covers the `internal_historical_elo` feature group reconstructed from the project's historical international match results. It does not use external World Football Elo snapshots.

## Configuration

- Starting Elo: `1500`
- Base K-factor: `20.00`
- K-factors used: `continental_championship=25.00, continental_qualifier=22.00, friendly=14.00, nations_league=20.00, other=20.00, world_cup=30.00, world_cup_qualifier=23.00`
- Home advantage: `50.00` Elo points for non-neutral matches
- Neutral venue handling: home advantage is set to `0` when `neutral=True`.
- Goal-difference formula: `1.0 for one-goal margins; ln(min(abs(goal_diff), 4) + 1.0) for larger margins`
- Penalty shootout handling: drawn matches with a shootout winner use `0.75` for the shootout winner and `0.25` for the loser.

## Coverage

- Number of teams: `336`
- Date range: `1872-11-30 to 2018-09-06`
- Historical training coverage: `1.000`
- World Cup 2026 coverage: `1.000`

## Validation Comparison

- Compared feature sets: `core_football_only` vs `core_plus_internal_elo`
- Improves log loss over `core_football_only`: `False`
- Log-loss delta (`core_plus_internal_elo` - `core_football_only`): `0.004040`
- Notes: Lower log loss is better.
