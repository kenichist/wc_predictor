# Odds Conversion Benchmark Report

- Decimal odds are converted to implied probabilities and evaluated on odds-covered historical World Cup matches.
- Lower log loss is better.
- Selected conversion method for market benchmarks: `multiplicative`

## Methods Compared

- `multiplicative`: raw inverse odds divided by their row sum.
- `additive`: equal overround subtraction followed by clipping and normalization.
- `power`: row-wise exponent chosen so adjusted probabilities sum to one.
- `shin`: approximate Shin insider-trading adjustment with a row-wise solved parameter.
- `favorite_longshot`: fixed power-style favourite-longshot-bias adjustment.

## Results

| odds_conversion_method | accuracy | log_loss | brier_score | ranked_probability_score | calibration_error | n_matches |
| --- | --- | --- | --- | --- | --- | --- |
| multiplicative | 0.553191 | 0.966467 | 0.569686 | 0.201729 | 0.065499 | 188 |
| favorite_longshot | 0.553191 | 0.967234 | 0.569716 | 0.201893 | 0.088955 | 188 |
| power | 0.553191 | 0.968515 | 0.570068 | 0.202098 | 0.091211 | 188 |
| shin | 0.553191 | 0.968631 | 0.570006 | 0.201987 | 0.063485 | 188 |
| additive | 0.553191 | 0.969928 | 0.570191 | 0.202106 | 0.089645 | 188 |
