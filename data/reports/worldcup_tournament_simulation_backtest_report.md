# World Cup Tournament Simulation Backtest

- Tournaments: 2010, 2014, 2018, 2022 FIFA World Cups
- Each tournament fold trains only on matches before that tournament's opening match.
- Group membership and knockout bracket slots are inferred from the historical fixture order.
- Group-stage matches are simulated from model WDL probabilities; knockout winners use decisive win/loss probability after removing draw mass.

## Metrics

| world_cup_year | model_name | actual_champion | champion_probability_assigned | actual_finalists_probability | semifinalist_probability_recall | round_of_16_brier_score | group_qualification_accuracy | group_match_log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2010 | final_model | Spain | 0.196000 | 0.245000 | 0.500000 | 0.160176 | 0.812500 | 0.981597 |
| 2010 | elo_only | Spain | 0.239000 | 0.272500 | 0.500000 | 0.177531 | 0.750000 | 1.103924 |
| 2014 | final_model | Germany | 0.122000 | 0.203000 | 0.750000 | 0.184356 | 0.625000 | 0.901995 |
| 2014 | elo_only | Germany | 0.127000 | 0.204000 | 0.750000 | 0.232563 | 0.687500 | 1.045095 |
| 2014 | bookmaker_odds_only | Germany | 0.129000 | 0.213500 | 0.750000 | 0.238097 | 0.500000 | 0.952901 |
| 2014 | market_blend | Germany | 0.134000 | 0.240500 | 0.750000 | 0.222206 | 0.625000 | 0.930878 |
| 2018 | final_model | France | 0.070000 | 0.090500 | 0.000000 | 0.131538 | 0.812500 | 0.942663 |
| 2018 | elo_only | France | 0.080000 | 0.100000 | 0.000000 | 0.156700 | 0.812500 | 1.035081 |
| 2018 | bookmaker_odds_only | France | 0.095000 | 0.104500 | 0.250000 | 0.123914 | 0.875000 | 0.929146 |
| 2018 | market_blend | France | 0.095000 | 0.108500 | 0.250000 | 0.127810 | 0.875000 | 0.926595 |
| 2022 | final_model | Argentina | 0.150000 | 0.217500 | 0.500000 | 0.203976 | 0.625000 | 1.100672 |
| 2022 | elo_only | Argentina | 0.112000 | 0.203000 | 0.500000 | 0.193620 | 0.562500 | 1.076344 |
| 2022 | bookmaker_odds_only | Argentina | 0.166000 | 0.230000 | 0.500000 | 0.221797 | 0.625000 | 1.040835 |
| 2022 | market_blend | Argentina | 0.144000 | 0.229500 | 0.500000 | 0.208195 | 0.625000 | 1.041137 |

## Averages

| model_name | tournaments | champion_probability_assigned | actual_finalists_probability | semifinalist_probability_recall | round_of_16_brier_score | group_qualification_accuracy | group_match_log_loss | odds_coverage_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| market_blend | 3 | 0.124333 | 0.192833 | 0.500000 | 0.186070 | 0.708333 | 0.966203 | 0.979167 |
| bookmaker_odds_only | 3 | 0.130000 | 0.182667 | 0.500000 | 0.194603 | 0.666667 | 0.974294 | 0.979167 |
| final_model | 4 | 0.134500 | 0.189000 | 0.437500 | 0.170011 | 0.718750 | 0.981732 |  |
| elo_only | 4 | 0.139500 | 0.194875 | 0.437500 | 0.190103 | 0.703125 | 1.065111 |  |

## Known Limitations

- Historical group labels are inferred from the first 48 fixtures rather than sourced from an official tournament fixture table.
- Penalty shootout winners are read from the historical shootouts source for actual knockout outcomes.
- Simulated knockout pairings not seen in the actual fixture list use a rating-based probability fallback.
- Bookmaker-only tournament simulations use market probabilities where odds exist and Elo fallback where odds are missing, so odds coverage is reported for those rows.
- Market-blend tournament simulations use blend probabilities where odds exist and final-model fallback where odds are missing.
- The bracket backtest measures stage probabilities, not exact scoreline or exact bracket-path accuracy.
