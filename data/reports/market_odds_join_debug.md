# Market Odds Join Debug

- Odds file: `C:\Users\kenic\OneDrive\Desktop\wc_predictor\data\external\market_odds.csv`
- Active 2026 fixture rows inspected: `71`
- 2026 odds rows: `24`
- Join key: `date`, normalized `home_team`, normalized `away_team`
- Fallback: same normalized home/away team pair within `1` day when strict date join misses and the best candidate is unique.

## 2026 Coverage

- Before fix, strict same-date coverage: `12/71 (0.169)`
- After fix, join-function coverage: `23/71 (0.324)`
- Current generated advanced prediction coverage: `23/71 (0.324)`

## Main Findings

- The previous zero-coverage report was caused by mixed date formats coercing prediction dates to `NaT` during advanced feature generation.
- Most missing strict joins are one-day date shifts between `market_odds.csv` and the active fixture input.
- Team-name normalization also needed the `D.R. Congo` alias mapped to `DR Congo`.
- One 2026 odds row does not correspond to an active fixture: `Spain` vs `Cape Verde`.

## Unmatched 2026 Odds Rows After Fallback

| date | home_team | away_team | reason |
| --- | --- | --- | --- |
| 2026-06-15 | Spain | Cape Verde | team pair not in active 2026 fixtures |

## Active 2026 Fixture Rows Without Odds After Fallback

| date | home_team | away_team | match_id |
| --- | --- | --- | --- |
| 2026-06-18 | Mexico | South Korea | worldcup_2026_18039d224dfc896c |
| 2026-06-18 | Czechia | South Africa | worldcup_2026_45f9a6b643b56b62 |
| 2026-06-18 | Canada | Qatar | worldcup_2026_82c9eb470b79bab5 |
| 2026-06-18 | Switzerland | Bosnia and Herzegovina | worldcup_2026_bbb5b0c388d09070 |
| 2026-06-19 | Turkey | Paraguay | worldcup_2026_699cb10bc7fd6982 |
| 2026-06-19 | Brazil | Haiti | worldcup_2026_a52237e4369a1991 |
| 2026-06-19 | Scotland | Morocco | worldcup_2026_a63ef9f9d9153955 |
| 2026-06-19 | United States | Australia | worldcup_2026_c27d0b8f575488e5 |
| 2026-06-20 | Netherlands | Sweden | worldcup_2026_40a592d26f1b8725 |
| 2026-06-20 | Tunisia | Japan | worldcup_2026_6d94a19e0bbb6aae |
| 2026-06-20 | Germany | Ivory Coast | worldcup_2026_c959c3f16de53415 |
| 2026-06-21 | Spain | Saudi Arabia | worldcup_2026_2e5a7758d5750c6b |
| 2026-06-21 | Uruguay | Cape Verde | worldcup_2026_34e13fd5859cd71d |
| 2026-06-21 | New Zealand | Egypt | worldcup_2026_4de8578ff9306d66 |
| 2026-06-22 | Jordan | Algeria | worldcup_2026_61004bb27c9198ea |
| 2026-06-22 | Norway | Senegal | worldcup_2026_74c5ce53db094213 |
| 2026-06-22 | Argentina | Austria | football_data_537399 |
| 2026-06-22 | France | Iraq | football_data_537393 |
| 2026-06-23 | Norway | Senegal | football_data_537394 |
| 2026-06-23 | Colombia | DR Congo | worldcup_2026_1367527cc7812e22 |
| 2026-06-23 | Jordan | Algeria | football_data_537400 |
| 2026-06-23 | Portugal | Uzbekistan | football_data_537405 |
| 2026-06-23 | England | Ghana | football_data_537411 |
| 2026-06-23 | Panama | Croatia | football_data_537412 |
| 2026-06-24 | Bosnia and Herzegovina | Qatar | worldcup_2026_3fc5b62f6dcbe6aa |
| 2026-06-24 | Switzerland | Canada | worldcup_2026_953a8304429d6ca6 |
| 2026-06-24 | Morocco | Haiti | worldcup_2026_a0fd9126179a044f |
| 2026-06-24 | Scotland | Brazil | worldcup_2026_a45443f21a65220d |
| 2026-06-24 | Czechia | Mexico | worldcup_2026_becd9cfb3d883107 |
| 2026-06-24 | South Africa | South Korea | worldcup_2026_f8059483abe02329 |
| 2026-06-25 | Curaçao | Ivory Coast | worldcup_2026_554aa48269cf8a96 |
| 2026-06-25 | Ecuador | Germany | worldcup_2026_6ec1d75cfc147850 |
| 2026-06-25 | Japan | Sweden | worldcup_2026_7f55f1d4da4e4dd4 |
| 2026-06-25 | Turkey | United States | worldcup_2026_a7057edddb165a2c |
| 2026-06-25 | Tunisia | Netherlands | worldcup_2026_dbe2a2978cbe2991 |
| 2026-06-25 | Paraguay | Australia | worldcup_2026_f9e2f1b9aefbc4fb |
| 2026-06-26 | Egypt | Iran | worldcup_2026_0df4483ac45863d3 |
| 2026-06-26 | Cape Verde | Saudi Arabia | worldcup_2026_1bbd3038e76a3fe5 |
| 2026-06-26 | Uruguay | Spain | worldcup_2026_518d5ec9682489ec |
| 2026-06-26 | New Zealand | Belgium | worldcup_2026_8232d5288849367f |
| 2026-06-26 | Norway | France | worldcup_2026_a5679a517895a479 |
| 2026-06-26 | Senegal | Iraq | worldcup_2026_d146f7625a7093a2 |
| 2026-06-27 | Algeria | Austria | worldcup_2026_4b13c4bf184689bf |
| 2026-06-27 | Croatia | Ghana | worldcup_2026_5f5a94f72f45ee6c |
| 2026-06-27 | Jordan | Argentina | worldcup_2026_79a2cefc680f300b |
| 2026-06-27 | Colombia | Portugal | worldcup_2026_9b702abe8f8b5b5b |
| 2026-06-27 | Panama | England | worldcup_2026_a6bb428be169c03d |
| 2026-06-27 | DR Congo | Uzbekistan | worldcup_2026_f7b67dedd449377e |
