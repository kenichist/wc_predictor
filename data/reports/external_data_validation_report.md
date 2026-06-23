# External Data Validation Report

## File Existence

| dataset | exists | path | status |
| --- | --- | --- | --- |
| fifa_rankings | True | `C:\Users\kenic\OneDrive\Desktop\wc_predictor\data\external\fifa_rankings.csv` | warning |
| world_football_elo | True | `C:\Users\kenic\OneDrive\Desktop\wc_predictor\data\external\world_football_elo.csv` | warning |

## Required Columns And Values

| dataset | required_columns_present | missing_required_columns | row_count | missing_values | numeric_parse_errors | duplicate_date_team_rows |
| --- | --- | --- | --- | --- | --- | --- |
| fifa_rankings | True |  | 782 | date:0;team:0;rank:0;points:421 | 0 | 0 |
| world_football_elo | True |  | 242 | date:0;team:0;elo:0 | 0 | 0 |

## Date Range And Teams

| dataset | date_range | unique_teams | unmapped_team_count | unmapped_team_names |
| --- | --- | --- | --- | --- |
| fifa_rankings | 2010-05-26 to 2026-06-11 | 226 | 23 | Antigua & Barbuda,Brunei Darussalam,Cabo Verde (Cape Verde Islands),Caymen Islands,China PR,Chinese Taipei,Curacao (formerly Netherlands Antilles),FYR Macedonia,Hong Kong, China,Korea DPR,Kyrgyz Republic,St Kitts and Nevis,St Lucia,St Vincent and the Grenadines,Swaziland,Swaziland (Eswatini),São Tomé e Príncipe,The Gambia,The Netherlands,Timor Leste,Trinidad & Tobago,Turks & Caicos Islands,US Virgin Islands |
| world_football_elo | 2026-06-22 to 2026-06-22 | 242 | 14 | Chinese Taipei,Ireland,Micronesia,Niue,Palau,Reunion,Saba,Saint Barthelemy,Sint Eustatius,São Tomé e Príncipe,Turks and Caicos,US Virgin Islands,Vatican,Wallis and Futuna |

## Coverage

| dataset | historical_team_coverage | worldcup_2026_team_coverage |
| --- | --- | --- |
| fifa_rankings | 0.651 | 1.000 |
| world_football_elo | 0.731 | 1.000 |

## Warnings And Next Actions

| dataset | warnings | next_actions |
| --- | --- | --- |
| fifa_rankings | required columns contain missing values; some team names do not match known training or World Cup teams | check team names and update normalization if needed |
| world_football_elo | some team names do not match known training or World Cup teams | check team names and update normalization if needed |
