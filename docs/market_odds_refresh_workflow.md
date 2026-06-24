# Market Odds Refresh Workflow

This workflow keeps market odds separate from official model predictions until they are validated and safely merged.

Do not use Club World Cup keys for national-team World Cup predictions. Do not guess odds. Do not overwrite `data/external/market_odds.csv` manually without a backup.

## API Workflow

1. Discover provider competitions:

```powershell
python -m src.cli discover-odds-competitions
```

2. Check:

```text
data/reports/odds_provider_competition_discovery.md
```

3. If needed, update:

```text
config/odds_provider_keys.json
```

For your paid API-Football subscription, use `API_FOOTBALL_KEY` in `.env`. Do not put the key itself in `config/odds_provider_keys.json`.

If discovery finds a national-team World Cup league ID for API-Football, add that ID here:

```json
{
  "api_football": {
    "enabled": true,
    "league_ids": [1]
  }
}
```

Use the ID from `data/reports/odds_provider_competition_discovery.md`; the `1` above is only an example. Do not use Club World Cup IDs for national-team World Cup predictions.

4. Refresh market odds:

```powershell
python -m src.cli refresh-market-odds
```

To use only the paid API-Football provider:

```powershell
python -m src.cli refresh-market-odds --provider api_football
```

5. Check:

```text
data/reports/refresh_market_odds_report.md
```

6. Refresh Streamlit or click **Refresh data**.

If API providers return zero useful rows, use the manual template workflow.

## Manual Workflow

1. Open Developer Mode -> Market Odds Coverage.

2. Download the missing odds template, or generate it:

```powershell
python -m src.cli generate-missing-odds-template
```

3. Fill only verified decimal 1X2 odds in:

```text
data/staging/missing_market_odds_template_2026.csv
```

4. Validate the manual file:

```powershell
python -m src.cli validate-manual-market-odds --input data/staging/missing_market_odds_template_2026.csv
```

5. If validation succeeds, merge:

```powershell
python -m src.cli merge-staged-market-odds --input data/staging/manual_market_odds_validated_2026.csv
```

6. Regenerate reports and live predictions:

```powershell
python -m src.cli feature-null-report
python -m src.cli generate-live-predictions
```

7. Refresh Streamlit.

## Safety Rules

- Do not fabricate odds.
- Do not estimate odds.
- Do not use scores as odds.
- Do not use Club World Cup odds for national-team World Cup predictions.
- Keep `date`, `home_team`, and `away_team` aligned with active fixture rows.
- Odds must be decimal odds greater than 1.
- Rows without draw odds are rejected.
