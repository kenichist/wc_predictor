# Paid API Data Ingestion

This project supports staged ingestion from official paid APIs for market odds and live injury/suspension data. The ingestion system is intentionally conservative:

- no website scraping
- no hardcoded API keys
- no direct overwrite of production odds during fetch
- no merge of live injury data into official model features
- no model retraining from ingestion commands

The official safe prediction path remains `football_only_ensemble` with `core_football_only`.

## Supported Providers

- API-Football / API-Sports: recommended first for World Cup odds and injury probes
- The Odds API: optional odds fallback
- Sportmonks: optional odds and injury fallback

The Odds API cannot solve 2010 World Cup odds coverage because its public historical odds availability starts from 2020.

## Environment Variables

Set keys in your shell or `.env` file:

```powershell
$env:API_FOOTBALL_KEY="..."
$env:THE_ODDS_API_KEY="..."
$env:SPORTMONKS_API_TOKEN="..."
```

The code uses `python-dotenv` only if it is already installed. API keys are never printed in full.

## Staging Files

Fetched market odds are written first to:

- `data/staging/api_football_market_odds_2026.csv`
- `data/staging/api_football_market_odds_2010_probe.csv`
- `data/staging/the_odds_api_market_odds_2026.csv`
- `data/staging/sportmonks_market_odds_2026.csv`

Fetched live injuries are written first to:

- `data/staging/api_football_injuries_live.csv`
- `data/staging/sportmonks_injuries_live.csv`

Raw JSON responses are saved under:

- `data/raw/api_football/`
- `data/raw/the_odds_api/`
- `data/raw/sportmonks/`

## Required Market Odds Schema

```text
date,home_team,away_team,home_odds,draw_odds,away_odds,bookmaker,source,updated_at,api_provider,api_fixture_id,raw_home_team,raw_away_team,raw_payload_file
```

Only 1X2 markets with home, draw, and away decimal odds are accepted. Win/lose markets without draw are skipped.

## Required Injury Schema

```text
date,team,player,status,reason,fixture_id,fixture_date,source,updated_at,api_provider,raw_team,raw_player,importance,scenario_only
```

`scenario_only` is always `True` by default. Live injuries are not official model training data unless historical as-of backtests later prove they improve performance.

## Commands

Check API availability:

```powershell
python -m src.cli api-coverage-check
```

Fetch API-Football odds:

```powershell
python -m src.cli fetch-api-football-odds --season 2026 --competition "World Cup"
```

Probe API-Football 2010 odds:

```powershell
python -m src.cli fetch-api-football-odds --season 2010 --competition "World Cup"
```

Fetch API-Football live injuries:

```powershell
python -m src.cli fetch-api-football-injuries --season 2026 --competition "World Cup"
```

Fetch The Odds API odds if a World Cup/international soccer sport key is available:

```powershell
python -m src.cli fetch-the-odds-api-odds
```

Fetch Sportmonks odds and injuries if available:

```powershell
python -m src.cli fetch-sportmonks-football-data
```

Validate staged API files:

```powershell
python -m src.cli validate-staged-api-data
```

Check 2010 odds coverage:

```powershell
python -m src.cli check-2010-odds-coverage
```

Merge validated staged odds into production odds:

```powershell
python -m src.cli merge-staged-market-odds --input data/staging/api_football_market_odds_2026.csv
```

Use `--prefer-api` only when you explicitly want a staged API row to replace an existing curated duplicate row:

```powershell
python -m src.cli merge-staged-market-odds --input data/staging/api_football_market_odds_2026.csv --prefer-api
```

## Merge Process

The merge command:

1. validates the staged schema and odds values
2. creates `data/external/market_odds_backup_before_api_merge.csv`
3. reads existing `data/external/market_odds.csv`
4. drops exact duplicate date/home_team/away_team keys
5. preserves curated existing rows unless `--prefer-api` is used
6. writes `data/reports/market_odds_api_merge_report.md`

Production `market_odds.csv` is not modified by fetch commands.

## Reports

The ingestion system writes:

- `data/reports/api_coverage_check_report.md`
- `data/reports/api_market_odds_ingestion_report.md`
- `data/reports/api_injury_ingestion_report.md`
- `data/reports/market_odds_api_merge_report.md`
- `data/reports/market_odds_2010_coverage_report.md`

## After Merge

After a successful merge, rerun the normal project reports and benchmarks:

```powershell
python -m src.cli build-advanced-features
python -m src.cli feature-null-report
python -m src.cli backtest-world-cups --model catboost --feature-set core_football_only
python -m src.cli backtest-worldcup-tournaments --model catboost --feature-set core_football_only --n-sims 1000
python -m src.cli full-sota-pipeline --model catboost --feature-set core_football_only --n-sims 100000
python -m pytest
```

Do not claim 2010 is solved unless all 64 valid 2010 World Cup rows are present. Do not claim `market_blend` is production-ready unless 2026 odds coverage is at least 90%. Do not claim live injuries improve accuracy unless historical as-of validation proves it.

