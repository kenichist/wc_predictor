# World Cup 2026 Prediction Dashboard

This Streamlit dashboard is a live prediction and research UI for the World Cup 2026 project.

It has two modes:

- **User Mode**: clean match cards, win/draw/loss probabilities, knockout advancement when relevant, tournament winner view, and plain-English context.
- **Developer Mode**: raw CSVs, reports, diagnostics, live API status, market blend benchmark, backtests, significance, feature/data quality, and model metadata.

This is not a betting platform. It has no deposits, wallets, bet slips, affiliate links, betting CTAs, or wagering actions.

## Install

```powershell
pip install -r dashboard/requirements.txt
```

## Run

```powershell
streamlit run dashboard/app.py
```

Run from the project root.

## Environment Variables

Optional live/API variables:

```powershell
$env:API_FOOTBALL_KEY="..."
$env:THE_ODDS_API_KEY="..."
$env:SPORTMONKS_API_TOKEN="..."
$env:DASHBOARD_DEV_PASSWORD="..."
```

If `DASHBOARD_DEV_PASSWORD` is set, Developer Mode requires it. If it is not set, Developer Mode is allowed locally and the app shows a warning.

## Official vs Live/Scenario Outputs

Official production output remains:

- model: `football_only_ensemble`
- feature set: `core_football_only`

Official files are read, not overwritten:

- `data/predictions/actual_team_match_predictions.csv`
- `data/simulation/worldcup_2026_simulation_results.csv`
- `data/external/market_odds.csv`

Live/scenario files are written separately under `data/live/`:

- `data/live/live_fixtures.csv`
- `data/live/live_odds.csv`
- `data/live/live_injuries.csv`
- `data/live/live_lineups.csv`
- `data/live/live_scores.csv`
- `data/live/live_predictions.csv`
- `data/live/live_prediction_factors.csv`
- `data/live/reports/live_api_status_report.md`
- `data/live/reports/live_prediction_report.md`

Live-adjusted predictions are scenario-based and are not official model outputs.

## Risk & Bankroll Lab

The dashboard includes an educational **Risk & Bankroll Lab**. It is a bankroll simulation and risk exposure tool only. It can ingest read-only live market data from the project's staged odds files, but it does not place bets, create bet slips, link to sportsbooks, handle deposits or withdrawals, or provide financial or gambling advice.

Permanent warning shown in the app:

> DISCLAIMER: This simulator ingests live market data strictly for quantitative risk modeling and educational purposes. It does not facilitate wagering or constitute financial advice. Online gambling is illegal in Indonesia.

IDR inputs are for budgeting/risk visualization only.

### User Mode

User Mode shows a simplified risk panel inside Match Detail:

- read-only market event dropdown
- market line selector
- outcome selector: home win, draw, or away win
- current decimal odds from staged market data
- raw implied probability using `(1 / decimal_odds) * 100`
- bankroll in IDR
- risk profile
- maximum stake cap
- maximum stake percentage of bankroll
- daily and tournament loss caps
- optional toggle for hypothetical stake simulation

The default hypothetical stake is `0 IDR`. Stake sizing remains `0 IDR` until the user enters a bankroll and explicitly enables hypothetical stake simulation.

### Developer Mode

Developer Mode has a **Risk Quant Lab** tab with:

- read-only market event dropdown and refresh button
- single-match risk table
- Kelly criterion details
- market edge details
- raw implied probability display
- probability and odds sensitivity tables
- Monte Carlo bankroll simulation
- portfolio exposure diagnostics
- raw risk output tables
- methodology notes

### Kelly Criterion

The lab uses decimal-odds Kelly sizing:

```text
f* = (bp - q) / b
b = decimal_odds - 1
p = model probability
q = 1 - p
```

The full Kelly fraction is reduced by the selected risk profile:

- Very conservative: `0.05 Kelly`
- Conservative: `0.10 Kelly`
- Balanced: `0.25 Kelly`
- Aggressive: `0.50 Kelly`

Fractional Kelly is used because full Kelly can create large drawdowns. Caps are then applied by bankroll percentage, user stake cap, daily loss cap, tournament loss cap, and bankroll size.

### Why Stake Can Be 0

The hypothetical stake can be `0 IDR` when:

- hypothetical stake simulation is disabled
- bankroll is missing or `<= 0`
- model probability or market odds are missing
- decimal odds are invalid
- edge is not positive
- expected value is not positive
- capped stake rounds below `1,000 IDR`

This is expected behavior and is not a betting recommendation.

### Risk Outputs

Risk calculations may be saved only to live/scenario files:

- `data/live/live_risk_analysis.csv`
- `data/live/reports/live_risk_report.md`

Official prediction files are not modified.

The live market refresh button uses the project's existing read-only market-data refresh path. It must not be extended with sportsbook account, wager placement, affiliate, deposit, withdrawal, or external betting-link functionality.

## Refresh Live Data

From the app, click **Refresh live data**. API calls only happen when you click the button.

From the CLI:

```powershell
python -m src.cli refresh-live-worldcup-data
```

Lineups can use extra API quota:

```powershell
python -m src.cli refresh-live-worldcup-data --fetch-lineups
```

Generate live predictions from existing local live CSVs:

```powershell
python -m src.cli generate-live-predictions
```

If `API_FOOTBALL_KEY` is missing, the dashboard and CLI use local CSV/report outputs only and continue gracefully.

## Files Read

The dashboard reads:

- official prediction CSVs from `data/predictions/`
- simulation outputs from `data/simulation/`
- reports from `data/reports/`
- backtests from `data/backtests/`
- market odds from `data/external/market_odds.csv`
- live/scenario files from `data/live/`

## Live API Endpoints

Implemented primary provider: API-Football / API-Sports.

Used endpoints:

- `/leagues`
- `/fixtures`
- `/odds`
- `/injuries`
- `/fixtures/lineups`

Raw API JSON payloads are saved to `data/live/raw/`.

The Odds API and Sportmonks keys are displayed in Developer Mode if configured, but the live dashboard refresh path currently uses API-Football as the primary provider.

## Safety Notes

- Predictions are uncertain.
- This is not a betting platform.
- The Risk & Bankroll Lab is an educational risk model, not betting advice.
- The app does not place bets, create bet slips, handle money, or link to sportsbooks.
- Do not claim true SOTA.
- Do not claim market blend is production-ready unless active fixture odds coverage is at least 90%.
- Do not claim live injuries improve accuracy unless historical as-of validation proves it.
- Live/scenario data is never silently merged into official model files.

## Risk Tests

Run the focused risk tests from the project root:

```powershell
python -m compileall dashboard
python -m pytest tests/test_risk_sizing.py
```
