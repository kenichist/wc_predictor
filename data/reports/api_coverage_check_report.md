# API Coverage Check Report

This report uses official API endpoints only. It does not modify production files.

## Availability

- API_FOOTBALL_AVAILABLE=True
- THE_ODDS_API_AVAILABLE=False
- SPORTMONKS_AVAILABLE=False
- API_FOOTBALL_2026_ODDS_AVAILABLE=True
- API_FOOTBALL_2010_ODDS_AVAILABLE=False
- API_FOOTBALL_INJURIES_AVAILABLE=True
- THE_ODDS_API_2026_ODDS_AVAILABLE=Unknown
- THE_ODDS_API_2010_ODDS_AVAILABLE=False
- SPORTMONKS_2026_ODDS_AVAILABLE=Unknown
- SPORTMONKS_2010_ODDS_AVAILABLE=Unknown
- SPORTMONKS_INJURIES_AVAILABLE=Unknown

## Recommendations

- Recommended provider for 2026 odds: `api_football`
- Recommended provider for live injuries: `api_football`
- Whether 2010 still needs manual/archive collection: `True` unless a provider returns all 64 valid 2010 1X2 odds rows.
- The Odds API 2010 odds: `False`; public historical availability starts from 2020.