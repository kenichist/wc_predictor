# world_football_elo Acquisition Failure

- status: failed
- output_path: `C:\Users\kenic\OneDrive\Desktop\wc_predictor\data\external\world_football_elo.csv`
- raw_path: `None`
- preserved_existing: False
- errors: [WinError 2] The system cannot find the file specified
- warnings: 

## Manual Import Instructions

Provide a real CSV/Excel/JSON/HTML file for world_football_elo with schema date,team,elo,source,retrieved_at at C:\Users\kenic\OneDrive\Desktop\wc_predictor\data\external\world_football_elo.csv, or run `python -m src.cli acquire-world-football-elo --input "C:\Users\kenic\Downloads\world_football_elo.csv"` after replacing the example path with your actual downloaded file path.

- Validate after import: `python -m src.cli validate-external-data`
- Rebuild features: `python -m src.cli build-advanced-features`
- Check activation: `python -m src.cli feature-null-report`
- Rerun ablation: `python -m src.cli run-ablation`
