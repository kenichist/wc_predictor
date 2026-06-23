# fifa_rankings Acquisition Failure

- status: failed
- output_path: `C:\Users\kenic\OneDrive\Desktop\wc_predictor\data\external\fifa_rankings.csv`
- raw_path: `None`
- preserved_existing: False
- errors: [WinError 2] The system cannot find the file specified
- warnings: 

## Manual Import Instructions

Provide a real CSV/Excel/JSON/HTML file for fifa_rankings with schema date,team,rank,points,source,retrieved_at at C:\Users\kenic\OneDrive\Desktop\wc_predictor\data\external\fifa_rankings.csv, or run `python -m src.cli acquire-fifa-rankings --input "C:\Users\kenic\Downloads\fifa_rankings.csv"` after replacing the example path with your actual downloaded file path.

- Validate after import: `python -m src.cli validate-external-data`
- Rebuild features: `python -m src.cli build-advanced-features`
- Check activation: `python -m src.cli feature-null-report`
- Rerun ablation: `python -m src.cli run-ablation`
