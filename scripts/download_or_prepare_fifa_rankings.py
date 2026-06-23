from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from src.config import load_config, resolve_project_path
from src.io_utils import download_file
from src.sources.fifa_rankings import prepare_fifa_rankings


def prepare_from_config(input_path: str | Path | None = None, config: dict[str, Any] | None = None) -> Path | None:
    cfg = config or load_config()
    source_cfg = cfg.get("external_sources", {}).get("fifa_rankings", {})
    output_path = resolve_project_path(source_cfg.get("output_path", "data/external/fifa_rankings.csv"))
    raw_dir = resolve_project_path(source_cfg.get("raw_dir", "data/raw/fifa_rankings"))
    template_path = output_path.with_name("fifa_rankings_template.csv")
    _write_template_if_missing(template_path)

    if input_path:
        prepare_fifa_rankings(Path(input_path), output_path)
        print(f"Prepared FIFA rankings at {output_path}")
        return output_path

    source_url = source_cfg.get("source_url")
    if source_url:
        raw_path = _download_source(str(source_url), raw_dir, "fifa_rankings")
        prepare_fifa_rankings(raw_path, output_path)
        print(f"Downloaded and prepared FIFA rankings at {output_path}")
        return output_path

    print("No FIFA rankings source_url is configured.")
    print(f"Place a real CSV at: {output_path}")
    print(f"A sample template is available at: {template_path}")
    print("Template rows are examples only and should not be used as final model inputs.")
    return None


def _download_source(source_url: str, raw_dir: Path, stem: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(source_url).path).suffix or ".csv"
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d%H%M%S")
    return download_file(source_url, raw_dir / f"{stem}_{timestamp}{suffix}")


def _write_template_if_missing(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"date": "2024-12-19", "team": "Argentina", "rank": 1, "points": 1867.25, "source": "manual_template"},
            {"date": "2024-12-19", "team": "France", "rank": 2, "points": 1859.78, "source": "manual_template"},
            {"date": "2024-12-19", "team": "Spain", "rank": 3, "points": 1853.27, "source": "manual_template"},
        ]
    ).to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download or prepare FIFA men's ranking history")
    parser.add_argument("--input", default=None, help="Optional raw CSV/XLSX/JSON file to normalize")
    args = parser.parse_args(argv)
    prepare_from_config(args.input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
