from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from src.config import load_config, resolve_project_path
from src.io_utils import download_file, write_dataframe
from src.sources.tabular_external import read_external_table
from src.sources.world_football_elo import (
    normalize_world_football_elo,
    prepare_world_football_elo,
    validate_world_football_elo,
)


def prepare_from_config(input_path: str | Path | None = None, config: dict[str, Any] | None = None) -> Path | None:
    cfg = config or load_config()
    source_cfg = cfg.get("external_sources", {}).get("world_football_elo", {})
    output_path = resolve_project_path(source_cfg.get("output_path", "data/external/world_football_elo.csv"))
    raw_dir = resolve_project_path(source_cfg.get("raw_dir", "data/raw/world_football_elo"))
    template_path = output_path.with_name("world_football_elo_template.csv")
    _write_template_if_missing(template_path)
    allow_html = bool(source_cfg.get("allow_html", False))

    if input_path:
        _prepare_input(Path(input_path), output_path, allow_html=allow_html)
        print(f"Prepared World Football Elo at {output_path}")
        return output_path

    source_url = source_cfg.get("source_url")
    if source_url:
        raw_path = _download_source(str(source_url), raw_dir, "world_football_elo")
        _prepare_input(raw_path, output_path, allow_html=allow_html)
        print(f"Downloaded and prepared World Football Elo at {output_path}")
        return output_path

    print("No World Football Elo source_url is configured.")
    print(f"Place a real CSV at: {output_path}")
    print(f"A sample template is available at: {template_path}")
    print("Template rows are examples only and should not be used as final model inputs.")
    return None


def _prepare_input(input_path: Path, output_path: Path, *, allow_html: bool) -> None:
    if input_path.suffix.lower() in {".html", ".htm"} and allow_html:
        raw = read_external_table(input_path, allow_html=True)
        errors = validate_world_football_elo(raw)
        if errors:
            raise ValueError("Invalid World Football Elo input: " + "; ".join(errors))
        write_dataframe(normalize_world_football_elo(raw), output_path)
        return
    prepare_world_football_elo(input_path, output_path)


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
            {"date": "2024-12-31", "team": "Argentina", "elo": 2145, "source": "manual_template"},
            {"date": "2024-12-31", "team": "France", "elo": 2098, "source": "manual_template"},
            {"date": "2024-12-31", "team": "Spain", "elo": 2075, "source": "manual_template"},
        ]
    ).to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download or prepare World Football Elo rating history")
    parser.add_argument("--input", default=None, help="Optional raw CSV/XLSX/JSON file to normalize")
    args = parser.parse_args(argv)
    prepare_from_config(args.input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
