from __future__ import annotations

import argparse

from src.config import load_config
from src.sources.external_rating_acquisition import acquire_world_football_elo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire real World Football Elo data")
    parser.add_argument("--input", default=None, help="Optional local CSV/XLSX/JSON/HTML file")
    parser.add_argument("--url", default=None, help="Optional source URL override")
    parser.add_argument("--force", action="store_true", help="Replace existing external file with newly validated data")
    parser.add_argument("--skip-post-checks", action="store_true", help="Skip downstream activation checks")
    args = parser.parse_args(argv)
    acquire_world_football_elo(
        input_path=args.input,
        url=args.url,
        force=args.force,
        config=load_config(),
        run_post_checks=not args.skip_post_checks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
