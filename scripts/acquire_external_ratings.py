from __future__ import annotations

import argparse

from src.config import load_config
from src.sources.external_rating_acquisition import acquire_external_ratings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire real FIFA rankings and World Football Elo data")
    parser.add_argument("--force", action="store_true", help="Replace existing external files with newly validated data")
    parser.add_argument("--skip-post-checks", action="store_true", help="Skip downstream activation checks")
    args = parser.parse_args(argv)
    acquire_external_ratings(force=args.force, config=load_config(), run_post_checks=not args.skip_post_checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
