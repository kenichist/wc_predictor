from __future__ import annotations

import logging
import sys


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console logging once for CLI usage."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
