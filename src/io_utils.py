from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests


logger = logging.getLogger(__name__)


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    joined = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def download_file(
    url: str,
    output_path: str | Path,
    *,
    timeout_seconds: int = 30,
    retry_count: int = 3,
    sleep_seconds: float = 1.0,
) -> Path:
    """Download a URL to disk with conservative retries."""
    path = ensure_parent_dir(output_path)
    last_error: Exception | None = None
    for attempt in range(1, retry_count + 1):
        try:
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            path.write_bytes(response.content)
            logger.info("Downloaded %s to %s", url, path)
            return path
        except requests.RequestException as exc:
            last_error = exc
            if attempt == retry_count:
                break
            delay = sleep_seconds * attempt
            logger.warning("Download failed on attempt %s/%s; retrying in %.1fs", attempt, retry_count, delay)
            time.sleep(delay)
    raise RuntimeError(f"Failed to download {url}") from last_error


def write_json(data: Any, path: str | Path) -> Path:
    resolved = ensure_parent_dir(path)
    with resolved.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=str)
    return resolved


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_dataframe(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    resolved = ensure_parent_dir(path)
    suffix = resolved.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(resolved, index=index)
    elif suffix == ".csv":
        df.to_csv(resolved, index=index)
    else:
        raise ValueError(f"Unsupported dataframe output format: {resolved}")
    logger.info("Wrote %s rows to %s", len(df), resolved)
    return resolved


def read_dataframe(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".parquet":
        try:
            return pd.read_parquet(resolved)
        except Exception as exc:
            try:
                logger.warning("Could not read %s as parquet (%s); retrying with single-threaded pyarrow", resolved, exc)
                return pd.read_parquet(resolved, engine="pyarrow", use_threads=False)
            except Exception as retry_exc:
                logger.warning("Single-threaded parquet retry failed for %s: %s", resolved, retry_exc)
            csv_fallback = resolved.with_suffix(".csv")
            if csv_fallback.exists():
                logger.warning("Could not read %s as parquet (%s); falling back to %s", resolved, exc, csv_fallback)
                return pd.read_csv(csv_fallback, low_memory=False)
            raise
    if suffix == ".csv":
        return pd.read_csv(resolved, low_memory=False)
    raise ValueError(f"Unsupported dataframe input format: {resolved}")
