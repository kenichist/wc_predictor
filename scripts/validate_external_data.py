from __future__ import annotations

from src.config import load_config
from src.sources.external_data_validation import validate_external_data


if __name__ == "__main__":
    report = validate_external_data(load_config())
    print(report.to_string(index=False))
