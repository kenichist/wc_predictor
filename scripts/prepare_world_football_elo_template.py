from __future__ import annotations

from src.config import load_config
from src.sources.external_data_validation import prepare_external_templates


if __name__ == "__main__":
    paths = prepare_external_templates(load_config())
    print(paths["world_football_elo_template"])
