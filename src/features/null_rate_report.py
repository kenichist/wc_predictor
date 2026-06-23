from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import config_path, load_config
from src.features.dynamic_ratings import DYNAMIC_RATING_COLUMNS
from src.features.feature_sets import FEATURE_GROUPS
from src.features.internal_historical_elo import INTERNAL_HISTORICAL_ELO_COLUMNS
from src.features.pi_ratings import PI_RATING_COLUMNS
from src.features.rolling_team_form import ROLLING_FEATURE_COLUMNS
from src.io_utils import read_dataframe, write_dataframe
from src.sources.fifa_rankings import FIFA_RANKING_COLUMNS
from src.sources.injury_interface import INJURY_FEATURE_COLUMNS
from src.sources.player_stats_interface import PLAYER_FEATURE_COLUMNS
from src.sources.world_football_elo import WORLD_FOOTBALL_ELO_COLUMNS
from src.sources.xg_interface import XG_FEATURE_COLUMNS
from src.validation import CORE_COLUMNS, TARGET_COLUMNS


GROUP_COLUMN_MAP: dict[str, set[str]] = {
    "metadata": set(CORE_COLUMNS),
    "target": set(TARGET_COLUMNS),
    "rolling_form": set(ROLLING_FEATURE_COLUMNS),
    "basic_elo": {"home_elo_pre_match", "away_elo_pre_match", "elo_diff"},
    "baseline": set(FEATURE_GROUPS["baseline"]),
    "dynamic_ratings": set(DYNAMIC_RATING_COLUMNS),
    "pi_ratings": set(PI_RATING_COLUMNS),
    "fifa_rankings": set(FIFA_RANKING_COLUMNS),
    "external_elo": set(WORLD_FOOTBALL_ELO_COLUMNS),
    "internal_historical_elo": set(INTERNAL_HISTORICAL_ELO_COLUMNS),
    "xg": set(XG_FEATURE_COLUMNS),
    "squad": set(PLAYER_FEATURE_COLUMNS),
    "injuries": set(INJURY_FEATURE_COLUMNS),
    "market": set(FEATURE_GROUPS["market"]),
}


def generate_feature_null_rate_report(config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or load_config()
    train_path = config_path(cfg, "match_training_dataset_advanced_parquet")
    pred_path = config_path(cfg, "worldcup_2026_prediction_input_advanced")
    if not train_path.exists():
        train_path = config_path(cfg, "match_training_dataset_parquet")
    training_df = read_dataframe(train_path)
    frames = [training_df.assign(_dataset="advanced_training")]
    prediction_df = pd.DataFrame()
    if pred_path.exists():
        prediction_df = read_dataframe(pred_path)
        frames.append(prediction_df.assign(_dataset="advanced_prediction"))
    df = pd.concat(frames, ignore_index=True, sort=False)

    rows = []
    for column in df.columns:
        if column == "_dataset":
            continue
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        is_numeric = numeric.notna().any()
        rows.append(
            {
                "feature_name": column,
                "feature_group": feature_group_for_column(column),
                "total_rows": len(series),
                "null_count": int(series.isna().sum()),
                "null_rate": float(series.isna().mean()) if len(series) else 0.0,
                "non_null_count": int(series.notna().sum()),
                "unique_values": int(series.nunique(dropna=True)),
                "min": float(numeric.min()) if is_numeric else np.nan,
                "max": float(numeric.max()) if is_numeric else np.nan,
                "mean": float(numeric.mean()) if is_numeric else np.nan,
            }
        )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("feature_group", as_index=False)
        .agg(
            average_null_rate=("null_rate", "mean"),
            fully_null_columns=("null_rate", lambda values: int((values == 1.0).sum())),
            usable_columns=("non_null_count", lambda values: int((values > 0).sum())),
            total_columns=("feature_name", "count"),
        )
    )
    coverage_rows = []
    for group, columns in GROUP_COLUMN_MAP.items():
        coverage_rows.append(
            {
                "feature_group": group,
                "historical_training_coverage": _row_coverage(training_df, columns),
                "worldcup_2026_coverage": _row_coverage(prediction_df, columns),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    summary = summary.merge(coverage, on="feature_group", how="left")
    summary["warning"] = summary.apply(_summary_warning, axis=1)
    write_dataframe(detail, config_path(cfg, "feature_null_rate_report_csv"))
    _write_markdown(detail, summary, config_path(cfg, "feature_null_rate_report_md"))
    return detail, summary


def feature_group_for_column(column: str) -> str:
    for group, columns in GROUP_COLUMN_MAP.items():
        if column in columns:
            return group
    if column in FEATURE_GROUPS["baseline"]:
        return "baseline"
    return "metadata"


def _row_coverage(df: pd.DataFrame, columns: set[str]) -> float:
    if df.empty:
        return 0.0
    available = [column for column in columns if column in df.columns]
    if not available:
        return 0.0
    return float(df[available].notna().any(axis=1).mean())


def _summary_warning(row: pd.Series) -> str:
    warnings: list[str] = []
    if row["average_null_rate"] > 0.5:
        warnings.append("average null rate above 50%")
    if row["feature_group"] in {"fifa_rankings", "external_elo"}:
        if row["worldcup_2026_coverage"] < 0.9:
            warnings.append("World Cup 2026 coverage below 90%")
        if row["historical_training_coverage"] < 0.5:
            warnings.append("historical training coverage below 50%")
        if row["usable_columns"] == 0:
            warnings.append("add the real external CSV to activate this group")
    return "; ".join(warnings)


def _write_markdown(detail: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Feature Null Rate Report",
        "",
        "## Summary By Feature Group",
        "",
        "| feature_group | average_null_rate | fully_null_columns | usable_columns | total_columns | historical_training_coverage | worldcup_2026_coverage | warning |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.feature_group} | {row.average_null_rate:.3f} | {row.fully_null_columns} | "
            f"{row.usable_columns} | {row.total_columns} | {row.historical_training_coverage:.3f} | "
            f"{row.worldcup_2026_coverage:.3f} | {row.warning} |"
        )
    lines.extend(["", "## Fully Null Columns", ""])
    fully_null = detail[detail["null_rate"].eq(1.0)][["feature_name", "feature_group"]]
    if fully_null.empty:
        lines.append("No fully null columns.")
    else:
        for row in fully_null.itertuples(index=False):
            lines.append(f"- `{row.feature_name}` ({row.feature_group})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
