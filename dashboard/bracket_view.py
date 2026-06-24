from __future__ import annotations

import pandas as pd


STAGE_LABELS = {
    "round_of_32": "Round Of 32",
    "round_of_16": "Round Of 16",
    "quarterfinal": "Quarterfinal",
    "semifinal": "Semifinal",
    "third_place": "Third-Place",
    "final": "Final",
    "champion": "Champion",
}


def display_stage(stage: str) -> str:
    return STAGE_LABELS.get(str(stage), str(stage).replace("_", " ").title())


def add_matchup_label(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    if {"team_a", "team_b"}.issubset(output.columns):
        output["matchup"] = output["team_a"].astype(str) + " vs " + output["team_b"].astype(str)
    return output


def stage_probability_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "champion_probability",
        "final_probability",
        "semifinal_probability",
        "quarterfinal_probability",
        "round_of_16_probability",
        "round_of_32_probability",
    ]
    return [column for column in preferred if column in df.columns]


def validate_stage_sums(df: pd.DataFrame) -> pd.DataFrame:
    expected = {
        "champion_probability": 1,
        "final_probability": 2,
        "semifinal_probability": 4,
        "quarterfinal_probability": 8,
        "round_of_16_probability": 16,
        "round_of_32_probability": 32,
    }
    rows = []
    for column, target in expected.items():
        if column not in df.columns:
            continue
        actual = float(pd.to_numeric(df[column], errors="coerce").sum())
        rows.append(
            {
                "column": column,
                "expected_sum": target,
                "actual_sum": actual,
                "absolute_error": abs(actual - target),
                "within_tolerance": abs(actual - target) <= 0.05,
            }
        )
    return pd.DataFrame(rows)


def format_percent_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = df.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce").map(lambda value: f"{value:.2%}" if pd.notna(value) else "")
    return output
