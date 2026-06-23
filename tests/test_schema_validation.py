import pandas as pd

from src.validation import REQUIRED_TRAINING_COLUMNS, validate_training_schema


def test_training_schema_contains_required_columns() -> None:
    df = pd.DataFrame([{column: None for column in REQUIRED_TRAINING_COLUMNS}])

    missing = validate_training_schema(df)

    assert missing == []
