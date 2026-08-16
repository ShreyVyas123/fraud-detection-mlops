import pandas as pd
import pytest
from src.validation import (
    validate_schema,
    validate_nulls,
    validate_target,
    validate_class_balance,
    validate_dataset,
    EXPECTED_COLUMNS,
)


def make_valid_df(n_normal=1000, n_fraud=2):
    data = {col: [0.0] * (n_normal + n_fraud) for col in EXPECTED_COLUMNS}
    data["Class"] = [0] * n_normal + [1] * n_fraud
    return pd.DataFrame(data)


def test_validate_schema_passes_with_correct_columns():
    df = make_valid_df()
    validate_schema(df)


def test_validate_schema_fails_with_missing_column():
    df = make_valid_df().drop(columns=["V1"])
    with pytest.raises(ValueError):
        validate_schema(df)


def test_validate_nulls_passes_with_no_nulls():
    df = make_valid_df()
    validate_nulls(df)


def test_validate_nulls_fails_with_nulls():
    df = make_valid_df()
    df.loc[0, "Amount"] = None
    with pytest.raises(ValueError):
        validate_nulls(df)


def test_validate_target_passes_with_binary_classes():
    df = make_valid_df()
    validate_target(df)


def test_validate_target_fails_with_single_class():
    df = make_valid_df(n_normal=1000, n_fraud=0)
    with pytest.raises(ValueError):
        validate_target(df)


def test_validate_class_balance_passes_within_range():
    df = make_valid_df(n_normal=1000, n_fraud=2)
    validate_class_balance(df)


def test_validate_class_balance_fails_when_zero_fraud():
    df = make_valid_df(n_normal=1000, n_fraud=0)
    with pytest.raises(ValueError):
        validate_class_balance(df)


def test_validate_dataset_full_pipeline_passes():
    df = make_valid_df()
    assert validate_dataset(df) is True
