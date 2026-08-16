import pandas as pd


EXPECTED_COLUMNS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27",
    "V28",
    "Amount",
    "Class"
]


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the dataset has the expected columns."""

    actual_columns = list(df.columns)

    if actual_columns != EXPECTED_COLUMNS:
        missing = set(EXPECTED_COLUMNS) - set(actual_columns)
        unexpected = set(actual_columns) - set(EXPECTED_COLUMNS)

        raise ValueError(
            f"Schema validation failed.\n"
            f"Missing columns: {missing}\n"
            f"Unexpected columns: {unexpected}"
        )

    print("✓ Schema validation passed")


def validate_nulls(df: pd.DataFrame) -> None:
    """Check for missing values."""

    null_count = int(df.isnull().sum().sum())

    if null_count > 0:
        raise ValueError(
            f"Null validation failed: {null_count} missing values found."
        )

    print("✓ Null validation passed")


def validate_target(df: pd.DataFrame) -> None:
    """Validate the fraud target column."""

    if "Class" not in df.columns:
        raise ValueError("Target validation failed: 'Class' column not found.")

    unique_values = set(df["Class"].unique())

    if not unique_values.issubset({0, 1}):
        raise ValueError(
            f"Target validation failed. Unexpected Class values: {unique_values}"
        )

    if df["Class"].nunique() < 2:
        raise ValueError(
            "Target validation failed: dataset contains only one class."
        )

    print("✓ Target validation passed")


def validate_class_balance(
    df: pd.DataFrame,
    minimum_fraud_rate: float = 0.001,
    maximum_fraud_rate: float = 0.01,
) -> None:
    """
    Validate that the fraud class exists within an expected range.

    The current dataset has approximately 0.173% fraud.
    """

    fraud_rate = float(df["Class"].mean())

    print(f"Fraud rate: {fraud_rate:.4%}")

    if not (minimum_fraud_rate <= fraud_rate <= maximum_fraud_rate):
        raise ValueError(
            f"Class-balance validation failed. "
            f"Fraud rate {fraud_rate:.4%} is outside the expected range "
            f"{minimum_fraud_rate:.2%} - {maximum_fraud_rate:.2%}."
        )

    print("✓ Class-balance validation passed")


def validate_dataset(df: pd.DataFrame) -> bool:
    """Run every validation check."""

    print("\n==============================")
    print("DATA VALIDATION GATE")
    print("==============================")

    validate_schema(df)
    validate_nulls(df)
    validate_target(df)
    validate_class_balance(df)

    print("==============================")
    print("✓ ALL VALIDATION CHECKS PASSED")
    print("✓ DATA IS SAFE TO CONTINUE")
    print("==============================\n")

    return True
