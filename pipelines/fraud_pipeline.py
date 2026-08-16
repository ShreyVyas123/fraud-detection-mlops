import pandas as pd
import numpy as np

from zenml import pipeline, step

from src.validation import validate_dataset


@step
def ingest_data() -> pd.DataFrame:
    """Load the fraud detection dataset."""

    data_path = "data/creditcard.csv"

    df = pd.read_csv(data_path)

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


@step
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data quality gate.

    If any validation fails, this step raises an exception
    and ZenML stops the pipeline.
    """

    validate_dataset(df)

    return df


@step
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features and target.

    The dataset contains:
    - Time
    - V1 ... V28
    - Amount
    - Class (target)
    """

    print("Starting transformation...")

    # Remove target from features
    X = df.drop(columns=["Class"]).copy()

    # Target
    y = df["Class"].copy()

    # Standardize Time and Amount
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X[["Time", "Amount"]] = scaler.fit_transform(
        X[["Time", "Amount"]]
    )

    transformed = X.copy()
    transformed["Class"] = y.values

    print(f"Transformation completed: {transformed.shape}")

    return transformed


@step
def train_model(df: pd.DataFrame) -> None:
    """
    Train a baseline Logistic Regression model
    using class weighting.
    """

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print(
        f"Training fraud rate: {y_train.mean():.4%}"
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    print("✓ Logistic Regression training completed")
    print("✓ Class weighting: balanced")


@step
def evaluate_model(df: pd.DataFrame) -> None:
    """Evaluate the trained fraud detection model."""

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        average_precision_score,
    )

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n==============================")
    print("MODEL EVALUATION")
    print("==============================")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")

    print("==============================")
    print("✓ MODEL EVALUATION COMPLETE")
    print("==============================")


@pipeline
def fraud_detection_pipeline():
    """Complete fraud detection MLOps pipeline."""

    data = ingest_data()

    validated_data = validate_data(data)

    transformed_data = transform_data(validated_data)

    # IMPORTANT:
    # Evaluation now depends on training.
    # This guarantees the correct pipeline order.
    trained = train_model(transformed_data)

    evaluate_model(transformed_data, after=trained)


if __name__ == "__main__":
    fraud_detection_pipeline()
