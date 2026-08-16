import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from imblearn.over_sampling import SMOTE


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a model."""

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    cm = confusion_matrix(y_test, predictions)

    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def main():

    print("Loading dataset...")

    df = pd.read_csv("data/creditcard.csv")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Scale Time and Amount
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X[["Time", "Amount"]] = scaler.fit_transform(
        X[["Time", "Amount"]]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")

    print(f"Original training fraud rate: {y_train.mean():.4%}")

    # -------------------------------------------------
    # EXPERIMENT 1: CLASS WEIGHT
    # -------------------------------------------------

    class_weight_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    class_weight_result = evaluate_model(
        "CLASS WEIGHT",
        class_weight_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # -------------------------------------------------
    # EXPERIMENT 2: SMOTE
    # -------------------------------------------------

    print("\nApplying SMOTE...")

    smote = SMOTE(
        random_state=42,
        sampling_strategy=1.0,
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train,
    )

    print(
        f"SMOTE training rows: {len(X_train_smote)}"
    )

    print(
        f"SMOTE fraud rate: {y_train_smote.mean():.4%}"
    )

    smote_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    smote_result = evaluate_model(
        "SMOTE",
        smote_model,
        X_train_smote,
        X_test,
        y_train_smote,
        y_test,
    )

    # -------------------------------------------------
    # COMPARISON
    # -------------------------------------------------

    results = pd.DataFrame(
        [
            class_weight_result,
            smote_result,
        ]
    )

    print("\n")
    print("=" * 70)
    print("CLASS WEIGHT vs SMOTE")
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    results.to_csv(
        "reports/model_comparison.csv",
        index=False,
    )

    print("\n✓ Results saved to reports/model_comparison.csv")


if __name__ == "__main__":
    main()
