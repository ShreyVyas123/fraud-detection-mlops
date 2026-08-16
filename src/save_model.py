import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/creditcard.csv"
MODEL_PATH = "models/fraud_model.joblib"


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"]).copy()
    y = df["Class"]

    # Same preprocessing used during experimentation.
    scaler = StandardScaler()

    X[["Time", "Amount"]] = scaler.fit_transform(
        X[["Time", "Amount"]]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Best parameters discovered by Optuna.
    model = LogisticRegression(
        C=0.0016655055320374391,
        solver="liblinear",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    print("Training final model...")

    model.fit(
        X_train,
        y_train,
    )

    # Save both model and scaler.
    artifact = {
        "model": model,
        "scaler": scaler,
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    print(
        f"✓ Model saved to {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
