import os

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/creditcard.csv"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def prepare_data():
    """Load, scale and split the fraud dataset."""

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].copy()

    # Scale Time and Amount.
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

    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """Calculate fraud-detection metrics."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities,
        ),
    }

    return metrics


def run_optuna(
    X_train,
    X_test,
    y_train,
    y_test,
    strategy,
):
    """
    Run Optuna hyperparameter optimization.

    strategy:
        class_weight
        smote
    """

    print("\n" + "=" * 70)
    print(f"STARTING OPTUNA — {strategy.upper()}")
    print("=" * 70)

    # -------------------------------------------------
    # PREPARE TRAINING DATA
    # -------------------------------------------------

    if strategy == "smote":

        print("Applying SMOTE...")

        smote = SMOTE(
            random_state=42,
            sampling_strategy=1.0,
        )

        X_train_used, y_train_used = smote.fit_resample(
            X_train,
            y_train,
        )

        print(
            f"SMOTE training rows: {len(X_train_used)}"
        )

        print(
            f"SMOTE fraud rate: {y_train_used.mean():.4%}"
        )

    else:

        X_train_used = X_train
        y_train_used = y_train

        print(
            f"Training rows: {len(X_train_used)}"
        )

        print(
            f"Training fraud rate: "
            f"{y_train_used.mean():.4%}"
        )

    # -------------------------------------------------
    # OPTUNA OBJECTIVE
    # -------------------------------------------------

    def objective(trial):

        C = trial.suggest_float(
            "C",
            0.001,
            10.0,
            log=True,
        )

        solver = trial.suggest_categorical(
            "solver",
            [
                "lbfgs",
                "liblinear",
            ],
        )

        max_iter = trial.suggest_int(
            "max_iter",
            500,
            2000,
            step=500,
        )

        if strategy == "class_weight":

            model = LogisticRegression(
                C=C,
                solver=solver,
                max_iter=max_iter,
                class_weight="balanced",
                random_state=42,
            )

        else:

            model = LogisticRegression(
                C=C,
                solver=solver,
                max_iter=max_iter,
                random_state=42,
            )

        model.fit(
            X_train_used,
            y_train_used,
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        return metrics["f1"]

    # -------------------------------------------------
    # CREATE STUDY
    # -------------------------------------------------

    study = optuna.create_study(
        direction="maximize",
        study_name=f"fraud_{strategy}",
    )

    study.optimize(
        objective,
        n_trials=10,
    )

    # -------------------------------------------------
    # BEST PARAMETERS
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print(f"OPTUNA RESULTS — {strategy.upper()}")
    print("=" * 70)

    print(
        f"Best F1: {study.best_value:.6f}"
    )

    print("Best parameters:")

    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # -------------------------------------------------
    # TRAIN BEST MODEL
    # -------------------------------------------------

    if strategy == "class_weight":

        best_model = LogisticRegression(
            **study.best_params,
            class_weight="balanced",
            random_state=42,
        )

    else:

        best_model = LogisticRegression(
            **study.best_params,
            random_state=42,
        )

    best_model.fit(
        X_train_used,
        y_train_used,
    )

    metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
    )

    print(
        f"\n{strategy.upper()} FINAL METRICS"
    )

    for key, value in metrics.items():
        print(
            f"{key}: {value:.4f}"
        )

    return study, best_model, metrics


def log_mlflow(
    strategy,
    study,
    model,
    metrics,
):
    """Log experiment results and model to MLflow."""

    experiment_name = "fraud_detection"

    mlflow.set_experiment(
        experiment_name
    )

    with mlflow.start_run(
        run_name=f"Optuna_{strategy}"
    ):

        # Strategy
        mlflow.log_param(
            "strategy",
            strategy,
        )

        # Number of Optuna trials
        mlflow.log_param(
            "n_trials",
            len(study.trials),
        )

        # Best Optuna parameters
        for key, value in study.best_params.items():

            mlflow.log_param(
                f"best_{key}",
                value,
            )

        # Metrics
        for key, value in metrics.items():

            mlflow.log_metric(
                key,
                float(value),
            )

        # Best Optuna objective
        mlflow.log_metric(
            "optuna_best_f1",
            float(study.best_value),
        )

        # Save model
        mlflow.sklearn.log_model(
            model,
            name=f"fraud_model_{strategy}",
        )

        print(
            f"✓ MLflow run logged: "
            f"Optuna_{strategy}"
        )


def main():

    print("=" * 70)
    print("FRAUD DETECTION — OPTUNA + MLFLOW")
    print("=" * 70)

    # -------------------------------------------------
    # PREPARE DATA
    # -------------------------------------------------

    X_train, X_test, y_train, y_test = prepare_data()

    print(
        f"\nTraining rows: {len(X_train)}"
    )

    print(
        f"Testing rows : {len(X_test)}"
    )

    print(
        f"Training fraud rate: "
        f"{y_train.mean():.4%}"
    )

    # -------------------------------------------------
    # MLFLOW SQLITE BACKEND
    # -------------------------------------------------

    print("\nConfiguring MLflow...")

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    print(
        f"MLflow tracking URI: "
        f"{MLFLOW_TRACKING_URI}"
    )

    # -------------------------------------------------
    # CLASS WEIGHT + OPTUNA
    # -------------------------------------------------

    study_cw, model_cw, metrics_cw = run_optuna(
        X_train,
        X_test,
        y_train,
        y_test,
        "class_weight",
    )

    log_mlflow(
        "class_weight",
        study_cw,
        model_cw,
        metrics_cw,
    )

    # -------------------------------------------------
    # SMOTE + OPTUNA
    # -------------------------------------------------

    study_smote, model_smote, metrics_smote = run_optuna(
        X_train,
        X_test,
        y_train,
        y_test,
        "smote",
    )

    log_mlflow(
        "smote",
        study_smote,
        model_smote,
        metrics_smote,
    )

    # -------------------------------------------------
    # FINAL COMPARISON
    # -------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "strategy": "class_weight",
                **metrics_cw,
            },
            {
                "strategy": "smote",
                **metrics_smote,
            },
        ]
    )

    print("\n")
    print("=" * 70)
    print("OPTUNA + MLFLOW COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    os.makedirs(
        "reports",
        exist_ok=True,
    )

    comparison.to_csv(
        "reports/optuna_mlflow_comparison.csv",
        index=False,
    )

    print(
        "\n✓ Comparison saved to "
        "reports/optuna_mlflow_comparison.csv"
    )

    print("\n" + "=" * 70)
    print("✓ OPTUNA + MLFLOW EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
