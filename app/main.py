import os
import joblib
import pandas as pd
from fastapi import FastAPI, Response
from pydantic import BaseModel
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(
    title="Fraud Detection API",
    description="Credit card fraud detection API",
    version="1.0.0",
)

MODEL_PATH = "models/fraud_model.joblib"


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


artifact = load_model()
if artifact is not None:
    model = artifact["model"]
    scaler = artifact["scaler"]
else:
    model = None
    scaler = None


# ── Prometheus metrics ──────────────────────────────────────
FLAGGED_FRAUD_COUNTER = Counter(
    "flagged_fraud_total",
    "Total number of transactions flagged as fraud by the model",
)

TOTAL_PREDICTIONS_COUNTER = Counter(
    "predictions_total",
    "Total number of prediction requests made",
)

FALSE_POSITIVE_PROXY_COUNTER = Counter(
    "false_positive_proxy_total",
    "Proxy count of low-confidence fraud flags (0.5-0.7 probability), "
    "used as an approximate false-positive signal since true labels "
    "are not available at inference time",
)


@app.get("/")
def root():
    return {
        "service": "Fraud Detection API",
        "status": "running",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        return {
            "error": "Model is not loaded"
        }

    data = pd.DataFrame(
        [transaction.model_dump()]
    )
    data[["Time", "Amount"]] = scaler.transform(
        data[["Time", "Amount"]]
    )

    probability = float(
        model.predict_proba(data)[0][1]
    )
    prediction = int(
        probability >= 0.5
    )

    TOTAL_PREDICTIONS_COUNTER.inc()
    if prediction == 1:
        FLAGGED_FRAUD_COUNTER.inc()
        if probability < 0.7:
            FALSE_POSITIVE_PROXY_COUNTER.inc()

    return {
        "fraud_prediction": prediction,
        "fraud_probability": probability,
        "is_fraud": bool(prediction),
    }
