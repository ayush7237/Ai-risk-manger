from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from xgboost import XGBClassifier

app = FastAPI(title="AI Risk Manager - Fraud Detection API")

# Load model artifacts once at startup
model = XGBClassifier()
model.load_model("fraud_model_v2.json")
feature_order = joblib.load("model_features.pkl")
threshold = joblib.load("model_threshold.pkl")


class Transaction(BaseModel):
    step: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type: str  # "CASH_OUT" or "TRANSFER"


def engineer_features(txn: Transaction) -> pd.DataFrame:
    """Recreate the same features used during training."""
    errorBalanceDest = txn.oldbalanceDest + txn.amount - txn.newbalanceDest
    dest_balance_was_zero = int(txn.oldbalanceDest == 0)
    type_TRANSFER = int(txn.type == "TRANSFER")

    row = {
        "step": txn.step,
        "amount": txn.amount,
        "oldbalanceOrg": txn.oldbalanceOrg,
        "oldbalanceDest": txn.oldbalanceDest,
        "newbalanceDest": txn.newbalanceDest,
        "dest_balance_was_zero": dest_balance_was_zero,
        "errorBalanceDest": errorBalanceDest,
        "type_TRANSFER": type_TRANSFER,
    }

    df = pd.DataFrame([row])
    # Ensure column order matches training exactly
    df = df.reindex(columns=feature_order, fill_value=0)
    return df


@app.get("/")
def root():
    return {"status": "ok", "message": "Fraud Risk Manager API is running"}


@app.post("/score")
def score_transaction(txn: Transaction):
    X = engineer_features(txn)
    proba = float(model.predict_proba(X)[0, 1])
    flagged = bool(proba >= threshold)

    # Simple human-readable reasoning based on top known signals
    reasons = []
    if txn.newbalanceDest == 0 and txn.oldbalanceDest == 0:
        reasons.append("destination account has zero balance before and after")
    if (txn.oldbalanceDest + txn.amount) != txn.newbalanceDest:
        reasons.append("destination balance does not match expected post-transaction value")
    if txn.type == "TRANSFER" or txn.type == "CASH_OUT":
        reasons.append(f"transaction type '{txn.type}' is the only type fraud occurs in for this model")

    return {
        "risk_score": round(proba, 4),
        "flagged": flagged,
        "threshold_used": threshold,
        "reasons": reasons if flagged else ["no strong risk signals detected"],
    }
