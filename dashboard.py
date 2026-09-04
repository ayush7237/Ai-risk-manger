import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

st.set_page_config(page_title="AI Risk Manager - Fraud Dashboard", layout="wide")

# ---------- Load model artifacts ----------
@st.cache_resource
def load_artifacts():
    model = XGBClassifier()
    model.load_model("fraud_model_v2.json")
    feature_order = joblib.load("model_features.pkl")
    threshold = joblib.load("model_threshold.pkl")
    return model, feature_order, threshold

model, feature_order, threshold = load_artifacts()


def engineer_features(row: dict) -> pd.DataFrame:
    """Same feature engineering used in training / the API."""
    errorBalanceDest = row["oldbalanceDest"] + row["amount"] - row["newbalanceDest"]
    dest_balance_was_zero = int(row["oldbalanceDest"] == 0)
    type_TRANSFER = int(row["type"] == "TRANSFER")

    features = {
        "step": row["step"],
        "amount": row["amount"],
        "oldbalanceOrg": row["oldbalanceOrg"],
        "oldbalanceDest": row["oldbalanceDest"],
        "newbalanceDest": row["newbalanceDest"],
        "dest_balance_was_zero": dest_balance_was_zero,
        "errorBalanceDest": errorBalanceDest,
        "type_TRANSFER": type_TRANSFER,
    }
    df = pd.DataFrame([features])
    return df.reindex(columns=feature_order, fill_value=0)


def score_row(row: dict):
    X = engineer_features(row)
    proba = float(model.predict_proba(X)[0, 1])
    return proba, proba >= threshold


# ---------- Header ----------
st.title("AI Risk Manager — Fraud Detection Dashboard")
st.caption(
    "XGBoost model trained on PaySim transaction data. "
    "Scoped to CASH_OUT / TRANSFER transactions, the only types fraud occurs in for this dataset."
)

# ---------- Headline metrics ----------
st.subheader("Model Performance (held-out test set)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precision", "70%")
col2.metric("Recall", "82%")
col3.metric("PR-AUC", "0.88")
col4.metric("Operating threshold", f"{threshold:.3f}")

col5, col6 = st.columns(2)
col5.metric("Fraud value caught", "98.4%")
col6.metric("Legit value unnecessarily flagged", "0.201%")

st.divider()

# ---------- Sample batch ----------
st.subheader("Batch Scoring Demo")
st.caption("A mix of sample transactions — some designed to look suspicious, some normal.")

sample_transactions = [
    {"label": "Large transfer, account drained, dest stuck at zero",
     "step": 200, "amount": 500000, "oldbalanceOrg": 500000, "newbalanceOrig": 0,
     "oldbalanceDest": 0, "newbalanceDest": 0, "type": "TRANSFER"},
    {"label": "Normal transfer between active accounts",
     "step": 200, "amount": 5000, "oldbalanceOrg": 50000, "newbalanceOrig": 45000,
     "oldbalanceDest": 20000, "newbalanceDest": 25000, "type": "TRANSFER"},
    {"label": "Cash-out, balance mismatch at destination",
     "step": 150, "amount": 250000, "oldbalanceOrg": 260000, "newbalanceOrig": 10000,
     "oldbalanceDest": 0, "newbalanceDest": 0, "type": "CASH_OUT"},
    {"label": "Small routine transfer",
     "step": 100, "amount": 1200, "oldbalanceOrg": 8000, "newbalanceOrig": 6800,
     "oldbalanceDest": 3000, "newbalanceDest": 4200, "type": "TRANSFER"},
    {"label": "Mid-size cash-out, healthy balances",
     "step": 300, "amount": 15000, "oldbalanceOrg": 100000, "newbalanceOrig": 85000,
     "oldbalanceDest": 40000, "newbalanceDest": 55000, "type": "CASH_OUT"},
]

results = []
for txn in sample_transactions:
    proba, flagged = score_row(txn)
    results.append({
        "Description": txn["label"],
        "Type": txn["type"],
        "Amount": txn["amount"],
        "Risk Score": round(proba, 4),
        "Flagged": "🚩 Yes" if flagged else "✅ No",
    })

results_df = pd.DataFrame(results)
st.dataframe(results_df, use_container_width=True, hide_index=True)

st.divider()

# ---------- Manual test ----------
st.subheader("Score Your Own Transaction")
with st.form("manual_score"):
    c1, c2, c3 = st.columns(3)
    with c1:
        step = st.number_input("Step (time unit)", value=200, min_value=1)
        amount = st.number_input("Amount", value=10000.0, min_value=0.0)
    with c2:
        oldbalanceOrg = st.number_input("Sender balance before", value=50000.0, min_value=0.0)
        newbalanceOrig = st.number_input("Sender balance after", value=40000.0, min_value=0.0)
    with c3:
        oldbalanceDest = st.number_input("Receiver balance before", value=10000.0, min_value=0.0)
        newbalanceDest = st.number_input("Receiver balance after", value=20000.0, min_value=0.0)
    txn_type = st.selectbox("Transaction type", ["TRANSFER", "CASH_OUT"])
    submitted = st.form_submit_button("Score Transaction")

if submitted:
    manual_txn = {
        "step": step, "amount": amount, "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig, "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest, "type": txn_type,
    }
    proba, flagged = score_row(manual_txn)
    if flagged:
        st.error(f"🚩 FLAGGED — risk score: {proba:.4f} (threshold: {threshold:.3f})")
    else:
        st.success(f"✅ Not flagged — risk score: {proba:.4f} (threshold: {threshold:.3f})")
