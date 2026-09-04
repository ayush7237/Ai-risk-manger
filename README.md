# AI Risk Manager — Fraud Detection

Built for the Razorpay AI Buildathon (Track 2: AI Risk Manager).

## Problem

Payment platforms need to flag fraudulent transactions in real time — but a
naive model that just chases 100% accuracy is dangerous, because false
positives block legitimate customers and false negatives let fraud through.
This project builds a fraud-risk scoring model, evaluates it honestly on a
held-out test set, and wraps it in a working API + dashboard.

## Dataset

[PaySim synthetic mobile-money dataset](https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset)
— 6.36M transactions, 8,213 labeled as fraud (0.13%). Fraud in this dataset
occurs exclusively in `TRANSFER` and `CASH_OUT` transaction types, so the
model is scoped to those two types.

## The leakage story (why this matters)

The first model trained on all available features hit **100% precision and
recall** — a result too good to be true. Investigating feature importances
showed two engineered features (`errorBalanceOrig`, `orig_balance_drained`)
were near-deterministic artifacts of how PaySim *simulates* fraud, not
generalizable real-world fraud signals. Removing them and re-checking
importances revealed a second, subtler version of the same issue
(`newbalanceOrig` alone carried 58% of the decision weight). After removing
both rounds of leaky/artifact features, the final model relies on a healthy
spread of 8 features (no single feature above 38% importance) and produces
a defensible, non-suspicious metric.

| Version | Features removed | Precision | Recall | PR-AUC |
|---|---|---|---|---|
| v1 — Leaky | none | 1.00 | 1.00 | 1.00 |
| v2 — Moderate | errorBalanceOrig, orig_balance_drained | 0.88 | 0.94 | 0.98 |
| **v3 — Final (used in API)** | + newbalanceOrig, amount_to_balance_ratio | **0.70** | **0.82** | **0.88** |

## Final model performance (held-out test set, time-based split)

- **Precision:** 70% at operating threshold 0.742
- **Recall:** 82%
- **PR-AUC:** 0.88
- **Fraud value caught:** 98.4% of total fraud value in the test set
- **Legit value unnecessarily flagged:** 0.201% of total legit value

The threshold (0.742) was deliberately chosen from a precision/recall
tradeoff curve, not left at the default 0.5, to balance catching fraud
against creating friction for legitimate transactions.

## Architecture

```
Colab notebook (training)
   → fraud_model_v2.json (XGBoost model, native format)
   → model_features.pkl / model_threshold.pkl
        ↓
   main.py (FastAPI)  →  POST /score endpoint
        ↓
   dashboard.py (Streamlit)  →  batch + manual scoring UI
```

## Features used by the final model

- `step`, `amount`, `oldbalanceOrg`, `oldbalanceDest`, `newbalanceDest`
- `type_TRANSFER` (one-hot encoded transaction type)
- `dest_balance_was_zero` (receiver balance was zero before the transaction)
- `errorBalanceDest` (mismatch between expected and actual receiver balance)

## How to run

### 1. API
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
Visit `http://localhost:8000/docs` for the interactive API tester.

### 2. Dashboard
```bash
streamlit run dashboard.py
```
Visit `http://localhost:8501`.

## What we'd improve with more time

- Test on real (non-synthetic) transaction data to confirm the feature
  importances hold outside PaySim's simulation quirks.
- Add SHAP-based per-prediction explanations instead of rule-based reasons.
- Build a feedback loop where confirmed false positives/negatives retrain
  the model over time.
