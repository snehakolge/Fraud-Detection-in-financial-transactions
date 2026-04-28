import os
import pickle
import pandas as pd

# ── PATHS ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH   = os.path.join(BASE_DIR, "fraud_model.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "fraud_scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "fraud_le.pkl")

# ── LOAD MODEL (DICT) ───────────────────
with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

# 🔥 SAFE LOAD (handles both cases)
if isinstance(artifact, dict):
    model = artifact["model"]
    FEATURE_NAMES = artifact["feature_names"]
else:
    # fallback (just in case)
    model = artifact
    FEATURE_NAMES = [
        'step', 'type', 'amount',
        'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest',
        'amount_ratio',
        'OriginalBalanceError',
        'OriginalBalanceWError',
        'DestBalanceError',
        'IsFullTransfer'
    ]

# ── LOAD SCALER & ENCODER ───────────────
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

TRANSACTION_TYPES = list(encoder.classes_)

# ── PREPROCESS ──────────────────────────
def preprocess(data: dict):
    df = pd.DataFrame([data])

    # feature engineering
    df['amount_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['OriginalBalanceError'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
    df['OriginalBalanceWError'] = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
    df['DestBalanceError'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['IsFullTransfer'] = (df['newbalanceOrig'] == 0).astype(int)

    # encode
    df['type'] = encoder.transform(df['type'])

    # enforce exact columns
    df = df[FEATURE_NAMES]

    return scaler.transform(df)

# ── PREDICT ─────────────────────────────
def predict(data: dict):
    try:
        processed = preprocess(data)

        pred = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0][1]

        return {
            "label": "Fraud" if pred == 1 else "Not Fraud",
            "probability": float(prob)
        }

    except Exception as e:
        return {"error": str(e)}