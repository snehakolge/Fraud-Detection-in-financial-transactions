"""
train_model.py
--------------
Run this ONCE to train the model and export:
  - fraud_model.pkl
  - fraud_scaler.pkl
  - fraud_le.pkl

Usage:
    python train_model.py --data PS_20174392719_1491204439457_log.csv
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['amount_ratio']           = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['OriginalBalanceError']   = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
    df['OriginalBalanceWError']  = df['oldbalanceOrg'] - df['amount'] - df['newbalanceOrig']
    df['DestBalanceError']       = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['IsFullTransfer']         = (df['newbalanceOrig'] == 0).astype(int)
    return df


def train(data_path: str, output_dir: str = "."):
    print(f"[1/6] Loading data from {data_path} ...")
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    print(f"      Shape: {df.shape}")

    print("[2/6] Feature engineering ...")
    df = feature_engineer(df)
    df = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])

    print("[3/6] Label-encoding 'type' ...")
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type'])

    X = df.drop('isFraud', axis=1)
    y = df['isFraud']
    feature_names = list(X.columns)

    print("[4/6] Scaling features ...")
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    print("[5/6] Training RandomForestClassifier ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    rfc = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rfc.fit(X_train, y_train)

    y_pred  = rfc.predict(X_test)
    y_score = rfc.predict_proba(X_test)[:, 1]
    print("\n--- Evaluation ---")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_score))

    print(f"\n[6/6] Saving artefacts to {output_dir}/ ...")

    with open(f"{output_dir}/fraud_model.pkl", "wb") as f:
        pickle.dump(rfc, f)  # ✅ ONLY MODEL

    with open(f"{output_dir}/fraud_scaler.pkl", "wb") as f:
        pickle.dump(sc, f)

    with open(f"{output_dir}/fraud_le.pkl", "wb") as f:
        pickle.dump(le, f)

    print("Done!  fraud_model.pkl  |  fraud_scaler.pkl  |  fraud_le.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True, help="Path to CSV dataset")
    parser.add_argument("--outdir", default=".",  help="Where to write .pkl files")
    args = parser.parse_args()
    train(args.data, args.outdir)
