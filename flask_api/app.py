from flask import Flask, request, jsonify
import sys
import os

# allow import from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from predictor import predict, TRANSACTION_TYPES

app = Flask(__name__)

# ── HEALTH ──────────────────────────────
@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "RandomForestClassifier"
    })

# ── TRANSACTION TYPES ───────────────────
@app.get("/transaction-types")
def types():
    return jsonify({
        "types": TRANSACTION_TYPES
    })

# ── VALIDATION ──────────────────────────
def validate(data):
    required = [
        "step", "type", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest"
    ]

    for field in required:
        if field not in data:
            return f"Missing field: {field}"

    return None

# ── PREDICT ─────────────────────────────
@app.post("/predict")
def predict_single():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No input provided"}), 400

    err = validate(data)
    if err:
        return jsonify({"error": err}), 422

    result = predict(data)

    if "error" in result:
        return jsonify(result), 400

    return jsonify({
        "prediction": result["label"],
        "probability": result["probability"]
    })

# ── MAIN ────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)