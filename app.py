"""
Bank GoodCredit — Credit Risk Scorecard Flask Application
REST API + Interactive Frontend
"""

from flask import Flask, request, jsonify, render_template
import joblib
import json
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

BASE      = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "model")

# ── Load artifacts ────────────────────────────────────────────────────────────
model          = joblib.load(f"{MODEL_DIR}/model.pkl")
imputer        = joblib.load(f"{MODEL_DIR}/imputer.pkl")
label_encoders = joblib.load(f"{MODEL_DIR}/label_encoders.pkl")
with open(f"{MODEL_DIR}/metadata.json") as f:
    META = json.load(f)

FEATURE_NAMES = META["feature_names"]
METRICS       = META["metrics"]

# Features the UI exposes directly (subset; rest filled with median defaults)
UI_FEATURES = [
    # Account-derived
    "num_accounts", "max_dpd_max", "cnt_30_59_sum", "cnt_60_89_sum",
    "cnt_90_plus_sum", "is_30_plus_ever_sum", "ratio_currbalance_creditlimit",
    "cur_balance_amt_sum", "creditlimit_sum", "amt_past_due_sum",
    "payment_history_length_mean",
    # Enquiry-derived
    "num_enquiries", "count_enquiry_recency_90", "count_enquiry_recency_365",
    "total_enq_amt", "n_unique_purpose",
    # Demographics (most predictive ones)
    "feature_7", "feature_3", "feature_21", "feature_20",
    "feature_1", "feature_2", "feature_4", "feature_5",
]


def build_feature_vector(data: dict) -> np.ndarray:
    """Convert a UI payload into the full feature vector the model expects."""
    row = {}
    # Fill all features with 0 as default
    for f in FEATURE_NAMES:
        row[f] = 0.0

    # Override with values from request
    for key, val in data.items():
        if key in FEATURE_NAMES:
            try:
                row[key] = float(val)
            except (TypeError, ValueError):
                row[key] = 0.0

    X = pd.DataFrame([row], columns=FEATURE_NAMES)
    X_imp = pd.DataFrame(imputer.transform(X), columns=FEATURE_NAMES)
    return X_imp


def risk_label(prob: float) -> dict:
    """Convert raw probability to risk tier."""
    if prob < 0.02:
        return {"tier": "Low",      "color": "#22c55e", "icon": "✅"}
    elif prob < 0.06:
        return {"tier": "Medium",   "color": "#f59e0b", "icon": "⚠️"}
    elif prob < 0.12:
        return {"tier": "High",     "color": "#f97316", "icon": "🔶"}
    else:
        return {"tier": "Very High","color": "#ef4444", "icon": "🚨"}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", metrics=METRICS, meta=META)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Body: JSON with customer features
    Returns: { prob_bad, risk_tier, gini_score, ... }
    """
    try:
        data = request.get_json(force=True)
        X    = build_feature_vector(data)
        prob = float(model.predict_proba(X)[0][1])
        risk = risk_label(prob)
        score = int((1 - prob) * 1000)   # credit score: higher = better

        return jsonify({
            "success":       True,
            "prob_bad":      round(prob, 6),
            "prob_good":     round(1 - prob, 6),
            "bad_pct":       round(prob * 100, 2),
            "credit_score":  score,
            "risk_tier":     risk["tier"],
            "risk_color":    risk["color"],
            "risk_icon":     risk["icon"],
            "model_gini":    METRICS["gini"],
            "benchmark_gini":METRICS["benchmark_gini"],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/metrics", methods=["GET"])
def metrics():
    return jsonify(META)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": META["best_model_name"]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
