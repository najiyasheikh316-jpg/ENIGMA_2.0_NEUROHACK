"""
╔══════════════════════════════════════════════════════════════╗
║         NeuroDetect — Flask REST API Backend                ║
║  Serves predictions to the frontend UI                      ║
╚══════════════════════════════════════════════════════════════╝
Run:  python app.py
API:  http://localhost:5000
"""

import os, json, time
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

MODEL_PATH   = "model/neurodetect_model.pkl"
METRICS_PATH = "model/metrics.json"

# ── Load model & metrics at startup ──────────────────────────
model   = None
metrics = {}

def load_assets():
    global model, metrics
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"  ✅  Model loaded: {MODEL_PATH}")
    else:
        print("  ⚠   Model not found — run train_model.py first")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        print(f"  ✅  Metrics loaded: {METRICS_PATH}")

load_assets()

# ── Helpers ───────────────────────────────────────────────────
def risk_label(score):
    if score >= 70: return "HIGH",     "#FF6B6B"
    if score >= 40: return "MODERATE", "#FFD166"
    return "LOW", "#00D4B8"

def interpret_features(feature_vec, feature_names):
    """Return top 3 abnormal biomarker flags."""
    flags = []
    name_map = {n: v for n, v in zip(feature_names, feature_vec)}
    gamma_keys = [k for k in name_map if "Gamma" in k]
    alpha_keys = [k for k in name_map if "Alpha" in k]
    coh_keys   = [k for k in name_map if "Coher" in k]
    asym_keys  = [k for k in name_map if "Asym" in k]

    if gamma_keys and np.mean([name_map[k] for k in gamma_keys]) < 0.5:
        flags.append({"marker":"Gamma Power","status":"Reduced","detail":"40 Hz oscillations suppressed — cognitive disruption marker"})
    if alpha_keys and np.mean([name_map[k] for k in alpha_keys]) < 0.8:
        flags.append({"marker":"Alpha Power","status":"Reduced","detail":"Parietal alpha suppression — attention/awareness impairment"})
    if coh_keys and np.mean([name_map[k] for k in coh_keys]) < 0.4:
        flags.append({"marker":"Connectivity","status":"Disrupted","detail":"Frontal-parietal coherence collapse — disconnection syndrome"})
    if asym_keys and np.mean([name_map[k] for k in asym_keys]) > 0.35:
        flags.append({"marker":"Frontal Asymmetry","status":"Elevated","detail":"Hemispheric imbalance detected in frontal lobe"})
    return flags[:3]

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/api/metrics")
def get_metrics():
    """Return training metrics for the dashboard."""
    return jsonify(metrics)

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accept EEG feature values and return risk score.
    Body: { "features": [f1, f2, ..., fn] }
          OR { "eeg_channels": { "delta": [...], "theta": [...], ... } }
          OR multipart file upload (CSV row)
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(silent=True)
    feat_names = metrics.get("feature_names", [f"f{i}" for i in range(20)])

    # ── Mode 1: raw feature array ─────────────────────────────
    if data and "features" in data:
        features = np.array(data["features"], dtype=float).reshape(1, -1)

    # ── Mode 2: EEG channel dict (from device / Streamlit) ───
    elif data and "eeg_channels" in data:
        ch = data["eeg_channels"]
        # Flatten band → channel values in order matching training features
        features = []
        for band in ["delta","theta","alpha","beta","gamma"]:
            for loc in ["F3","F4","Fz"]:
                key = f"{band}_{loc}"
                features.append(ch.get(key, ch.get(band, {}).get(loc, np.random.randn())))
        # Add coherence + asymmetry (4 extra)
        for k in ["coh_F3F4","coh_F3Pz","coh_F4Pz"]:
            features.append(ch.get(k, np.random.randn() * 0.5 + 0.5))
        features.append(ch.get("asymmetry_frontal", np.random.randn() * 0.2))
        features = np.array(features).reshape(1, -1)

    # ── Mode 3: CSV file upload ───────────────────────────────
    elif request.files.get("file"):
        f   = request.files["file"]
        df  = pd.read_csv(f)
        num = df.select_dtypes(include=[np.number])
        features = num.iloc[0:1].values

    else:
        return jsonify({"error": "Send JSON with 'features' array or 'eeg_channels' dict"}), 400

    # ── Predict ───────────────────────────────────────────────
    try:
        n_expected = model.n_features_in_
        if features.shape[1] != n_expected:
            # Pad or truncate to match model input
            if features.shape[1] < n_expected:
                pad = np.zeros((1, n_expected - features.shape[1]))
                features = np.hstack([features, pad])
            else:
                features = features[:, :n_expected]

        prob       = model.predict_proba(features)[0]
        risk_score = round(float(prob[1]) * 100, 1)
        prediction = int(prob[1] >= 0.5)
        label, color = risk_label(risk_score)
        flags = interpret_features(features[0].tolist(), feat_names)

        return jsonify({
            "risk_score":   risk_score,
            "prediction":   prediction,
            "label":        "Schizophrenia Risk Detected" if prediction else "Low Risk — Likely Healthy",
            "risk_level":   label,
            "risk_color":   color,
            "confidence":   round(float(max(prob)) * 100, 1),
            "probabilities":{"healthy": round(float(prob[0])*100,1),
                             "schizophrenia": round(float(prob[1])*100,1)},
            "biomarker_flags": flags,
            "disclaimer":   "Clinical support tool only. Not a substitute for professional diagnosis.",
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/csv", methods=["POST"])
def predict_csv():
    """Batch prediction from uploaded CSV file."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f   = request.files["file"]
    df  = pd.read_csv(f)
    num = df.select_dtypes(include=[np.number])
    X   = num.values
    n   = model.n_features_in_
    if X.shape[1] < n:  X = np.hstack([X, np.zeros((len(X), n-X.shape[1]))])
    else:               X = X[:, :n]

    probs  = model.predict_proba(X)[:, 1]
    preds  = (probs >= 0.5).astype(int)
    results = []
    for i, (p, pr) in enumerate(zip(preds, probs)):
        lbl, col = risk_label(pr*100)
        results.append({
            "row":        i + 1,
            "risk_score": round(float(pr)*100, 1),
            "risk_level": lbl,
            "prediction": int(p),
            "risk_color": col,
        })
    return jsonify({"total": len(results), "results": results})


@app.route("/api/sample")
def sample_input():
    """Return a sample input for testing the predict endpoint."""
    feat_names = metrics.get("feature_names", [])
    n = model.n_features_in_ if model else 20
    sample = (np.random.randn(n) * 0.8 + 0.5).tolist()
    return jsonify({
        "features":      sample,
        "feature_names": feat_names,
        "note": "POST this to /api/predict"
    })


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🧠  NeuroDetect API  →  http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
