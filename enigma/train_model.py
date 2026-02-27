"""
╔══════════════════════════════════════════════════════════════╗
║        NeuroDetect — ML Training Script                     ║
║  Supports real Kaggle EEG dataset + fallback synthetic data  ║
╚══════════════════════════════════════════════════════════════╝

HOW TO GET THE REAL KAGGLE DATASET:
1. Go to: https://www.kaggle.com/datasets/broach/button-tone-sz
   OR:     https://www.kaggle.com/datasets/shashwatwork/eeg-psychiatric-disorders-dataset
2. Download → place the CSV in this folder
3. Update DATASET_PATH below with filename
4. Run: python train_model.py

If no dataset found, script auto-generates realistic synthetic data.
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────
DATASET_PATH = "eeg_dataset.csv"   # ← Change to your Kaggle CSV name
MODEL_OUT    = "model/neurodetect_model.pkl"
METRICS_OUT  = "model/metrics.json"
PLOT_OUT     = "model/training_report.png"
os.makedirs("model", exist_ok=True)

# ── COLORS ───────────────────────────────────────────────────
BG     = "#0A0F2C"
TEAL   = "#00D4B8"
NAVY   = "#0F1A3E"
RED    = "#FF6B6B"
YELLOW = "#FFD166"
MUTED  = "#8899BB"

# ═══════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════

def load_kaggle_dataset(path):
    """Load and normalise a Kaggle EEG CSV."""
    df = pd.read_csv(path)
    print(f"  Loaded '{path}': {df.shape[0]} rows, {df.shape[1]} cols")

    # Auto-detect label column (last column, or named label/class/target/diagnosis)
    label_keywords = ["label", "class", "target", "diagnosis", "schiz", "disorder"]
    label_col = None
    for col in df.columns:
        if any(k in col.lower() for k in label_keywords):
            label_col = col; break
    if label_col is None:
        label_col = df.columns[-1]          # fallback: last column
    print(f"  Label column detected: '{label_col}'")

    # Drop non-numeric cols except label
    feature_cols = [c for c in df.columns if c != label_col
                    and pd.api.types.is_numeric_dtype(df[c])]
    df = df.dropna(subset=feature_cols + [label_col])

    X = df[feature_cols].values
    y = pd.factorize(df[label_col])[0]     # 0 = healthy, 1 = schizophrenia
    return X, y, feature_cols


def generate_synthetic_data():
    """Realistic synthetic EEG features (used as fallback)."""
    print("  ⚠  No Kaggle CSV found — generating realistic synthetic EEG data.")
    print("     Download from Kaggle and set DATASET_PATH to use real data.\n")
    np.random.seed(42)
    feat_names = [
        "Delta_F3","Delta_F4","Delta_Fz","Theta_F3","Theta_F4","Theta_Fz",
        "Alpha_P3","Alpha_P4","Alpha_Pz","Beta_F3","Beta_F4","Beta_Fz",
        "Gamma_F3","Gamma_F4","Gamma_Fz","Coherence_F3F4","Coherence_F3Pz",
        "Coherence_F4Pz","Asymmetry_Frontal","Asymmetry_Parietal"
    ]
    healthy = np.random.randn(300, 20) * 0.8 + [
        1.2,1.1,1.3, 0.9,0.9,0.8, 1.5,1.4,1.6,
        1.0,1.1,1.0, 0.8,0.9,0.8, 0.7,0.6,0.7, 0.1,0.1
    ]
    schizo  = np.random.randn(300, 20) * 0.8 + [
        2.1,2.0,2.2, 1.4,1.3,1.5, 0.7,0.6,0.7,
        1.2,1.3,1.2, 0.3,0.3,0.3, 0.3,0.2,0.3, 0.5,0.4
    ]
    X = np.vstack([healthy, schizo])
    y = np.array([0]*300 + [1]*300)
    return X, y, feat_names


# ── Load ──────────────────────────────────────────────────────
print("=" * 60)
print("   🧠  NeuroDetect — Model Training")
print("=" * 60)

if os.path.exists(DATASET_PATH):
    X, y, feature_names = load_kaggle_dataset(DATASET_PATH)
else:
    X, y, feature_names = generate_synthetic_data()

print(f"\n  Samples : {len(X)}")
print(f"  Features: {len(feature_names)}")
print(f"  Healthy : {(y==0).sum()} | Schizophrenia: {(y==1).sum()}")

# ═══════════════════════════════════════════════════════════════
# STEP 2 — TRAIN / EVALUATE
# ═══════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(
        n_estimators=300, max_depth=12,
        min_samples_split=4, random_state=42, n_jobs=-1
    ))
])
pipe.fit(X_train, y_train)

# Metrics
preds = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1]
acc   = accuracy_score(y_test, preds)
auc   = roc_auc_score(y_test, proba)
cv    = cross_val_score(pipe, X, y, cv=StratifiedKFold(5), scoring="accuracy")
cm    = confusion_matrix(y_test, preds)
fpr, tpr, _ = roc_curve(y_test, proba)
imp   = pipe.named_steps["clf"].feature_importances_

print(f"\n{'='*60}")
print(f"  ✅  Accuracy   : {acc*100:.2f}%")
print(f"  ✅  ROC-AUC    : {auc:.4f}")
print(f"  ✅  CV Mean    : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
print(f"{'='*60}")
print(classification_report(y_test, preds, target_names=["Healthy","Schizophrenia"]))

# ── Save model ────────────────────────────────────────────────
joblib.dump(pipe, MODEL_OUT)
print(f"\n  💾  Model saved → {MODEL_OUT}")

# ── Save metrics JSON (read by Flask API) ─────────────────────
top_feats = sorted(zip(feature_names, imp.tolist()),
                   key=lambda x: x[1], reverse=True)[:10]
metrics = {
    "accuracy": round(acc * 100, 2),
    "auc":      round(auc, 4),
    "cv_mean":  round(cv.mean() * 100, 2),
    "cv_std":   round(cv.std() * 100, 2),
    "confusion_matrix": cm.tolist(),
    "fpr": fpr.tolist(),
    "tpr": tpr.tolist(),
    "top_features": [{"name": n, "importance": round(v, 4)} for n, v in top_feats],
    "feature_names": feature_names,
    "n_samples": len(X),
    "n_healthy": int((y==0).sum()),
    "n_schizo":  int((y==1).sum()),
}
with open(METRICS_OUT, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  📊  Metrics saved → {METRICS_OUT}")

# ═══════════════════════════════════════════════════════════════
# STEP 3 — TRAINING REPORT PLOT
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(BG)
for ax in axes.flat:
    ax.set_facecolor(NAVY)
    for sp in ax.spines.values(): sp.set_color(TEAL + "44")

# 1. Feature Importance
ax = axes[0, 0]
top_n = min(10, len(feature_names))
idx  = np.argsort(imp)[::-1][:top_n]
cols = [TEAL if imp[i] > imp[idx].mean() else "#1C7293" for i in idx[::-1]]
ax.barh(range(top_n), imp[idx[::-1]], color=cols)
ax.set_yticks(range(top_n))
ax.set_yticklabels([feature_names[i] for i in idx[::-1]], fontsize=8, color="white")
ax.set_xlabel("Importance", color=MUTED); ax.tick_params(colors="white")
ax.set_title("Top EEG Biomarkers", color=TEAL, fontsize=12, fontweight="bold")

# 2. Confusion Matrix
ax = axes[0, 1]
im = ax.imshow(cm, cmap="YlOrRd", aspect="auto")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=22, fontweight="bold", color="white" if cm[i,j] < cm.max()/2 else "black")
ax.set_xticks([0,1]); ax.set_xticklabels(["Healthy","Schizophrenia"], color="white")
ax.set_yticks([0,1]); ax.set_yticklabels(["Healthy","Schizophrenia"], color="white", rotation=90, va="center")
ax.set_xlabel("Predicted", color=MUTED); ax.set_ylabel("Actual", color=MUTED)
ax.set_title("Confusion Matrix", color=TEAL, fontsize=12, fontweight="bold")

# 3. ROC Curve
ax = axes[0, 2]
ax.plot(fpr, tpr, color=TEAL, lw=2.5, label=f"AUC = {auc:.3f}")
ax.plot([0,1],[0,1], color=MUTED, lw=1, linestyle="--", label="Random")
ax.fill_between(fpr, tpr, alpha=0.12, color=TEAL)
ax.set_xlabel("False Positive Rate", color=MUTED); ax.set_ylabel("True Positive Rate", color=MUTED)
ax.tick_params(colors="white"); ax.legend(fontsize=10)
ax.set_title("ROC Curve", color=TEAL, fontsize=12, fontweight="bold")

# 4. Risk Score Distribution
ax = axes[1, 0]
risk_h = pipe.predict_proba(X_test[y_test==0])[:,1]*100
risk_s = pipe.predict_proba(X_test[y_test==1])[:,1]*100
ax.hist(risk_h, bins=20, color=TEAL, alpha=0.75, label="Healthy", edgecolor=BG)
ax.hist(risk_s, bins=20, color=RED,  alpha=0.75, label="Schizophrenia", edgecolor=BG)
ax.axvline(50, color=YELLOW, lw=2, linestyle="--", label="Threshold 50%")
ax.set_xlabel("Risk Score (%)", color=MUTED); ax.set_ylabel("Count", color=MUTED)
ax.tick_params(colors="white"); ax.legend(fontsize=9)
ax.set_title("Risk Score Distribution", color=TEAL, fontsize=12, fontweight="bold")

# 5. Cross-Validation
ax = axes[1, 1]
bar_c = [TEAL if s >= cv.mean() else "#1C7293" for s in cv]
ax.bar([f"Fold {i+1}" for i in range(5)], cv*100, color=bar_c, edgecolor=BG)
ax.axhline(cv.mean()*100, color=YELLOW, lw=2, linestyle="--", label=f"Mean {cv.mean()*100:.1f}%")
ax.set_ylim(70, 100); ax.set_ylabel("Accuracy (%)", color=MUTED)
ax.tick_params(colors="white"); ax.legend(fontsize=9)
ax.set_title("5-Fold Cross Validation", color=TEAL, fontsize=12, fontweight="bold")

# 6. Summary
ax = axes[1, 2]; ax.axis("off")
for sp in ax.spines.values(): sp.set_visible(False)
metrics_display = [
    ("ACCURACY",  f"{acc*100:.1f}%", TEAL),
    ("ROC-AUC",   f"{auc:.3f}",      YELLOW),
    ("CV SCORE",  f"{cv.mean()*100:.1f}%", "#02C39A"),
]
for i, (lbl, val, col) in enumerate(metrics_display):
    ax.text(0.5, 0.82-i*0.28, val, ha="center", va="center",
            transform=ax.transAxes, fontsize=34, color=col, fontweight="bold")
    ax.text(0.5, 0.82-i*0.28-0.09, lbl, ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color=MUTED)
ax.set_title("Model Summary", color=TEAL, fontsize=12, fontweight="bold")

plt.suptitle("🧠 NeuroDetect — Training Report", fontsize=16,
             color="white", fontweight="bold", y=0.98)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(PLOT_OUT, dpi=140, bbox_inches="tight", facecolor=BG)
print(f"  📈  Plot saved → {PLOT_OUT}")
print(f"\n✅  Training complete!  Run: python app.py\n")
