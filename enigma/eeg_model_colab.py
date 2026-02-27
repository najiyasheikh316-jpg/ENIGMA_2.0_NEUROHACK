# ============================================================
# 🧠 NeuroDetect — EEG-Based Schizophrenia Detection
# Google Colab Ready | Hackathon 2026
# ============================================================
# STEP 0: Install dependencies (run this cell first in Colab)
# !pip install scikit-learn pandas matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# SECTION 1 — LOAD DATASET
# ─────────────────────────────────────────────────────────
# If you have a real Kaggle EEG dataset, replace with:
#   data = pd.read_csv("your_dataset.csv")
#
# For demo purposes, we generate realistic synthetic EEG features
# representing common extracted features: band power, connectivity,
# coherence values across EEG channels

np.random.seed(42)
n_samples = 500

# Feature groups (realistic EEG-derived features)
feature_names = [
    "Delta_F3", "Delta_F4", "Delta_Fz",
    "Theta_F3", "Theta_F4", "Theta_Fz",
    "Alpha_P3", "Alpha_P4", "Alpha_Pz",
    "Beta_F3",  "Beta_F4",  "Beta_Fz",
    "Gamma_F3", "Gamma_F4", "Gamma_Fz",
    "Coherence_F3_F4", "Coherence_F3_Pz", "Coherence_F4_Pz",
    "Asymmetry_Frontal", "Asymmetry_Parietal"
]

# Simulate healthy controls (label=0)
healthy = np.random.randn(250, 20) * 0.8 + np.array([
    1.2, 1.1, 1.3,   # Delta: normal
    0.9, 0.9, 0.8,   # Theta: normal
    1.5, 1.4, 1.6,   # Alpha: normal
    1.0, 1.1, 1.0,   # Beta: normal
    0.8, 0.9, 0.8,   # Gamma: normal
    0.7, 0.6, 0.7,   # Coherence: normal
    0.1, 0.1          # Asymmetry: low
])

# Simulate schizophrenia patients (label=1)
# Known EEG signatures: reduced alpha, reduced gamma, higher delta, disrupted coherence
schizo = np.random.randn(250, 20) * 0.8 + np.array([
    2.1, 2.0, 2.2,   # Delta: elevated
    1.4, 1.3, 1.5,   # Theta: elevated
    0.7, 0.6, 0.7,   # Alpha: reduced
    1.2, 1.3, 1.2,   # Beta: slightly elevated
    0.3, 0.3, 0.3,   # Gamma: strongly reduced (key biomarker!)
    0.3, 0.2, 0.3,   # Coherence: disrupted
    0.5, 0.4          # Asymmetry: elevated
])

X = np.vstack([healthy, schizo])
y = np.array([0]*250 + [1]*250)

data = pd.DataFrame(X, columns=feature_names)
data["Label"] = y

print("=" * 55)
print("     🧠 NeuroDetect — EEG Schizophrenia Detection")
print("=" * 55)
print(f"\n📊 Dataset shape: {data.shape}")
print(f"   Healthy subjects  : {(y==0).sum()}")
print(f"   Schizophrenia pts : {(y==1).sum()}")
print(f"\n📋 Features: {len(feature_names)} EEG-derived features")
print(f"   (Band power, coherence, asymmetry across channels)\n")

# ─────────────────────────────────────────────────────────
# SECTION 2 — PREPROCESSING
# ─────────────────────────────────────────────────────────
X_features = data[feature_names].values
y_labels = data["Label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)
print(f"✅ Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

# ─────────────────────────────────────────────────────────
# SECTION 3 — TRAIN MODEL
# ─────────────────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────────────────
# SECTION 4 — EVALUATE
# ─────────────────────────────────────────────────────────
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities[:, 1])
cv_scores = cross_val_score(model, X_scaled, y_labels, cv=5)

print("\n" + "=" * 55)
print("              📈 MODEL PERFORMANCE")
print("=" * 55)
print(f"  ✅ Accuracy       : {accuracy*100:.2f}%")
print(f"  ✅ ROC-AUC Score  : {auc:.4f}")
print(f"  ✅ Cross-Val Mean : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print("=" * 55)

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, predictions,
      target_names=["Healthy", "Schizophrenia"]))

# Risk scores for first 5 test subjects
print("\n🔬 Sample Risk Scores (first 5 test subjects):")
print("-" * 45)
for i in range(min(5, len(X_test))):
    risk = probabilities[i][1] * 100
    actual = "Schizophrenia" if y_test[i] == 1 else "Healthy"
    flag = "⚠️ HIGH RISK" if risk > 60 else ("🟡 MODERATE" if risk > 30 else "✅ LOW RISK")
    print(f"  Subject {i+1}: Risk Score = {risk:.1f}%  |  Actual: {actual}  {flag}")

# ─────────────────────────────────────────────────────────
# SECTION 5 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#0A0F2C")

# ── Plot 1: Feature Importance ──
ax1 = fig.add_subplot(2, 3, 1)
importance = model.feature_importances_
indices = np.argsort(importance)[::-1][:10]  # Top 10
colors = ["#00D4B8" if importance[i] > 0.07 else "#1C7293" for i in indices]
bars = ax1.barh(range(10), importance[indices][::-1], color=colors[::-1])
ax1.set_yticks(range(10))
ax1.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=8, color="white")
ax1.set_xlabel("Importance Score", color="#8899BB")
ax1.set_title("🔍 Top 10 EEG Biomarkers", color="#00D4B8", fontsize=12, fontweight="bold")
ax1.set_facecolor("#0F1A3E")
ax1.tick_params(colors="white")
for spine in ax1.spines.values(): spine.set_color("#1C7293")

# ── Plot 2: Confusion Matrix ──
ax2 = fig.add_subplot(2, 3, 2)
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=["Healthy", "Schizophrenia"],
            yticklabels=["Healthy", "Schizophrenia"],
            ax=ax2, linewidths=2, linecolor="#0A0F2C",
            annot_kws={"size": 16, "weight": "bold"})
ax2.set_title("🎯 Confusion Matrix", color="#00D4B8", fontsize=12, fontweight="bold")
ax2.set_xlabel("Predicted", color="#8899BB")
ax2.set_ylabel("Actual", color="#8899BB")
ax2.tick_params(colors="white")

# ── Plot 3: Risk Score Distribution ──
ax3 = fig.add_subplot(2, 3, 3)
risk_healthy = probabilities[y_test == 0, 1] * 100
risk_schizo  = probabilities[y_test == 1, 1] * 100
ax3.hist(risk_healthy, bins=20, alpha=0.75, color="#00D4B8", label="Healthy", edgecolor="#0A0F2C")
ax3.hist(risk_schizo,  bins=20, alpha=0.75, color="#FF6B6B", label="Schizophrenia", edgecolor="#0A0F2C")
ax3.axvline(50, color="#FFD166", linestyle="--", lw=2, label="Decision Threshold (50%)")
ax3.set_xlabel("Risk Score (%)", color="#8899BB")
ax3.set_ylabel("Count", color="#8899BB")
ax3.set_title("📊 Risk Score Distribution", color="#00D4B8", fontsize=12, fontweight="bold")
ax3.legend(fontsize=9)
ax3.set_facecolor("#0F1A3E")
ax3.tick_params(colors="white")
for spine in ax3.spines.values(): spine.set_color("#1C7293")

# ── Plot 4: Cross-Validation ──
ax4 = fig.add_subplot(2, 3, 4)
folds = [f"Fold {i+1}" for i in range(5)]
bar_colors = ["#00D4B8" if s >= cv_scores.mean() else "#1C7293" for s in cv_scores]
ax4.bar(folds, cv_scores * 100, color=bar_colors, edgecolor="#0A0F2C", linewidth=1.5)
ax4.axhline(cv_scores.mean() * 100, color="#FFD166", linestyle="--", lw=2,
            label=f"Mean: {cv_scores.mean()*100:.1f}%")
ax4.set_ylim(70, 100)
ax4.set_ylabel("Accuracy (%)", color="#8899BB")
ax4.set_title("📉 5-Fold Cross-Validation", color="#00D4B8", fontsize=12, fontweight="bold")
ax4.legend(fontsize=9)
ax4.set_facecolor("#0F1A3E")
ax4.tick_params(colors="white")
for spine in ax4.spines.values(): spine.set_color("#1C7293")

# ── Plot 5: EEG Band Comparison ──
ax5 = fig.add_subplot(2, 3, 5)
bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
healthy_means = [
    data[data.Label==0][["Delta_F3","Delta_F4","Delta_Fz"]].values.mean(),
    data[data.Label==0][["Theta_F3","Theta_F4","Theta_Fz"]].values.mean(),
    data[data.Label==0][["Alpha_P3","Alpha_P4","Alpha_Pz"]].values.mean(),
    data[data.Label==0][["Beta_F3","Beta_F4","Beta_Fz"]].values.mean(),
    data[data.Label==0][["Gamma_F3","Gamma_F4","Gamma_Fz"]].values.mean(),
]
schizo_means = [
    data[data.Label==1][["Delta_F3","Delta_F4","Delta_Fz"]].values.mean(),
    data[data.Label==1][["Theta_F3","Theta_F4","Theta_Fz"]].values.mean(),
    data[data.Label==1][["Alpha_P3","Alpha_P4","Alpha_Pz"]].values.mean(),
    data[data.Label==1][["Beta_F3","Beta_F4","Beta_Fz"]].values.mean(),
    data[data.Label==1][["Gamma_F3","Gamma_F4","Gamma_Fz"]].values.mean(),
]
x = np.arange(len(bands))
w = 0.35
ax5.bar(x - w/2, healthy_means, w, label="Healthy", color="#00D4B8", alpha=0.85)
ax5.bar(x + w/2, schizo_means, w, label="Schizophrenia", color="#FF6B6B", alpha=0.85)
ax5.set_xticks(x)
ax5.set_xticklabels(bands, color="white")
ax5.set_ylabel("Mean Power (μV²/Hz)", color="#8899BB")
ax5.set_title("🧠 EEG Band Power Comparison", color="#00D4B8", fontsize=12, fontweight="bold")
ax5.legend(fontsize=9)
ax5.set_facecolor("#0F1A3E")
ax5.tick_params(colors="white")
for spine in ax5.spines.values(): spine.set_color("#1C7293")

# ── Plot 6: Big accuracy callout ──
ax6 = fig.add_subplot(2, 3, 6)
ax6.set_facecolor("#0F1A3E")
ax6.axis("off")
for spine in ax6.spines.values(): spine.set_color("#1C7293")

ax6.text(0.5, 0.85, "MODEL SUMMARY", ha="center", va="center",
         transform=ax6.transAxes, fontsize=11, color="#8899BB",
         fontweight="bold", style="italic")

metrics = [
    ("ACCURACY", f"{accuracy*100:.1f}%", "#00D4B8"),
    ("ROC-AUC",  f"{auc:.3f}",           "#FFD166"),
    ("CV MEAN",  f"{cv_scores.mean()*100:.1f}%", "#02C39A"),
]
for i, (lbl, val, col) in enumerate(metrics):
    ax6.text(0.5, 0.62 - i*0.22, val, ha="center", va="center",
             transform=ax6.transAxes, fontsize=28, color=col, fontweight="bold")
    ax6.text(0.5, 0.62 - i*0.22 - 0.08, lbl, ha="center", va="center",
             transform=ax6.transAxes, fontsize=9, color="#8899BB")

ax6.text(0.5, 0.04, "Phase 2: CNN-LSTM for temporal-spatial learning",
         ha="center", va="center", transform=ax6.transAxes,
         fontsize=8.5, color="#FFD166", style="italic")

plt.suptitle("🧠 NeuroDetect — EEG Schizophrenia Detection Dashboard",
             fontsize=15, color="white", fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("neurodetect_results.png", dpi=150, bbox_inches="tight",
            facecolor="#0A0F2C")
plt.show()
print("\n✅ Dashboard saved as 'neurodetect_results.png'")
print("\n🎯 TAKE A SCREENSHOT OF THIS OUTPUT — it's your working prototype!")
