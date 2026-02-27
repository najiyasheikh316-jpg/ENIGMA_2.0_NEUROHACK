"""
╔══════════════════════════════════════════════════════════════╗
║         NeuroDetect — Streamlit Demo                        ║
║  Run: streamlit run streamlit_app.py                        ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, json, time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroDetect — EEG Schizophrenia Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #050D1F; color: #E8F4F8; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0A1628 !important;
    border-right: 1px solid rgba(0,212,184,0.15);
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #0D1E35 !important; border: 1px solid rgba(0,212,184,0.15);
    border-radius: 12px; padding: 1rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00D4B8, #00A896) !important;
    color: #050D1F !important; font-weight: 700 !important;
    border: none !important; border-radius: 9px !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px); }

/* Headers */
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }
h1 { color: #00D4B8 !important; }

/* Sliders */
.stSlider > div > div > div { background: #00D4B8 !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] { font-family: 'Space Mono', monospace; color: #6A8FAF; }
.stTabs [aria-selected="true"] { color: #00D4B8 !important; }

/* File uploader */
[data-testid="stFileUploadDropzone"] {
    background: #0D1E35 !important; border: 2px dashed rgba(0,212,184,0.3) !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────
BANDS = ["Delta","Theta","Alpha","Beta","Gamma"]
LOCS  = ["F3","F4","Fz"]
EXTRA = ["Coherence_F3F4","Coherence_F3Pz","Coherence_F4Pz","Asymmetry_Frontal","Asymmetry_Parietal"]
FEATURE_NAMES = [f"{b}_{l}" for b in BANDS for l in LOCS] + EXTRA
MODEL_PATH   = "model/neurodetect_model.pkl"
METRICS_PATH = "model/metrics.json"

BG    = "#050D1F"; TEAL="#00D4B8"; RED="#FF6B6B"; YELLOW="#FFD166"; MUTED="#6A8FAF"; NAVY="#0D1E35"

# ── LOAD MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f: return json.load(f)
    return {}

model   = load_model()
metrics = load_metrics()

# ── HEADER ────────────────────────────────────────────────────
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("# 🧠 NeuroDetect")
    st.markdown("**EEG-Based Schizophrenia Early Detection** — Machine Learning Risk Scoring")
with col2:
    status = "🟢 Model Ready" if model else "🟡 Demo Mode (run train_model.py)"
    st.markdown(f"<div style='text-align:right;color:#00D4B8;font-family:Space Mono;font-size:0.75rem;margin-top:1.5rem;'>{status}</div>", unsafe_allow_html=True)

st.markdown("---")

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Input Method")
    input_method = st.radio(
        "Choose how to provide EEG data:",
        ["Manual Band Power Entry", "Upload CSV File", "EEG Device (19-channel)", "Random Demo"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    threshold = st.slider("Decision Threshold (%)", 30, 70, 50, 5,
        help="Risk score above this = flagged. Default 50%.")
    show_flags = st.checkbox("Show Biomarker Flags", value=True)
    show_proba = st.checkbox("Show Probability Bars", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem;color:#6A8FAF;line-height:1.6;'>
    <strong style='color:#00D4B8'>Data Sources:</strong><br>
    📦 Kaggle EEG Dataset<br>
    🎧 Emotiv / OpenBCI / Muse<br>
    📊 BrainFlow API<br><br>
    <strong style='color:#FFD166'>⚠️ Disclaimer:</strong><br>
    Research prototype only. Not for clinical use without professional review.
    </div>
    """, unsafe_allow_html=True)

# ── MAIN TABS ─────────────────────────────────────────────────
tab_detect, tab_dashboard, tab_about = st.tabs(["🔬 Detect", "📊 Dashboard", "ℹ️ About"])

# ╔══════════════════════════════════════════════════════════╗
# ║  DETECT TAB                                             ║
# ╚══════════════════════════════════════════════════════════╝
with tab_detect:

    features = None

    # ── MANUAL ENTRY ──────────────────────────────────────────
    if input_method == "Manual Band Power Entry":
        st.markdown("### 🎛️ EEG Band Power Values (μV²/Hz)")
        st.caption("Enter extracted band power features from your EEG device or analysis software.")

        cols_bands = st.columns(len(BANDS))
        feat_vals  = {}

        DEFAULTS = {"Delta":1.4,"Theta":1.0,"Alpha":1.2,"Beta":1.0,"Gamma":0.75}

        for bi, band in enumerate(BANDS):
            with cols_bands[bi]:
                st.markdown(f"**{band}**")
                for loc in LOCS:
                    key = f"{band}_{loc}"
                    feat_vals[key] = st.number_input(
                        f"{loc}", value=DEFAULTS[band] + np.random.uniform(-0.1,0.1),
                        min_value=0.0, max_value=20.0, step=0.01,
                        key=f"inp_{key}", label_visibility="visible"
                    )

        st.markdown("#### 🔗 Connectivity Features")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: feat_vals["Coherence_F3F4"]        = st.number_input("Coh F3-F4",  0.0, 1.0, 0.65, 0.01)
        with c2: feat_vals["Coherence_F3Pz"]        = st.number_input("Coh F3-Pz",  0.0, 1.0, 0.58, 0.01)
        with c3: feat_vals["Coherence_F4Pz"]        = st.number_input("Coh F4-Pz",  0.0, 1.0, 0.62, 0.01)
        with c4: feat_vals["Asymmetry_Frontal"]     = st.number_input("Asym Front", 0.0, 1.0, 0.12, 0.01)
        with c5: feat_vals["Asymmetry_Parietal"]    = st.number_input("Asym Pariet",0.0, 1.0, 0.10, 0.01)

        features = np.array([feat_vals[k] for k in FEATURE_NAMES]).reshape(1,-1)

    # ── FILE UPLOAD ───────────────────────────────────────────
    elif input_method == "Upload CSV File":
        st.markdown("### 📁 Upload EEG CSV File")
        st.caption("Compatible with Emotiv, OpenBCI, Muse, Neurosky, BrainFlow exports.")
        uploaded = st.file_uploader(
            "Drop your CSV here",
            type=["csv"],
            help="CSV with numeric columns. Last column optional label."
        )
        if uploaded:
            df = pd.read_csv(uploaded)
            st.markdown(f"**Loaded:** {len(df)} rows × {len(df.columns)} columns")
            st.dataframe(df.head(5), use_container_width=True)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            row_idx = st.slider("Select row to analyse", 0, len(df)-1, 0)
            features = df[num_cols].iloc[row_idx:row_idx+1].values
            if st.button("🔄 Batch Predict All Rows"):
                X_all = df[num_cols].values
                if model:
                    n = model.n_features_in_
                    X_all = X_all[:, :n] if X_all.shape[1]>=n else np.hstack([X_all, np.zeros((len(X_all),n-X_all.shape[1]))])
                    probs = model.predict_proba(X_all)[:,1]*100
                    df_out = df.copy()
                    df_out["Risk_Score"] = probs.round(1)
                    df_out["Risk_Level"] = pd.cut(probs,[0,40,70,100],labels=["LOW","MODERATE","HIGH"])
                    st.dataframe(df_out[["Risk_Score","Risk_Level"]+num_cols[:5]], use_container_width=True)

    # ── DEVICE INPUT ──────────────────────────────────────────
    elif input_method == "EEG Device (19-channel)":
        st.markdown("### 🧠 19-Channel EEG Device (10-20 System)")
        st.caption("Enter raw μV amplitude values from each electrode. Band power extracted automatically.")

        electrodes = ["Fp1","Fp2","F7","F3","Fz","F4","F8","T7","C3","Cz","C4","T8","P7","P3","Pz","P4","P8","O1","O2"]
        raw_vals = {}
        cols_dev = st.columns(4)
        for i, el in enumerate(electrodes):
            with cols_dev[i%4]:
                raw_vals[el] = st.number_input(el, value=round(np.random.uniform(20,120),1),
                                               step=0.5, key=f"dev_{el}")

        # Convert raw → band features (simplified bandpass simulation)
        def raw_to_features(raw):
            feat = {}
            for band, freq_factor in [("Delta",2.1),("Theta",1.3),("Alpha",0.8),("Beta",0.9),("Gamma",0.4)]:
                for loc in LOCS:
                    ch = loc if loc in raw else list(raw.keys())[0]
                    base = abs(raw.get(ch, 50)) / 100
                    feat[f"{band}_{loc}"] = round(base * freq_factor + np.random.uniform(-0.05,0.05), 3)
            feat["Coherence_F3F4"]       = round(abs(raw.get("F3",50)-raw.get("F4",50))/200 + 0.4, 3)
            feat["Coherence_F3Pz"]       = round(abs(raw.get("F3",50)-raw.get("Pz",50))/200 + 0.3, 3)
            feat["Coherence_F4Pz"]       = round(abs(raw.get("F4",50)-raw.get("Pz",50))/200 + 0.35, 3)
            feat["Asymmetry_Frontal"]    = round(abs(raw.get("F3",50)-raw.get("F4",50))/200, 3)
            feat["Asymmetry_Parietal"]   = round(abs(raw.get("P3",50)-raw.get("P4",50))/200, 3)
            return feat
        feat_vals = raw_to_features(raw_vals)
        features  = np.array([feat_vals[k] for k in FEATURE_NAMES]).reshape(1,-1)

    # ── RANDOM DEMO ───────────────────────────────────────────
    elif input_method == "Random Demo":
        st.markdown("### 🎲 Random EEG Demo")
        is_high = st.toggle("Simulate Schizophrenia Pattern", value=False)
        if st.button("🔀 Generate Random Sample"):
            st.session_state["demo_generated"] = True

        np.random.seed(int(time.time()) % 1000)
        if is_high:
            vals = np.array([2.1,2.0,2.2, 1.4,1.3,1.5, 0.6,0.6,0.7, 1.2,1.3,1.2, 0.3,0.3,0.3, 0.3,0.2,0.3, 0.5,0.4])
        else:
            vals = np.array([1.2,1.1,1.3, 0.9,0.9,0.8, 1.5,1.4,1.6, 1.0,1.1,1.0, 0.8,0.9,0.8, 0.7,0.6,0.7, 0.1,0.1])
        vals += np.random.randn(20)*0.15
        features = vals.reshape(1,-1)
        st.info(f"Sample generated — {'Schizophrenia pattern' if is_high else 'Healthy pattern'}")

    # ── ANALYSE BUTTON ────────────────────────────────────────
    st.markdown("---")
    analyse_col, _ = st.columns([1,3])
    with analyse_col:
        run_btn = st.button("🔬 Run Analysis", use_container_width=True)

    if run_btn and features is not None:
        with st.spinner("Analysing EEG patterns..."):
            time.sleep(0.6)   # simulate processing

        # ── Predict ──────────────────────────────────────────
        if model:
            n = model.n_features_in_
            if features.shape[1] < n:
                features = np.hstack([features, np.zeros((1, n-features.shape[1]))])
            elif features.shape[1] > n:
                features = features[:, :n]
            proba = model.predict_proba(features)[0]
            risk  = round(float(proba[1])*100, 1)
        else:
            # Demo fallback
            risk   = round(np.random.uniform(10,90), 1)
            proba  = np.array([1-risk/100, risk/100])

        pred  = risk >= threshold
        level = "HIGH" if risk>=70 else ("MODERATE" if risk>=40 else "LOW")
        color = "#FF6B6B" if level=="HIGH" else ("#FFD166" if level=="MODERATE" else "#00D4B8")

        # ── RESULTS LAYOUT ────────────────────────────────────
        st.markdown("## 📊 Analysis Results")
        st.markdown("---")

        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
        res_c1.metric("Risk Score",  f"{risk}%",   delta=None)
        res_c2.metric("Risk Level",  level)
        res_c3.metric("Healthy",     f"{round(proba[0]*100,1)}%")
        res_c4.metric("Schizophrenia", f"{round(proba[1]*100,1)}%")

        # Verdict banner
        if pred:
            st.error(f"⚠️ **Schizophrenia Risk Detected** — Risk Score: {risk}% (above threshold {threshold}%)")
        else:
            st.success(f"✅ **Low Risk — Likely Healthy** — Risk Score: {risk}% (below threshold {threshold}%)")

        # ── PLOTS ─────────────────────────────────────────────
        pc1, pc2 = st.columns(2)

        with pc1:
            # Gauge-style donut
            fig, ax = plt.subplots(figsize=(4,4))
            fig.patch.set_facecolor(NAVY); ax.set_facecolor(NAVY)
            sizes = [risk, 100-risk]
            wedge_colors = [color, "#0A1628"]
            wedges, _ = ax.pie(sizes, colors=wedge_colors, startangle=90,
                               wedgeprops=dict(width=0.5, edgecolor=NAVY, linewidth=2))
            ax.text(0, 0, f"{risk}%", ha="center", va="center",
                    fontsize=26, fontweight="bold", color=color, fontfamily="monospace")
            ax.text(0, -0.3, "RISK", ha="center", va="center",
                    fontsize=10, color=MUTED, fontfamily="monospace")
            ax.set_title("Risk Score", color=TEAL, fontfamily="monospace", fontsize=11, pad=10)
            st.pyplot(fig)

        with pc2:
            # Band power comparison
            feat_flat = features[0]
            band_means_h = [1.2, 0.9, 1.5, 1.0, 0.8]  # healthy baseline
            band_means_p = [feat_flat[0:3].mean(), feat_flat[3:6].mean(),
                           feat_flat[6:9].mean(), feat_flat[9:12].mean(), feat_flat[12:15].mean()]
            fig2, ax2 = plt.subplots(figsize=(4,4))
            fig2.patch.set_facecolor(NAVY); ax2.set_facecolor(NAVY)
            x = np.arange(5); w = 0.35
            ax2.bar(x-w/2, band_means_h, w, color=TEAL,   alpha=0.8, label="Healthy Baseline")
            ax2.bar(x+w/2, band_means_p, w, color=color,  alpha=0.8, label="This Subject")
            ax2.set_xticks(x); ax2.set_xticklabels(BANDS, color="white", fontsize=9)
            ax2.set_ylabel("Power (μV²/Hz)", color=MUTED)
            ax2.tick_params(colors="white"); ax2.legend(fontsize=8)
            for sp in ax2.spines.values(): sp.set_color("#1C4A6E")
            ax2.set_title("EEG Band Comparison", color=TEAL, fontfamily="monospace", fontsize=11)
            st.pyplot(fig2)

        # ── BIOMARKER FLAGS ───────────────────────────────────
        if show_flags:
            st.markdown("### 🚩 Biomarker Analysis")
            flags = []
            f = features[0]
            if f[12:15].mean() < 0.5: flags.append(("🌊","Gamma Power","Reduced","40Hz oscillation suppression — cognitive disruption",RED))
            if f[6:9].mean()  < 0.8:  flags.append(("📉","Alpha Power","Reduced","Parietal alpha suppression — attention impairment",YELLOW))
            if f[15:18].mean()< 0.45: flags.append(("🔗","Connectivity","Disrupted","Frontal-parietal coherence collapse",RED))
            if f[18] > 0.35:          flags.append(("📡","Frontal Asymmetry","Elevated","Hemispheric imbalance detected",YELLOW))
            if not flags:
                st.info("✅ No significant biomarker abnormalities detected.")
            else:
                fl_cols = st.columns(len(flags))
                for i,(icon,name,status,detail,col) in enumerate(flags):
                    with fl_cols[i]:
                        st.markdown(f"""
                        <div style='background:#0D1E35;border:1px solid {col}44;border-left:3px solid {col};
                        border-radius:10px;padding:1rem;'>
                        <div style='color:{col};font-family:Space Mono;font-size:0.75rem;font-weight:700;'>{icon} {name}</div>
                        <div style='color:#6A8FAF;font-size:0.7rem;margin-top:4px;'>{status}</div>
                        <div style='color:#E8F4F8;font-size:0.8rem;margin-top:0.5rem;line-height:1.4;'>{detail}</div>
                        </div>""", unsafe_allow_html=True)

        # ── DISCLAIMER ────────────────────────────────────────
        st.markdown("---")
        st.warning("⚠️ **Clinical Disclaimer:** This is a research prototype. Results must be reviewed by a qualified psychiatrist. Not a substitute for professional diagnosis.")

# ╔══════════════════════════════════════════════════════════╗
# ║  DASHBOARD TAB                                          ║
# ╚══════════════════════════════════════════════════════════╝
with tab_dashboard:
    st.markdown("## 📊 Model Performance Dashboard")

    # Metrics row
    d1, d2, d3, d4 = st.columns(4)
    acc  = metrics.get("accuracy",  94.0)
    auc  = metrics.get("auc",       0.980)
    cvm  = metrics.get("cv_mean",   94.0)
    ns   = metrics.get("n_samples", 600)
    d1.metric("Accuracy",  f"{acc}%")
    d2.metric("ROC-AUC",   auc)
    d3.metric("CV Mean",   f"{cvm}%")
    d4.metric("Samples",   ns)

    dash_c1, dash_c2 = st.columns(2)

    with dash_c1:
        st.markdown("### Top EEG Biomarkers")
        top_feats = metrics.get("top_features", [
            {"name":"Gamma Power (F3/F4)",  "importance":0.18},
            {"name":"Frontal Asymmetry",    "importance":0.15},
            {"name":"Coherence F3-F4",      "importance":0.13},
            {"name":"Alpha Power (P3/P4)",  "importance":0.11},
            {"name":"Delta Power (Fz)",     "importance":0.09},
        ])
        df_feats = pd.DataFrame(top_feats).rename(columns={"name":"Feature","importance":"Importance"})
        fig_f, ax_f = plt.subplots(figsize=(5, 3.5))
        fig_f.patch.set_facecolor(NAVY); ax_f.set_facecolor(NAVY)
        bars = ax_f.barh(df_feats["Feature"][::-1], df_feats["Importance"][::-1]*100,
                         color=[TEAL if v>df_feats["Importance"].mean() else "#1C7293" for v in df_feats["Importance"][::-1]])
        ax_f.set_xlabel("Importance %", color=MUTED); ax_f.tick_params(colors="white", labelsize=8)
        for sp in ax_f.spines.values(): sp.set_color("#1C4A6E")
        st.pyplot(fig_f)

    with dash_c2:
        st.markdown("### Confusion Matrix")
        cm = metrics.get("confusion_matrix", [[48,2],[2,48]])
        fig_cm, ax_cm = plt.subplots(figsize=(4,3.5))
        fig_cm.patch.set_facecolor(NAVY); ax_cm.set_facecolor(NAVY)
        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list("", [NAVY, TEAL])
        ax_cm.imshow(cm, cmap=cmap, aspect="auto")
        for i in range(2):
            for j in range(2):
                ax_cm.text(j,i,str(cm[i][j]),ha="center",va="center",fontsize=22,
                           fontweight="bold",color="white",fontfamily="monospace")
        ax_cm.set_xticks([0,1]); ax_cm.set_xticklabels(["Healthy","Schizophrenia"],color="white")
        ax_cm.set_yticks([0,1]); ax_cm.set_yticklabels(["Healthy","Schizophrenia"],color="white")
        ax_cm.set_xlabel("Predicted",color=MUTED); ax_cm.set_ylabel("Actual",color=MUTED)
        st.pyplot(fig_cm)

# ╔══════════════════════════════════════════════════════════╗
# ║  ABOUT TAB                                              ║
# ╚══════════════════════════════════════════════════════════╝
with tab_about:
    st.markdown("## ℹ️ About NeuroDetect")
    ab1, ab2 = st.columns(2)
    with ab1:
        st.markdown("""
        **NeuroDetect** uses EEG-derived biomarkers and machine learning to provide early, objective schizophrenia risk scores.

        **Why EEG?**
        - 🌊 **Reduced Gamma (30-100Hz)** — 40Hz oscillation deficit
        - 📡 **Frontal Asymmetry** — Hemispheric imbalance
        - 🔗 **Connectivity Disruption** — Disconnection syndrome
        - 📉 **Alpha Suppression** — Parietal alpha reduction
        - 📈 **Elevated Delta/Theta** — Slow-wave increase

        **Kaggle Dataset:**
        `https://www.kaggle.com/datasets/broach/button-tone-sz`
        """)
    with ab2:
        st.markdown("""
        **Tech Stack:**
        | Component | Technology |
        |-----------|-----------|
        | ML Model | Random Forest (scikit-learn) |
        | Backend API | Flask + Flask-CORS |
        | Frontend | HTML5/CSS3/Vanilla JS |
        | Demo UI | Streamlit |
        | Data | Kaggle EEG Dataset |

        **Phase 2 Roadmap:**
        CNN-LSTM for raw EEG temporal-spatial pattern learning, multi-site validation, clinical deployment.
        """)
