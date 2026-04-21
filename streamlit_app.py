"""
============================================================
KOA PAIN LEVEL PREDICTOR — STREAMLIT APP
Pure inference only. Loads pre-trained koa_model.joblib
from GitHub. No training occurs here.
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import urllib.request
import tempfile
import os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="KOA Pain Level Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
    padding: 2rem; border-radius: 12px; color: white; margin-bottom: 2rem;
}
.metric-card {
    background: #f0f4f8; padding: .65rem 1rem; border-radius: 8px;
    border-left: 4px solid #2d6a9f; margin: .35rem 0; font-size: .88rem;
}
.section-hdr {
    color: #1e3a5f; font-size: 1.05rem; font-weight: 600;
    border-bottom: 2px solid #2d6a9f; padding-bottom: .3rem;
    margin-top: 1rem; margin-bottom: .75rem;
}
.result-box {
    border-radius: 12px; padding: 2rem; text-align: center;
    border-width: 2px; border-style: solid;
}
.disclaimer {
    background: #fff3cd; border: 1px solid #ffc107;
    border-radius: 8px; padding: 1rem; margin-top: 1rem; font-size: .85rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

PAIN_LABELS = {1: "No Pain", 2: "Mild Pain", 3: "Moderate Pain", 4: "Severe Pain"}
PAIN_COLORS = {1: "#27ae60", 2: "#f39c12", 3: "#e67e22", 4: "#e74c3c"}
PAIN_EMOJI  = {1: "✅", 2: "🟡", 3: "🟠", 4: "🔴"}
PAIN_BG     = {1: "#d4efdf", 2: "#fef9e7", 3: "#fdebd0", 4: "#fadbd8"}

CHANNELS   = ['Fp1', 'Fp2', 'Fz', 'F7', 'F8', 'T7', 'T8', 'Pz']
BANDS      = ['Alpha', 'Beta', 'Theta', 'Delta', 'Gamma']
CONDITIONS = ['Pre', 'Post', 'Standing', 'Sitting']

MODEL_URL = (
    "https://raw.githubusercontent.com/focuzdev/"
    "Pain-Level-Prediction/master/koa_model.joblib"
)

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (must match train_and_save.py exactly)
# ─────────────────────────────────────────────────────────────

def engineer_features(src_dict: dict) -> pd.DataFrame:
    src = pd.DataFrame([src_dict])
    feats = {}

    def safe_log(arr):
        arr = np.array(arr, dtype=float)
        return np.log1p(np.where(arr < 0, 0, arr))

    def col(name):
        return src[name].values if name in src.columns else np.zeros(len(src))

    for cond in CONDITIONS:
        v1 = col(f'Fp1_Alpha_{cond}')
        v2 = col(f'Fp2_Alpha_{cond}')
        feats[f'FAA_{cond}'] = (v1 - v2) / (v1 + v2 + 1e-6)

    for cond in CONDITIONS:
        feats[f'TBR_Fz_{cond}'] = safe_log(
            col(f'Fz_Theta_{cond}') / (col(f'Fz_Beta_{cond}') + 1e-6))

    for cond in CONDITIONS:
        vals = np.stack([col(f'{ch}_Alpha_{cond}') for ch in CHANNELS], axis=1)
        feats[f'Global_Alpha_{cond}'] = safe_log(vals.mean(axis=1))

    for cond in CONDITIONS:
        vals = np.stack([col(f'{ch}_Theta_{cond}') for ch in CHANNELS], axis=1)
        feats[f'Global_Theta_{cond}'] = safe_log(vals.mean(axis=1))

    for cond in CONDITIONS:
        vals = np.stack([col(f'T7_Delta_{cond}'), col(f'T8_Delta_{cond}')], axis=1)
        feats[f'Temporal_Delta_{cond}'] = safe_log(vals.mean(axis=1))

    for cond in CONDITIONS:
        feats[f'Pz_Alpha_{cond}'] = safe_log(col(f'Pz_Alpha_{cond}'))

    for cond in ['Pre', 'Post']:
        vals = np.stack([col(f'Fp1_Gamma_{cond}'), col(f'Fp2_Gamma_{cond}')], axis=1)
        feats[f'Frontal_Gamma_{cond}'] = safe_log(vals.mean(axis=1))

    for cond in ['Pre', 'Post']:
        d = np.stack([col(f'{ch}_Delta_{cond}') for ch in CHANNELS], axis=1).mean(axis=1)
        a = np.stack([col(f'{ch}_Alpha_{cond}') for ch in CHANNELS], axis=1).mean(axis=1)
        feats[f'Delta_Alpha_Ratio_{cond}'] = safe_log(d / (a + 1e-6))

    feats['Alpha_Change_PrePost']  = (np.array(feats['Global_Alpha_Post'])
                                      - np.array(feats['Global_Alpha_Pre']))
    feats['Alpha_Change_StandSit'] = (np.array(feats['Global_Alpha_Standing'])
                                      - np.array(feats['Global_Alpha_Sitting']))

    for c in ['KL_grade', 'McGill', 'Age', 'BMI', 'Duration_of_symptom',
              'PRI', 'WMI', 'PSI', 'QoL']:
        feats[c] = col(c)

    feats['Cognitive_Composite'] = (col('PRI') + col('WMI') + col('PSI')) / 3.0

    return pd.DataFrame(feats)

# ─────────────────────────────────────────────────────────────
# LOAD BUNDLE
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_bundle():
    tmp = tempfile.mktemp(suffix='.joblib')
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        bundle = joblib.load(tmp)
        # Validate expected keys exist
        required = ['scaler', 'model', 'feature_names', 'model_name',
                    'loo_accuracy', 'loo_f1_macro', 'all_model_results']
        missing = [k for k in required if k not in bundle]
        if missing:
            return None, (
                f"Model file is outdated — missing keys: {missing}.\n\n"
                f"Please re-run `train_and_save.py` (latest version) in Colab "
                f"and re-upload `koa_model.joblib` to GitHub."
            )
        return bundle, None
    except Exception as e:
        return None, str(e)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

def main():

    st.markdown("""
    <div class="main-header">
        <h1 style="margin:0;font-size:2rem;">🧠 KOA Pain Level Predictor</h1>
        <p style="margin:.5rem 0 0;opacity:.9;font-size:1rem;">
            EEG + clinical machine learning — Knee Osteoarthritis pain classification
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ───────────────────────────────────────────
    with st.spinner("Loading pre-trained model..."):
        bundle, err = load_bundle()

    if err:
        st.error("❌ Could not load model.")
        st.markdown(f"**Reason:** {err}")
        st.markdown("""
        **Fix — follow these steps:**
        1. Open `train_and_save.py` in Google Colab (use the **latest version**)
        2. Run `!pip install imbalanced-learn` then run the full script
        3. Download `koa_model.joblib` from Google Drive → `MyDrive/KOA_PainPredictor/`
        4. In your GitHub repo → **delete** the old `koa_model.joblib`
        5. Upload the new `koa_model.joblib` → Commit
        6. Reload this app
        """)
        st.stop()

    st.success(f"✅ Model loaded: **{bundle['model_name']}** — "
               f"trained on {bundle['n_samples']} patients, LOO-CV validated")

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📊 Model Performance (LOO-CV)")
        st.markdown(f"""
        <div class="metric-card"><b>Deployed model:</b> {bundle['model_name']}</div>
        <div class="metric-card"><b>Accuracy:</b> {bundle['loo_accuracy']*100:.1f}%</div>
        <div class="metric-card"><b>Balanced Accuracy:</b> {bundle['loo_balanced_accuracy']*100:.1f}%</div>
        <div class="metric-card"><b>F1 Macro:</b> {bundle['loo_f1_macro']*100:.1f}%</div>
        <div class="metric-card"><b>F1 Weighted:</b> {bundle['loo_f1_weighted']*100:.1f}%</div>
        <div class="metric-card"><b>Cohen's Kappa:</b> {bundle['loo_kappa']:.3f}</div>
        <div class="metric-card"><b>Training N:</b> {bundle['n_samples']} patients</div>
        <div class="metric-card"><b>Features:</b> {bundle['n_features']}</div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📈 All Models Compared")
        for mname, mres in bundle['all_model_results'].items():
            deployed = "🟢 " if mname == bundle['model_name'] else ""
            st.markdown(f"**{deployed}{mname}**")
            st.progress(float(mres['f1_macro']))
            st.caption(
                f"F1: {mres['f1_macro']*100:.1f}%  "
                f"Acc: {mres['accuracy']*100:.1f}%  "
                f"κ: {mres['kappa']:.3f}"
            )

        feat_imp = bundle.get('feature_importances', {})
        if feat_imp:
            st.markdown("---")
            st.markdown("### 🏆 Top 5 Features")
            for i, (fname, fimp) in enumerate(list(feat_imp.items())[:5]):
                bar = "█" * max(1, int(fimp * 80))
                st.caption(f"**{i+1}.** `{fname[:28]}`\n{bar} {fimp:.3f}")

        st.markdown("---")
        st.markdown("### 🗂 Training Class Distribution")
        dist = bundle.get('class_dist', {})
        for k in sorted(dist):
            pct = dist[k] / bundle['n_samples'] * 100
            st.caption(f"Class {k} — {PAIN_LABELS[k]}: n={dist[k]} ({pct:.1f}%)")

        st.markdown("---")
        st.caption(
            "Model trained offline on N=62 KOA patients. "
            "Loaded from GitHub — fixed weights, reproducible predictions."
        )

    # ── Input tabs ───────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🩺 Clinical Features",
        "🧠 EEG Features",
        "📂 Upload CSV Row",
        "🎯 Predict"
    ])
    inputs = {}

    # TAB 1 — CLINICAL
    with tab1:
        st.markdown('<p class="section-hdr">Clinical & Demographic Information</p>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            inputs['KL_grade'] = st.selectbox(
                "KL Grade (radiological severity)",
                options=[0, 1, 2, 3, 4],
                format_func=lambda x: {
                    0: "0 — Normal",  1: "1 — Doubtful",
                    2: "2 — Minimal", 3: "3 — Moderate",
                    4: "4 — Severe"
                }[x],
                index=2
            )
            inputs['McGill'] = st.slider("McGill Pain Score (0–5)", 0, 5, 2)
            inputs['QoL']    = st.slider("Quality of Life (0–10)", 0, 10, 5)
        with c2:
            inputs['Age'] = st.number_input("Age (years)", 20, 90, 55, step=1)
            inputs['BMI'] = st.number_input("BMI (kg/m²)", 15.0, 50.0, 27.0,
                                             step=0.1, format="%.1f")
            inputs['Duration_of_symptom'] = st.slider("Symptom Duration (years)", 0, 15, 3)
        with c3:
            inputs['PRI'] = st.number_input("PRI — Perceptual Reasoning", 0, 200, 90)
            inputs['WMI'] = st.number_input("WMI — Working Memory",        0, 200, 95)
            inputs['PSI'] = st.number_input("PSI — Processing Speed",      0, 200, 85)
        st.info("💡 **Strongest predictors:** KL Grade & McGill Score (p<0.001), "
                "Symptom Duration (p<0.001), BMI (p=0.002)")

    # TAB 2 — EEG
    with tab2:
        st.markdown('<p class="section-hdr">EEG Band Power Values (μV²)</p>',
                    unsafe_allow_html=True)
        st.caption(
            "Enter raw power spectral density values per channel × band × condition. "
            "If only Pre-task recordings are available, use the same values for all conditions."
        )
        for ch in CHANNELS:
            with st.expander(f"📡  Channel: **{ch}**"):
                for band in BANDS:
                    st.markdown(f"<small><b>{band}</b></small>", unsafe_allow_html=True)
                    cols = st.columns(4)
                    for i, cond in enumerate(CONDITIONS):
                        key = f'{ch}_{band}_{cond}'
                        with cols[i]:
                            inputs[key] = st.number_input(
                                cond, min_value=0.0, value=1000.0,
                                step=100.0, key=key, format="%.1f"
                            )
        st.info("🔬 Key biomarkers: Frontal Alpha Asymmetry (Fp1 vs Fp2), "
                "Theta/Beta Ratio (Fz), Temporal Delta power (T7+T8).")

    # TAB 3 — CSV UPLOAD
    with tab3:
        st.markdown('<p class="section-hdr">Upload a Patient Row (CSV)</p>',
                    unsafe_allow_html=True)
        st.markdown(
            "Upload a single-row CSV in the same format as `koa_erd_processed.csv`. "
            "Values will override the manual inputs above."
        )
        uploaded = st.file_uploader("Choose CSV file", type=["csv"])
        if uploaded:
            try:
                dfup = pd.read_csv(uploaded)
                dfup.columns = (dfup.columns.str.strip()
                                .str.replace(' ', '_')
                                .str.replace('-', '_')
                                .str.replace('Siiting', 'Sitting')
                                .str.replace('Posting', 'Post'))
                row = dfup.iloc[0].to_dict()
                inputs.update({
                    k: float(v) if isinstance(v, (int, float, np.integer, np.floating))
                    else v for k, v in row.items()
                })
                st.success(f"✅ Loaded {len(row)} columns.")
                st.dataframe(
                    pd.DataFrame([row]).T.rename(columns={0: "Value"}),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")

    # TAB 4 — PREDICT
    with tab4:
        st.markdown('<p class="section-hdr">Patient Summary</p>', unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"- **KL Grade:** {inputs.get('KL_grade', '—')}")
            st.markdown(f"- **McGill:** {inputs.get('McGill', '—')}  |  "
                        f"**QoL:** {inputs.get('QoL', '—')}")
            bmi_val = float(inputs.get('BMI', 0))
            st.markdown(f"- **Age:** {inputs.get('Age', '—')} yrs  |  "
                        f"**BMI:** {bmi_val:.1f} kg/m²")
            st.markdown(f"- **Duration:** {inputs.get('Duration_of_symptom', '—')} yrs")
        with s2:
            st.markdown(f"- **PRI / WMI / PSI:** "
                        f"{inputs.get('PRI','—')} / "
                        f"{inputs.get('WMI','—')} / "
                        f"{inputs.get('PSI','—')}")
            st.markdown(f"- **EEG:** {len(CHANNELS)} channels × "
                        f"{len(BANDS)} bands × {len(CONDITIONS)} conditions")

        st.markdown("---")

        if st.button("🔍  Predict Pain Level", use_container_width=True):
            with st.spinner("Running prediction..."):
                try:
                    scaler     = bundle['scaler']
                    model      = bundle['model']
                    feat_names = bundle['feature_names']

                    X_in = engineer_features(inputs)
                    for feat in feat_names:
                        if feat not in X_in.columns:
                            X_in[feat] = 0.0
                    X_in = X_in[feat_names]
                    X_sc = scaler.transform(X_in)

                    pred  = int(model.predict(X_sc)[0])
                    proba = {
                        int(c): float(p)
                        for c, p in zip(model.classes_,
                                        model.predict_proba(X_sc)[0])
                    }

                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Result")

                    r1, r2 = st.columns([1, 1])
                    with r1:
                        st.markdown(f"""
                        <div class="result-box"
                             style="background:{PAIN_BG[pred]};
                                    border-color:{PAIN_COLORS[pred]};">
                          <div style="font-size:3rem;">{PAIN_EMOJI[pred]}</div>
                          <div style="font-size:1.8rem;font-weight:700;
                               color:{PAIN_COLORS[pred]};">
                            {PAIN_LABELS[pred]}
                          </div>
                          <div style="color:#555;margin-top:.5rem;font-size:.95rem;">
                            Model confidence: <b>{proba.get(pred,0)*100:.1f}%</b>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                        note = {
                            1: "EEG and clinical markers within normal range.",
                            2: "Mild indicators present. Monitor KL grade and symptom progression.",
                            3: "Moderate pain burden. Significant functional impact likely.",
                            4: "Severe pain profile. Consistent with advanced KOA."
                        }
                        st.markdown(
                            f"<br><small><i>{note[pred]}</i></small>",
                            unsafe_allow_html=True
                        )

                    with r2:
                        st.markdown("**Probability per class**")
                        for cls in [1, 2, 3, 4]:
                            p    = proba.get(cls, 0)
                            pct  = int(p * 100)
                            mark = " ◀ predicted" if cls == pred else ""
                            st.markdown(f"""
                            <div style="margin:7px 0;">
                              <div style="display:flex;justify-content:space-between;
                                   font-size:.85rem;">
                                <span>{PAIN_LABELS[cls]}
                                  <small style="color:#999;">{mark}</small></span>
                                <b>{pct}%</b>
                              </div>
                              <div style="background:#e8e8e8;border-radius:4px;
                                   height:10px;margin-top:3px;">
                                <div style="background:{PAIN_COLORS[cls]};
                                     width:{pct}%;height:10px;
                                     border-radius:4px;"></div>
                              </div>
                            </div>""", unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("### 🏥 Key Clinical Risk Factors")
                    kl  = inputs.get('KL_grade', 0)
                    mc  = inputs.get('McGill', 0)
                    dur = inputs.get('Duration_of_symptom', 0)
                    bmi = float(inputs.get('BMI', 0))
                    kl_r = ("⚠️ High"      if kl  >= 3 else
                            "⬆️ Moderate"  if kl  >= 2 else "✅ Low")
                    mc_r = ("⚠️ High"      if mc  >= 4 else
                            "⬆️ Moderate"  if mc  >= 2 else "✅ Low")
                    dr_r = "⚠️ Long"    if dur >= 5 else "✅ Short"
                    bm_r = ("⚠️ Obese"     if bmi >= 30 else
                            "⬆️ Overweight" if bmi >= 25 else "✅ Normal")
                    st.dataframe(
                        pd.DataFrame({
                            'Feature'    : ['KL Grade', 'McGill Score',
                                            'Symptom Duration', 'BMI'],
                            'Value'      : [kl, mc, f"{dur} yrs", f"{bmi:.1f} kg/m²"],
                            'Risk Level' : [kl_r, mc_r, dr_r, bm_r]
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.markdown("""
                    <div class="disclaimer">
                      ⚠️ <b>Clinical Disclaimer:</b> For research and educational
                      purposes only. Based on N=62 patients. Not a substitute for
                      clinical judgement. Always consult a qualified healthcare professional.
                    </div>""", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.exception(e)

    st.markdown("---")
    st.caption(
        f"KOA Pain Level Predictor  |  Model: {bundle['model_name']}  |  "
        f"LOO-CV Validated  |  Inference only — model trained offline  |  Research use only"
    )

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
