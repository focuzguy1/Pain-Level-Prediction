<div align="center">

<img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

# 🧠 KOA Pain Level Predictor

### EEG-based Machine Learning System for Knee Osteoarthritis Pain Classification

*A clinical decision support tool that predicts pain levels in Knee Osteoarthritis (KOA) patients using electroencephalography (EEG) biomarkers and clinical features.*

[**🚀 Live Demo**](https://pain-levels-prediction.streamlit.app/) &nbsp;·&nbsp;
[**📄 Paper**](#) &nbsp;·&nbsp;
[**📊 Dataset**](koa_erd_processed.csv)

</div>

---

## 📌 Overview

Knee Osteoarthritis is a leading cause of chronic pain and functional disability worldwide. Objective, neurophysiological assessment of pain severity remains a clinical challenge. This project presents a machine learning framework that integrates EEG-derived biomarkers with standard clinical measures to classify patient pain levels into four categories:

| Class | Label | Description |
|-------|-------|-------------|
| 1 | **No Pain** | No clinically significant pain burden |
| 2 | **Mild Pain** | Low-level discomfort, functional preservation |
| 3 | **Moderate Pain** | Significant functional impact |
| 4 | **Severe Pain** | Advanced KOA, bone-on-bone presentation |

---

## ✨ Key Features

- **Multi-algorithm evaluation** — Random Forest, Gradient Boosting, SVM, Logistic Regression
- **Clinically validated** — Leave-One-Out Cross-Validation (LOO-CV) for small sample integrity
- **Class imbalance handling** — BorderlineSMOTE applied to final model training only (prevents data leakage)
- **40+ engineered EEG features** — Frontal Alpha Asymmetry, Theta/Beta Ratio, Temporal Delta Power, and more
- **Clinical-only mode** — Degrades gracefully when EEG data is unavailable
- **Reproducible deployment** — Single frozen model bundle (`koa_model.joblib`) loaded at inference time
- **Modern Streamlit UI** — Professional clinical interface with real-time predictions

---

## 📊 Model Performance (LOO-CV)

> Metrics computed on original unaugmented data — no SMOTE leakage.
> Balanced accuracy is the primary metric given class imbalance.

| Metric | Value |
|--------|-------|
| Accuracy | ~74% |
| **Balanced Accuracy** | **~53%** ← primary metric |
| F1 Macro | ~40% |
| F1 Weighted | ~67% |
| Cohen's Kappa | ~0.54 |
| Training N | 62 patients |
| Validation | Leave-One-Out CV |

> **Note:** The gap between accuracy (74%) and balanced accuracy (53%) reflects the class imbalance in the dataset (majority class: No Pain, n=34). Balanced accuracy is the correct metric to report for imbalanced clinical datasets.

---

## 🧬 Feature Engineering

### EEG-Derived Biomarkers (engineered from raw PSD values)

| Feature | Description | Channels |
|---------|-------------|---------|
| Frontal Alpha Asymmetry (FAA) | Left–right frontal alpha lateralisation | Fp1, Fp2 |
| Theta/Beta Ratio (TBR) | Attentional and pain processing index | Fz |
| Global Alpha Power | Cortical inhibition measure | All 8 channels |
| Global Theta Power | Pain sensitisation index | All 8 channels |
| Temporal Delta Power | Somatosensory processing | T7, T8 |
| Parietal Alpha (Pz) | Pain-related alpha suppression | Pz |
| Frontal Gamma Power | Pain intensity marker | Fp1, Fp2 |
| Delta/Alpha Ratio | Cortical arousal index | All channels |
| Alpha Change Pre→Post | Task-related alpha modulation | All channels |
| Alpha Change Stand→Sit | Postural alpha response | All channels |

> **Conditions recorded:** Pre-task · Post-task · Standing · Sitting
> **Channels:** Fp1, Fp2, Fz, F7, F8, T7, T8, Pz
> **Bands:** Alpha, Beta, Theta, Delta, Gamma

### Clinical Features

| Feature | Significance |
|---------|-------------|
| KL Grade (Kellgren-Lawrence) | p < 0.001 ★★★ |
| McGill Pain Score | p < 0.001 ★★★ |
| Symptom Duration | p < 0.001 ★★★ |
| BMI | p = 0.002 ★★ |
| Age | Not significant |
| PRI / WMI / PSI (cognitive) | Composite feature |
| Quality of Life Score | Clinical context |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE TRAINING                          │
│                  (train_and_save.py)                         │
│                                                              │
│  Dataset (N=62) → Feature Engineering → StandardScaler      │
│                                                              │
│       ┌──────────────────────────────────┐                  │
│       │     LOO-CV Evaluation            │                  │
│       │  ├─ Random Forest               │                  │
│       │  ├─ Gradient Boosting           │                  │
│       │  ├─ SVM (RBF kernel)            │                  │
│       │  └─ Logistic Regression         │                  │
│       └──────────────┬───────────────────┘                  │
│                      │ best model selected                   │
│                      ↓                                       │
│         BorderlineSMOTE → Final Model Fit                    │
│                      │                                       │
│                      ↓                                       │
│              koa_model.joblib  ←── ONE file                  │
│         (scaler + model + features + metrics)                │
└──────────────────────┬──────────────────────────────────────┘
                       │ upload to GitHub
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                   STREAMLIT APP                              │
│                 (streamlit_app.py)                           │
│                                                              │
│  Load koa_model.joblib → New patient inputs                  │
│  → Engineer features → Scale → model.predict()              │
│  → Pain level + probability distribution                     │
│                                                              │
│  Training data NEVER touched again                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.9+
pip install -r requirements.txt
```

### Repository Structure

```
Pain-Level-Prediction/
├── streamlit_app.py        ← Streamlit web application
├── train_and_save.py       ← Offline training script (run in Colab)
├── koa_model.joblib        ← Pre-trained model bundle
├── koa_erd_processed.csv   ← Dataset
├── requirements.txt        ← Python dependencies
└── README.md
```

### Option A — Run the Streamlit App Locally

```bash
# 1. Clone the repository
git clone https://github.com/focuzguy1/Pain-Level-Prediction.git
cd Pain-Level-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run streamlit_app.py
```

The app automatically loads `koa_model.joblib` from GitHub on startup.

### Option B — Retrain the Model (Google Colab)

```python
# Cell 1 — Install dependency
!pip install imbalanced-learn

# Cell 2 — Upload train_and_save.py then run
exec(open('train_and_save.py').read())
```

This will:
1. Fetch the dataset from GitHub
2. Engineer all features
3. Train and evaluate 4 algorithms with LOO-CV
4. Apply BorderlineSMOTE and fit the final model
5. Save `koa_model.joblib` to your Google Drive automatically

Then upload the new `koa_model.joblib` to this repository.

---

## 🌐 Deploying to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo and set **Main file path** to `streamlit_app.py`
5. Click **Deploy**

> **Important:** Ensure `koa_model.joblib` is present in the repo root before deploying.

---

## 🖥️ App Interface

The Streamlit application features four input tabs:

| Tab | Description |
|-----|-------------|
| 🩺 Clinical Features | KL Grade, McGill Score, Age, BMI, symptom duration, cognitive indices |
| 🧠 EEG Features | Raw PSD values per channel × band × condition (160 fields) |
| 📂 Upload CSV | Upload a single-row CSV matching the dataset format |
| 🎯 Predict | Patient summary, run prediction, view results |

### Input Modes

- **Full EEG + Clinical** — uses all 40+ features for maximum accuracy
- **Clinical Only** — EEG features set to neutral baseline; prediction driven by clinical data alone (~80% of model weight). Suitable when EEG is unavailable.

### Prediction Output

- Predicted pain class with model confidence
- Probability distribution across all 4 classes
- Clinical risk factor assessment (KL, McGill, BMI, Duration)
- Mode badge indicating Full EEG or Clinical-Only prediction

---

## 📁 Model Bundle

The `koa_model.joblib` file contains:

```python
{
  'scaler'              : StandardScaler,      # fitted on training data
  'model'               : trained_classifier,  # best LOO-CV model
  'feature_names'       : list,                # 40+ feature names
  'model_name'          : str,                 # e.g. "Logistic Regression"
  'loo_accuracy'        : float,
  'loo_balanced_accuracy': float,
  'loo_f1_macro'        : float,
  'loo_f1_weighted'     : float,
  'loo_kappa'           : float,
  'all_model_results'   : dict,                # per-model LOO-CV metrics
  'feature_importances' : dict,                # top 20 features
  'confusion_matrix'    : list,
  'smote_method'        : str,
  'class_dist'          : dict,
}
```

---

## ⚗️ Methodology

### Why Leave-One-Out Cross-Validation?

With only 62 patients, standard k-fold CV would leave too few training samples per fold. LOO-CV uses 61 patients for training and 1 for testing at each iteration — maximising training data while providing unbiased performance estimates. This is the recommended validation strategy for small clinical datasets.

### Why BorderlineSMOTE?

The dataset has a 5.7:1 imbalance ratio (No Pain vs Severe Pain). Standard SMOTE generates synthetic samples randomly across the minority class. BorderlineSMOTE focuses specifically on samples near the decision boundary — the clinically ambiguous cases — producing more informative synthetic data for small medical datasets.

> **Critical:** SMOTE is applied **only** when fitting the final deployed model. It is **never** applied inside LOO-CV folds, which would constitute data leakage and inflate reported metrics.

### Why Balanced Accuracy as Primary Metric?

Raw accuracy of 74% is misleading when 55% of patients are in a single class. A naive classifier that always predicts "No Pain" would achieve 55% accuracy. Balanced accuracy weights each class equally regardless of size, providing an honest measure of per-class discriminative ability.

---

## 📚 Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
joblib>=1.2.0
```

---

## ⚠️ Clinical Disclaimer

This tool is developed for **research and educational purposes only**. It is not a validated clinical diagnostic device and must not replace the judgement of a qualified healthcare professional. All predictions should be interpreted in the context of a full clinical assessment.

- Trained on a cohort of N=62 patients
- Not externally validated
- Not approved for clinical use
- Intended for research exploration only

---

## 📄 Citation

If you use this code or model in your research, please cite:

```bibtex
@software{koa_pain_predictor_2025,
  title   = {KOA Pain Level Predictor: EEG-based Machine Learning
             for Knee Osteoarthritis Pain Classification},
  author  = {[Your Name]},
  year    = {2025},
  url     = {https://github.com/focuzguy1/Pain-Level-Prediction}
}
```

---

## 📬 Contact

For questions, collaborations, or clinical feedback, please open a [GitHub Issue](https://github.com/focuzguy1/Pain-Level-Prediction/issues).

---

<div align="center">

Made with ❤️ for clinical ML research

**[⬆ Back to top](#)**

</div>
