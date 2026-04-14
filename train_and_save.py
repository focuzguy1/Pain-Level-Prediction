"""
============================================================
KOA PAIN LEVEL PREDICTION — TRAINING SCRIPT
Run once in Google Colab. Produces ONE file: koa_model.joblib
Upload koa_model.joblib to your GitHub repo root.
============================================================

WHY ONE FILE:
  - Simpler deployment
  - No version mismatch between scaler/model/features
  - Everything stays in sync

HOW TO RUN IN COLAB:
  !pip install imbalanced-learn
  # Then run all cells
============================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score,
                              balanced_accuracy_score, cohen_kappa_score)
from imblearn.over_sampling import SMOTE

# ============================================================
# CONSTANTS
# ============================================================

DATASET_URL = (
    "https://raw.githubusercontent.com/focuzguy1/Pain-Level-Prediction/"
    "324cd6a74838e471a00f8537ad9376376da1938b/koa_erd_processed.csv"
)

PAIN_LABELS = {1: "No Pain", 2: "Mild Pain", 3: "Moderate Pain", 4: "Severe Pain"}
CHANNELS    = ['Fp1', 'Fp2', 'Fz', 'F7', 'F8', 'T7', 'T8', 'Pz']
CONDITIONS  = ['Pre', 'Post', 'Standing', 'Sitting']

# ============================================================
# SENSITIVITY & SPECIFICITY HELPER
# ============================================================

def compute_sens_spec(y_true, y_pred, labels):
    """
    Compute per-class and macro sensitivity + specificity
    using one-vs-rest from the confusion matrix.

    Sensitivity (Recall) = TP / (TP + FN)
      — of all patients who truly have this pain level,
        how many did the model correctly identify?

    Specificity = TN / (TN + FP)
      — of all patients who do NOT have this pain level,
        how many did the model correctly exclude?

    Both are computed per class (one-vs-rest) then macro-averaged.
    This is the standard approach for multiclass clinical ML papers.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n  = len(labels)

    per_class = {}
    for i, cls in enumerate(labels):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP          # true class predicted as other
        FP = cm[:, i].sum() - TP          # other class predicted as this
        TN = cm.sum() - TP - FN - FP      # correctly excluded

        sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        per_class[int(cls)] = {
            'sensitivity': float(sens),
            'specificity': float(spec),
            'TP': int(TP), 'FN': int(FN),
            'FP': int(FP), 'TN': int(TN)
        }

    macro_sens = float(np.mean([v['sensitivity'] for v in per_class.values()]))
    macro_spec = float(np.mean([v['specificity'] for v in per_class.values()]))

    return {
        'per_class'       : per_class,
        'macro_sensitivity': macro_sens,
        'macro_specificity': macro_spec
    }


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

print("=" * 60)
print("STEP 1: Loading dataset")
print("=" * 60)

df = pd.read_csv(DATASET_URL)
df.columns = (df.columns.str.strip()
              .str.replace(' ', '_')
              .str.replace('-', '_')
              .str.replace('Siiting', 'Sitting')
              .str.replace('Posting', 'Post'))

y    = df['Pain_Level'].values
dist = Counter(y)
classes = sorted(dist.keys())

print(f"  Total patients : {len(df)}")
print(f"  Class distribution:")
for k in classes:
    print(f"    Class {k} ({PAIN_LABELS[k]:<15}): "
          f"n={dist[k]:2d}  ({dist[k]/len(y)*100:.1f}%)")

min_class_n = min(dist.values())
print(f"\n  Smallest class: n={min_class_n}")

# ============================================================
# STEP 2 — FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Feature Engineering")
print("=" * 60)

def engineer_features(src: pd.DataFrame) -> pd.DataFrame:
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
            col(f'Fz_Theta_{cond}') / (col(f'Fz_Beta_{cond}') + 1e-6)
        )

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

    result = pd.DataFrame(feats)
    result = result.fillna(result.median())
    return result


X = engineer_features(df)
feature_names = list(X.columns)
print(f"  Features created : {len(feature_names)}")

# ============================================================
# STEP 3 — SCALE
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Fitting StandardScaler")
print("=" * 60)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("  Scaler fitted on full training set")

# ============================================================
# STEP 4 — LOO-CV EVALUATION (all 4 models)
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Leave-One-Out Cross-Validation (all models)")
print("  Sensitivity & Specificity computed per class (one-vs-rest)")
print("=" * 60)

models_to_eval = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42
    ),
    "SVM (RBF)": SVC(
        kernel='rbf', C=10, gamma='scale',
        class_weight='balanced', probability=True, random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        C=0.1, class_weight='balanced', solver='lbfgs',
        max_iter=2000, random_state=42
    ),
}

loo          = LeaveOneOut()
eval_results = {}
best_f1      = 0
best_model_name = None

for name, m in models_to_eval.items():
    yp = cross_val_predict(m, X_scaled, y, cv=loo)

    acc  = accuracy_score(y, yp)
    bal  = balanced_accuracy_score(y, yp)
    f1m  = f1_score(y, yp, average='macro',    zero_division=0)
    f1w  = f1_score(y, yp, average='weighted', zero_division=0)
    kap  = cohen_kappa_score(y, yp)

    # Sensitivity & Specificity (per class + macro)
    ss   = compute_sens_spec(y, yp, labels=classes)

    eval_results[name] = dict(
        accuracy=acc, balanced_accuracy=bal,
        f1_macro=f1m, f1_weighted=f1w, kappa=kap,
        macro_sensitivity=ss['macro_sensitivity'],
        macro_specificity=ss['macro_specificity'],
        per_class_sens_spec=ss['per_class'],
        y_pred=yp
    )

    marker = " <- best so far" if f1m > best_f1 else ""
    if f1m > best_f1:
        best_f1 = f1m
        best_model_name = name

    print(f"\n  [{name}]{marker}")
    print(f"    Accuracy          : {acc*100:.1f}%")
    print(f"    Balanced Accuracy : {bal*100:.1f}%")
    print(f"    F1 Macro          : {f1m*100:.1f}%")
    print(f"    Cohen's Kappa     : {kap:.3f}")
    print(f"    Sensitivity (macro): {ss['macro_sensitivity']*100:.1f}%")
    print(f"    Specificity (macro): {ss['macro_specificity']*100:.1f}%")
    print(f"    Per-class breakdown:")
    for cls, vals in ss['per_class'].items():
        print(f"      {PAIN_LABELS[cls]:<15} "
              f"Sens={vals['sensitivity']*100:.1f}%  "
              f"Spec={vals['specificity']*100:.1f}%  "
              f"(TP={vals['TP']} FP={vals['FP']} "
              f"TN={vals['TN']} FN={vals['FN']})")

print(f"\n  Best model: {best_model_name} (F1 Macro = {best_f1*100:.1f}%)")

# ============================================================
# STEP 5 — SELECT FINAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Selecting final model")
print("=" * 60)

final_model = models_to_eval[best_model_name]
best_res    = eval_results[best_model_name]

print(f"  Deployed model     : {best_model_name}")
print(f"  LOO-CV Accuracy    : {best_res['accuracy']*100:.1f}%")
print(f"  Balanced Accuracy  : {best_res['balanced_accuracy']*100:.1f}%")
print(f"  F1 Macro           : {best_res['f1_macro']*100:.1f}%")
print(f"  Cohen's Kappa      : {best_res['kappa']:.3f}")
print(f"  Sensitivity (macro): {best_res['macro_sensitivity']*100:.1f}%")
print(f"  Specificity (macro): {best_res['macro_specificity']*100:.1f}%")

print("\n  Full Classification Report (LOO-CV):")
print(classification_report(
    y, best_res['y_pred'],
    target_names=[PAIN_LABELS[i] for i in classes],
    zero_division=0
))

# ============================================================
# STEP 6 — TRAIN FINAL MODEL ON FULL DATA + SMOTE
# ============================================================

print("=" * 60)
print("STEP 6: Training final model on full data (with SMOTE)")
print("=" * 60)

try:
    k_neighbors = max(1, min(3, min_class_n - 1))
    sm          = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_res, y_res = sm.fit_resample(X_scaled, y)
    print(f"  SMOTE k_neighbors : {k_neighbors}")
    print(f"  After SMOTE       : {len(y_res)} samples")
    for k in sorted(Counter(y_res)):
        print(f"    Class {k}: n={Counter(y_res)[k]}")
except Exception as e:
    print(f"  SMOTE skipped ({e}) — using original data")
    X_res, y_res = X_scaled, y

final_model.fit(X_res, y_res)
print(f"  {best_model_name} trained on SMOTE-resampled data")

# Feature importances
feat_imp = {}
if hasattr(final_model, 'feature_importances_'):
    fi_series = pd.Series(
        final_model.feature_importances_, index=feature_names
    ).sort_values(ascending=False)
    feat_imp = fi_series.head(20).to_dict()
    print(f"\n  Top 10 features:")
    for feat, imp in list(feat_imp.items())[:10]:
        bar = "█" * max(1, int(imp * 100))
        print(f"    {feat:<38} {bar} {imp:.4f}")

# ============================================================
# STEP 7 — SAVE BUNDLE
# ============================================================

print("\n" + "=" * 60)
print("STEP 7: Saving model bundle -> koa_model.joblib")
print("=" * 60)

bundle = {
    # ── Core inference objects ────────────────────────────
    'scaler'        : scaler,
    'model'         : final_model,
    'feature_names' : feature_names,

    # ── Identity ──────────────────────────────────────────
    'model_name'    : best_model_name,
    'n_samples'     : len(y),
    'n_features'    : len(feature_names),
    'pain_labels'   : PAIN_LABELS,
    'class_dist'    : {int(k): int(v) for k, v in dist.items()},

    # ── Overall LOO-CV metrics ─────────────────────────────
    'loo_accuracy'          : best_res['accuracy'],
    'loo_balanced_accuracy' : best_res['balanced_accuracy'],
    'loo_f1_macro'          : best_res['f1_macro'],
    'loo_f1_weighted'       : best_res['f1_weighted'],
    'loo_kappa'             : best_res['kappa'],

    # ── Sensitivity & Specificity ─────────────────────────
    'loo_macro_sensitivity' : best_res['macro_sensitivity'],
    'loo_macro_specificity' : best_res['macro_specificity'],
    'loo_per_class_sens_spec': best_res['per_class_sens_spec'],

    # ── Per-model breakdown (for sidebar comparison table) ─
    'all_model_results': {
        k: {mk: mv for mk, mv in v.items() if mk != 'y_pred'}
        for k, v in eval_results.items()
    },

    # ── Feature importances ───────────────────────────────
    'feature_importances': feat_imp,

    # ── Confusion matrix & classification report ──────────
    'confusion_matrix': confusion_matrix(y, best_res['y_pred'],
                                         labels=classes).tolist(),
    'classification_report': classification_report(
        y, best_res['y_pred'],
        target_names=[PAIN_LABELS[i] for i in classes],
        zero_division=0,
        output_dict=True
    ),
}

joblib.dump(bundle, 'koa_model.joblib', compress=3)
size_kb = os.path.getsize('koa_model.joblib') / 1024
print(f"  koa_model.joblib saved ({size_kb:.0f} KB)")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"""
  Model     : {best_model_name}
  File      : koa_model.joblib ({size_kb:.0f} KB)

  LOO-CV METRICS (report in paper):
    Accuracy          : {best_res['accuracy']*100:.1f}%
    Balanced Accuracy : {best_res['balanced_accuracy']*100:.1f}%
    F1 Macro          : {best_res['f1_macro']*100:.1f}%
    F1 Weighted       : {best_res['f1_weighted']*100:.1f}%
    Cohen's Kappa     : {best_res['kappa']:.3f}
    Sensitivity (macro): {best_res['macro_sensitivity']*100:.1f}%
    Specificity (macro): {best_res['macro_specificity']*100:.1f}%

  PER-CLASS SENSITIVITY / SPECIFICITY:""")

for cls, vals in best_res['per_class_sens_spec'].items():
    print(f"    {PAIN_LABELS[cls]:<15} "
          f"Sens={vals['sensitivity']*100:.1f}%  "
          f"Spec={vals['specificity']*100:.1f}%")

print("""
  NEXT STEPS:
    1. koa_model.joblib is being saved to Google Drive (below)
    2. Download it and upload to your GitHub repo root
    3. That is the ONLY file the Streamlit app needs
""")
print("=" * 60)

# ============================================================
# SAVE TO GOOGLE DRIVE
# ============================================================

print("\n" + "=" * 60)
print("SAVING TO GOOGLE DRIVE")
print("=" * 60)

try:
    from google.colab import drive
    import shutil

    print("  Mounting Google Drive...")
    drive.mount('/content/drive', force_remount=False)

    drive_folder = '/content/drive/MyDrive/KOA_PainPredictor'
    os.makedirs(drive_folder, exist_ok=True)

    drive_model_path = f'{drive_folder}/koa_model.joblib'
    shutil.copy('koa_model.joblib', drive_model_path)
    drive_size_kb = os.path.getsize(drive_model_path) / 1024

    try:
        shutil.copy('/content/train_and_save.py', f'{drive_folder}/train_and_save.py')
        script_saved = True
    except Exception:
        script_saved = False

    print(f"\n  Files saved to Google Drive:")
    print(f"  MyDrive/KOA_PainPredictor/")
    print(f"    koa_model.joblib   ({drive_size_kb:.0f} KB)  <- upload this to GitHub")
    if script_saved:
        print(f"    train_and_save.py  (training script backup)")

    print("""
  HOW TO GET IT ONTO GITHUB:

  Option A - via Google Drive:
    1. Open drive.google.com
    2. Go to MyDrive > KOA_PainPredictor
    3. Right-click koa_model.joblib -> Download
    4. Go to your GitHub repo -> Add file -> Upload files
    5. Drag in koa_model.joblib -> Commit changes

  Option B - directly from Colab sidebar:
    1. Click the folder icon in the left Colab panel
    2. Find koa_model.joblib -> right-click -> Download
    3. Upload to GitHub as above
""")

except ModuleNotFoundError:
    print(f"  Not running in Colab.")
    print(f"  koa_model.joblib at: {os.path.abspath('koa_model.joblib')}")

except Exception as e:
    print(f"  Could not save to Drive: {e}")
    print(f"  Download from Colab sidebar (folder icon) -> right-click -> Download")

print("\n" + "=" * 60)
print("ALL DONE")
print("=" * 60)
