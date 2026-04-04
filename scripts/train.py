"""
train.py - End-to-end training pipeline.

Run from the project root:
    python scripts/train.py

Steps
------
1. Load raw data
2. Feature engineering
3. Train / test split
4. Fit & save preprocessor
5. Transform features
6. Train LightGBM
7. Find F1-optimal threshold
8. Evaluate & save model + threshold
"""

import sys
from pathlib import Path

# Allow imports from src/ regardless of where the script is called from
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
)

from src.config import (
    DATA_PATH,
    DROP_COLS,
    LGBM_PARAMS,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    THRESHOLD_PATH,
)
from src.feature_engineering import add_features
from src.preprocessing import fit_and_save_preprocessor


# ─── 1. Load data ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  Credit Default Prediction — Training Pipeline")
print("=" * 55)

print(f"\n[1/7] Loading data from {DATA_PATH} …")
df = pd.read_csv(DATA_PATH)
print(f"      Shape: {df.shape}")

# ─── 2. Feature engineering ───────────────────────────────────────────────────
print("\n[2/7] Engineering features …")
df = add_features(df)

X = df.drop(columns=[TARGET] + DROP_COLS)
y = df[TARGET]

# ─── 3. Train / test split ────────────────────────────────────────────────────
print("\n[3/7] Splitting data …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)
print(f"      Train: {X_train.shape}   Test: {X_test.shape}")

# ─── 4. Fit & save preprocessor ───────────────────────────────────────────────
print("\n[4/7] Fitting preprocessor …")
preprocessor = fit_and_save_preprocessor(X_train)

# ─── 5. Transform ─────────────────────────────────────────────────────────────
print("\n[5/7] Transforming features …")
X_train_proc = preprocessor.transform(X_train)
X_test_proc  = preprocessor.transform(X_test)
print(f"      Processed train shape: {X_train_proc.shape}")

# ─── 6. Train LightGBM ────────────────────────────────────────────────────────
print("\n[6/7] Training LightGBM …")
model = lgb.LGBMClassifier(**LGBM_PARAMS)
model.fit(X_train_proc, y_train)

# ─── 7. Find F1-optimal threshold & evaluate ──────────────────────────────────
print("\n[7/7] Evaluating …")
y_scores = model.predict_proba(X_test_proc)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
f1_scores    = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx     = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])

y_pred = (y_scores >= best_threshold).astype(int)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC        : {roc_auc_score(y_test, y_scores):.4f}")
print(f"Best Threshold : {best_threshold:.4f}  (F1 = {f1_scores[best_idx]:.4f})")

# ─── Save model & threshold ───────────────────────────────────────────────────
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model,          MODEL_PATH)
joblib.dump(best_threshold, THRESHOLD_PATH)

print(f"\n✅  Model saved     → {MODEL_PATH}")
print(f"✅  Threshold saved → {THRESHOLD_PATH}")
print("\nDone! 🎉")
