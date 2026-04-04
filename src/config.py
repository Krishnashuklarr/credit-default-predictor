"""
config.py - Central configuration for the Credit Default Prediction project.
All paths, model parameters, and constants live here.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_PATH            = DATA_DIR / "UCI_Credit_Card.csv"
PREPROCESSOR_PATH    = MODEL_DIR / "preprocessor.pkl"
MODEL_PATH           = MODEL_DIR / "lgb_model.pkl"
THRESHOLD_PATH       = MODEL_DIR / "best_threshold.pkl"

# ─── Data ─────────────────────────────────────────────────────────────────────
TARGET       = "default.payment.next.month"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ─── Feature groups ───────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]

NUMERIC_FEATURES = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

ENGINEERED_FEATURES = [
    "debt_to_age",
    "debt_to_limit",
    "payment_ratio",
    "avg_bill",
    "pay_ratio",
    "bill_to_limit",
    "pay_ratio_2",
    "avg_pay_ratio",
    "bill_amnt_diff",
]

# Columns to drop before modelling (not useful for prediction)
DROP_COLS = ["ID"]

# ─── LightGBM parameters ──────────────────────────────────────────────────────
LGBM_PARAMS = {
    "random_state"  : RANDOM_STATE,
    "class_weight"  : "balanced",
    "n_estimators"  : 500,
    "learning_rate" : 0.05,
    "num_leaves"    : 31,
}
