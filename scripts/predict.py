"""
predict.py - Make predictions on new data (replaces 04_new_prediction.ipynb).

Usage
-----
Predict on a CSV file of new records:
    python scripts/predict.py --input data/new_customers.csv --output outputs/predictions.csv

Use built-in sample data (no CSV needed):
    python scripts/predict.py --sample
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH, OUTPUT_DIR, PREPROCESSOR_PATH, THRESHOLD_PATH
from src.feature_engineering import add_features


# ─── Sample data (mirrors test_prediction_sample.py) ─────────────────────────
SAMPLE_DATA = {
    "ID":        [1,     2,      3,      4,      5],
    "LIMIT_BAL": [50000, 100000, 20000,  80000,  30000],
    "SEX":       [2,     1,      2,      1,      2],
    "EDUCATION": [2,     1,      3,      2,      1],
    "MARRIAGE":  [1,     2,      1,      1,      2],
    "AGE":       [30,    45,     25,     35,     50],
    "PAY_0":     [0,    -1,      1,      0,      2],
    "PAY_2":     [0,    -1,      1,      0,      2],
    "PAY_3":     [0,    -1,      1,      0,      2],
    "PAY_4":     [0,    -1,      1,      0,      2],
    "PAY_5":     [0,    -1,      1,      0,      2],
    "PAY_6":     [0,    -1,      1,      0,      2],
    "BILL_AMT1": [20000, 0,      15000,  40000,  25000],
    "BILL_AMT2": [18000, 0,      14000,  38000,  23000],
    "BILL_AMT3": [16000, 0,      13000,  36000,  21000],
    "BILL_AMT4": [14000, 0,      12000,  34000,  19000],
    "BILL_AMT5": [12000, 0,      11000,  32000,  17000],
    "BILL_AMT6": [10000, 0,      10000,  30000,  15000],
    "PAY_AMT1":  [2000,  5000,   1000,   3000,   1500],
    "PAY_AMT2":  [1500,  4500,   900,    2500,   1200],
    "PAY_AMT3":  [1000,  4000,   800,    2000,   1000],
    "PAY_AMT4":  [500,   3500,   700,    1500,   800],
    "PAY_AMT5":  [300,   3000,   600,    1000,   600],
    "PAY_AMT6":  [200,   2500,   500,    800,    400],
}


def load_artifacts():
    """Load preprocessor, model, and threshold. Raise early if missing."""
    for p in [PREPROCESSOR_PATH, MODEL_PATH, THRESHOLD_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing artifact: {p}\n"
                "Run  python scripts/train.py  first."
            )
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model        = joblib.load(MODEL_PATH)
    threshold    = joblib.load(THRESHOLD_PATH)
    print(f"✅  Loaded preprocessor, model, threshold ({threshold:.4f})")
    return preprocessor, model, threshold


def predict(df: pd.DataFrame, preprocessor, model, threshold: float) -> pd.DataFrame:
    """
    Run the full prediction pipeline on a raw DataFrame.

    Returns the original DataFrame with two extra columns:
        Default_Probability  – float probability of default
        Prediction           – 0 (no default) or 1 (default)
    """
    df = add_features(df)
    X_proc = preprocessor.transform(df)

    proba  = model.predict_proba(X_proc)[:, 1]
    labels = (proba >= threshold).astype(int)

    result = df.copy()
    result["Default_Probability"] = proba.round(4)
    result["Prediction"]          = labels
    result["Prediction_Label"]    = result["Prediction"].map({0: "No Default", 1: "Default"})
    return result


def main():
    parser = argparse.ArgumentParser(description="Credit Default Predictor")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true",
                       help="Run on built-in 5-row sample data")
    group.add_argument("--input",  type=Path,
                       help="Path to a CSV file with new customer records")
    parser.add_argument("--output", type=Path,
                        default=OUTPUT_DIR / "predictions.csv",
                        help="Where to save the prediction CSV (default: outputs/predictions.csv)")
    args = parser.parse_args()

    # ── Load model artifacts ────────────────────────────────────────────────
    preprocessor, model, threshold = load_artifacts()

    # ── Load input data ─────────────────────────────────────────────────────
    if args.sample:
        print("\nUsing built-in sample data (5 rows) …")
        df = pd.DataFrame(SAMPLE_DATA)
    else:
        print(f"\nLoading input data from {args.input} …")
        df = pd.read_csv(args.input)
        print(f"  {df.shape[0]} rows loaded")

    print("\nRaw input:")
    print(df.to_string(index=False))

    # ── Predict ─────────────────────────────────────────────────────────────
    results = predict(df, preprocessor, model, threshold)

    # ── Print results ────────────────────────────────────────────────────────
    print("\n=== Prediction Results ===")
    display_cols = ["ID", "Default_Probability", "Prediction", "Prediction_Label"]
    print(results[display_cols].to_string(index=False))

    # ── Save results ─────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"\n✅  Predictions saved → {args.output}")


if __name__ == "__main__":
    main()
