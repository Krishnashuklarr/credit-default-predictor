"""
src/logger.py - Saves every prediction to outputs/prediction_log.csv
"""

import pandas as pd
from datetime import datetime
from src.config import OUTPUT_DIR

LOG_PATH = OUTPUT_DIR / "prediction_log.csv"

COLUMNS = [
    "timestamp", "customer_id",
    "limit_bal", "age", "sex", "education", "marriage",
    "avg_pay_status", "avg_bill", "avg_payment",
    "default_probability", "prediction", "risk_level"
]


def _risk(proba, threshold):
    if proba < 0.3:       return "LOW"
    elif proba < threshold: return "MEDIUM"
    else:                  return "HIGH"


def log_prediction(row: dict, proba: float, threshold: float, customer_id: str = "—"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pamt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    pay_cols  = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]

    record = {
        "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "customer_id":        customer_id,
        "limit_bal":          row.get("LIMIT_BAL", 0),
        "age":                row.get("AGE", 0),
        "sex":                "Male" if row.get("SEX") == 1 else "Female",
        "education":          {1:"Graduate",2:"University",3:"High School",4:"Other"}.get(row.get("EDUCATION"), "—"),
        "marriage":           {1:"Married",2:"Single",3:"Other"}.get(row.get("MARRIAGE"), "—"),
        "avg_pay_status":     round(sum(row.get(c,0) for c in pay_cols) / 6, 2),
        "avg_bill":           round(sum(row.get(c,0) for c in bill_cols) / 6, 2),
        "avg_payment":        round(sum(row.get(c,0) for c in pamt_cols) / 6, 2),
        "default_probability": round(proba, 4),
        "prediction":         int(proba >= threshold),
        "risk_level":         _risk(proba, threshold),
    }

    df_new = pd.DataFrame([record], columns=COLUMNS)
    if LOG_PATH.exists():
        df_new.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(LOG_PATH, mode="w", header=True, index=False)


def load_log() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(LOG_PATH, parse_dates=["timestamp"])