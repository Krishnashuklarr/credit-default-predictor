"""
feature_engineering.py - All feature engineering in one place.

Every function here is PURE: it takes a DataFrame and returns a new one.
Both training and prediction call the same function so features are guaranteed
to match.
"""

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a DataFrame.

    IMPORTANT: Call this AFTER the raw data is loaded but BEFORE
    any preprocessing / scaling.  The function never mutates in-place —
    it returns a copy.
    """
    df = df.copy()

    # Ratio of credit limit to age  (how much limit per year of life)
    df["debt_to_age"] = df["LIMIT_BAL"] / (df["AGE"] + 1)

    # How much of the credit limit is used this month
    df["debt_to_limit"] = df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1)

    # Fraction of this month's bill that was paid last month
    df["payment_ratio"] = df["PAY_AMT1"] / (df["BILL_AMT1"] + 1)

    # Average bill amount across all 6 months
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    df["avg_bill"] = df[bill_cols].mean(axis=1)

    # Average repayment status across the 3 most recent months
    df["pay_ratio"] = (df["PAY_0"] + df["PAY_2"] + df["PAY_3"]) / 3

    # Sum of 3-month bills relative to limit
    df["bill_to_limit"] = (
        df["BILL_AMT1"] + df["BILL_AMT2"] + df["BILL_AMT3"]
    ) / (df["LIMIT_BAL"] + 1)

    # Payment ratio for month 2
    df["pay_ratio_2"] = df["PAY_AMT2"] / (df["BILL_AMT2"] + 1)

    # Average payment ratio across month 1 and month 2
    df["avg_pay_ratio"] = df[["pay_ratio", "pay_ratio_2"]].mean(axis=1)

    # Month-over-month change in bill amount (positive = bill grew)
    df["bill_amnt_diff"] = df["BILL_AMT1"] - df["BILL_AMT2"]

    # Replace any inf / -inf introduced by division with NaN
    # (the downstream scaler/imputer will handle NaN)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df
