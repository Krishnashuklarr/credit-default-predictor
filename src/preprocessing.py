"""
preprocessing.py - Build, fit, and save the sklearn ColumnTransformer.

The preprocessor handles:
  - Numeric features  → median imputation + standard scaling
  - Categorical features → most-frequent imputation + one-hot encoding
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
    PREPROCESSOR_PATH,
)


def build_preprocessor() -> ColumnTransformer:
    """Return an *unfitted* ColumnTransformer."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    all_numeric = NUMERIC_FEATURES + ENGINEERED_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,  all_numeric),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # silently drops ID and anything else not listed
    )
    return preprocessor


def fit_and_save_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit the preprocessor on training data and persist it to disk."""
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"✅  Preprocessor saved → {PREPROCESSOR_PATH}")
    return preprocessor


def load_preprocessor() -> ColumnTransformer:
    """Load the preprocessor that was saved during training."""
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {PREPROCESSOR_PATH}. "
            "Run scripts/train.py first."
        )
    return joblib.load(PREPROCESSOR_PATH)
