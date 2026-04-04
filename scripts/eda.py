"""
eda.py - Exploratory Data Analysis (replaces 01_data_exploration.ipynb).

Run from the project root:
    python scripts/eda.py

All plots are saved to outputs/eda/
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import DATA_PATH, OUTPUT_DIR, TARGET

PLOT_DIR = OUTPUT_DIR / "eda"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def save(fig: plt.Figure, name: str) -> None:
    path = PLOT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ─── Load ──────────────────────────────────────────────────────────────────────
print(f"Loading {DATA_PATH} …")
df = pd.read_csv(DATA_PATH)

print(f"\nDataset Shape : {df.shape}")
print(f"\nMissing Values:\n{df.isnull().sum().to_string()}")
print(f"\nData Types:\n{df.dtypes.to_string()}")

# ─── Class distribution ────────────────────────────────────────────────────────
print(f"\nClass Distribution:\n{df[TARGET].value_counts()}")
print(f"\nClass Distribution (%):\n{df[TARGET].value_counts(normalize=True).mul(100).round(2).to_string()}")

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=TARGET, data=df, palette="Set2", ax=ax)
ax.set_title("Class Distribution")
save(fig, "01_class_distribution.png")

# ─── Histograms ───────────────────────────────────────────────────────────────
fig = df.hist(bins=30, figsize=(16, 12), color="steelblue", edgecolor="white")[0][0].figure
fig.suptitle("Feature Histograms", y=1.01)
save(fig, "02_histograms.png")

# ─── Boxplot LIMIT_BAL ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x=df["LIMIT_BAL"], color="skyblue", ax=ax)
ax.set_title("LIMIT_BAL Distribution")
save(fig, "03_limit_bal_boxplot.png")

# ─── Correlation heatmap ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.3, ax=ax)
ax.set_title("Correlation Heatmap")
save(fig, "04_correlation_heatmap.png")

print(f"\nTop correlations with {TARGET}:")
print(corr[TARGET].sort_values(ascending=False).head(10).to_string())

# ─── Categorical features vs target ───────────────────────────────────────────
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
for cat in categorical_features:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=cat, hue=TARGET, data=df, palette="Set1", ax=ax)
    ax.set_title(f"{cat} vs {TARGET}")
    save(fig, f"05_{cat.lower()}_vs_target.png")

# ─── LIMIT_BAL by target ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(x=TARGET, y="LIMIT_BAL", data=df, palette="muted", ax=ax)
ax.set_title("LIMIT_BAL by Default Status")
save(fig, "06_limit_bal_by_target.png")

print(f"\n✅  All plots saved to {PLOT_DIR}")
