"""Load and clean the PaySim sample, and define the two feature sets.

Usage:
    from src.data import load_clean, AUTH_FEATURES, ALL_FEATURES
    df = load_clean()
"""

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "fraud_data.csv"

TARGET = "isFraud"
RULE_FLAG = "isFlaggedFraud"  # existing rule; a baseline, never a model input
TIME_COL = "step"  # hour of the simulation (1..743); used for the time split

# Fraud only occurs in these two types, so the model is trained on them only.
FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]

# Old index columns and the unreliable date column.
DROP_COLS = ["Unnamed: 0.1", "Unnamed: 0", "date"]

# Known when the transaction is requested (before it is authorised).
AUTH_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "is_transfer",
    "hour",
]

# Only known after the transaction has gone through.
POST_FEATURES = [
    "newbalanceOrig",
    "newbalanceDest",
]

ALL_FEATURES = AUTH_FEATURES + POST_FEATURES


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Read the CSV as-is."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unused columns, keep fraud-bearing types, add simple features."""
    out = df.drop(columns=DROP_COLS, errors="ignore")
    out = out[out["type"].isin(FRAUD_TYPES)].copy()

    out["is_transfer"] = (out["type"] == "TRANSFER").astype(int)
    out["hour"] = out[TIME_COL] % 24

    return out.sort_values(TIME_COL, kind="stable").reset_index(drop=True)


def load_clean(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Read and clean in one call."""
    return clean(load_raw(path))


# Train on the first 300 simulated hours, test on the rest (301..743).
SPLIT_STEP = 300


def time_split(
    df: pd.DataFrame, split_step: int = SPLIT_STEP
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: rows with step <= split_step train, later rows test."""
    train = df[df[TIME_COL] <= split_step]
    test = df[df[TIME_COL] > split_step]
    return train, test


if __name__ == "__main__":
    data = load_clean()
    print("shape:", data.shape)
    print("frauds:", int(data[TARGET].sum()))
    print("fraud rate: {:.2%}".format(data[TARGET].mean()))
    print("step range:", data[TIME_COL].min(), "-", data[TIME_COL].max())
