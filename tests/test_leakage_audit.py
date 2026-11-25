"""Smoke test for src/leakage_audit.py on synthetic data (no CSV needed)."""

import numpy as np
import pandas as pd

from src.data import ALL_FEATURES, TARGET
from src.leakage_audit import fit_and_score


def test_fit_and_score_returns_valid_metrics():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({col: rng.random(n) for col in ALL_FEATURES})
    df[TARGET] = (df["amount"] > 0.8).astype(int)

    train, test = df.iloc[:300], df.iloc[300:]
    _, metrics = fit_and_score(train, test, ALL_FEATURES)

    assert set(metrics) == {"roc_auc", "pr_auc"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
    assert metrics["roc_auc"] > 0.9  # signal is easy to learn
