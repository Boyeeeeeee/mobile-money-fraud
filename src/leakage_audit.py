"""Part A: train the same model on two feature sets and compare test scores.

Run:
    python -m src.leakage_audit
Writes reports/leakage_audit.csv.
"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import (
    ALL_FEATURES,
    AUTH_FEATURES,
    TARGET,
    load_clean,
    time_split,
)

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "leakage_audit.csv"

FEATURE_SETS = {
    "all": ALL_FEATURES,
    "auth_time": AUTH_FEATURES,
    # Auth-time minus the sender's balance. In PaySim no normal transaction
    # empties the sender's account but 98% of frauds do, so oldbalanceOrg is a
    # simulator shortcut. This set shows the score without it.
    "auth_no_sender_balance": [f for f in AUTH_FEATURES if f != "oldbalanceOrg"],
}


def make_model() -> lgb.LGBMClassifier:
    """Same model and settings for every feature set."""
    return lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    )


def fit_and_score(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[lgb.LGBMClassifier, dict]:
    """Fit on train, return the model and test metrics."""
    model = make_model()
    model.fit(train[features], train[TARGET])
    p = model.predict_proba(test[features])[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(test[TARGET], p),
        "pr_auc": average_precision_score(test[TARGET], p),
    }
    return model, metrics


def run() -> pd.DataFrame:
    df = load_clean()
    train, test = time_split(df)
    print(
        f"train: {len(train):,} rows, {int(train[TARGET].sum()):,} frauds | "
        f"test: {len(test):,} rows, {int(test[TARGET].sum()):,} frauds"
    )

    rows = []
    for name, features in FEATURE_SETS.items():
        model, metrics = fit_and_score(train, test, features)
        importance = pd.Series(
            model.booster_.feature_importance("gain"), index=features
        )
        top = importance.sort_values(ascending=False).index[0]
        rows.append({"feature_set": name, **metrics, "top_feature": top})

    results = pd.DataFrame(rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_PATH, index=False)
    return results


if __name__ == "__main__":
    print(run().to_string(index=False, float_format="{:.4f}".format))
