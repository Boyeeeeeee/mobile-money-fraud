"""Part B: judge alert strategies in naira, at the real fraud rate.

Costs
- Every alert is reviewed by an analyst: REVIEW_COST each (caught fraud is stopped).
- A missed fraud costs its full amount.

Re-weighting
The sample kept every fraud but only a fraction of normal transactions. Each
normal row in the test set is weighted by NORMAL_WEIGHT so counts and costs
match the full PaySim data (assumes normal rows were sampled at random).

Thresholds
The model is trained on steps 1..VALID_START-1. Its single threshold is chosen on
the validation period (VALID_START..SPLIT_STEP), never on the test period.

Run:
    python -m src.cost_eval
Writes reports/cost_results.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import AUTH_FEATURES, RULE_FLAG, SPLIT_STEP, TARGET, TIME_COL, load_clean, time_split
from src.leakage_audit import make_model

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "cost_results.csv"

REVIEW_COST = 500.0  # naira per alert

# Full PaySim: TRANSFER 532,909 + CASH_OUT 2,237,500 rows, 8,213 of them fraud.
FULL_NORMAL = 532_909 + 2_237_500 - 8_213  # 2,762,196
SAMPLE_NORMAL = 43_491  # normal TRANSFER/CASH_OUT rows in our sample
NORMAL_WEIGHT = FULL_NORMAL / SAMPLE_NORMAL  # ~63.5 real normals per sample row

VALID_START = 241  # validation = steps 241..300

MODEL_SETS = {
    "auth_time": AUTH_FEATURES,
    "auth_no_sender_balance": [f for f in AUTH_FEATURES if f != "oldbalanceOrg"],
}


def row_weights(y: np.ndarray, normal_weight: float = NORMAL_WEIGHT) -> np.ndarray:
    """1 for each fraud, normal_weight for each normal transaction."""
    return np.where(np.asarray(y) == 1, 1.0, normal_weight)


def to_true_rate(p: np.ndarray, normal_weight: float = NORMAL_WEIGHT) -> np.ndarray:
    """Convert sample probabilities to probabilities at the real fraud rate."""
    p = np.asarray(p, dtype=float)
    return p / (p + (1 - p) * normal_weight)


def evaluate(y, flag, amount, weight, review_cost: float = REVIEW_COST) -> dict:
    """Weighted outcome and naira cost of one set of alerts."""
    y = np.asarray(y)
    flag = np.asarray(flag).astype(bool)
    amount = np.asarray(amount, dtype=float)
    weight = np.asarray(weight, dtype=float)
    fraud = y == 1

    alerts = weight[flag].sum()
    caught = (flag & fraud).sum()
    false_alarms = weight[flag & ~fraud].sum()
    missed_amount = amount[fraud & ~flag].sum()
    review = alerts * review_cost
    return {
        "alerts": alerts,
        "false_alarms": false_alarms,
        "frauds_caught": int(caught),
        "frauds_total": int(fraud.sum()),
        "precision": caught / alerts if alerts else 0.0,
        "fraud_amount_caught": amount[fraud & flag].sum(),
        "missed_fraud_cost": missed_amount,
        "review_cost": review,
        "total_cost": missed_amount + review,
    }


def best_threshold(y, p, amount, weight, review_cost: float = REVIEW_COST) -> float:
    """Single probability threshold with the lowest total cost."""
    grid = np.linspace(0.0, 1.0, 1001)
    costs = [evaluate(y, p >= t, amount, weight, review_cost)["total_cost"] for t in grid]
    return float(grid[int(np.argmin(costs))])


def run() -> pd.DataFrame:
    df = load_clean()
    train_all, test = time_split(df)
    fit = train_all[train_all[TIME_COL] < VALID_START]
    valid = train_all[train_all[TIME_COL] >= VALID_START]

    y_te, amt_te = test[TARGET].to_numpy(), test["amount"].to_numpy()
    w_te = row_weights(y_te)
    w_va = row_weights(valid[TARGET])

    rows = [
        {"strategy": "no alerts", **evaluate(y_te, np.zeros(len(test)), amt_te, w_te)},
        {"strategy": "isFlaggedFraud rule", **evaluate(y_te, test[RULE_FLAG], amt_te, w_te)},
    ]

    for name, features in MODEL_SETS.items():
        model = make_model().fit(fit[features], fit[TARGET])
        p_va = model.predict_proba(valid[features])[:, 1]
        p_te = model.predict_proba(test[features])[:, 1]

        t = best_threshold(valid[TARGET], p_va, valid["amount"], w_va)
        rows.append(
            {"strategy": f"{name}: threshold {t:.3f}", **evaluate(y_te, p_te >= t, amt_te, w_te)}
        )

        expected_loss = to_true_rate(p_te) * amt_te
        rows.append(
            {
                "strategy": f"{name}: expected loss > N{REVIEW_COST:.0f}",
                **evaluate(y_te, expected_loss > REVIEW_COST, amt_te, w_te),
            }
        )

    results = pd.DataFrame(rows)
    baseline = results.loc[0, "total_cost"]
    results["naira_saved"] = baseline - results["total_cost"]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_PATH, index=False)
    return results


if __name__ == "__main__":
    print(f"normal weight: {NORMAL_WEIGHT:.2f} | validation steps {VALID_START}-{SPLIT_STEP}")
    res = run()
    cols = ["strategy", "alerts", "frauds_caught", "precision", "review_cost",
            "missed_fraud_cost", "total_cost", "naira_saved"]
    with pd.option_context("display.width", 200):
        print(res[cols].to_string(index=False, formatters={
            "alerts": "{:,.0f}".format,
            "precision": "{:.1%}".format,
            "review_cost": "{:,.0f}".format,
            "missed_fraud_cost": "{:,.0f}".format,
            "total_cost": "{:,.0f}".format,
            "naira_saved": "{:,.0f}".format,
        }))
