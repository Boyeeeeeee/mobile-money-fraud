"""What if the review team can only handle so many alerts an hour?

For each hourly budget, pick the expected-loss cut-off on the validation period
so that alerts per hour stay within the budget, then apply it unchanged to the
test period. The cut-off is never below REVIEW_COST (alerting below that loses
money on average). The decision uses only the model score, never the label.

Run:
    python -m src.capacity
Writes reports/capacity_results.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.cost_eval import (
    MODEL_SETS,
    REVIEW_COST,
    VALID_START,
    evaluate,
    row_weights,
    to_true_rate,
)
from src.data import SPLIT_STEP, TARGET, TIME_COL, load_clean, time_split
from src.leakage_audit import make_model

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "capacity_results.csv"

BUDGETS = [10, 25, 50, 100, 150, 200]  # alerts per hour; None = no limit


def cutoff_for_budget(score, weight, hours: int, budget: float | None,
                      floor: float = REVIEW_COST) -> float:
    """Lowest cut-off (>= floor) whose weighted alerts per hour fit the budget."""
    score = np.asarray(score, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if budget is None:
        return floor

    order = np.argsort(-score)
    s, cum = score[order], np.cumsum(weight[order]) / hours
    fits = cum <= budget
    if not fits.any():
        return float(s[0]) + 1.0  # budget too small: alert on nothing
    # The k highest scores fit; the cut-off sits just above the next score.
    k = int(np.argmin(fits)) if not fits.all() else len(s)
    cut = float(s[k]) if k < len(s) else floor
    return max(cut, floor)


def run() -> pd.DataFrame:
    df = load_clean()
    train_all, test = time_split(df)
    fit = train_all[train_all[TIME_COL] < VALID_START]
    valid = train_all[train_all[TIME_COL] >= VALID_START]

    valid_hours = SPLIT_STEP - VALID_START + 1
    test_hours = test[TIME_COL].max() - SPLIT_STEP

    y_va, y_te = valid[TARGET].to_numpy(), test[TARGET].to_numpy()
    w_va, w_te = row_weights(y_va), row_weights(y_te)
    amt_te = test["amount"].to_numpy()
    no_alert_cost = amt_te[y_te == 1].sum()

    rows = []
    for name, features in MODEL_SETS.items():
        model = make_model().fit(fit[features], fit[TARGET])
        s_va = to_true_rate(model.predict_proba(valid[features])[:, 1]) * valid["amount"].to_numpy()
        s_te = to_true_rate(model.predict_proba(test[features])[:, 1]) * amt_te

        for budget in BUDGETS + [None]:
            cut = cutoff_for_budget(s_va, w_va, valid_hours, budget)
            out = evaluate(y_te, s_te > cut, amt_te, w_te)
            rows.append({
                "model": name,
                "budget_per_hour": budget if budget is not None else "no limit",
                "cutoff": cut,
                "alerts_per_hour": out["alerts"] / test_hours,
                "frauds_caught": out["frauds_caught"],
                "precision": out["precision"],
                "total_cost": out["total_cost"],
                "naira_saved": no_alert_cost - out["total_cost"],
                "share_of_loss_saved": 1 - out["total_cost"] / no_alert_cost,
            })

    results = pd.DataFrame(rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_PATH, index=False)
    return results


if __name__ == "__main__":
    res = run()
    with pd.option_context("display.width", 200):
        print(res.to_string(index=False, formatters={
            "cutoff": "{:,.0f}".format,
            "alerts_per_hour": "{:,.1f}".format,
            "precision": "{:.1%}".format,
            "total_cost": "{:,.0f}".format,
            "naira_saved": "{:,.0f}".format,
            "share_of_loss_saved": "{:.1%}".format,
        }))
