"""Run the whole pipeline: Part A, Part B, capacity scenario, figures.

Run:
    python -m src.run_all
"""

import pandas as pd

from src import capacity, cost_eval, figures, leakage_audit

pd.set_option("display.float_format", "{:,.3f}".format)
pd.set_option("display.width", 200)


def main() -> None:
    print("1/4 Part A: leakage audit")
    print(leakage_audit.run().to_string(index=False), "\n")

    print("2/4 Part B: cost in naira")
    res = cost_eval.run()
    print(res[["strategy", "frauds_caught", "total_cost", "naira_saved"]].to_string(index=False), "\n")

    print("3/4 Capacity scenario")
    cap = capacity.run()
    print(cap[["model", "budget_per_hour", "alerts_per_hour", "share_of_loss_saved"]]
          .to_string(index=False), "\n")

    print("4/4 Figures")
    figures.FIG_DIR.mkdir(parents=True, exist_ok=True)
    figures.leakage_figure()
    figures.capacity_figure()
    print("saved to", figures.FIG_DIR)


if __name__ == "__main__":
    main()
