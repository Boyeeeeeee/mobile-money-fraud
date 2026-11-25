"""Make the README figures from the saved result CSVs.

Run (after src.leakage_audit and src.capacity):
    python -m src.figures
Writes reports/figures/leakage_audit.png and reports/figures/capacity.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"
FIG_DIR = REPORTS / "figures"

# Reference palette (light mode).
SERIES = {"auth_time": "#257de0", "auth_no_sender_balance": "#f47230"}
TEXT, TEXT_2 = "#191919", "#666666"
GRID, AXIS = "#e7e7e7", "#c2c2c2"

LABELS = {
    "all": "All features",
    "auth_time": "Auth-time",
    "auth_no_sender_balance": "Auth-time, no sender balance",
}


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_2, labelsize=10)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _title(fig, title, subtitle):
    fig.text(0.02, 0.95, title, fontsize=13, fontweight="bold", color=TEXT, va="top")
    fig.text(0.02, 0.88, subtitle, fontsize=10, color=TEXT_2, va="top")


def leakage_figure():
    df = pd.read_csv(REPORTS / "leakage_audit.csv").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    fig.subplots_adjust(left=0.30, right=0.92, top=0.74, bottom=0.17)
    labels = [LABELS[n] for n in df["feature_set"]]
    ax.barh(labels, df["pr_auc"], height=0.5, color=SERIES["auth_time"])
    for y, v in enumerate(df["pr_auc"]):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=10, color=TEXT)
    ax.set_xlim(0.8, 1.0)
    ax.set_xlabel("PR AUC on test period (x-axis starts at 0.8)", color=TEXT_2)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=TEXT, length=0)
    _style(ax)
    ax.grid(axis="y", visible=False)
    _title(fig, "Removing post-transaction balances barely matters",
           "The big drop comes from removing the sender's balance, a PaySim shortcut")
    fig.savefig(FIG_DIR / "leakage_audit.png", dpi=150, facecolor="white")
    plt.close(fig)


def capacity_figure():
    df = pd.read_csv(REPORTS / "capacity_results.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.80, bottom=0.13)
    for name, color in SERIES.items():
        d = df[df["model"] == name].drop_duplicates(["alerts_per_hour", "frauds_caught"])
        d = d.sort_values("alerts_per_hour")
        y = d["share_of_loss_saved"] * 100
        ax.plot(d["alerts_per_hour"], y, color=color, linewidth=2,
                marker="o", markersize=8, markeredgecolor="white", markeredgewidth=2)
        ax.text(d["alerts_per_hour"].iloc[-1], y.iloc[-1] + 0.45, LABELS[name],
                color=TEXT, fontsize=10, ha="right", va="bottom")
    ax.set_xlim(0, None)
    ax.set_ylim(None, 101)
    ax.set_xlabel("Alerts reviewed per hour (test period)", color=TEXT_2)
    ax.set_ylabel("Fraud loss saved (%)", color=TEXT_2)
    _style(ax)
    _title(fig, "A small review team already saves most of the loss",
           "Each point is one hourly alert budget. The isFlaggedFraud rule saves 0.9%.")
    fig.savefig(FIG_DIR / "capacity.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    leakage_figure()
    capacity_figure()
    print("saved:", *sorted(p.name for p in FIG_DIR.glob("*.png")))
