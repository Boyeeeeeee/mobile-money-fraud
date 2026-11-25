# Mobile-Money Fraud Detection: Judged in Naira, Audited for Leakage

A fraud detection project on mobile-money transactions that asks two questions most fraud notebooks skip:

1. **Is the model's score real, or is it leakage?** Some columns are only known *after* a transaction completes. A model that uses them looks near-perfect but could never run at authorisation time.
2. **What is the model worth in money?** Accuracy means little when fraud is rare. This project picks the alert threshold by cost (missed fraud vs. analyst review) and reports results as naira saved, not percentages.

## Dataset

A sample of [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1), a synthetic mobile-money simulation built from real transaction logs.

| | |
|---|---|
| Rows | 108,213 transactions |
| Fraud cases | 8,213 (7.6% of the sample) |
| Fraud rate in full PaySim | ~0.13% |
| Transaction types | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |

The file is not committed. Place it at `data/raw/fraud_data.csv`.

### What an initial look showed

- **Fraud only happens in TRANSFER and CASH_OUT.** The other three types have zero fraud cases.
- **The sample is rebalanced.** All frauds were kept and normal transactions were cut down, so fraud is ~58x more common here than in the full data. Metrics like precision must be re-weighted back to the real base rate before they mean anything.
- **Balance columns leak the label.** In 98% of frauds the amount equals the sender's full balance (the account is drained). About half of frauds leave the receiver's balance at zero before and after, which happens in 0.02% of normal transfers.
- **The `date` column is unreliable.** It does not line up with `step` (the hour of the simulation), so `step` is used as the time axis and `date` is dropped.
- **The existing rule flag is weak.** `isFlaggedFraud` catches 16 of 8,213 frauds. That is the baseline to beat.
- **Account IDs almost never repeat**, so per-account history and network features are not possible with this sample.

## Plan

**Part A — Leakage audit**
- Train the same model on two feature sets: everything, and only what is known at authorisation time.
- Show how much of the "near-perfect" score comes from leaked columns.

**Part B — Cost-based evaluation**
- Re-weight results to the real ~0.13% fraud rate.
- Cost of a missed fraud = the transaction amount; cost of a false alarm = a fixed review cost (e.g. ₦500).
- Choose the threshold that minimises total cost and compare against the `isFlaggedFraud` rule.
- Split by `step` (train on earlier hours, test on later) so the test mimics real deployment.

## Project structure

```
mobile-money-fraud/
├── data/raw/            # fraud_data.csv goes here (git-ignored)
├── src/                 # pipeline scripts
├── tests/               # pytest tests
├── reports/figures/     # generated plots
├── requirements.txt
└── README.md
```

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Status

- [x] Repo scaffold, environment, initial data check
- [ ] Data loading and cleaning
- [ ] Part A: leakage audit
- [ ] Part B: cost-based threshold and naira-saved results
- [ ] Tests
