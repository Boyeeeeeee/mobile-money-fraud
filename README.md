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
- **Balance columns almost give the label away.** In 98% of frauds the amount equals the sender's full balance (the account is drained); no normal transaction does this. About half of frauds leave the receiver's balance at zero before and after, which happens in 0.02% of normal transfers. Part A tests whether this is true leakage or something else.
- **The `date` column is unreliable.** It does not line up with `step` (the hour of the simulation), so `step` is used as the time axis and `date` is dropped.
- **The existing rule flag is weak.** `isFlaggedFraud` catches 16 of 8,213 frauds. That is the baseline to beat.
- **Account IDs almost never repeat**, so per-account history and network features are not possible with this sample.

## Data preparation (`src/data.py`)

- Drops the two old index columns and `date`.
- Keeps only TRANSFER and CASH_OUT, since no other type has fraud. That leaves **51,704 rows with all 8,213 frauds (15.9%)**.
- Sorts by `step` so later splits can follow time order.
- Adds `is_transfer` (1 = TRANSFER, 0 = CASH_OUT) and `hour` (`step % 24`).
- Defines two feature sets for Part A:

| Set | Columns | Meaning |
|---|---|---|
| Auth-time | `amount`, `oldbalanceOrg`, `oldbalanceDest`, `is_transfer`, `hour` | Known when the transaction is requested |
| All | auth-time + `newbalanceOrig`, `newbalanceDest` | Adds balances that only exist after the transaction goes through |

`isFraud` is the label. `isFlaggedFraud` is kept only as the rule baseline and is never a model input.

Run `python -m src.data` to print a summary, and `pytest` to run the tests.

### Train/test split by time

The model trains on the first 300 simulated hours (`step` 1–300) and is tested on the later hours (301–743). It never sees the future.

| | Rows | Frauds | Fraud rate |
|---|---|---|---|
| Train (steps 1–300) | 31,498 | 3,401 | 10.8% |
| Test (steps 301–743) | 20,206 | 4,812 | 23.8% |

The test set has a higher fraud rate because PaySim's normal transaction volume drops sharply after step ~400 while fraud stays steady. Part B re-weights to the real base rate, so this does not distort the cost results.

## Part A results: leakage audit (`src/leakage_audit.py`)

The same LightGBM model (300 trees, fixed seed) was trained on three feature sets and scored on the later test period.

| Feature set | Columns | ROC AUC | PR AUC |
|---|---|---|---|
| All | auth-time + post-transaction balances | 0.9998 | 0.9992 |
| Auth-time | known when the transaction is requested | 0.9987 | 0.9958 |
| Auth-time without sender balance | auth-time minus `oldbalanceOrg` | 0.9439 | 0.9021 |

**What this shows**
- **Leakage in time is small.** Removing the post-transaction balances (`newbalanceOrig`, `newbalanceDest`) barely changes the score (PR AUC 0.9992 → 0.9958).
- **The near-perfect score comes from a simulator shortcut, not leakage.** In PaySim, fraud almost always empties the sender's account and normal transactions never do. The sender's starting balance (`oldbalanceOrg`) is known at authorisation time, so it is fair to use, but this clean split is unlikely in real fraud.
- **Without that shortcut, the model is good but not perfect** (PR AUC 0.90). This is a more realistic picture of what the other signals can do.

Results are saved to `reports/leakage_audit.csv`. Run with `python -m src.leakage_audit`.

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
- [x] Data loading and cleaning
- [x] Part A: leakage audit
- [ ] Part B: cost-based threshold and naira-saved results
- [ ] Tests
