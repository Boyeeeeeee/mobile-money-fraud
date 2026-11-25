"""Tests for src/data.py.

The cleaning tests use a tiny hand-made frame, so they run without the real CSV.
The last test checks the real file and is skipped if it is not present.
"""

import pandas as pd
import pytest

from src.data import (
    ALL_FEATURES,
    AUTH_FEATURES,
    POST_FEATURES,
    RAW_PATH,
    RULE_FLAG,
    TARGET,
    clean,
    load_clean,
    time_split,
)


@pytest.fixture
def toy():
    return pd.DataFrame(
        {
            "Unnamed: 0.1": [0, 1, 2, 3],
            "Unnamed: 0": [0, 1, 2, 3],
            "step": [30, 1, 5, 2],
            "type": ["TRANSFER", "PAYMENT", "CASH_OUT", "CASH_IN"],
            "amount": [100.0, 50.0, 200.0, 10.0],
            "nameOrig": ["C1", "C2", "C3", "C4"],
            "oldbalanceOrg": [100.0, 60.0, 500.0, 0.0],
            "newbalanceOrig": [0.0, 10.0, 300.0, 10.0],
            "nameDest": ["C5", "M6", "C7", "C8"],
            "oldbalanceDest": [0.0, 0.0, 10.0, 5.0],
            "newbalanceDest": [0.0, 0.0, 210.0, 0.0],
            "isFraud": [1, 0, 0, 0],
            "isFlaggedFraud": [0, 0, 0, 0],
            "date": ["2021-01-01"] * 4,
        }
    )


def test_drops_unused_columns(toy):
    out = clean(toy)
    for col in ["Unnamed: 0.1", "Unnamed: 0", "date"]:
        assert col not in out.columns


def test_keeps_only_transfer_and_cash_out(toy):
    out = clean(toy)
    assert set(out["type"]) == {"TRANSFER", "CASH_OUT"}
    assert len(out) == 2


def test_sorted_by_step_and_features_added(toy):
    out = clean(toy)
    assert out["step"].tolist() == [5, 30]
    assert out["is_transfer"].tolist() == [0, 1]
    assert out["hour"].tolist() == [5, 6]  # 30 % 24 == 6


def test_feature_sets_are_consistent(toy):
    out = clean(toy)
    assert set(ALL_FEATURES) <= set(out.columns)
    assert ALL_FEATURES == AUTH_FEATURES + POST_FEATURES
    # Post-transaction balances must never be in the auth-time set.
    assert not set(POST_FEATURES) & set(AUTH_FEATURES)
    # Label and rule flag are never model inputs.
    assert TARGET not in ALL_FEATURES
    assert RULE_FLAG not in ALL_FEATURES


def test_time_split_has_no_overlap(toy):
    out = clean(toy)
    train, test = time_split(out, split_step=10)
    assert train["step"].max() <= 10 < test["step"].min()
    assert len(train) + len(test) == len(out)


@pytest.mark.skipif(not RAW_PATH.exists(), reason="raw data not present")
def test_real_data_keeps_every_fraud():
    df = load_clean()
    assert len(df) == 51_704
    assert int(df[TARGET].sum()) == 8_213
    assert df[ALL_FEATURES].isna().sum().sum() == 0
