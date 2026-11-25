"""Tests for src/cost_eval.py on small hand-made inputs."""

import numpy as np
import pytest

from src.cost_eval import best_threshold, evaluate, row_weights, to_true_rate


def test_row_weights():
    assert row_weights([1, 0, 0], normal_weight=10).tolist() == [1, 10, 10]


def test_to_true_rate_lowers_probability():
    # 50/50 in the sample becomes 1 in 11 when normals are 10x more common.
    assert to_true_rate(np.array([0.5]), normal_weight=10)[0] == pytest.approx(1 / 11)
    assert to_true_rate(np.array([0.0, 1.0]), normal_weight=10).tolist() == [0.0, 1.0]


def test_evaluate_costs():
    y = np.array([1, 1, 0, 0])
    flag = np.array([1, 0, 1, 0])
    amount = np.array([1000.0, 2000.0, 50.0, 50.0])
    weight = row_weights(y, normal_weight=10)

    out = evaluate(y, flag, amount, weight, review_cost=5)
    assert out["alerts"] == 11  # 1 fraud + 1 normal row worth 10
    assert out["frauds_caught"] == 1
    assert out["missed_fraud_cost"] == 2000
    assert out["review_cost"] == 55
    assert out["total_cost"] == 2055


def test_no_alerts_costs_all_fraud():
    y = np.array([1, 0, 1])
    amount = np.array([10.0, 99.0, 30.0])
    out = evaluate(y, np.zeros(3), amount, row_weights(y))
    assert out["total_cost"] == 40
    assert out["review_cost"] == 0


def test_best_threshold_separates_clean_scores():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    amount = np.array([10.0, 10.0, 1e6, 1e6])
    t = best_threshold(y, p, amount, row_weights(y, 10), review_cost=1)
    assert 0.2 < t <= 0.8
