"""Tests for src/capacity.py."""

import numpy as np

from src.capacity import cutoff_for_budget


def test_no_limit_uses_floor():
    assert cutoff_for_budget([1, 2, 3], [1, 1, 1], hours=1, budget=None, floor=500) == 500


def test_budget_limits_alerts_per_hour():
    score = np.array([9000.0, 8000.0, 7000.0, 6000.0])
    weight = np.ones(4)
    cut = cutoff_for_budget(score, weight, hours=2, budget=1, floor=500)
    # 1 alert/hour over 2 hours = 2 alerts: only 9000 and 8000 pass "> cut".
    assert (score > cut).sum() == 2


def test_cutoff_never_below_floor():
    score = np.array([900.0, 100.0, 50.0])
    cut = cutoff_for_budget(score, np.ones(3), hours=1, budget=100, floor=500)
    assert cut == 500


def test_tiny_budget_alerts_nothing():
    score = np.array([900.0, 800.0])
    weight = np.array([60.0, 60.0])
    cut = cutoff_for_budget(score, weight, hours=1, budget=10, floor=500)
    assert (score > cut).sum() == 0
