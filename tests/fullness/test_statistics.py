"""The Wilson interval: how a count of failures becomes a bound that can be stated."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from beverage_disposal.core.statistics import wilson_interval, wilson_upper


def test_no_failures_in_381_trials_bounds_the_rate_just_under_one_percent():
    # z^2 / (n + z^2) with z = 1.96: 3.8416 / 384.8416, worked by hand.
    assert wilson_upper(0, 381) == pytest.approx(3.8416 / 384.8416, abs=1e-9)
    assert wilson_upper(0, 381) < 0.01


def test_no_failures_in_100_trials_only_bounds_the_rate_at_3_7_percent():
    assert wilson_upper(0, 100) == pytest.approx(0.036995, abs=1e-5)


def test_five_failures_in_100_trials_hand_worked():
    # centre (0.05 + 1.9208/100) / 1.038416; half-width 1.96 * sqrt(0.000475 + 0.00009604) / 1.0384.
    low, high = wilson_interval(5, 100)
    assert (low, high) == pytest.approx((0.02158, 0.11174), abs=2e-4)


def test_a_wider_confidence_asks_for_a_higher_bound():
    assert wilson_upper(0, 100, z=2.576) > wilson_upper(0, 100, z=1.96)


@given(failures=st.integers(0, 200), extra=st.integers(0, 2000))
def test_the_interval_contains_the_observed_rate_and_stays_within_zero_and_one(failures, extra):
    trials = failures + extra + 1
    low, high = wilson_interval(failures, trials)
    assert 0.0 <= low <= failures / trials <= high <= 1.0


@given(failures=st.integers(0, 99))
def test_one_more_failure_never_lowers_the_upper_bound(failures):
    assert wilson_upper(failures + 1, 100) >= wilson_upper(failures, 100)


def test_impossible_counts_are_rejected():
    with pytest.raises(ValueError, match="trials"):
        wilson_upper(0, 0)
    with pytest.raises(ValueError, match="failures"):
        wilson_upper(5, 3)
    with pytest.raises(ValueError, match="failures"):
        wilson_upper(-1, 3)
