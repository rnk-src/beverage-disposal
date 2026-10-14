"""Step-response metrics, checked against responses whose textbook answers are known."""

import numpy as np
import pytest

from beverage_disposal.core.step_response import step_response_metrics

DT = 1e-3
TIMES = np.arange(0, 3, DT)


def first_order(tau):
    return 1 - np.exp(-TIMES / tau)


def second_order(zeta, omega):
    # Standard underdamped unit-step response.
    damped = omega * np.sqrt(1 - zeta**2)
    return 1 - np.exp(-zeta * omega * TIMES) * (
        np.cos(damped * TIMES) + zeta / np.sqrt(1 - zeta**2) * np.sin(damped * TIMES)
    )


def test_a_first_order_response_rises_in_ln9_time_constants_and_never_overshoots():
    tau = 0.1
    metrics = step_response_metrics(TIMES, first_order(tau), start=0.0, target=1.0)
    assert metrics["rise_time"] == pytest.approx(tau * np.log(9), abs=2 * DT)
    assert metrics["overshoot_percent"] == pytest.approx(0, abs=1e-9)


def test_a_first_order_response_settles_within_two_percent_after_ln50_time_constants():
    tau = 0.1
    metrics = step_response_metrics(TIMES, first_order(tau), start=0.0, target=1.0)
    assert metrics["settling_time"] == pytest.approx(tau * np.log(50), abs=2 * DT)
    assert metrics["steady_state_error"] == pytest.approx(0, abs=1e-6)


def test_an_underdamped_second_order_response_overshoots_by_the_textbook_amount():
    # exp(-pi zeta / sqrt(1 - zeta^2)), as a percentage.
    zeta = 0.3
    metrics = step_response_metrics(TIMES, second_order(zeta, 20.0), start=0.0, target=1.0)
    expected = 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2))
    assert metrics["overshoot_percent"] == pytest.approx(expected, abs=0.1)


def test_a_response_that_ends_short_reports_the_missing_part_as_the_steady_state_error():
    # Gravity sag: the joint settles 0.05 of the step short of the command.
    metrics = step_response_metrics(TIMES, 0.95 * first_order(0.05), start=0.0, target=1.0)
    assert metrics["steady_state_error"] == pytest.approx(0.05, abs=1e-6)


def test_a_downward_step_is_measured_like_an_upward_one():
    metrics = step_response_metrics(TIMES, 2.0 - first_order(0.1), start=2.0, target=1.0)
    assert metrics["rise_time"] == pytest.approx(0.1 * np.log(9), abs=2 * DT)
    assert metrics["overshoot_percent"] == pytest.approx(0, abs=1e-9)


def test_the_steady_state_error_is_a_size_in_the_units_of_the_signal_whichever_side_it_misses_on():
    long_by = step_response_metrics(TIMES, 2.0 - first_order(0.1) - 0.02, start=2.0, target=1.0)
    short_by = step_response_metrics(TIMES, 2.0 - first_order(0.1) + 0.02, start=2.0, target=1.0)
    assert long_by["steady_state_error"] == pytest.approx(0.02, abs=1e-6)
    assert short_by["steady_state_error"] == pytest.approx(0.02, abs=1e-6)
