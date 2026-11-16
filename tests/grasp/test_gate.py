"""The gate's measurements on hand-built episodes, and its pass bar at each edge."""

import math

import numpy as np

from beverage_disposal.core.gate import GateBar, GateRow, grip_slip_m, passes, swing_settle_s
from beverage_disposal.core.trajectory import Trajectory

BAR = GateBar(
    max_slip_m=0.002, max_swing_settle_s=2.0, required_places=12, swing_band_rad=0.035, seeds=12
)
STEP = 0.05  # seconds per row


def episode(tip, container, tilt_rad=None):
    """A recorded episode with these tip and container rows; every other signal is quiet."""
    n = len(tip)
    tilt = np.zeros(n) if tilt_rad is None else np.asarray(tilt_rad, dtype=float)
    return Trajectory(
        time=np.arange(n) * STEP,
        tip=np.asarray(tip, dtype=float),
        gripper=np.zeros(n),
        container=np.asarray(container, dtype=float),
        upright=np.cos(tilt),
        speed=np.zeros(n),
        arm_force=np.zeros(n),
        action=np.zeros((n, 4)),
        finished=True,
    )


def row(**changes):
    values = {
        "size": "pet_500ml",
        "fill": "empty",
        "slips": (0.001,) * 12,
        "settle_times": (1.0,) * 12,
        "places_upright": (True,) * 12,
    }
    return GateRow(**{**values, **changes})


def test_a_container_carried_rigidly_with_the_tip_has_not_slipped():
    tip = [[0.1 + 0.01 * i, 0.0, 0.2] for i in range(10)]
    container = [[0.1 + 0.01 * i, 0.0, 0.15] for i in range(10)]
    assert grip_slip_m(episode(tip, container), 2, 9) == 0.0


def test_a_slide_of_3_mm_in_the_grip_is_a_slip_of_3_mm():
    tip = [[0.1, 0.0, 0.2]] * 10
    # It sits still for four rows, then hangs 3 mm lower in the jaws.
    container = [[0.1, 0.0, 0.15]] * 4 + [[0.1, 0.0, 0.147]] * 6
    assert math.isclose(grip_slip_m(episode(tip, container), 2, 9), 0.003, abs_tol=1e-12)


def test_movement_before_the_grip_settled_does_not_count_as_slip():
    tip = [[0.1, 0.0, 0.2]] * 10
    # The jaws centre the container by 5 mm while closing (rows 0 and 1), then hold it.
    container = [[0.105, 0.0, 0.15]] * 2 + [[0.1, 0.0, 0.15]] * 8
    assert grip_slip_m(episode(tip, container), 2, 9) == 0.0


def test_the_slip_is_read_on_the_last_row_before_the_release():
    tip = [[0.1, 0.0, 0.2]] * 10
    # The container drops away only once the jaws have opened, at row 8.
    container = [[0.1, 0.0, 0.15]] * 8 + [[0.1, 0.0, 0.05]] * 2
    assert grip_slip_m(episode(tip, container), 2, 8) == 0.0


def test_the_swing_settles_at_the_last_time_the_tilt_is_outside_the_band():
    tilt = np.zeros(60)
    tilt[10:38] = 0.10  # swinging
    tilt[20:24] = 0.0  # passes through the band once, on the way
    # The carry ends at row 10; the tilt is last outside the band at row 37, 27 rows later, so it
    # has settled one row after that: 28 rows of 0.05 s.
    flat = [[0.0, 0.0, 0.0]] * 60
    assert math.isclose(swing_settle_s(episode(flat, flat, tilt), 10, 0.035), 1.4)


def test_a_container_that_never_swings_settles_at_once():
    flat = [[0.0, 0.0, 0.0]] * 30
    assert swing_settle_s(episode(flat, flat), 10, 0.035) == 0.0


def test_a_swing_that_is_still_going_at_the_end_never_settles():
    tilt = 0.2 * np.sin(np.arange(60))  # leaves the band of its last value on the last rows too
    flat = [[0.0, 0.0, 0.0]] * 60
    assert swing_settle_s(episode(flat, flat, tilt), 10, 0.035) == math.inf


def test_a_run_that_never_got_to_the_carry_has_no_measurement():
    flat = [[0.0, 0.0, 0.0]] * 5
    assert swing_settle_s(episode(flat, flat), 5, 0.035) == math.inf
    assert grip_slip_m(episode(flat, flat), 5, 5) == math.inf


def test_a_row_inside_every_limit_passes():
    assert passes(row(), BAR)


def test_one_place_short_of_twelve_fails():
    assert not passes(row(places_upright=(True,) * 11 + (False,)), BAR)
    assert not passes(
        row(slips=(0.001,) * 11, settle_times=(1.0,) * 11, places_upright=(True,) * 11), BAR
    )


def test_a_slip_or_a_settle_time_at_the_limit_passes_and_just_over_fails():
    assert passes(row(slips=(0.002,) * 12), BAR)
    assert not passes(row(slips=(0.001,) * 11 + (0.0021,)), BAR)
    assert passes(row(settle_times=(2.0,) * 12), BAR)
    assert not passes(row(settle_times=(1.0,) * 11 + (2.01,)), BAR)
