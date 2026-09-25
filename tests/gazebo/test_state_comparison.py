"""How two Gazebo state readouts are compared: exact counts, approximate floats, same fields."""

import copy

from states import same_state

STATE = {"iterations": 100, "sim_time": 0.1, "poses": {"ball": [0.0, 0.0, 9.9504595]}}


def changed(**edits):
    state = copy.deepcopy(STATE)
    state.update(edits)
    return state


def test_identical_states_are_the_same():
    assert same_state(STATE, copy.deepcopy(STATE))


def test_a_last_bit_difference_in_a_position_is_the_same():
    # 1e-9 relative is far below pytest.approx's default 1e-6 and above double-precision rounding.
    nudged = changed(poses={"ball": [0.0, 0.0, 9.9504595 * (1 + 1e-9)]})
    assert same_state(STATE, nudged)


def test_a_visible_difference_in_a_position_is_not_the_same():
    assert not same_state(STATE, changed(poses={"ball": [0.0, 0.0, 9.9504595 * (1 + 1e-4)]}))


def test_a_different_iteration_count_is_not_the_same_however_close_the_times():
    assert not same_state(STATE, changed(iterations=101))


def test_an_extra_field_is_not_the_same():
    assert not same_state(STATE, changed(wall=123.4))


def test_a_field_that_differs_between_two_runs_is_not_the_same():
    # The fault suite found this: a non-deterministic field present in both readouts got through.
    assert not same_state(changed(wall=1.0), changed(wall=2.0))


def test_an_extra_field_that_matches_is_the_same():
    assert same_state(changed(wall=1.0), changed(wall=1.0))


def test_a_missing_model_is_not_the_same():
    assert not same_state(STATE, changed(poses={}))
