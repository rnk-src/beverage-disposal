import math

import pytest

from beverage_disposal_bringup.bin_geometry import (
    BIN_AZIMUTH,
    BIN_MODEL_X,
    BIN_MODEL_Y,
    BIN_PITCH,
    BIN_RADIUS,
    BIN_RELEASE_HEIGHT_M,
    BIN_SUCCESS_X_RANGE,
    BIN_SUCCESS_Y_RANGE,
    BIN_SUCCESS_Z_RANGE,
    BIN_WALL_TOP_Z,
    bin_hover_joint_targets,
    bin_release_joint_targets,
)
from beverage_disposal_bringup.grasp_geometry import WRIST_ROLL
from beverage_disposal_bringup.kinematics import solve_arm_ik


def test_bin_azimuth_and_radius_match_world_file_pose():
    """A literal check that BIN_AZIMUTH/BIN_RADIUS reconstruct from
    BIN_MODEL_X/BIN_MODEL_Y -- these have to be kept in sync by hand with
    the world file's real bin <pose> (same documented gap class as
    can_mass_kg between spawn_arm.launch.py and pick_and_lift.py), so a
    future edit to one without the other should be caught here."""
    assert BIN_AZIMUTH == pytest.approx(-math.atan2(BIN_MODEL_Y, BIN_MODEL_X))
    assert BIN_RADIUS == pytest.approx(math.hypot(BIN_MODEL_X, BIN_MODEL_Y))


def test_bin_hover_joint_targets_matches_manual_ik_composition():
    expected = solve_arm_ik(BIN_RADIUS, 0.33, BIN_PITCH, WRIST_ROLL, elbow_up=False)
    actual = bin_hover_joint_targets()
    assert actual == pytest.approx(expected, abs=1e-12)


def test_bin_release_joint_targets_matches_manual_ik_composition():
    expected = solve_arm_ik(
        BIN_RADIUS, BIN_RELEASE_HEIGHT_M, BIN_PITCH, WRIST_ROLL, elbow_up=False)
    actual = bin_release_joint_targets()
    assert actual == pytest.approx(expected, abs=1e-12)


def test_bin_hover_and_release_are_reachable():
    """Real regression guard: today both are reachable with real margin
    from every joint limit (see CLAUDE.md's Iteration 5 entry), but a
    future change to BIN_HOVER_HEIGHT_M/BIN_RELEASE_HEIGHT_M/BIN_PITCH
    should be caught here before it ever reaches live sim -- solve_arm_ik
    raises ValueError on an unreachable target."""
    bin_hover_joint_targets()
    bin_release_joint_targets()


def test_bin_success_ranges_are_well_ordered_and_inside_the_bin():
    """Sanity guard on the literal success bounds themselves: each range's
    low bound must actually be below its high bound (a swapped pair would
    silently make every check fail), and the whole box must sit strictly
    inside the bin's real interior span and height band, not just
    overlap it."""
    for lo, hi in (BIN_SUCCESS_X_RANGE, BIN_SUCCESS_Y_RANGE, BIN_SUCCESS_Z_RANGE):
        assert lo < hi

    assert BIN_SUCCESS_Z_RANGE[1] < BIN_WALL_TOP_Z
    assert BIN_SUCCESS_Z_RANGE[0] > 0.0
