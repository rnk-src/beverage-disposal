import pytest

from beverage_disposal_bringup.grasp_geometry import (
    CALIBRATION_CAN_HEIGHT,
    CALIBRATION_CAN_RADIUS,
    grasp_pose_for_can,
    hover_pose_for_can,
)
from beverage_disposal_bringup.kinematics import forward_kinematics

WRIST_ROLL = 1.5708

# This iteration's own empirically-verified, live-sim-proven poses (see
# commit-notes/10-iteration-3-pick-and-lift.md) -- the ground truth these
# functions are calibrated against and must reproduce exactly at the
# calibration can position.
KNOWN_GOOD_GRASP_JOINTS = (-0.35, -0.35, 1.6)
KNOWN_GOOD_HOVER_JOINTS = (-0.55, -0.55, 1.6)


def test_grasp_pose_recovers_known_good_target_at_calibration_position():
    """At the exact can position this module's offsets were calibrated
    against, grasp_pose_for_can must reproduce the known-good grasp's real
    (radius, height, pitch) -- not approximately close, exactly, since the
    offsets are *defined* as the difference between that pose and the
    calibration can position."""
    expected = forward_kinematics(*KNOWN_GOOD_GRASP_JOINTS, WRIST_ROLL)
    actual = grasp_pose_for_can(CALIBRATION_CAN_RADIUS, CALIBRATION_CAN_HEIGHT)
    for a, e in zip(actual, expected):
        assert a == pytest.approx(e, abs=1e-9)


def test_hover_pose_recovers_known_good_target_at_calibration_position():
    expected = forward_kinematics(*KNOWN_GOOD_HOVER_JOINTS, WRIST_ROLL)
    actual = hover_pose_for_can(CALIBRATION_CAN_RADIUS, CALIBRATION_CAN_HEIGHT)
    for a, e in zip(actual, expected):
        assert a == pytest.approx(e, abs=1e-9)


@pytest.mark.parametrize('delta_radius,delta_height', [
    (0.01, 0.0), (-0.01, 0.0), (0.0, 0.01), (0.0, -0.01), (0.008, -0.006),
])
def test_grasp_pose_tracks_the_cans_actual_measured_position(delta_radius, delta_height):
    """The entire reason this module exists (see its docstring and
    commit-notes/10-iteration-3-pick-and-lift.md's SAFE_RAISE-disturbance
    finding): if the can ends up somewhere other than its original spawn
    position -- which real trials show it reliably does, by a small,
    run-to-run-variable amount -- the grasp target must shift by exactly
    the same amount, not stay pinned to the original calibration position.
    A fixed joint-angle constant (what this replaces) cannot do this by
    construction; this checks the replacement actually does."""
    baseline = grasp_pose_for_can(CALIBRATION_CAN_RADIUS, CALIBRATION_CAN_HEIGHT)
    shifted = grasp_pose_for_can(
        CALIBRATION_CAN_RADIUS + delta_radius, CALIBRATION_CAN_HEIGHT + delta_height)

    assert shifted[0] == pytest.approx(baseline[0] + delta_radius, abs=1e-9)
    assert shifted[1] == pytest.approx(baseline[1] + delta_height, abs=1e-9)
    assert shifted[2] == pytest.approx(baseline[2], abs=1e-9)  # pitch is can-position-independent


def test_hover_pose_tracks_the_cans_actual_measured_position():
    baseline = hover_pose_for_can(CALIBRATION_CAN_RADIUS, CALIBRATION_CAN_HEIGHT)
    shifted = hover_pose_for_can(CALIBRATION_CAN_RADIUS + 0.015, CALIBRATION_CAN_HEIGHT - 0.01)

    assert shifted[0] == pytest.approx(baseline[0] + 0.015, abs=1e-9)
    assert shifted[1] == pytest.approx(baseline[1] - 0.01, abs=1e-9)
    assert shifted[2] == pytest.approx(baseline[2], abs=1e-9)


def test_grasp_and_hover_poses_are_reachable_via_solve_arm_ik():
    """Both targets must actually be usable, not just internally
    consistent -- i.e. solve_arm_ik must accept them (this is really a
    regression guard: forward_kinematics(*JOINTS, ...) is trivially always
    reachable by construction, but a future change to the offsets/pitch
    constants could produce a target outside this arm's real workspace)."""
    from beverage_disposal_bringup.kinematics import solve_arm_ik

    grasp = grasp_pose_for_can(CALIBRATION_CAN_RADIUS, CALIBRATION_CAN_HEIGHT)
    hover = hover_pose_for_can(CALIBRATION_CAN_RADIUS, CALIBRATION_CAN_HEIGHT)
    solve_arm_ik(*grasp, WRIST_ROLL, elbow_up=False)
    solve_arm_ik(*hover, WRIST_ROLL, elbow_up=False)
