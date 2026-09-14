import pytest

from beverage_disposal_bringup.grasp_geometry import (
    CALIBRATION_CAN_HEIGHT,
    CALIBRATION_CAN_RADIUS,
    grasp_joint_targets,
    grasp_pose_for_can,
    hover_joint_targets,
    hover_pose_for_can,
)
from beverage_disposal_bringup.kinematics import forward_kinematics, solve_arm_ik

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


# grasp_joint_targets/hover_joint_targets (below) exist to formalize what
# every throwaway pipeline script in commit-notes/10-iteration-3-pick-and-
# lift.md's "closure-stall" investigation had to hand-write inline: compose
# grasp_pose_for_can/hover_pose_for_can with solve_arm_ik to get the arm's
# three pitch-joint values directly, rather than every caller repeating that
# two-step composition (and its clamp_near_limits reasoning -- both the
# calibrated GRASP and HOVER poses sit exactly at wrist_flex's hard limit,
# see grasp_geometry.py's module docstring) itself.
def test_grasp_joint_targets_matches_manual_ik_composition():
    can_r, can_z = CALIBRATION_CAN_RADIUS + 0.008, CALIBRATION_CAN_HEIGHT - 0.006
    expected = solve_arm_ik(*grasp_pose_for_can(can_r, can_z), WRIST_ROLL, elbow_up=False,
                             clamp_near_limits=0.1)
    actual = grasp_joint_targets(can_r, can_z)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_hover_joint_targets_matches_manual_ik_composition():
    can_r, can_z = CALIBRATION_CAN_RADIUS - 0.005, CALIBRATION_CAN_HEIGHT + 0.004
    expected = solve_arm_ik(*hover_pose_for_can(can_r, can_z), WRIST_ROLL, elbow_up=False,
                             clamp_near_limits=0.1)
    actual = hover_joint_targets(can_r, can_z)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_grasp_joint_targets_recovers_known_good_joints_at_calibration_position():
    """Regression guard tying this all the way back to the real, live-sim-
    verified joint tuple this whole module is calibrated from."""
    actual = grasp_joint_targets(CALIBRATION_CAN_RADIUS, CALIBRATION_CAN_HEIGHT)
    assert actual == pytest.approx(KNOWN_GOOD_GRASP_JOINTS, abs=1e-6)


def test_hover_joint_targets_recovers_known_good_joints_at_calibration_position():
    actual = hover_joint_targets(CALIBRATION_CAN_RADIUS, CALIBRATION_CAN_HEIGHT)
    assert actual == pytest.approx(KNOWN_GOOD_HOVER_JOINTS, abs=1e-6)
