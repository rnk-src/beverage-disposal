import math

import pytest

from beverage_disposal_bringup.kinematics import forward_kinematics, solve_arm_ik

WRIST_ROLL = 1.5708

# shoulder_lift, elbow_flex, wrist_flex used by the old fold-based grasp
# heuristic's GRASP waypoint -- known, from many prior live-sim trials, to
# put the gripper at a workable grasp position/orientation. Recovering
# these exactly (within numerical tolerance) from their own forward
# kinematics is this module's main correctness check, since it ties the
# new IK back to a configuration this project has actually run in Gazebo.
KNOWN_GOOD_GRASP = (-0.35, -0.35, 1.6)

# A handful of other joint configurations, spanning the arm's working
# range, used to check that forward_kinematics -> solve_arm_ik recovers
# the original angles in general, not just for the one hand-picked case
# above.
SAMPLE_CONFIGURATIONS = [
    (-0.35, -0.35, 1.6),
    (-0.55, -0.55, 1.6),
    (-0.65, -0.24, 0.9),
    (0.0, -0.4, 0.8),
    (-0.9, -1.0, -0.5),
    (0.2, -0.2, 1.3),
]


def test_recovers_known_good_grasp_configuration():
    radius, height, pitch = forward_kinematics(*KNOWN_GOOD_GRASP, WRIST_ROLL)

    shoulder_lift, elbow_flex, wrist_flex = solve_arm_ik(
        radius, height, pitch, WRIST_ROLL, elbow_up=False)

    assert shoulder_lift == pytest.approx(KNOWN_GOOD_GRASP[0], abs=1e-4)
    assert elbow_flex == pytest.approx(KNOWN_GOOD_GRASP[1], abs=1e-4)
    assert wrist_flex == pytest.approx(KNOWN_GOOD_GRASP[2], abs=1e-4)


@pytest.mark.parametrize('shoulder_lift,elbow_flex,wrist_flex', SAMPLE_CONFIGURATIONS)
def test_forward_then_inverse_round_trip(shoulder_lift, elbow_flex, wrist_flex):
    """solve_arm_ik should recover a configuration that reaches the exact
    same (radius, height, pitch) that forward_kinematics computed for the
    original joint values -- not necessarily the same joint values
    themselves, since the elbow-up/elbow-down branches are both valid,
    but the same physical gripper pose."""
    target_radius, target_height, target_pitch = forward_kinematics(
        shoulder_lift, elbow_flex, wrist_flex, WRIST_ROLL)

    recovered = None
    for elbow_up in (True, False):
        try:
            recovered = solve_arm_ik(
                target_radius, target_height, target_pitch, WRIST_ROLL, elbow_up=elbow_up)
            break
        except ValueError:
            continue

    assert recovered is not None, 'neither elbow branch reproduced a reachable solution'
    radius2, height2, pitch2 = forward_kinematics(*recovered, WRIST_ROLL)
    assert radius2 == pytest.approx(target_radius, abs=1e-6)
    assert height2 == pytest.approx(target_height, abs=1e-6)
    assert pitch2 == pytest.approx(target_pitch, abs=1e-6)


def test_known_good_grasp_has_no_vertical_slack_at_constant_pitch():
    """Documents a real, non-obvious fact found while building the actual
    descent path: KNOWN_GOOD_GRASP's wrist_flex is exactly 1.6, this
    arm's hard upper limit, so raising the gripper at that exact radius
    and pitch is never possible, not even by a millimetre -- the descent
    integration has to change pitch (and typically radius) on the way up
    to a hover point, it can't just lift straight up from this pose."""
    target_radius, target_height, target_pitch = forward_kinematics(*KNOWN_GOOD_GRASP, WRIST_ROLL)
    with pytest.raises(ValueError):
        solve_arm_ik(target_radius, target_height + 0.001, target_pitch, WRIST_ROLL, elbow_up=False)


def test_unreachable_target_raises_value_error():
    with pytest.raises(ValueError):
        solve_arm_ik(target_radius=5.0, target_height=5.0, target_pitch=0.0,
                     wrist_roll=WRIST_ROLL)


def test_clamp_near_limits_recovers_small_boundary_overshoot():
    """The real, intended use of clamp_near_limits: a straight Cartesian
    line between HOVER and KNOWN_GOOD_GRASP (both using wrist_flex=1.6)
    bulges a fraction of a degree past that limit partway along the line
    -- clamping should silently fix that, but a genuinely large overshoot
    should still raise."""
    target_radius, target_height, target_pitch = forward_kinematics(*KNOWN_GOOD_GRASP, WRIST_ROLL)

    # Without clamping, a tiny overshoot still raises.
    with pytest.raises(ValueError):
        solve_arm_ik(target_radius, target_height + 0.001, target_pitch, WRIST_ROLL,
                     elbow_up=False, clamp_near_limits=0.0)

    # With enough clamp allowance, the tiny overshoot is absorbed and
    # wrist_flex comes back pinned exactly at its limit.
    _, _, wrist_flex = solve_arm_ik(target_radius, target_height + 0.001, target_pitch,
                                     WRIST_ROLL, elbow_up=False, clamp_near_limits=0.01)
    assert wrist_flex == pytest.approx(1.6, abs=1e-9)

    # A large, genuine overshoot still raises even with clamping allowed.
    with pytest.raises(ValueError):
        solve_arm_ik(target_radius, target_height + 0.10, target_pitch, WRIST_ROLL,
                     elbow_up=False, clamp_near_limits=0.01)


def test_target_requiring_out_of_limit_joint_raises_value_error():
    # Straight out in front at the arm's own height, pointing level --
    # reachable as a bare position, but only via joint angles this arm's
    # real joint limits don't allow.
    with pytest.raises(ValueError):
        solve_arm_ik(target_radius=0.35, target_height=0.30, target_pitch=0.0,
                     wrist_roll=WRIST_ROLL, elbow_up=True)


def test_vertical_descent_at_constant_radius_and_pitch():
    """The actual motivation for this module: a vertical descent should
    be expressible as a constant (radius, pitch) with only height
    varying, and solve_arm_ik should track that line at every step
    without drifting sideways -- unlike the old joint-space heuristic
    this replaces (see commit-notes/10-iteration-3-pick-and-lift.md).

    Deliberately not anchored on KNOWN_GOOD_GRASP: that configuration's
    wrist_flex sits exactly at its 1.6 rad hard limit, leaving zero
    vertical slack at constant pitch (any height change at all requires
    exceeding the limit) -- a real, previously-unnoticed fact about that
    specific pose, not a limitation of the IK. See commit-notes/10-
    iteration-3-pick-and-lift.md for how the actual grasp descent's hover
    point was chosen given that constraint (short answer: it doesn't try
    to hold radius/pitch constant from that exact pose)."""
    target_radius, target_height, target_pitch = forward_kinematics(-0.65, -0.24, 0.9, WRIST_ROLL)
    hover_height = target_height + 0.02

    radii_seen = []
    for step in range(11):
        frac = step / 10
        height = hover_height + (target_height - hover_height) * frac
        shoulder_lift, elbow_flex, wrist_flex = solve_arm_ik(
            target_radius, height, target_pitch, WRIST_ROLL, elbow_up=False)
        radius_at_step, height_at_step, pitch_at_step = forward_kinematics(
            shoulder_lift, elbow_flex, wrist_flex, WRIST_ROLL)
        radii_seen.append(radius_at_step)
        assert radius_at_step == pytest.approx(target_radius, abs=1e-6)
        assert pitch_at_step == pytest.approx(target_pitch, abs=1e-6)
        assert height_at_step == pytest.approx(height, abs=1e-6)

    assert max(radii_seen) - min(radii_seen) < 1e-6
