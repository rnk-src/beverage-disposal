"""Numerical IK: a TCP position (and a top-down pitch) to the four joints that move the tip."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from support.planar_ik import planar_solutions
from support.so101 import EXPECTED_RANGES, JOINTS

from beverage_disposal.core.ik import solve_ik
from beverage_disposal.core.kinematics import HOME, fk, jacobian, pitch

POSITION_TOLERANCE = 1e-4  # about 0.1 mm
PITCH_TOLERANCE = 1e-3  # about 0.06 degrees
LIMIT_SLACK = 1e-6


def joint_limits():
    return np.array([EXPECTED_RANGES[joint] for joint in JOINTS[:5]]).T


def poses_near_home(count, seed, spread):
    """Poses a little way from HOME on the four IK joints, kept inside the limits (roll stays 0)."""
    rng = np.random.default_rng(seed)
    low, high = joint_limits()
    poses = np.tile(HOME, (count, 1))
    poses[:, :4] += rng.uniform(-spread, spread, size=(count, 4))
    # A micro-radian inside the limits, so that every pose is one the solver is allowed to reach.
    return np.clip(poses, low + LIMIT_SLACK, high - LIMIT_SLACK)


def test_position_only_ik_reaches_targets_that_forward_kinematics_says_exist():
    # Targets are the tips of known poses, so a solution is certain to exist; fk is the oracle.
    for pose in poses_near_home(100, seed=0, spread=0.5):
        target = fk(pose)[:3, 3]
        result = solve_ik(target, target_pitch=None, start=HOME)
        assert result.converged
        np.testing.assert_allclose(
            fk(result.angles)[:3, 3], target, rtol=0, atol=POSITION_TOLERANCE
        )
        assert result.position_error < POSITION_TOLERANCE


def turn_between(after, before):
    # Pitch is an angle; two readings either side of the +-180 degree seam differ by a small turn.
    difference = after - before
    return np.arctan2(np.sin(difference), np.cos(difference))


def test_ik_also_reaches_the_pitch_of_a_known_pose_when_the_arm_has_the_freedom():
    for pose in poses_near_home(100, seed=1, spread=0.5):
        target, target_pitch = fk(pose)[:3, 3], pitch(pose)
        result = solve_ik(target, target_pitch=target_pitch, start=HOME)
        assert result.converged
        assert abs(turn_between(pitch(result.angles), target_pitch)) < PITCH_TOLERANCE
        assert result.pitch_error < PITCH_TOLERANCE
        np.testing.assert_allclose(
            fk(result.angles)[:3, 3], target, rtol=0, atol=POSITION_TOLERANCE
        )


def top_down_poses(count, seed):
    """Poses with the tool straight down: wrist_flex is whatever makes lift + elbow + flex = 90."""
    rng = np.random.default_rng(seed)
    poses = []
    while len(poses) < count:
        pan, lift, elbow = rng.uniform([-0.8, -0.5, -0.5], [0.8, 0.8, 1.0])
        flex = np.pi / 2 - lift - elbow
        if abs(flex) < np.radians(95):
            poses.append([pan, lift, elbow, flex, 0.0])
    return np.array(poses)


def test_ik_points_the_tool_straight_down_by_default_at_targets_where_that_is_possible():
    for pose in top_down_poses(100, seed=2):
        target = fk(pose)[:3, 3]
        result = solve_ik(target, start=HOME)
        assert result.converged
        assert abs(turn_between(pitch(result.angles), np.pi / 2)) < PITCH_TOLERANCE


targets_anywhere = st.tuples(
    st.floats(-0.6, 0.8), st.floats(-0.6, 0.6), st.floats(-0.2, 0.7), st.floats(-np.pi, np.pi)
)


@settings(max_examples=50, derandomize=True)
@given(target_and_pitch=targets_anywhere)
def test_ik_never_leaves_the_joint_limits_and_never_touches_the_roll(target_and_pitch):
    # Many targets are out of reach or inside the arm's own base: the limits must hold anyway.
    *target, target_pitch = target_and_pitch
    start = HOME.copy()
    start[4] = 0.7
    result = solve_ik(np.array(target), target_pitch=target_pitch, start=start)
    low, high = joint_limits()
    # The MJCF rounds the +-95 degree wrist limit to six figures, hence a micro-radian of slack.
    assert np.all(result.angles >= low - LIMIT_SLACK) and np.all(
        result.angles <= high + LIMIT_SLACK
    )
    assert result.angles[4] == 0.7
    assert np.all(np.isfinite(result.angles))


# The elbow angle that lines the forearm up with the upper arm (a straight, fully stretched arm).
# Link offsets from 1.2's hand work: upper arm (0.028, 0.11257), forearm (0.1349, 0.0052) in x, z.
ELBOW_STRAIGHT = np.arctan2(0.0052, 0.1349) - np.arctan2(0.11257, 0.028)
SHOULDER_ANCHOR = np.array([0.0692345, -0.0182778, 0.1166])


def stretched_arm_and_a_target_beyond_its_reach():
    # Lift and elbow make the arm straight; the wrist angle that puts the tip furthest from the
    # shoulder lines the tool up as well, which is the fully stretched, singular pose.
    flex_angles = np.linspace(-1.65, 1.65, 3301)
    reach = [
        np.linalg.norm(fk(np.array([0.0, 1.0, ELBOW_STRAIGHT, flex, 0.0]))[:3, 3] - SHOULDER_ANCHOR)
        for flex in flex_angles
    ]
    start = np.array([0.0, 1.0, ELBOW_STRAIGHT, flex_angles[int(np.argmax(reach))], 0.0])
    tip = fk(start)[:3, 3]
    outward = (tip - SHOULDER_ANCHOR) / np.linalg.norm(tip - SHOULDER_ANCHOR)
    return start, tip + 0.05 * outward


def test_damping_keeps_a_step_from_a_stretched_arm_small_where_plain_inversion_explodes():
    start, target = stretched_arm_and_a_target_beyond_its_reach()
    plain_step = np.linalg.pinv(jacobian(start)[:3, :4]) @ (target - fk(start)[:3, 3])
    # Precondition: this really is a singular pose (the undamped inverse asks for a huge step).
    assert np.linalg.norm(plain_step) > 10
    # The cap is lifted so that only the damping is being tested. A damped inverse never asks for
    # more than error / (2 * damping): a singular value s gives s / (s^2 + d^2), at most 1 / (2d).
    damping = 0.05
    error = np.linalg.norm(target - fk(start)[:3, 3])
    result = solve_ik(
        target, target_pitch=None, start=start, max_iterations=1, max_step=100.0, damping=damping
    )
    assert np.all(np.isfinite(result.angles))
    assert np.linalg.norm(result.angles - start) <= error / (2 * damping) + 1e-9


@pytest.mark.parametrize("max_step", [0.2, 0.05])
def test_one_iteration_never_moves_the_joints_further_than_the_step_cap(max_step):
    start = HOME
    target = fk(np.array([0.9, 0.6, 0.6, 0.0, 0.0]))[:3, 3]
    uncapped = solve_ik(target, target_pitch=None, start=start, max_iterations=1, max_step=100.0)
    # Precondition: the uncapped step is bigger than the cap, so the cap has something to do.
    assert np.linalg.norm(uncapped.angles - start) > max_step
    capped = solve_ik(target, target_pitch=None, start=start, max_iterations=1, max_step=max_step)
    assert np.linalg.norm(capped.angles - start) <= max_step + 1e-12


def test_an_unreachable_target_gives_the_best_effort_and_converged_false_not_an_exception():
    target = np.array([1.0, 0.0, 0.2])
    start_error = np.linalg.norm(target - fk(HOME)[:3, 3])
    result = solve_ik(target, target_pitch=None, start=HOME)
    assert not result.converged
    assert 0 < result.position_error < start_error
    assert np.all(np.isfinite(result.angles))


def test_ik_is_deterministic():
    target = fk(poses_near_home(1, seed=9, spread=0.5)[0])[:3, 3]
    first, second = solve_ik(target, start=HOME), solve_ik(target, start=HOME)
    np.testing.assert_array_equal(first.angles, second.angles)


def test_position_has_strict_priority_over_a_pitch_the_arm_cannot_reach():
    # A tool pointing straight back is out of reach at these tips. The position must still be met,
    # to the same accuracy as when the pitch task is off.
    for pose in poses_near_home(25, seed=10, spread=0.5):
        target = fk(pose)[:3, 3]
        with_pitch = solve_ik(target, target_pitch=-np.pi, start=HOME)
        without_pitch = solve_ik(target, target_pitch=None, start=HOME)
        assert with_pitch.position_error < POSITION_TOLERANCE
        assert without_pitch.position_error < POSITION_TOLERANCE
        assert with_pitch.pitch_error > 0.5
        assert not with_pitch.converged


def test_the_solution_stays_near_the_start_for_a_nearby_target():
    # Warm starting keeps the arm on its elbow configuration: a small move asks for a small turn.
    # The starts are spread over the workspace, away from HOME, so that a solver which forgot its
    # start (and began from HOME) would end up somewhere else.
    rng = np.random.default_rng(11)
    low, high = joint_limits()
    middle, half = (low + high) / 2, (high - low) / 2
    for _ in range(100):
        start = middle + rng.uniform(-0.6, 0.6, 5) * half
        start[4] = 0.0
        moved = start.copy()
        moved[:4] += rng.uniform(-0.2, 0.2, 4)
        # A target outside the joint limits would not be reachable, whatever the solver does.
        moved = np.clip(moved, low + LIMIT_SLACK, high - LIMIT_SLACK)
        result = solve_ik(fk(moved)[:3, 3], target_pitch=pitch(moved), start=start)
        assert result.converged
        assert np.max(np.abs(result.angles - start)) < 0.6


def test_ik_matches_the_closed_form_planar_solution_in_the_arms_plane():
    for pose in poses_near_home(50, seed=13, spread=0.5):
        pose[0] = 0.0
        tip = fk(pose)[:3, 3]
        expected, _ = planar_solutions(tip[[0, 2]], pitch(pose), HOME)
        result = solve_ik(
            tip,
            target_pitch=pitch(pose),
            start=HOME,
            max_iterations=300,
            damping=1e-4,
            position_tolerance=1e-10,
            pitch_tolerance=1e-10,
        )
        assert result.converged
        np.testing.assert_allclose(result.angles[1:4], expected, rtol=0, atol=1e-6)
        assert abs(result.angles[0]) < 1e-6


def test_a_target_the_arm_is_already_at_takes_no_iterations():
    result = solve_ik(fk(HOME)[:3, 3], start=HOME)
    assert result.converged
    assert result.iterations == 0
    np.testing.assert_array_equal(result.angles, HOME)


def test_the_iteration_count_is_reported_and_stays_small_for_the_moves_the_policy_will_ask_for():
    # Measured over 300 targets each: at most 5 iterations near home, at most 12 for top-down
    # poses spread over the workspace. 20 leaves room without hiding a convergence regression.
    for pose in top_down_poses(50, seed=14):
        result = solve_ik(fk(pose)[:3, 3], start=HOME)
        assert result.converged
        assert 0 < result.iterations <= 20


def test_the_iteration_count_stops_at_the_cap_when_the_target_cannot_be_reached():
    result = solve_ik(np.array([1.0, 0.0, 0.2]), target_pitch=None, max_iterations=7)
    assert not result.converged
    assert result.iterations == 7
