import math
import random

import pytest

from beverage_disposal_bringup.kinematics import (
    ELBOW_FLEX_LIMITS,
    SHOULDER_LIFT_LIMITS,
    WRIST_FLEX_LIMITS,
    forward_kinematics,
    solve_arm_ik,
)

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


# --- Hardening: broader edge-case coverage -----------------------------
#
# The tests above (recovering a known-good pose, a handful of hand-picked
# round trips, the one clamp/reachability case each) proved the module
# isn't obviously wrong. The tests below go further, the way a real
# robotics IK implementation earns trust: many random configurations
# instead of a few hand-picked ones, explicit checks at the exact edges of
# the reachable workspace (not just "clearly outside"), an independent
# second solving method to catch algebra mistakes unit tests alone could
# share a blind spot with, and an explicit, justified choice between the
# two mathematically valid elbow solutions.

def _numeric_ik(target_radius, target_height, target_pitch, wrist_roll, initial_guess):
    """A from-scratch, independent way to solve the same problem, used only
    to cross-check solve_arm_ik's closed-form answer -- if the analytical
    derivation had an algebra mistake, there's a real chance a unit test
    built with the same mental model as the implementation would share the
    same blind spot, so an unrelated method (numeric Newton-Raphson via a
    finite-difference Jacobian, using forward_kinematics as the only
    "ground truth") is a meaningfully independent check, not a duplicate
    of the analytical proof.
    """
    x = list(initial_guess)
    for _ in range(100):
        r, h, p = forward_kinematics(*x, wrist_roll)
        residual = (target_radius - r, target_height - h, target_pitch - p)
        if max(abs(v) for v in residual) < 1e-9:
            return tuple(x)
        eps = 1e-6
        jac = []
        for i in range(3):
            xp = list(x)
            xp[i] += eps
            rp, hp, pp = forward_kinematics(*xp, wrist_roll)
            jac.append(((rp - r) / eps, (hp - h) / eps, (pp - p) / eps))
        # Solve the 3x3 system jac^T * delta = residual via Cramer's rule
        # (small and fixed-size -- not worth a numpy dependency this
        # project otherwise avoids).
        m = [[jac[0][k], jac[1][k], jac[2][k]] for k in range(3)]

        def det3(a):
            return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                    - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                    + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))

        d = det3(m)
        if abs(d) < 1e-12:
            break
        delta = []
        for col in range(3):
            m2 = [row[:] for row in m]
            for row in range(3):
                m2[row][col] = residual[row]
            delta.append(det3(m2) / d)
        x = [x[i] + delta[i] for i in range(3)]
    raise AssertionError('numeric IK did not converge')


@pytest.mark.parametrize('shoulder_lift,elbow_flex,wrist_flex', [
    (-0.35, -0.35, 1.6), (-0.5, -0.6, 1.2), (-0.2, -0.8, 0.5),
    (0.3, -0.3, 1.0), (-1.0, -0.9, -0.3),
])
def test_analytical_ik_agrees_with_independent_numeric_solver(
        shoulder_lift, elbow_flex, wrist_flex):
    """Cross-validates solve_arm_ik against a from-scratch numeric solver
    that shares none of its derivation -- a real algebra mistake in the
    closed-form solution would have to coincidentally also satisfy Newton-
    Raphson's independent iteration to slip through both, which is a much
    stronger guarantee than either method alone."""
    target_radius, target_height, target_pitch = forward_kinematics(
        shoulder_lift, elbow_flex, wrist_flex, WRIST_ROLL)

    analytical = solve_arm_ik(target_radius, target_height, target_pitch, WRIST_ROLL, elbow_up=False)
    numeric = _numeric_ik(target_radius, target_height, target_pitch, WRIST_ROLL,
                           initial_guess=analytical)

    # 1e-4 tolerance, not 1e-6: the numeric solver's own finite-difference
    # Jacobian and fixed iteration count have real, expected precision
    # limits of their own -- this comparison is meant to catch the
    # analytical solution being *wrong* (an algebra mistake would show up
    # as an error orders of magnitude larger than this), not to demand
    # agreement tighter than the reference method's own precision.
    for a, n in zip(analytical, numeric):
        assert a == pytest.approx(n, abs=1e-4)


def test_elbow_up_and_elbow_down_are_distinct_and_both_valid():
    """The law-of-cosines elbow bend has two mathematically valid roots
    (elbow_up=True/False) -- this test proves both branches are actually
    implemented (not one silently aliasing the other), give different
    joint values for the same physical point, and both independently
    satisfy forward kinematics. This project always calls with
    elbow_up=False for real grasp/motion planning (see
    test_recovers_known_good_grasp_configuration and every live-sim-
    verified pose in commit-notes/10) because that is the branch that
    reproduces this arm's actual, empirically-verified working
    configuration -- elbow_up=True is mathematically valid but was never
    the physical configuration this arm has actually run in Gazebo, so
    using it for real paths would be unverified territory, not just an
    arbitrary preference.

    Not every reachable target has both branches within this arm's real
    joint limits (a random-sampling check found only about 1 in 5 do --
    see test_random_forward_then_inverse_round_trip_across_full_joint_range
    -- and KNOWN_GOOD_GRASP itself is one of the 4 in 5 that elbow_up=True
    cannot reach within limits), so this test deliberately uses a target
    confirmed by that same sampling to admit both, rather than assuming
    any arbitrary reachable target will."""
    target_radius, target_height, target_pitch = forward_kinematics(
        0.9155305926326398, -1.6831974476759124, -0.1747609790246354, WRIST_ROLL)

    down = solve_arm_ik(target_radius, target_height, target_pitch, WRIST_ROLL, elbow_up=False)
    up = solve_arm_ik(target_radius, target_height, target_pitch, WRIST_ROLL, elbow_up=True)

    assert down != pytest.approx(up, abs=1e-3)
    for solution in (down, up):
        r, h, p = forward_kinematics(*solution, WRIST_ROLL)
        assert r == pytest.approx(target_radius, abs=1e-6)
        assert h == pytest.approx(target_height, abs=1e-6)
        assert p == pytest.approx(target_pitch, abs=1e-6)


def test_boundary_reach_exactly_at_max_wrist_distance_is_solvable():
    """The reachability check (r > l1+l2 or r < abs(l1-l2): raise) uses a
    small tolerance around the true limits (see reach_tol in
    solve_arm_ik), so a target sitting exactly at full extension -- right
    at the edge of a mathematically singular configuration (cos_gamma ==
    -1 exactly) and the classic place a naive IK implementation divides by
    zero or raises an acos() domain error -- should solve cleanly, not
    spuriously raise from floating-point roundoff in how the boundary
    target itself was computed. Neither elbow branch should behave
    differently here since both converge to the same fully-extended
    configuration at this exact boundary."""
    p_shoulder_r, l1, l2, l3 = _reference_link_lengths()

    for elbow_up in (False, True):
        target_pitch = 0.3
        target_radius = p_shoulder_r[0] + (l1 + l2 + l3 * math.cos(target_pitch))
        target_height = p_shoulder_r[2] + l3 * math.sin(target_pitch)
        shoulder_lift, elbow_flex, wrist_flex = solve_arm_ik(
            target_radius, target_height, target_pitch, WRIST_ROLL, elbow_up=elbow_up)
        r, h, p = forward_kinematics(shoulder_lift, elbow_flex, wrist_flex, WRIST_ROLL)
        assert r == pytest.approx(target_radius, abs=1e-6)
        assert h == pytest.approx(target_height, abs=1e-6)


def test_boundary_reach_exactly_at_min_wrist_distance_correctly_rejected():
    """The opposite boundary (link1 folded almost back onto link2, wrist
    center at its minimum possible distance from the shoulder) passes the
    simple *position* reachability check, but a broad scan (every 0.1 rad
    of pitch from -1.5 to 1.5, both elbow branches) found this arm's real
    joint limits reject every single one of them -- reaching the position-
    only minimum requires relative joint angles more extreme than
    shoulder_lift/elbow_flex actually allow. This documents that as a real
    property of this arm (position-only reachability is necessarily a
    superset of what's actually achievable once joint limits and a
    required orientation are added, not a bug), and pins down that the
    module fails for the *right* reason here -- a joint-limit ValueError,
    not a reach-distance one, proving it got as far as actually solving
    the geometry rather than rejecting the position prematurely."""
    p_shoulder_r, l1, l2, l3 = _reference_link_lengths()
    target_pitch = 0.3
    target_radius = p_shoulder_r[0] + (abs(l1 - l2) + l3 * math.cos(target_pitch))
    target_height = p_shoulder_r[2] + l3 * math.sin(target_pitch)

    with pytest.raises(ValueError, match='joint limit'):
        solve_arm_ik(target_radius, target_height, target_pitch, WRIST_ROLL, elbow_up=False)


def _reference_link_lengths():
    from beverage_disposal_bringup.kinematics import _link_geometry
    p_shoulder, l1, l2, l3, *_ = _link_geometry(WRIST_ROLL)
    return p_shoulder, l1, l2, l3


def test_just_beyond_max_reach_raises():
    _, l1, l2, l3 = _reference_link_lengths()
    p_shoulder = _reference_link_lengths()[0]
    target_pitch = 0.3
    # 5mm past the true boundary -- clearly outside, not a boundary-
    # rounding false positive.
    wrist_dist = l1 + l2 + 0.005
    target_radius = p_shoulder[0] + wrist_dist + l3 * math.cos(target_pitch)
    target_height = p_shoulder[2] + l3 * math.sin(target_pitch)
    with pytest.raises(ValueError):
        solve_arm_ik(target_radius, target_height, target_pitch, WRIST_ROLL)


def test_just_inside_min_reach_raises():
    _, l1, l2, l3 = _reference_link_lengths()
    p_shoulder = _reference_link_lengths()[0]
    target_pitch = 0.3
    wrist_dist = max(abs(l1 - l2) - 0.005, 0.0)
    target_radius = p_shoulder[0] + wrist_dist + l3 * math.cos(target_pitch)
    target_height = p_shoulder[2] + l3 * math.sin(target_pitch)
    with pytest.raises(ValueError):
        solve_arm_ik(target_radius, target_height, target_pitch, WRIST_ROLL)


def test_random_forward_then_inverse_round_trip_across_full_joint_range():
    """The hand-picked SAMPLE_CONFIGURATIONS above happen to all be
    reachable and well clear of any limit. This test samples many more
    configurations at random across each joint's *actual, full* mechanical
    range (not a hand-picked comfortable subset) to check the round trip
    holds broadly across the real workspace, not just at a few points
    someone happened to try. Seeded for reproducibility -- a random-
    looking failure should be exactly reproducible when investigated, not
    a new roll of the dice each run."""
    rng = random.Random(20260910)
    checked = 0
    for _ in range(200):
        shoulder_lift = rng.uniform(*SHOULDER_LIFT_LIMITS)
        elbow_flex = rng.uniform(*ELBOW_FLEX_LIMITS)
        wrist_flex = rng.uniform(*WRIST_FLEX_LIMITS)
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
        if recovered is None:
            # A randomly-sampled joint configuration can land at a pose
            # that's only reachable via joint values *outside* this arm's
            # own limits from the other elbow branch -- not every random
            # sample is guaranteed invertible, so this is allowed, just
            # not silently ignored: track how many actually got checked.
            continue

        r2, h2, p2 = forward_kinematics(*recovered, WRIST_ROLL)
        assert r2 == pytest.approx(target_radius, abs=1e-6)
        assert h2 == pytest.approx(target_height, abs=1e-6)
        assert p2 == pytest.approx(target_pitch, abs=1e-6)
        checked += 1

    # Guards against a vacuous pass (e.g. a bug that made every solve
    # raise, which would make the loop above "pass" 200 times over doing
    # nothing).
    assert checked > 150


def test_zero_and_negative_radius_targets_are_rejected_not_miscomputed():
    """radius is a physical distance from the shoulder_pan axis -- zero or
    negative isn't a valid arm target (it's behind or on top of the arm's
    own rotation axis) and should be rejected the same way any other
    unreachable target is, not produce a silently-wrong answer from
    unexpected trig signs."""
    for target_radius in (0.0, -0.1):
        with pytest.raises(ValueError):
            solve_arm_ik(target_radius, target_height=0.25, target_pitch=0.0, wrist_roll=WRIST_ROLL)
