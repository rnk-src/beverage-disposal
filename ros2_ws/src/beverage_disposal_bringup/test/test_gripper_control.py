"""Unit tests for the gripper's contact-vs-free-settle decision logic.

No ROS/Gazebo dependency on purpose -- this is exactly the class of logic
this project's commit-notes/10 identified as needing a real fix (the old
proximity-to-target taper can suppress grip force when a stall happens to
land close to the nominal target), and it's pure math/state, so it's
tested the same way kinematics.py/grasp_geometry.py are: plain pytest,
no sim needed.
"""
import pytest

from beverage_disposal_bringup.gripper_control import (
    GripperCloseController,
    GripperCloseStatus,
    GripperControlParams,
)


def default_params(**overrides):
    values = dict(
        p_gain=10.0,
        i_gain=0.0,
        d_gain=1.2,
        max_effort=5.0,
        goal_tolerance=0.05,
        stall_position_threshold=0.01,
        contact_confirm_time=0.1,
        near_target_threshold=0.15,
        near_target_max_effort=0.05,
        contact_hold_effort=2.0,
    )
    values.update(overrides)
    return GripperControlParams(**values)


def run_steps(controller, positions, dt=0.02):
    result = None
    for position in positions:
        result = controller.step(position, dt)
    return result


def test_starts_in_tracking_status():
    controller = GripperCloseController(target=0.0, params=default_params())
    result = controller.step(position=1.5, dt=0.02)
    assert result.status == GripperCloseStatus.TRACKING


def test_reaches_free_space_target_without_ever_stalling():
    """Smooth, uninterrupted convergence to within goal_tolerance -- the
    free-space open/close case test_gripper.py exercises. Must end
    REACHED_FREE, not CONTACT_DETECTED."""
    controller = GripperCloseController(target=0.0, params=default_params())
    positions = [1.5 - 0.1 * i for i in range(20)]  # 1.5 -> -0.4, real progress every tick
    result = run_steps(controller, positions)
    assert result.status == GripperCloseStatus.REACHED_FREE


def test_detects_contact_when_position_stops_progressing_far_from_target():
    """The joint's position stops changing well short of the commanded
    target and stays there -- the real signature of hitting an object,
    not of arriving somewhere it should settle. Must be recognized as
    contact, not just eventually reported as a plain failure."""
    controller = GripperCloseController(target=0.0, params=default_params())
    positions = [1.5 - 0.1 * i for i in range(6)]  # -> position 0.9, real progress
    positions += [0.9] * 10  # stopped dead, no further progress
    result = run_steps(controller, positions)
    assert result.status == GripperCloseStatus.CONTACT_DETECTED


def test_ignores_instantaneous_velocity_and_uses_real_position_progress():
    """Real bug found live (2026-09-13): gripper_joint's reported /joint_states
    velocity for this effort-controlled joint is unreliable -- observed
    pinned at its declared +-10 rad/s velocity limit on nearly every
    control tick regardless of the joint's *actual* motion (confirmed via
    live position traces showing genuinely smooth, small per-tick position
    changes while /joint_states.velocity read +-10 the whole time). A
    stall criterion based on instantaneous velocity can therefore almost
    never fire reliably for this joint. This is why step() no longer takes
    a velocity argument at all: real *position* progress over
    contact_confirm_time is the only signal used now. This test reproduces
    a real captured live trace (see commit-notes/10) -- position genuinely
    stable (drifting a few thousandths of a radian per tick, consistent
    with real sustained contact) for well over contact_confirm_time,
    which must be detected as contact regardless of what any velocity
    reading would have said."""
    controller = GripperCloseController(target=0.0, params=default_params())
    # Fast real approach, then a long, nearly-flat real plateau (adapted
    # from an actual captured trace: 0.3769, 0.3770, 0.3773, 0.3775,
    # 0.3776, 0.3747, 0.3749, ... drifting by ~0.001-0.003 per tick).
    positions = [1.4 - 0.15 * i for i in range(9)]  # -> ~0.05, real progress
    plateau = [0.3769, 0.3770, 0.3773, 0.3775, 0.3776, 0.3747, 0.3749, 0.3750,
               0.3753, 0.3755, 0.3757, 0.3757, 0.3760, 0.3761, 0.3763, 0.3766]
    result = run_steps(controller, positions + plateau)
    assert result.status == GripperCloseStatus.CONTACT_DETECTED


def test_contact_hold_effort_is_not_suppressed_by_near_target_taper():
    """The exact bug documented in commit-notes/10 (2026-09-10 entry): a
    stall whose position happens to land within near_target_threshold of
    the *original* nominal target must still get full contact_hold_effort,
    not the tiny near_target_max_effort meant for gentle free-space
    settling. Use a stall position within near_target_threshold (0.15) of
    the target (0.0) but outside goal_tolerance (0.05) -- exactly the
    "arrived close, but blocked, not settled" scenario."""
    params = default_params()
    controller = GripperCloseController(target=0.0, params=params)
    positions = [0.5 - 0.05 * i for i in range(8)]  # -> position 0.1, real progress
    positions += [0.1] * 10  # stalled at error=0.1: inside near_target_threshold,
    # outside goal_tolerance -- the taper-zone trap.
    result = run_steps(controller, positions)
    assert result.status == GripperCloseStatus.CONTACT_DETECTED
    assert abs(result.effort) == pytest.approx(
        min(params.contact_hold_effort, params.max_effort))


def test_free_space_taper_still_applies_while_tracking_near_target():
    """Approaching a free-space target normally (never stalling) but
    already within near_target_threshold: still TRACKING (hasn't reached
    goal_tolerance yet), and effort should already be tapered down to
    avoid the overshoot/limit-slam this taper was originally added to fix
    -- unaffected by the contact-detection path."""
    params = default_params()
    controller = GripperCloseController(target=0.0, params=params)
    # First step lands well inside near_target_threshold but outside
    # goal_tolerance -- one cycle isn't enough to trip contact_confirm_time
    # regardless of how it's measured.
    result = controller.step(position=0.1, dt=0.02)
    assert result.status == GripperCloseStatus.TRACKING
    assert abs(result.effort) <= params.near_target_max_effort + 1e-9


def test_contact_status_locks_in_and_keeps_holding():
    """Once contact is detected, later steps (even if the hold effort
    itself produces some small position movement) must never revert to
    TRACKING/REACHED_FREE -- the whole point is a stable, sustained grip,
    not a controller that lets go the moment something wiggles."""
    controller = GripperCloseController(target=0.0, params=default_params())
    positions = [0.9] * 13  # stopped dead immediately, no real progress at all
    run_steps(controller, positions)
    # A little post-contact motion, well away from goal_tolerance -- would
    # look like normal tracking progress if status weren't locked.
    result = controller.step(position=0.88, dt=0.02)
    assert result.status == GripperCloseStatus.CONTACT_DETECTED


def test_large_first_dt_does_not_falsely_trigger_contact():
    """Real bug, found by independently re-running test_gripper.py fresh
    (not hypothesized): the very first control tick after a fresh node/goal
    start can see an anomalously large dt -- e.g. the sim-clock subscription
    catching up right after node startup, before the joint has had any real
    chance to move. Before this fix, one such tick (dt=5.0, far exceeding
    contact_confirm_time=0.1) instantly and permanently locked the
    controller into CONTACT_DETECTED at the starting position, with the
    commanded goal nowhere near reached -- this is exactly what made
    gripper_action_server stop applying any real effort after a single
    tick, reproducing the 5/5 test_gripper.py failures."""
    controller = GripperCloseController(target=1.5, params=default_params())
    result = controller.step(position=0.0, dt=5.0)
    assert result.status == GripperCloseStatus.TRACKING


def test_contact_hold_pushes_in_original_closing_direction():
    """The hold effort must keep squeezing toward the original target
    direction (a real, physically motivated sustained push), not drift
    toward zero or reverse -- regardless of which direction closing
    happens to be for this particular goal."""
    closing_positive = GripperCloseController(target=1.5, params=default_params())
    positions = [0.5 + 0.1 * i for i in range(6)] + [1.1] * 10
    result = run_steps(closing_positive, positions)
    assert result.status == GripperCloseStatus.CONTACT_DETECTED
    assert result.effort > 0

    closing_negative = GripperCloseController(target=0.0, params=default_params())
    positions = [1.5 - 0.1 * i for i in range(6)] + [0.9] * 10
    result = run_steps(closing_negative, positions)
    assert result.status == GripperCloseStatus.CONTACT_DETECTED
    assert result.effort < 0


def test_small_jitter_within_threshold_does_not_reset_the_stall_window():
    """A stalled joint isn't perfectly motionless in reality (see the real
    captured trace in test_ignores_instantaneous_velocity...) -- small
    jitter below stall_position_threshold must still accumulate toward
    contact_confirm_time, not reset the window every tick just because the
    position isn't bit-for-bit identical each time."""
    controller = GripperCloseController(target=0.0, params=default_params())
    positions = [1.5 - 0.1 * i for i in range(6)]  # -> 0.9, real progress
    # Tiny back-and-forth jitter, each step well under stall_position_threshold
    # (0.01) relative to the position a few ticks back.
    jitter = [0.900, 0.902, 0.899, 0.901, 0.903, 0.900, 0.898, 0.901, 0.902, 0.900]
    result = run_steps(controller, positions + jitter)
    assert result.status == GripperCloseStatus.CONTACT_DETECTED
