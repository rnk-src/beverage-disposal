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
        stall_velocity_threshold=0.001,
        contact_confirm_time=0.1,
        near_target_threshold=0.15,
        near_target_max_effort=0.05,
        contact_hold_effort=2.0,
    )
    values.update(overrides)
    return GripperControlParams(**values)


def run_steps(controller, steps, dt=0.02):
    """steps: list of (position, velocity) pairs, one per control cycle."""
    result = None
    for position, velocity in steps:
        result = controller.step(position, velocity, dt)
    return result


def test_starts_in_tracking_status():
    controller = GripperCloseController(target=0.0, params=default_params())
    result = controller.step(position=1.5, velocity=-2.0, dt=0.02)
    assert result.status == GripperCloseStatus.TRACKING


def test_reaches_free_space_target_without_ever_stalling():
    """Smooth, uninterrupted convergence to within goal_tolerance, with
    real nonzero velocity the whole way (nothing resisting it) -- the
    free-space open/close case test_gripper.py exercises. Must end
    REACHED_FREE, not CONTACT_DETECTED."""
    controller = GripperCloseController(target=0.0, params=default_params())
    steps = [(1.5 - 0.1 * i, -0.5) for i in range(20)]  # 1.5 -> -0.4, never stalls
    result = run_steps(controller, steps)
    assert result.status == GripperCloseStatus.REACHED_FREE


def test_detects_contact_when_stalled_far_from_target():
    """The joint stops dead (velocity ~0) well short of the commanded
    target and stays stopped -- the real signature of hitting an object,
    not of arriving somewhere it should settle. Must be recognized as
    contact, not just eventually reported as a plain failure."""
    controller = GripperCloseController(target=0.0, params=default_params())
    # Moving normally, then slams into something at position=0.9 (error=0.9,
    # far outside goal_tolerance=0.05 and near_target_threshold=0.15).
    steps = [(1.5 - 0.1 * i, -0.5) for i in range(6)]  # -> position 0.9
    steps += [(0.9, 0.0)] * 10  # stalled, velocity pinned at 0
    result = run_steps(controller, steps)
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
    steps = [(0.5 - 0.05 * i, -0.5) for i in range(8)]  # -> position 0.1
    steps += [(0.1, 0.0)] * 10  # stalled at error=0.1: inside near_target_threshold,
    # outside goal_tolerance -- the taper-zone trap.
    result = run_steps(controller, steps)
    assert result.status == GripperCloseStatus.CONTACT_DETECTED
    assert abs(result.effort) == pytest.approx(
        min(params.contact_hold_effort, params.max_effort))


def test_free_space_taper_still_applies_while_tracking_near_target():
    """Approaching a free-space target normally (never stalling) but
    already within near_target_threshold: still TRACKING (hasn't reached
    goal_tolerance yet), and effort should already be tapered down to
    avoid the overshoot/limit-slam this taper was originally added to fix
    -- unaffected by the new contact-detection path."""
    params = default_params()
    controller = GripperCloseController(target=0.0, params=params)
    # First step lands well inside near_target_threshold but outside
    # goal_tolerance, with clearly nonzero velocity (still moving, not
    # stalled) -- one cycle isn't enough to trip contact_confirm_time.
    result = controller.step(position=0.1, velocity=-0.5, dt=0.02)
    assert result.status == GripperCloseStatus.TRACKING
    assert abs(result.effort) <= params.near_target_max_effort + 1e-9


def test_contact_status_locks_in_and_keeps_holding():
    """Once contact is detected, later steps (even if the hold effort
    itself produces some small nonzero velocity) must never revert to
    TRACKING/REACHED_FREE -- the whole point is a stable, sustained grip,
    not a controller that lets go the moment something wiggles."""
    controller = GripperCloseController(target=0.0, params=default_params())
    steps = [(0.9, -0.5)] * 3 + [(0.9, 0.0)] * 10
    run_steps(controller, steps)
    # A little post-contact motion, well away from goal_tolerance and with
    # real velocity -- would look like normal tracking if status weren't
    # locked.
    result = controller.step(position=0.88, velocity=-0.05, dt=0.02)
    assert result.status == GripperCloseStatus.CONTACT_DETECTED


def test_contact_hold_pushes_in_original_closing_direction():
    """The hold effort must keep squeezing toward the original target
    direction (a real, physically motivated sustained push), not drift
    toward zero or reverse -- regardless of which direction closing
    happens to be for this particular goal."""
    closing_positive = GripperCloseController(target=1.5, params=default_params())
    steps = [(0.5 + 0.1 * i, 0.5) for i in range(6)] + [(1.1, 0.0)] * 10
    result = run_steps(closing_positive, steps)
    assert result.status == GripperCloseStatus.CONTACT_DETECTED
    assert result.effort > 0

    closing_negative = GripperCloseController(target=0.0, params=default_params())
    steps = [(1.5 - 0.1 * i, -0.5) for i in range(6)] + [(0.9, 0.0)] * 10
    result = run_steps(closing_negative, steps)
    assert result.status == GripperCloseStatus.CONTACT_DETECTED
    assert result.effort < 0
