"""Pure decision logic for gripper_joint's PID-based closing behavior.

Extracted out of gripper_action_server.py so the actual control decision
-- specifically, telling "arrived where it should settle, nothing to hold
against" apart from "stopped early because something real is blocking it"
-- can be unit tested without spinning up ROS or Gazebo, the same way
kinematics.py/grasp_geometry.py already are.

Why this distinction matters (see commit-notes/10-iteration-3-pick-and-lift.md,
2026-09-10 entry): the previous design drove toward a fixed commanded
target and tapered the allowed effort down once *close to that target
number*. That taper was meant to stop the joint chattering as it settled
in free space, but it can't tell "arrived gently" apart from "blocked,
needs real force" -- if a grasp happens to stall at a position that falls
within the taper zone of the original nominal target, the taper caps grip
force down to a value tuned for gentle free-space settling, not for
holding an object.

The fix here uses the stall itself as the signal, not proximity to a
stored target number: while still tracking, a stall (near-zero velocity)
that persists for contact_confirm_time while the joint is still far
outside goal_tolerance is contact, and the response is a dedicated,
sustained hold effort -- not a distance-based taper. Reaching
goal_tolerance normally, without an early stall, is the free-space case,
and keeps the original taper behavior (nothing has changed there; it
still exists because it works and test_gripper.py already depends on it).
"""
import enum
from dataclasses import dataclass


class GripperCloseStatus(enum.Enum):
    TRACKING = 'tracking'
    REACHED_FREE = 'reached_free'
    CONTACT_DETECTED = 'contact_detected'


@dataclass
class GripperControlParams:
    p_gain: float
    i_gain: float
    d_gain: float
    max_effort: float
    goal_tolerance: float
    stall_velocity_threshold: float
    contact_confirm_time: float
    near_target_threshold: float
    near_target_max_effort: float
    contact_hold_effort: float


@dataclass
class GripperStepResult:
    effort: float
    status: GripperCloseStatus
    error: float


class GripperCloseController:
    """One goal execution's worth of PID + contact-detection state.

    Call step() once per control cycle. status starts at TRACKING and
    transitions exactly once, to either REACHED_FREE or CONTACT_DETECTED
    -- after that it's locked for the rest of this controller's life, so a
    caller can keep calling step() through a subsequent hold phase without
    the grip being released by a momentary wiggle.
    """

    def __init__(self, target, params):
        self.target = target
        self.params = params
        self.status = GripperCloseStatus.TRACKING
        self._integral = 0.0
        self._previous_error = None
        self._close_direction = 1.0
        self._stall_elapsed = 0.0

    def step(self, position, velocity, dt):
        p = self.params
        error = self.target - position

        if self._previous_error is None:
            self._previous_error = error
            # Sign of the direction closing actually moves this joint for
            # this goal -- fixed once at the start, since it's a property
            # of (target, starting position), not of where a stall later
            # happens to occur.
            self._close_direction = 1.0 if error >= 0 else -1.0

        derivative = (error - self._previous_error) / dt if dt > 0 else 0.0
        self._integral += error * dt
        self._previous_error = error

        if self.status == GripperCloseStatus.TRACKING:
            if abs(error) <= p.goal_tolerance:
                self.status = GripperCloseStatus.REACHED_FREE
            elif abs(velocity) < p.stall_velocity_threshold:
                self._stall_elapsed += dt
                if self._stall_elapsed >= p.contact_confirm_time:
                    self.status = GripperCloseStatus.CONTACT_DETECTED
            else:
                self._stall_elapsed = 0.0

        if self.status == GripperCloseStatus.CONTACT_DETECTED:
            # Sustained squeeze in the original closing direction, at a
            # fixed real effort -- not a position-error PID term (which
            # would relax toward zero the instant the stall position is
            # treated as the new setpoint), and not the free-space taper
            # (tuned for settling gently, not for holding against
            # resistance).
            effort = self._close_direction * min(p.contact_hold_effort, p.max_effort)
        else:
            effective_max = p.max_effort
            if abs(error) < p.near_target_threshold:
                effective_max = min(p.max_effort, p.near_target_max_effort)
            effort_raw = p.p_gain * error + p.i_gain * self._integral + p.d_gain * derivative
            effort = max(-effective_max, min(effective_max, effort_raw))

        return GripperStepResult(effort=effort, status=self.status, error=error)
