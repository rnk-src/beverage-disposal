"""The safety supervisor: coded checks that stop or veto, and report by light, sound and message.

It never decides where a container goes. It can refuse a pick, set a container aside, or stop the
arm; when a check itself fails it stops, because a supervisor that cannot be trusted must not let
the arm go on.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

import numpy as np

from beverage_disposal.core.containers import ContainerRecord
from beverage_disposal.core.interfaces import Observation
from beverage_disposal.core.safety_config import SafetyLimits


class Outcome(Enum):
    PROCEED = auto()
    SET_ASIDE = auto()  # leave the container in the set-aside area
    SAFE_STOP = auto()  # hold still and report; a person looks


@dataclass(frozen=True)
class Event:
    code: str
    message: str


@dataclass(frozen=True)
class Decision:
    outcome: Outcome
    event: Event | None = None


PROCEED = Decision(Outcome.PROCEED)


class Indicators(Protocol):
    """The three ways the supervisor tells a person something. Hardware implements these in 9.5."""

    def light(self, color: str) -> None: ...

    def sound(self, pattern: str) -> None: ...

    def message(self, text: str) -> None: ...


class RecordingIndicators:
    """The simulated indicators: they only keep a log of (kind, value)."""

    def __init__(self):
        self.log: list[tuple[str, str]] = []

    def light(self, color: str) -> None:
        self.log.append(("light", color))

    def sound(self, pattern: str) -> None:
        self.log.append(("sound", pattern))

    def message(self, text: str) -> None:
        self.log.append(("message", text))


class Supervisor:
    def __init__(self, limits: SafetyLimits, indicators: Indicators):
        self._limits = limits
        self._indicators = indicators
        self.events: list[Event] = []

    def _trip(self, outcome: Outcome, code: str, message: str) -> Decision:
        event = Event(code, message)
        self.events.append(event)
        stop = outcome is Outcome.SAFE_STOP
        self._indicators.light("red" if stop else "amber")
        self._indicators.sound("stop" if stop else "set_aside")
        self._indicators.message(message)
        return Decision(outcome, event)

    def check_container(
        self, record: ContainerRecord | None, *, diameter: float, mass_kg: float
    ) -> Decision:
        """Whether a container may be picked at all."""
        if diameter > self._limits.max_container_diameter:
            return self._trip(
                Outcome.SAFE_STOP,
                "too_wide",
                f"container {diameter * 1000:.1f} mm wide is over the "
                f"{self._limits.max_container_diameter * 1000:.0f} mm the pads can take",
            )
        if mass_kg > self._limits.max_container_mass:
            return self._trip(
                Outcome.SAFE_STOP,
                "too_heavy",
                f"container of {mass_kg:.3f} kg is over the "
                f"{self._limits.max_container_mass} kg limit",
            )
        if record is None:
            return self._trip(
                Outcome.SET_ASIDE, "unknown_type", "unknown container type: set aside for enrolment"
            )
        return PROCEED

    def check_test_lift(self, *, gain_m: float, asked_m: float) -> Decision:
        """Whether a small first lift raised the container as far as asked."""
        if gain_m < self._limits.min_test_lift_fraction * asked_m:
            return self._trip(
                Outcome.SAFE_STOP,
                "test_lift_failed",
                f"test lift raised the container {gain_m * 1000:.1f} mm "
                f"of the {asked_m * 1000:.1f} mm asked",
            )
        return PROCEED

    def check_forces(self, *, arm_force_n: float, grip_torque_nm: float) -> Decision:
        if arm_force_n > self._limits.max_arm_force:
            return self._trip(
                Outcome.SAFE_STOP, "arm_force", f"arm pressing with {arm_force_n:.1f} N"
            )
        if grip_torque_nm > self._limits.max_grip_torque:
            return self._trip(
                Outcome.SAFE_STOP, "grip_torque", f"gripper asked for {grip_torque_nm:.2f} N m"
            )
        return PROCEED

    def check_observation(self, observation: Observation | None) -> Decision:
        """Whether the sensors can be trusted: something arrived, and every number is finite."""
        if observation is None:
            return self._trip(Outcome.SAFE_STOP, "sensor_dropout", "no observation arrived")
        values = [
            observation.joints,
            observation.tip,
            observation.container,
            observation.goal,
            observation.previous_action,
            [observation.gripper, observation.time_remaining],
        ]
        if not all(np.all(np.isfinite(np.asarray(v, dtype=float))) for v in values):
            return self._trip(
                Outcome.SAFE_STOP, "non_finite_observation", "an observation held NaN or infinity"
            )
        return PROCEED

    def check_time(self, seconds: float) -> Decision:
        if seconds > self._limits.max_episode_seconds:
            return self._trip(
                Outcome.SAFE_STOP,
                "timeout",
                f"{seconds:.1f} s is over the {self._limits.max_episode_seconds:.0f} s allowed",
            )
        return PROCEED

    def check_container_position(self, seen: np.ndarray, now: np.ndarray | None) -> Decision:
        """Whether the container is still where it was first seen."""
        if now is None:
            return self._trip(Outcome.SAFE_STOP, "container_missing", "the container is not seen")
        moved = float(np.linalg.norm(np.asarray(now) - np.asarray(seen)))
        if moved > self._limits.container_position_tolerance:
            return self._trip(
                Outcome.SAFE_STOP,
                "container_moved",
                f"the container is {moved * 1000:.0f} mm from where it was seen",
            )
        return PROCEED

    def guard(self, check: Callable[[], Decision]) -> Decision:
        """Run a check; if the check itself fails, stop. Nothing escapes to the caller."""
        try:
            return check()
        except Exception as error:  # noqa: BLE001 - the whole point is to catch every failure
            return self._trip(Outcome.SAFE_STOP, "supervisor_fault", f"supervisor fault: {error}")

    def filter_action(self, action: np.ndarray) -> tuple[np.ndarray, Decision]:
        """The action the arm may be given: clipped to [-1, 1], or standing still if not finite."""
        action = np.asarray(action, dtype=float)
        if not np.all(np.isfinite(action)):
            decision = self._trip(
                Outcome.SAFE_STOP, "non_finite_action", "the controller gave a non-finite action"
            )
            return np.zeros_like(action), decision
        return np.clip(action, -1.0, 1.0), PROCEED
