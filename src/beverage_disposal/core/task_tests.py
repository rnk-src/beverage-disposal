"""The task as pure functions on a recorded episode: pass-fail tests and indicative numbers.

Pass-fail tests give 0 or 1 and define the goal; indicators are real numbers that say how close a
failing run came. Thresholds were set from measurements of the scripted controller in 2.3 and
2.5, not guessed; where one is an assumption it says so.
"""

import math
from dataclasses import dataclass

import numpy as np

from beverage_disposal.core.config import SceneConfig
from beverage_disposal.core.layout import Goal
from beverage_disposal.core.trajectory import Trajectory

LIFT_HEIGHT = 0.02  # the container's centre must rise this far, metres
LIFT_STEPS = 5  # ... and stay up for this many consecutive control steps
HELD_DISTANCE = 0.08  # ... within this distance of the tool tip, metres (held: 0.037 to 0.048)
# Scripted runs peak at 1.08 m/s (measured over 24 episodes); the drop into the bin from 7 cm is
# 1.17 m/s in free fall. More than twice that counts as a throw (assumed).
MAX_SPEED = 2.5  # m/s
MIN_UPRIGHT = math.cos(math.radians(60.0))  # a tilt beyond 60 degrees counts as knocked over
MAX_ARM_FORCE = 20.0  # N the arm may press on the table, bin or pad (assumed; scripted runs: 0 N)
# Where a container may end up and still count as put down.
REST_HEIGHT_TOLERANCE = 0.003  # metres
MAX_TILT_DEG = 5.0
MAX_REST_SPEED = 0.01  # m/s


def final_state_problem(
    position: np.ndarray, upright: float, speed: float, goal: Goal, config: SceneConfig
) -> str | None:
    """Why a container at rest here is not properly on `goal`, or None when it is."""
    radius = config.container.diameter / 2
    if not goal.holds(position[0], position[1], radius):
        return "not inside the goal"
    if abs(position[2] - (goal.rest_z + config.container.height / 2)) >= REST_HEIGHT_TOLERANCE:
        return "not resting on the goal's surface"
    if upright <= math.cos(math.radians(MAX_TILT_DEG)):
        return "not upright"
    if speed >= MAX_REST_SPEED:
        return "not at rest"
    return None


@dataclass(frozen=True)
class TaskVerdict:
    passed: dict[str, bool]
    indicators: dict[str, float]

    @property
    def success(self) -> bool:
        return all(self.passed.values())


def _longest_run(flags: np.ndarray) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def evaluate(trajectory: Trajectory, goal: Goal, config: SceneConfig) -> TaskVerdict:
    t = trajectory
    start_height = t.container[0, 2]
    distance = np.linalg.norm(t.container - t.tip, axis=1)
    lifted = (t.container[:, 2] >= start_height + LIFT_HEIGHT) & (distance <= HELD_DISTANCE)
    problem = final_state_problem(t.container[-1], t.upright[-1], t.speed[-1], goal, config)
    passed = {
        "grasped": _longest_run(lifted) >= LIFT_STEPS,
        "placed": problem is None,
        "not_thrown": bool(t.speed.max() <= MAX_SPEED),
        "not_knocked_over": bool(t.upright.min() >= MIN_UPRIGHT),
        "arm_safe": bool(t.arm_force.max() <= MAX_ARM_FORCE),
        "in_time": t.finished,
    }
    jumps = np.abs(np.diff(t.action, axis=0)).sum(axis=1)
    indicators = {
        "closest_approach": float(distance.min()),
        "height_gained": float(t.container[:, 2].max() - start_height),
        "final_distance": float(
            np.linalg.norm(t.container[-1, :2] - [goal.center_x, goal.center_y])
        ),
        "smoothness": float(jumps.mean()) if len(jumps) else 0.0,
        "duration": float(t.time[-1] - t.time[0]),
    }
    return TaskVerdict(passed, indicators)
