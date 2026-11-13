"""The limits the safety supervisor enforces."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from beverage_disposal.core.config import from_mapping


@dataclass(frozen=True)
class SafetyLimits:
    max_container_diameter: float  # metres: wider than the pads can take
    max_container_mass: float  # kg: heavier than the arm should lift
    min_test_lift_fraction: float  # a test lift must raise the container this fraction of the way
    max_arm_force: float  # N the arm may press on the table, bin or pad
    max_grip_torque: float  # N m the gripper servo may be asked for
    max_episode_seconds: float
    container_position_tolerance: float  # metres from where the container was first seen


@dataclass(frozen=True)
class _File:
    limits: SafetyLimits


def load_safety_limits(path: str | Path) -> SafetyLimits:
    with open(path, "rb") as file:
        return from_mapping(_File, tomllib.load(file), "safety").limits
