"""What the pick-and-place skill sees and the shape of anything that controls it."""

from dataclasses import dataclass, replace
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Observation:
    """One control step's view, in the arm's base frame. Lengths in metres, angles in radians.

    Positions of the container and the goal are relative to the tool tip, as the locked
    observation design has it. The version changes whenever a field is added or its meaning moves.
    """

    VERSION: ClassVar[int] = 1

    joints: np.ndarray  # five measured arm joint angles
    gripper: float  # measured gripper angle
    tip: np.ndarray  # tool tip from our forward kinematics
    container: np.ndarray  # container centre minus tip
    goal: np.ndarray  # goal (x, y, release height) minus tip
    previous_action: np.ndarray  # the last four-number action
    time_remaining: float  # seconds left in the episode


@runtime_checkable
class Controller(Protocol):
    """Anything that turns observations into the four-number action [dx, dy, dz, gripper]."""

    def reset(self, observation: Observation) -> None: ...

    def act(self, observation: Observation) -> np.ndarray: ...


class ObservationNoise(Protocol):
    def apply(self, observation: Observation, rng: np.random.Generator) -> Observation: ...


@dataclass(frozen=True)
class PositionNoise:
    """Gaussian noise on the perceived container position; the arm's own sensing stays exact."""

    sigma: float  # standard deviation per axis, metres

    def __post_init__(self):
        if self.sigma < 0:
            raise ValueError(f"sigma must not be negative, got {self.sigma}")

    def apply(self, observation: Observation, rng: np.random.Generator) -> Observation:
        if self.sigma == 0:
            return observation
        noise = rng.normal(0.0, self.sigma, size=3)
        return replace(observation, container=observation.container + noise)
