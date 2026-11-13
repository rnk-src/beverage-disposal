"""One recorded episode, a row per control step, all in the arm's base frame."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    time: np.ndarray  # (n,) seconds
    tip: np.ndarray  # (n, 3) tool tip, metres
    gripper: np.ndarray  # (n,) gripper angle, rad
    container: np.ndarray  # (n, 3) container centre, metres
    upright: np.ndarray  # (n,) cosine of the container's tilt: 1 standing, 0 on its side
    speed: np.ndarray  # (n,) container speed, m/s
    arm_force: np.ndarray  # (n,) largest force the arm presses on the table, bin or pad, N
    action: np.ndarray  # (n, 4) the action taken at each step
    finished: bool  # the controller reported done before the time ran out
