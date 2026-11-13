"""Can the arm reach every pose the pick and place needs, with a tilt it can hold?"""

from dataclasses import dataclass

import numpy as np

from beverage_disposal.core.config import SceneConfig
from beverage_disposal.core.ik import solve_ik

PITCH_STEP_DEG = 5


@dataclass(frozen=True)
class PoseReach:
    label: str
    position: tuple[float, float, float]  # TCP target in the base frame, metres
    best_pitch_deg: int | None  # the steepest pitch that converged, None when none did


def _poses(config: SceneConfig) -> list[tuple[str, tuple[float, float, float]]]:
    grasp = config.container.grasp_height
    reach = config.reach
    poses = []

    def site(name, x, y, base, action):
        poses.append((f"{name}:approach", (x, y, base + grasp + reach.approach_clearance)))
        poses.append((f"{name}:{action}", (x, y, base + grasp)))

    spawn = config.spawn
    for name, dx, dy in [
        ("spawn_centre", 0, 0),
        ("spawn_x_min", -1, 0),
        ("spawn_x_max", 1, 0),
        ("spawn_y_min", 0, -1),
        ("spawn_y_max", 0, 1),
    ]:
        x = config.container_x + dx * spawn.half_size
        y = config.container_y + dy * spawn.half_size
        site(name, x, y, 0.0, "grasp")
        poses.append((f"{name}:lift", (x, y, grasp + reach.lift_height)))
    site("pad", config.pad.center_x, config.pad.center_y, config.pad.height, "place")
    site("set_aside", config.set_aside.center_x, config.set_aside.center_y, 0.0, "place")
    release_z = config.bin.height + reach.release_clearance + grasp
    poses.append(("bin:release", (config.bin.center_x, config.bin.center_y, release_z)))
    return poses


def reach_report(config: SceneConfig) -> list[PoseReach]:
    """For every pose, the steepest pitch (at least `min_pitch_deg`) the IK reaches it with."""
    floor = round(config.reach.min_pitch_deg)
    report = []
    for label, position in _poses(config):
        best = None
        for pitch_deg in range(90, floor - 1, -PITCH_STEP_DEG):
            if solve_ik(np.array(position), np.radians(pitch_deg)).converged:
                best = pitch_deg
                break
        report.append(PoseReach(label, position, best))
    return report
