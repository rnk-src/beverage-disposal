"""Can the arm reach every pose the pick and place needs, with a tilt it can hold?"""

import dataclasses
from dataclasses import dataclass

import numpy as np

from beverage_disposal.core.config import SceneConfig
from beverage_disposal.core.containers import ContainerSpec
from beverage_disposal.core.ik import solve_ik
from beverage_disposal.core.scripted import CARRY_CLEARANCE, approach_height_m

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


def _reaches(position, pitch_deg: int) -> bool:
    return bool(solve_ik(np.array(position), np.radians(pitch_deg)).converged)


def container_poses(
    config: SceneConfig, spec: ContainerSpec
) -> list[tuple[str, tuple[float, float, float]]]:
    """The tip poses of one leg from the pad to the bin or the set-aside area, for `spec`."""
    grasp = spec.grasp_height
    pad, bin_, area, reach = config.pad, config.bin, config.set_aside, config.reach
    lift_z = pad.height + grasp + reach.lift_height
    release_z = bin_.height + reach.release_clearance + grasp
    approach = approach_height_m(dataclasses.replace(config, container=spec))
    return [
        (
            "pad:approach",
            (pad.center_x, pad.center_y, pad.height + approach),
        ),
        ("pad:grasp", (pad.center_x, pad.center_y, pad.height + grasp)),
        ("pad:lift", (pad.center_x, pad.center_y, lift_z)),
        # The script carries at the higher of the lift height and its clearance over the goal.
        ("bin:carry", (bin_.center_x, bin_.center_y, max(lift_z, release_z + CARRY_CLEARANCE))),
        ("bin:release", (bin_.center_x, bin_.center_y, release_z)),
        ("set_aside:carry", (area.center_x, area.center_y, max(lift_z, grasp + CARRY_CLEARANCE))),
        ("set_aside:place", (area.center_x, area.center_y, grasp)),
    ]


def steepest_common_pitch(
    poses: list[tuple[str, tuple[float, float, float]]], floor_deg: int = 0
) -> int | None:
    """The steepest pitch, in whole steps down to `floor_deg`, that reaches every pose."""
    for pitch_deg in range(90, floor_deg - 1, -PITCH_STEP_DEG):
        if all(_reaches(position, pitch_deg) for _, position in poses):
            return pitch_deg
    return None


def reach_for_sizes(
    config: SceneConfig, specs: dict[str, ContainerSpec]
) -> dict[str, list[PoseReach]]:
    """Per size, every pose with the steepest pitch from 90 down to 0 degrees that reaches it."""
    report = {}
    for size, spec in specs.items():
        rows = []
        for label, position in container_poses(config, spec):
            pitches = range(90, -1, -PITCH_STEP_DEG)
            best = next((pitch for pitch in pitches if _reaches(position, pitch)), None)
            rows.append(PoseReach(label, position, best))
        report[size] = rows
    return report
