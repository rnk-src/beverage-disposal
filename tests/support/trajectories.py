"""Builds a Trajectory from a few segments, so a test can state the story of an episode."""

import numpy as np

from beverage_disposal.core.trajectory import Trajectory

CONTROL_PERIOD = 0.05
HALF_HEIGHT = 0.061  # a can at rest on a surface, centre above it


def trajectory(segments, finished=True):
    """Each segment is a dict: steps, tip, container (xyz, or start and end), and optional
    upright, speed, arm_force, action. A value that is a pair of points is linearly interpolated."""
    rows = {name: [] for name in ["tip", "container", "upright", "speed", "arm_force", "action"]}
    gripper = []
    for segment in segments:
        n = segment["steps"]
        for name, default in [
            ("tip", None),
            ("container", None),
            ("upright", 1.0),
            ("speed", 0.0),
            ("arm_force", 0.0),
            ("action", (0.0, 0.0, 0.0, 0.0)),
        ]:
            value = segment.get(name, default)
            if name in ("tip", "container") and isinstance(value, tuple) and len(value) == 2:
                value = np.linspace(value[0], value[1], n)
            elif name in ("tip", "container"):
                value = np.tile(value, (n, 1))
            else:
                value = np.tile(np.asarray(value, dtype=float), (n,) + (1,) * np.ndim(value))
            rows[name].append(np.asarray(value, dtype=float))
        gripper.append(np.full(n, segment.get("gripper", 0.5)))
    count = sum(segment["steps"] for segment in segments)
    return Trajectory(
        time=np.arange(count) * CONTROL_PERIOD,
        tip=np.concatenate(rows["tip"]),
        gripper=np.concatenate(gripper),
        container=np.concatenate(rows["container"]),
        upright=np.concatenate(rows["upright"]),
        speed=np.concatenate(rows["speed"]),
        arm_force=np.concatenate(rows["arm_force"]),
        action=np.concatenate(rows["action"]),
        finished=finished,
    )
