"""A prescribed pick: straddle the container, squeeze, lift. No controller, no learning yet."""

import math
from dataclasses import dataclass

import numpy as np

from beverage_disposal.core.config import SceneConfig
from beverage_disposal.core.ik import solve_ik
from beverage_disposal.core.kinematics import HOME, fk
from beverage_disposal.sim.loop import ArmLoop
from beverage_disposal.sim.scene import build_scene

SETTLE_STEPS = 20  # control steps of holding still between phases
DESCEND_STEPS = 40
LIFT_STEPS = 40
# The TCP's distance from the fixed pad's inner face along the closing direction, metres (D-77).
FACE_TO_TCP = 0.0079


@dataclass(frozen=True)
class GraspResult:
    lift_gain: float  # how far the container rose during the lift, metres
    slip: float  # how far it moved relative to the tool tip during the lift, metres
    gripper_angle: float  # where the gripper stopped while squeezing, rad


def _pose(position, pitch, start):
    result = solve_ik(np.array(position), pitch, start)
    if not result.converged:
        raise RuntimeError(
            f"the pose {tuple(round(float(v), 3) for v in position)} is out of reach"
        )
    return result.angles


def prescribed_grasp(config: SceneConfig, mjcf, *, close: bool = True) -> GraspResult:
    """Pick the container standing at the config's position and report what the lift did."""
    grasp, reach = config.grasp, config.reach
    model = build_scene(config, mjcf)
    loop = ArmLoop(model)
    data = loop._data

    start = np.array(HOME, dtype=float)
    start[4] = grasp.wrist_roll
    pitch = math.radians(grasp.pitch_deg)
    cx, cy, height = config.container_x, config.container_y, config.container.grasp_height
    # The closing direction in the base frame: the tool's x axis, horizontal at a 90 degree roll.
    closing = fk(_pose((cx, cy, height), pitch, start))[:2, 0]
    closing = closing / np.linalg.norm(closing)
    # The container's centre sits half a diameter plus the clearance from the fixed pad's face,
    # and the face is 7.9 mm from the TCP on the side away from the moving jaw.
    to_centre = config.container.diameter / 2 + grasp.clearance - FACE_TO_TCP
    tcp_xy = np.array([cx, cy]) - to_centre * closing

    above = _pose((*tcp_xy, height + reach.approach_clearance), pitch, start)
    at = _pose((*tcp_xy, height), pitch, above)
    up = _pose((*tcp_xy, height + reach.lift_height), pitch, at)

    loop.reset(above[:5], grasp.open_angle)
    for _ in range(SETTLE_STEPS):
        loop.step(above[:5], grasp.open_angle)
    for i in range(1, DESCEND_STEPS + 1):
        loop.step(above[:5] + (at[:5] - above[:5]) * i / DESCEND_STEPS, grasp.open_angle)

    target = grasp.open_angle
    squeeze_steps = math.ceil((grasp.open_angle - grasp.close_target) / grasp.close_rate)
    for _ in range(squeeze_steps + SETTLE_STEPS):
        if close:
            target = max(grasp.close_target, target - grasp.close_rate)
        loop.step(at[:5], target)
    can = data.body("container")
    before, tip_before = can.xpos.copy(), loop.tip.copy()
    angle = loop.gripper

    for i in range(1, LIFT_STEPS + 1):
        loop.step(at[:5] + (up[:5] - at[:5]) * i / LIFT_STEPS, target)
    for _ in range(SETTLE_STEPS):
        loop.step(up[:5], target)
    after, tip_after = can.xpos.copy(), loop.tip.copy()
    if not np.all(np.isfinite(after)):
        return GraspResult(lift_gain=-math.inf, slip=math.inf, gripper_angle=angle)
    slip = float(np.linalg.norm((after - tip_after) - (before - tip_before)))
    return GraspResult(lift_gain=float(after[2] - before[2]), slip=slip, gripper_angle=angle)
