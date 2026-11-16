"""The scripted pick-and-place: a state machine that proves the task can be solved.

It reads container and goal positions straight from the observation (ground truth plus whatever
noise the observation carries) and answers with the same four-number action a learned policy
will give. It is the baseline the learned skill is compared against.
"""

import math
from enum import Enum, auto

import numpy as np

from beverage_disposal.core.config import SceneConfig
from beverage_disposal.core.controller import MAX_DELTA
from beverage_disposal.core.interfaces import Observation
from beverage_disposal.core.kinematics import fk, load_chain

# Fraction of the remaining distance asked for per control step. The delta controller adds each
# action to its previous target, so it integrates; with the servos 1-2 steps behind, a gain of 0.5
# made the tip swing 40 mm over a far can and knock it over. Swept over 381 spawns at 0 and 30 mL:
# 0.15, 0.2 and 0.25 never failed, 0.3 failed 5 of 20 hard seeds.
GAIN = 0.2
ABOVE_TOLERANCE = 0.004  # metres from the point above the container before descending
GRASP_TOLERANCE = 0.003  # metres from the grasp point before squeezing
LIFT_TOLERANCE = 0.005
CARRY_TOLERANCE = 0.004
LOWER_TOLERANCE = 0.003
RETREAT_TOLERANCE = 0.005
SETTLE_STEPS = 20  # control steps to wait after the gripper reaches its closing target
# Largest action while the container is held: fast moves swing it on the pads.
HELD_SPEED = 0.25
RELEASE_RATE = 0.1  # gripper angle gained per control step while letting go, rad
RELEASE_MARGIN = 0.004  # the container's bottom is lowered to this far above the release height
CARRY_CLEARANCE = 0.015  # extra height over the release point while moving across, metres
TOP_CLEARANCE = 0.015  # metres the tip stays above a container's top while coming in over it
OPEN_TOLERANCE = 0.05  # rad: the gripper counts as open this close to its open angle
# The TCP's distance from the fixed pad's inner face along the closing direction, metres (D-77).
FACE_TO_TCP = 0.0079
_GRIPPER_LOW, _GRIPPER_HIGH = load_chain()["gripper_range"]


def grasp_offset_m(config: SceneConfig) -> float:
    """Metres from the container's centre to the TCP along the closing direction, before closing."""
    return config.container.grasp_diameter / 2 + config.grasp.clearance - FACE_TO_TCP


def approach_height_m(config: SceneConfig) -> float:
    """Height of the tip over the container's bottom while it comes in above the container.

    The clearance over the grasp point, or just over the top when the container stands taller
    than that (a bottle taken by its body: the jaws come down around the neck).
    """
    container = config.container
    return max(
        container.grasp_height + config.reach.approach_clearance,
        container.height + TOP_CLEARANCE,
    )


class Phase(Enum):
    ABOVE = auto()
    DESCEND = auto()
    CLOSE = auto()
    LIFT = auto()
    CARRY = auto()
    LOWER = auto()
    RELEASE = auto()
    RETREAT = auto()
    DONE = auto()


class ScriptedController:
    """Hover, descend, close, lift, carry, lower, release, retreat. `squeeze=False` never closes."""

    def __init__(self, config: SceneConfig, *, squeeze: bool = True):
        self._config = config
        self._squeeze = squeeze
        grasp = config.grasp
        self._ramp_steps = math.ceil((grasp.open_angle - grasp.close_target) / grasp.close_rate)
        self.phase = Phase.ABOVE
        # Each phase with the control step it began on, for measurements on the recorded episode.
        self.phase_log: list[tuple[Phase, int]] = []

    @property
    def done(self) -> bool:
        return self.phase is Phase.DONE

    def reset(self, observation: Observation) -> None:
        self.phase = Phase.ABOVE
        self.phase_log = [(Phase.ABOVE, 0)]
        self._step = 0
        self._angle = self._config.grasp.open_angle
        self._ramp = 0  # control steps spent closing
        self._anchor = observation.tip.copy()  # where the current hold or climb started

    def act(self, observation: Observation) -> np.ndarray:
        tip = observation.tip
        container = tip + observation.container
        goal = tip + observation.goal
        config, grasp = self._config, self._config.grasp
        height = config.container.grasp_height
        bottom = container[2] - config.container.height / 2
        closing = fk(observation.joints)[:2, 0]
        closing = closing / np.linalg.norm(closing)
        beside = container[:2] - grasp_offset_m(config) * closing
        carry_z = max(
            self._anchor[2] + config.reach.lift_height,
            goal[2] + height + CARRY_CLEARANCE,
        )
        phase = self.phase

        if phase is Phase.ABOVE:
            wanted = np.array([*beside, bottom + approach_height_m(config)])
            if np.linalg.norm(wanted - tip) < ABOVE_TOLERANCE:
                phase = Phase.DESCEND
        if phase is Phase.DESCEND:
            wanted = np.array([*beside, bottom + height])
            if np.linalg.norm(wanted - tip) < GRASP_TOLERANCE:
                phase = Phase.CLOSE
                self._anchor = tip.copy()
                self._ramp = 0
        if phase is Phase.CLOSE:
            wanted = self._anchor
            if self._squeeze:
                self._angle = max(grasp.close_target, self._angle - grasp.close_rate)
            # The wait is the same with or without squeezing, so both scripts take equal time.
            self._ramp += 1
            if self._ramp >= self._ramp_steps + SETTLE_STEPS:
                phase = Phase.LIFT
        if phase is Phase.LIFT:
            wanted = np.array([*self._anchor[:2], carry_z])
            if abs(carry_z - tip[2]) < LIFT_TOLERANCE:
                phase = Phase.CARRY
        if phase is Phase.CARRY:
            wanted = np.array([*(tip[:2] + goal[:2] - container[:2]), carry_z])
            if np.linalg.norm(wanted[:2] - tip[:2]) < CARRY_TOLERANCE:
                phase = Phase.LOWER
        if phase is Phase.LOWER:
            gap = bottom - goal[2]
            wanted = np.array(
                [*(tip[:2] + goal[:2] - container[:2]), tip[2] - gap + RELEASE_MARGIN]
            )
            if abs(gap - RELEASE_MARGIN) < LOWER_TOLERANCE:
                phase = Phase.RELEASE
        if phase is Phase.RELEASE:
            wanted = tip
            self._angle = min(grasp.open_angle, self._angle + RELEASE_RATE)
            if observation.gripper >= grasp.open_angle - OPEN_TOLERANCE and (
                self._angle >= grasp.open_angle
            ):
                phase = Phase.RETREAT
                self._anchor = tip.copy()
        if phase is Phase.RETREAT:
            wanted = np.array(
                [*self._anchor[:2], self._anchor[2] + config.reach.approach_clearance]
            )
            if abs(wanted[2] - tip[2]) < RETREAT_TOLERANCE:
                phase = Phase.DONE
        if phase is Phase.DONE:
            wanted = tip

        if phase is not self.phase:
            self.phase_log.append((phase, self._step))
        self._step += 1
        self.phase = phase
        limit = HELD_SPEED if phase in (Phase.LIFT, Phase.CARRY, Phase.LOWER) else 1.0
        move = np.clip(GAIN * (wanted - tip) / MAX_DELTA, -limit, limit)
        return np.array([*move, self._action_for(self._angle)])

    @staticmethod
    def _action_for(angle: float) -> float:
        return 2 * (angle - _GRIPPER_LOW) / (_GRIPPER_HIGH - _GRIPPER_LOW) - 1
