"""Builds an Observation for a pose, so controller tests state only what they care about."""

import numpy as np

from beverage_disposal.core.interfaces import Observation
from beverage_disposal.core.kinematics import fk


def observation(
    joints,
    gripper,
    container_xyz,
    goal_xyz,
    previous_action=None,
    time_remaining=20.0,
    tip=None,
):
    """Container and goal positions are base-frame; the observation stores them tip-relative.

    `tip` defaults to the forward kinematics of `joints`; a stand-in world can give its own.
    """
    tip = fk(np.asarray(joints, dtype=float))[:3, 3] if tip is None else np.asarray(tip, float)
    return Observation(
        joints=np.asarray(joints, dtype=float),
        gripper=float(gripper),
        tip=tip,
        container=np.asarray(container_xyz, dtype=float) - tip,
        goal=np.asarray(goal_xyz, dtype=float) - tip,
        previous_action=np.zeros(4) if previous_action is None else np.asarray(previous_action),
        time_remaining=time_remaining,
    )
