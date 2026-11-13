"""Turns a running episode into a Trajectory, one row per control step."""

import mujoco
import numpy as np

from beverage_disposal.core.trajectory import Trajectory
from beverage_disposal.sim.loop import ArmLoop

# The scene's own fixtures: what "the arm pressing on the environment" is measured against.
ENVIRONMENT_GEOMS = ("table", "pad", "bin_floor", "bin_wall_north", "bin_wall_south")
ENVIRONMENT_GEOMS += ("bin_wall_east", "bin_wall_west")


class Recorder:
    def __init__(self, model: mujoco.MjModel):
        self._model = model
        self._environment = {model.geom(name).id for name in ENVIRONMENT_GEOMS}
        self._container = model.body("container").id
        self._rows: dict[str, list] = {
            name: []
            for name in [
                "time",
                "tip",
                "gripper",
                "container",
                "upright",
                "speed",
                "force",
                "action",
            ]
        }

    def record(self, loop: ArmLoop, action: np.ndarray) -> None:
        data = loop.data
        can = data.body("container")
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self._model, data, mujoco.mjtObj.mjOBJ_BODY, self._container, velocity, 0
        )
        rows = self._rows
        rows["time"].append(loop.time)
        rows["tip"].append(loop.tip)
        rows["gripper"].append(loop.gripper)
        rows["container"].append(can.xpos.copy())
        rows["upright"].append(can.xmat.reshape(3, 3)[2, 2])
        rows["speed"].append(np.linalg.norm(velocity[3:]))
        rows["force"].append(self._arm_force(data))
        rows["action"].append(np.asarray(action, dtype=float))

    def _arm_force(self, data) -> float:
        """The largest normal force between an arm part and the table, bin or pad, in newtons."""
        largest = 0.0
        wrench = np.zeros(6)
        for index in range(data.ncon):
            first, second = data.contact[index].geom
            if (first in self._environment) == (second in self._environment):
                continue
            part = second if first in self._environment else first
            body = int(self._model.geom_bodyid[part])
            if body in (0, self._container):
                continue
            mujoco.mj_contactForce(self._model, data, index, wrench)
            largest = max(largest, float(wrench[0]))
        return largest

    def trajectory(self, finished: bool) -> Trajectory:
        rows = self._rows
        return Trajectory(
            time=np.array(rows["time"]),
            tip=np.array(rows["tip"]),
            gripper=np.array(rows["gripper"]),
            container=np.array(rows["container"]),
            upright=np.array(rows["upright"]),
            speed=np.array(rows["speed"]),
            arm_force=np.array(rows["force"]),
            action=np.array(rows["action"]),
            finished=finished,
        )
