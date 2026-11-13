"""Runs one episode: a controller moves the arm in the scene and the result is judged."""

import dataclasses
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from beverage_disposal.core.config import SceneConfig
from beverage_disposal.core.controller import EEDeltaController
from beverage_disposal.core.ik import solve_ik
from beverage_disposal.core.interfaces import Controller, Observation, ObservationNoise
from beverage_disposal.core.kinematics import HOME
from beverage_disposal.core.layout import Goal, sample_spawn
from beverage_disposal.core.task_tests import final_state_problem
from beverage_disposal.core.trajectory import Trajectory
from beverage_disposal.sim.loop import CONTROL_PERIOD, ArmLoop
from beverage_disposal.sim.recorder import Recorder
from beverage_disposal.sim.scene import build_scene

MAX_STEPS = 600
# Where the tip starts each episode, base frame, metres: above the middle of the spawn box.
READY_TIP = (0.26, 0.0, 0.15)
# The controller's target box (base frame, metres): wide enough for every pose of the layout.
WORKSPACE = (np.array([0.10, -0.25, 0.0]), np.array([0.40, 0.25, 0.25]))
SETTLE_STEPS = 20  # control steps after the controller is done, so the container can come to rest


@dataclass(frozen=True)
class Episode:
    success: bool
    reason: str  # "ok", or what failed first
    container: np.ndarray  # final container centre, metres
    steps: int
    trajectory: Trajectory | None = None


def run_episode(
    config: SceneConfig,
    mjcf,
    controller: Controller,
    goal: Goal,
    *,
    seed: int,
    noise: ObservationNoise | None = None,
    max_steps: int = MAX_STEPS,
) -> Episode:
    """Spawn the container at a seeded position, run `controller`, and judge where it ended up."""
    x, y = sample_spawn(config, np.random.default_rng(seed))
    config = dataclasses.replace(config, container_x=x, container_y=y)
    loop = ArmLoop(build_scene(config, mjcf))
    grasp = config.grasp
    pitch = math.radians(grasp.pitch_deg)
    start = np.array(HOME, dtype=float)
    start[4] = grasp.wrist_roll
    ready = solve_ik(np.array(READY_TIP), pitch, start).angles
    loop.reset(ready[:5], grasp.open_angle)
    arm = EEDeltaController(workspace=WORKSPACE, pitch=pitch)
    arm.reset(loop.joints, loop.gripper)
    noise_rng = np.random.default_rng([seed, 1])
    previous = np.zeros(4)

    def observe(step: int) -> Observation:
        tip = loop.tip
        observation = Observation(
            joints=loop.joints,
            gripper=loop.gripper,
            tip=tip,
            container=loop.data.body("container").xpos - tip,
            goal=np.array([goal.center_x, goal.center_y, goal.release_z]) - tip,
            previous_action=previous,
            time_remaining=(max_steps - step) * CONTROL_PERIOD,
        )
        return observation if noise is None else noise.apply(observation, noise_rng)

    recorder = Recorder(loop.model)
    controller.reset(observe(0))
    step = 0
    while step < max_steps and not getattr(controller, "done", False):
        previous = np.asarray(controller.act(observe(step)), dtype=float)
        command = arm.step(previous, loop.joints, loop.gripper)
        loop.step(command.joints[:5], command.gripper)
        recorder.record(loop, previous)
        step += 1
    finished = bool(getattr(controller, "done", False))
    stop = getattr(controller, "stop_event", None)
    if stop is not None:
        trajectory = recorder.trajectory(False)
        can = loop.data.body("container").xpos.copy()
        return Episode(False, f"safe stop: {stop.code}", can, len(trajectory.time), trajectory)
    if finished:
        for _ in range(SETTLE_STEPS):
            loop.step(command.joints[:5], command.gripper)
            recorder.record(loop, np.zeros(4))
    trajectory = recorder.trajectory(finished)
    can = loop.data.body("container").xpos.copy()
    if not finished:
        return Episode(False, "time ran out", can, len(trajectory.time), trajectory)
    result = judge(config, loop.data, goal, len(trajectory.time))
    return dataclasses.replace(result, trajectory=trajectory)


def judge(config: SceneConfig, data, goal: Goal, steps: int = 0) -> Episode:
    """Whether the container rests upright on the goal's surface, wholly inside the goal."""
    can = data.body("container")
    position = can.xpos.copy()
    problem = final_state_problem(
        position, float(can.xmat.reshape(3, 3)[2, 2]), _container_speed(data), goal, config
    )
    return Episode(problem is None, problem or "ok", position, steps)


def _container_speed(data) -> float:
    velocity = np.zeros(6)
    body = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_BODY, "container")
    mujoco.mj_objectVelocity(data.model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
    return float(np.linalg.norm(velocity[3:]))
