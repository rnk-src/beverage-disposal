"""Step responses of the arm's position servos in MuJoCo: the numbers Gazebo is matched to later."""

import mujoco
import numpy as np

from beverage_disposal.core.kinematics import HOME
from beverage_disposal.core.step_response import step_response_metrics
from beverage_disposal.sim.loop import CONTROL_PERIOD, ArmLoop

GRIPPER_START = 0.5
STEP_SIZE = 0.2
SETTLE_SECONDS = 2.0
RECORD_SECONDS = 1.5


def _servo_parameters(model):
    servo = {}
    for actuator in range(model.nu):
        joint = model.joint(model.actuator_trnid[actuator, 0]).name
        servo[joint] = {
            "kp": float(model.actuator_gainprm[actuator, 0]),
            "kv": float(-model.actuator_biasprm[actuator, 2]),
            "force_limit": float(model.actuator_forcerange[actuator, 1]),
        }
    return servo


def _step_one_joint(model, joint, start_pose):
    """Hold `start_pose`, then step one joint by STEP_SIZE towards the middle of its range."""
    loop = ArmLoop(model)
    loop.reset(start_pose[:5], start_pose[5])
    targets = np.array(start_pose, dtype=float)
    for _ in range(round(SETTLE_SECONDS / CONTROL_PERIOD)):
        loop.step(targets[:5], targets[5])
    low, high = model.jnt_range[joint]
    start = targets[joint]
    targets[joint] = start + STEP_SIZE * (1 if start < (low + high) / 2 else -1)
    times, angles = [], []

    def record():
        times.append(loop.time)
        angles.append(loop.gripper if joint == 5 else loop.joints[joint])

    for _ in range(round(RECORD_SECONDS / CONTROL_PERIOD)):
        loop.step(targets[:5], targets[5], on_substep=record)
    times = np.array(times) - times[0] + model.opt.timestep
    return {
        "start": float(start),
        "target": float(targets[joint]),
        "metrics": step_response_metrics(times, angles, start, targets[joint]),
    }


def measure_step_responses(model):
    """Per joint: hold the home pose, step one joint by 0.2 rad, and measure the response."""
    start_pose = [*HOME, GRIPPER_START]
    joints = {
        model.joint(joint).name: _step_one_joint(model, joint, start_pose)
        for joint in range(model.njnt)
    }
    return {
        "protocol": {
            "timestep": float(model.opt.timestep),
            "control_period": CONTROL_PERIOD,
            "start_pose": [float(angle) for angle in HOME],
            "gripper_start": GRIPPER_START,
            "step_size": STEP_SIZE,
            "settle_seconds": SETTLE_SECONDS,
            "record_seconds": RECORD_SECONDS,
            "servo": _servo_parameters(model),
        },
        "generated_with": {"mujoco": mujoco.__version__},
        "joints": joints,
    }
