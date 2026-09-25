"""The vendored SO-101 steps in MuJoCo: it stays stable and holds its pose."""

import mujoco
import numpy as np
import pytest
from support.so101 import MJCF, SO101

STEPS = 1000

# Worst static gravity torque is about 0.6 kg * 9.81 m/s^2 * 0.4 m = 2.4 N*m against a servo
# stiffness of kp = 998 N*m/rad, so the arm sags at most about 0.0025 rad; 0.01 leaves margin.
REST_TOLERANCE = 0.01


def load():
    model = mujoco.MjModel.from_xml_path(str(SO101 / MJCF))
    return model, mujoco.MjData(model)


def step(model, data, count):
    for _ in range(count):
        mujoco.mj_step(model, data)


def warning_counts(data):
    return {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning)}


def test_a_thousand_steps_at_rest_stay_finite_and_raise_no_mujoco_warning():
    model, data = load()
    step(model, data, STEPS)
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    assert not any(warning_counts(data).values()), warning_counts(data)


def test_the_arm_holds_its_rest_pose_under_gravity():
    model, data = load()
    step(model, data, STEPS)
    assert np.abs(data.qpos).max() < REST_TOLERANCE


JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TARGET_TOLERANCE = 0.01
SETTLE_SECONDS = 1.0


@pytest.mark.parametrize("joint", JOINTS)
def test_a_position_command_moves_the_joint_to_its_target(joint):
    model, data = load()
    lower, upper = model.joint(joint).range
    target = lower + 0.75 * (upper - lower)
    data.ctrl[model.actuator(joint).id] = target
    step(model, data, round(SETTLE_SECONDS / model.opt.timestep))
    assert data.joint(joint).qpos[0] == pytest.approx(target, abs=TARGET_TOLERANCE)


# Must beat the servo force limit (3.35 N*m) plus the worst static gravity torque (about 2.4 N*m).
PUSH_TORQUE = 10.0
# MuJoCo enforces joint limits as soft constraints; measured penetration is about 0.001 rad.
STOP_TOLERANCE = 0.02
PUSH_SECONDS = 1.5
# The gripper touches the shoulder before the elbow reaches its upper stop (tested below).
BLOCKED_BY_THE_ARM_ITSELF = {("elbow_flex", 1)}


def push_joint(joint, direction):
    """Push one joint against its stop for PUSH_SECONDS; return (model, data)."""
    model, data = load()
    dof = model.joint(joint).dofadr[0]
    for _ in range(round(PUSH_SECONDS / model.opt.timestep)):
        data.qfrc_applied[dof] = direction * PUSH_TORQUE
        mujoco.mj_step(model, data)
    return model, data


@pytest.mark.parametrize("direction", [-1, 1], ids=["toward_lower_stop", "toward_upper_stop"])
@pytest.mark.parametrize("joint", JOINTS)
def test_a_hard_push_never_carries_the_joint_past_its_stop(joint, direction):
    model, data = push_joint(joint, direction)
    stop = model.joint(joint).range[0 if direction < 0 else 1]
    past = direction * (data.joint(joint).qpos[0] - stop)
    assert past <= STOP_TOLERANCE
    if (joint, direction) not in BLOCKED_BY_THE_ARM_ITSELF:
        # Arriving at the stop is what makes "never past it" mean something: a push too weak to
        # get there would pass the check above for any limit.
        assert past >= -STOP_TOLERANCE


def test_the_gripper_touches_the_shoulder_before_the_elbow_reaches_its_upper_stop():
    model, data = push_joint("elbow_flex", 1)
    stop = model.joint("elbow_flex").range[1]
    assert data.joint("elbow_flex").qpos[0] < stop - 0.05
    body_names = {
        frozenset(model.body(model.geom_bodyid[g]).name for g in (c.geom1, c.geom2))
        for c in data.contact[: data.ncon]
    }
    assert frozenset({"shoulder", "gripper"}) in body_names


# A moving joint carries momentum into the soft limit. The limit constraint is critically damped
# with time constant tau = 0.02 s and impedance d about 0.95, so penetration peaks at
# speed * d * tau / e = 6.5 rad/s (fastest servo) * 0.95 * 0.02 / 2.718 = 0.045 rad.
DYNAMIC_STOP_TOLERANCE = 0.06
LEG_SECONDS = 1.5  # the wrist roll needs about 1.1 s to cross its 5.6 rad range


@pytest.mark.parametrize("joint", JOINTS)
def test_a_full_range_step_command_reaches_both_ends_and_never_passes_them(joint):
    model, data = load()
    lower, upper = model.joint(joint).range
    actuator = model.actuator(joint).id
    seen = []
    for target in (lower, upper):
        data.ctrl[actuator] = target
        for _ in range(round(LEG_SECONDS / model.opt.timestep)):
            mujoco.mj_step(model, data)
            seen.append(data.joint(joint).qpos[0])
    assert np.isfinite(seen).all()
    assert min(seen) >= lower - DYNAMIC_STOP_TOLERANCE
    assert max(seen) <= upper + DYNAMIC_STOP_TOLERANCE
    # Reaching the ends is what makes "never passes them" mean something.
    assert min(seen) <= lower + DYNAMIC_STOP_TOLERANCE
    if (joint, 1) not in BLOCKED_BY_THE_ARM_ITSELF:
        assert max(seen) >= upper - DYNAMIC_STOP_TOLERANCE
