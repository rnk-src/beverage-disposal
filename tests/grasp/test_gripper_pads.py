"""The gripper's collision model: two flat pads facing each other square-on at the grasp opening."""

import dataclasses

import mujoco
import numpy as np
import pytest
from support.paths import ROOT
from support.so101 import MJCF, SO101

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.sim.scene import build_scene

CONFIG = load_scene_config(ROOT / "configs" / "scene.toml")
FIXED_FACE_X = -0.0079  # the fixed finger's inner face in the gripper frame, from 1.1's jaw tests


@pytest.fixture(scope="module")
def at_design_angle():
    model = build_scene(CONFIG, SO101 / MJCF)
    data = mujoco.MjData(model)
    data.qpos[model.joint("gripper").qposadr[0]] = CONFIG.gripper_pads.design_angle
    mujoco.mj_kinematics(model, data)
    return model, data


def in_gripper_frame(data, point):
    gripper = data.body("gripper")
    return gripper.xmat.reshape(3, 3).T @ (point - gripper.xpos)


def test_the_two_pad_faces_are_parallel_at_the_design_angle(at_design_angle):
    _, data = at_design_angle
    fixed = data.geom("fixed_pad").xmat.reshape(3, 3)[:, 0]
    moving = data.geom("moving_pad").xmat.reshape(3, 3)[:, 0]
    assert abs(fixed @ moving) > np.cos(np.radians(0.5))


def test_the_pad_faces_are_the_design_width_apart_at_the_design_angle(at_design_angle):
    model, data = at_design_angle
    pads = CONFIG.gripper_pads
    fixed_inner = in_gripper_frame(data, data.geom("fixed_pad").xpos)[0] + pads.thickness / 2
    moving_inner = in_gripper_frame(data, data.geom("moving_pad").xpos)[0] - pads.thickness / 2
    assert fixed_inner == pytest.approx(FIXED_FACE_X, abs=5e-4)
    assert moving_inner - fixed_inner == pytest.approx(pads.design_width, abs=5e-4)
    assert pads.design_width == 0.066  # the can's width: the figure the hardware is sized for


def test_the_pads_meet_the_can_at_the_same_height(at_design_angle):
    _, data = at_design_angle
    fixed_z = in_gripper_frame(data, data.geom("fixed_pad").xpos)[2]
    moving_z = in_gripper_frame(data, data.geom("moving_pad").xpos)[2]
    assert fixed_z == pytest.approx(CONFIG.gripper_pads.depth, abs=5e-4)
    assert moving_z == pytest.approx(fixed_z, abs=5e-4)


def test_only_the_pads_collide_on_the_gripper_and_the_jaw():
    model = build_scene(CONFIG, SO101 / MJCF)
    colliding = {
        model.geom(g).name or f"mesh{g}"
        for g in range(model.ngeom)
        if model.body(int(model.geom_bodyid[g])).name in ("gripper", "moving_jaw_so101_v1")
        and model.geom_contype[g] > 0
    }
    assert colliding == {"fixed_pad", "moving_pad"}


def test_the_gripper_servo_is_limited_to_the_configured_torque():
    model = build_scene(CONFIG, SO101 / MJCF)
    limit = CONFIG.grasp.force_limit
    assert model.actuator("gripper").forcerange == pytest.approx([-limit, limit])
    # The other joints keep the vendored limit.
    assert model.actuator("shoulder_pan").forcerange[1] > limit


def pad_gap_at_the_design_angle(config):
    model = build_scene(config, SO101 / MJCF)
    data = mujoco.MjData(model)
    data.qpos[model.joint("gripper").qposadr[0]] = config.gripper_pads.design_angle
    mujoco.mj_kinematics(model, data)
    thickness = config.gripper_pads.thickness
    return (
        in_gripper_frame(data, data.geom("moving_pad").xpos)[0]
        - thickness / 2
        - (in_gripper_frame(data, data.geom("fixed_pad").xpos)[0] + thickness / 2)
    )


def test_the_pads_are_hardware_so_the_container_does_not_move_them():
    other = dataclasses.replace(
        CONFIG.container, diameter=0.07, grasp_diameter=0.026, name="something_else"
    )
    gap = pad_gap_at_the_design_angle(dataclasses.replace(CONFIG, container=other))
    assert gap == pytest.approx(CONFIG.gripper_pads.design_width, abs=5e-4)


def test_the_configured_design_width_sets_the_pad_gap():
    pads = dataclasses.replace(CONFIG.gripper_pads, design_width=0.07)
    gap = pad_gap_at_the_design_angle(dataclasses.replace(CONFIG, gripper_pads=pads))
    assert gap == pytest.approx(0.07, abs=5e-4)


def test_the_configured_timestep_and_contact_softness_reach_the_model():
    config = dataclasses.replace(
        CONFIG,
        physics=dataclasses.replace(
            CONFIG.physics, timestep=0.001, contact_time_constant=0.03, contact_damping_ratio=0.8
        ),
    )
    model = build_scene(config, SO101 / MJCF)
    assert model.opt.timestep == 0.001
    for name in ["table", "container", "pad", "bin_floor", "fixed_pad", "moving_pad"]:
        assert model.geom(name).solref == pytest.approx([0.03, 0.8]), name


def test_the_pads_resist_twisting_with_the_configured_torsional_friction():
    model = build_scene(CONFIG, SO101 / MJCF)
    for name in ["fixed_pad", "moving_pad"]:
        geom = model.geom(name)
        assert geom.condim[0] == 4, name
        assert geom.friction[1] == pytest.approx(CONFIG.gripper_pads.torsional_friction), name
