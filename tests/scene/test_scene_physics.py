"""The simulated scene: a table, a can that stays where it is put, and the arm beside them."""

import dataclasses

import mujoco
import numpy as np
import pytest
from support.jaws import jaw_gap
from support.paths import ROOT
from support.so101 import MJCF, SO101

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.kinematics import HOME, fk
from beverage_disposal.sim.loop import ArmLoop
from beverage_disposal.sim.scene import build_scene

CONFIG = load_scene_config(ROOT / "configs" / "scene.toml")
# Closed: with the jaw swung open at home it reaches over to x = 0.29 m and shoves a can at 0.30 m.
GRIPPER_HOLD = 0.0
FULL_CAN_ML = 355.0
FLOOR_TO_CONTACT = 5e-4  # the can may rest this far from its ideal height, metres


def scene(config=CONFIG):
    return build_scene(config, SO101 / MJCF)


def home_data(model):
    """The arm held at HOME by its actuators, the can where the config puts it."""
    data = mujoco.MjData(model)
    data.qpos[:6] = [*HOME, GRIPPER_HOLD]
    data.ctrl[:6] = [*HOME, GRIPPER_HOLD]
    mujoco.mj_forward(model, data)
    return data


def run(model, data, seconds):
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def can_position(data):
    return data.body("container").xpos.copy()


@pytest.fixture(scope="module", params=[0.0, FULL_CAN_ML], ids=["empty", "full"])
def settled(request):
    config = dataclasses.replace(CONFIG, fill_ml=request.param)
    model = scene(config)
    data = home_data(model)
    run(model, data, 2.0)
    return model, data


def test_the_can_rests_at_half_its_height_on_the_table(settled):
    _, data = settled
    assert can_position(data)[2] == pytest.approx(CONFIG.container.height / 2, abs=FLOOR_TO_CONTACT)


def test_the_can_stays_where_it_is_for_ten_seconds(settled):
    model, data = settled
    before = can_position(data)
    run(model, data, 10.0)
    assert np.linalg.norm(can_position(data) - before) < 1e-3


def test_a_push_well_below_the_friction_limit_does_not_make_the_can_creep():
    # 0.5 N on a full can of 0.369 kg: the friction limit is mu * m * g, about 3.6 N, and the
    # push is below the tipping force too (about 2 N), so the can must not move at all.
    config = dataclasses.replace(CONFIG, fill_ml=FULL_CAN_ML)
    model = scene(config)
    data = home_data(model)
    run(model, data, 2.0)
    start = can_position(data)
    body = model.body("container").id
    data.xfrc_applied[body, 0] = 0.5
    run(model, data, 2.0)
    data.xfrc_applied[body, 0] = 0.0
    run(model, data, 3.0)
    assert np.linalg.norm(can_position(data) - start) < 1e-3


@pytest.mark.parametrize("fill_ml", [0.0, FULL_CAN_ML])
def test_the_model_mass_is_the_empty_mass_plus_a_gram_per_millilitre(fill_ml):
    model = scene(dataclasses.replace(CONFIG, fill_ml=fill_ml))
    expected = 0.014 + fill_ml * 1e-3  # the figures of CLAUDE.md, not read back from the config
    assert model.body("container").mass[0] == pytest.approx(expected, abs=1e-9)


def test_the_can_in_the_model_has_the_configured_width_and_height():
    model = scene()
    geom = model.geom("container").id
    assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert model.geom_size[geom][0] * 2 == pytest.approx(0.066, abs=1e-9)
    assert model.geom_size[geom][1] * 2 == pytest.approx(0.122, abs=1e-9)


def test_the_can_starts_where_the_config_puts_it():
    config = dataclasses.replace(CONFIG, container_x=0.27, container_y=-0.04)
    data = home_data(scene(config))
    assert can_position(data)[:2] == pytest.approx([0.27, -0.04], abs=1e-9)


def test_the_table_top_is_the_height_of_the_arms_base():
    model = scene()
    data = home_data(model)
    table = model.geom("table").id
    top = data.geom_xpos[table][2] + model.geom_size[table][2]
    assert top == pytest.approx(data.body("base").xpos[2], abs=1e-9)


def test_the_arm_at_home_touches_neither_the_can_nor_the_table_except_by_its_base():
    model = scene()
    data = home_data(model)
    run(model, data, 1.0)
    base = model.body("base").id
    names = set()
    for contact in data.contact[: data.ncon]:
        bodies = {int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)}
        if model.body("container").id in bodies or model.geom("table").bodyid[0] in bodies:
            others = bodies - {model.body("container").id, int(model.geom("table").bodyid[0])}
            if others and others != {base}:
                names |= {model.body(b).name for b in others}
    assert not names, f"touching the scene: {sorted(names)}"


def test_the_arm_loop_runs_in_the_scene_and_holds_its_pose():
    # The loop was written for the bare arm; the can adds free-joint coordinates after its six.
    loop = ArmLoop(scene())
    loop.reset(HOME, GRIPPER_HOLD)
    for _ in range(10):
        loop.step(HOME, GRIPPER_HOLD)
    assert loop.tip == pytest.approx(fk(HOME)[:3, 3], abs=5e-3)


def test_the_gripper_opens_wider_than_the_can_by_a_margin_well_inside_its_range():
    # The jaw gap is read where the TCP sits along the fingers. The margin is the room a grasp needs
    # for the arm's positioning error (assumed, 5 mm); the angle must leave the controller slack.
    model = scene()
    data = mujoco.MjData(model)
    address = model.joint("gripper").qposadr[0]
    low, high = model.jnt_range[model.joint("gripper").id]
    mujoco.mj_kinematics(model, data)
    depth = (
        (data.site("tcp").xpos - data.body("gripper").xpos)
        @ data.body("gripper").xmat.reshape(3, 3)
    )[2]
    needed = CONFIG.container.diameter + 0.005
    for angle in np.arange(0.0, high, 0.05):
        data.qpos[address] = angle
        mujoco.mj_kinematics(model, data)
        if jaw_gap(model, data, depth) > needed:
            break
    else:
        pytest.fail("the gripper never opens wide enough for the can")
    assert angle <= high - 0.5, f"only wide enough at {angle:.2f} rad of {high:.2f}"
