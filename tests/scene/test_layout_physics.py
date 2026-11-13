"""The bin, the pad and the set-aside area in the simulated scene."""

import dataclasses

import mujoco
import numpy as np
import pytest
from support.paths import ROOT
from support.so101 import MJCF, SO101

from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.kinematics import HOME
from beverage_disposal.sim.scene import build_scene

CONFIG = load_scene_config(ROOT / "configs" / "scene.toml")
CAN_HEIGHT = 0.122
DROP_HEIGHT = 0.20


def scene(config=CONFIG):
    return build_scene(config, SO101 / MJCF)


def data_with_can_at(model, x, y, z, fill_ml=0.0):
    data = mujoco.MjData(model)
    data.qpos[:6] = [*HOME, 0.0]
    data.ctrl[:6] = [*HOME, 0.0]
    address = model.joint("container").qposadr[0]
    data.qpos[address : address + 3] = [x, y, z]
    mujoco.mj_forward(model, data)
    return data


def settle(model, data, seconds):
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def can_xyz(data):
    return data.body("container").xpos.copy()


def test_a_can_dropped_over_the_bin_ends_up_on_the_bin_floor():
    model = scene()
    bin_ = CONFIG.bin
    data = data_with_can_at(model, bin_.center_x, bin_.center_y, DROP_HEIGHT)
    settle(model, data, 3.0)
    x, y, z = can_xyz(data)
    assert abs(x - bin_.center_x) <= bin_.inner_x / 2 and abs(y - bin_.center_y) <= bin_.inner_y / 2
    # The floor's top is `floor` above the table; an empty can stands on it at half its height.
    assert z == pytest.approx(bin_.floor + CAN_HEIGHT / 2, abs=5e-4)


def test_a_can_dropped_just_outside_the_bin_does_not_end_up_in_it():
    model = scene()
    bin_ = CONFIG.bin
    outside_x = bin_.center_x + bin_.inner_x / 2 + bin_.wall + 0.05
    data = data_with_can_at(model, outside_x, bin_.center_y, DROP_HEIGHT)
    settle(model, data, 3.0)
    assert can_xyz(data)[2] == pytest.approx(CAN_HEIGHT / 2, abs=5e-4)


def test_the_bin_walls_stand_at_the_configured_height_around_the_configured_centre():
    model = scene()
    data = data_with_can_at(model, 0.3, 0.0, CAN_HEIGHT / 2)
    bin_ = CONFIG.bin
    tops = []
    for side in ["north", "south", "east", "west"]:
        geom = model.geom(f"bin_wall_{side}").id
        tops.append(data.geom_xpos[geom][2] + model.geom_size[geom][2])
    assert tops == pytest.approx([bin_.height] * 4)
    xs = [data.geom_xpos[model.geom(f"bin_wall_{s}").id][0] for s in ["east", "west"]]
    assert np.mean(xs) == pytest.approx(bin_.center_x)
    assert xs[0] - xs[1] == pytest.approx(bin_.inner_x + bin_.wall)


def test_moving_the_bin_in_the_config_moves_its_walls():
    moved = dataclasses.replace(CONFIG, bin=dataclasses.replace(CONFIG.bin, center_y=-0.19))
    model = scene(moved)
    data = data_with_can_at(model, 0.3, 0.0, CAN_HEIGHT / 2)
    ys = [data.geom_xpos[model.geom(f"bin_wall_{s}").id][1] for s in ["north", "south"]]
    assert np.mean(ys) == pytest.approx(-0.19)
    assert ys[0] - ys[1] == pytest.approx(moved.bin.inner_y + moved.bin.wall)


def test_a_can_on_the_pad_rests_on_the_pad_top():
    model = scene()
    pad = CONFIG.pad
    data = data_with_can_at(model, pad.center_x, pad.center_y, pad.height + CAN_HEIGHT / 2 + 0.01)
    settle(model, data, 2.0)
    assert can_xyz(data)[2] == pytest.approx(pad.height + CAN_HEIGHT / 2, abs=5e-4)


def test_the_set_aside_marker_is_painted_on_and_does_not_hold_a_can_up():
    model = scene()
    marker = model.geom("set_aside_marker").id
    assert model.geom_contype[marker] == 0 and model.geom_conaffinity[marker] == 0
    area = CONFIG.set_aside
    data = data_with_can_at(model, area.center_x, area.center_y, CAN_HEIGHT / 2 + 0.01)
    settle(model, data, 2.0)
    assert can_xyz(data)[2] == pytest.approx(CAN_HEIGHT / 2, abs=5e-4)


def test_the_arm_at_home_touches_none_of_the_layout():
    model = scene()
    data = data_with_can_at(model, CONFIG.container_x, CONFIG.container_y, CAN_HEIGHT / 2)
    settle(model, data, 1.0)
    layout = {
        model.geom(name).id
        for name in [
            "pad",
            "bin_floor",
            *[f"bin_wall_{s}" for s in ["north", "south", "east", "west"]],
        ]
    }
    touching = [
        (model.body(int(model.geom_bodyid[a])).name, model.body(int(model.geom_bodyid[b])).name)
        for a, b in (tuple(c.geom) for c in data.contact[: data.ncon])
        if a in layout or b in layout
    ]
    assert not touching, touching
