"""A PET bottle in the MuJoCo scene: right mass, right liquid height, and it stands on the table.

The liquid's height is checked from the model's own centre of mass, against the closed-form
centroid in `core/bottles.py`, so the check does not depend on how the geoms are laid out.
"""

import dataclasses

import mujoco
import numpy as np
import pytest
from support.paths import ROOT
from support.so101 import MJCF, SO101

from beverage_disposal.core.bottle_family import load_family, sample_bottle, scene_for_bottle
from beverage_disposal.core.bottles import capacity_ml, liquid_centroid_m
from beverage_disposal.core.config import load_scene_config
from beverage_disposal.core.kinematics import HOME
from beverage_disposal.core.materials import load_materials
from beverage_disposal.sim.scene import build_scene

pytestmark = pytest.mark.medium

SCENE = load_scene_config(ROOT / "configs" / "scene.toml")
FAMILY = load_family(ROOT / "configs" / "bottles.toml")
PET = load_materials(ROOT / "configs" / "materials.toml")["PET"]
SIZES = list(FAMILY.classes)
PAYLOAD_KG = 0.60  # the safety limit's payload
# Far from the arm, which sits at x = 0 and reaches over the spawn box at 0.27 m.
FAR_X = 0.36
SETTLE_SECONDS = 1.0
WATCH_SECONDS = 1.0
CREEP = 5e-4  # metres the bottle may drift while it is being watched
GRIPPER_HOLD = 0.0
LEVEL_OFF_FLOOR = 1e-3  # metres the bottle may rest off its ideal height


def bottle_of(size, seed=0):
    return sample_bottle(FAMILY, size, PET, np.random.default_rng(seed))


def payload_ml(bottle):
    return min(capacity_ml(bottle.shape), (PAYLOAD_KG - bottle.empty_mass_kg) * 1000.0)


def scene_of(bottle, liquid_ml, density=1.0):
    scene = scene_for_bottle(SCENE, bottle, FAMILY.classes[bottle.size], liquid_ml, density)
    return dataclasses.replace(scene, container_x=FAR_X, container_y=0.0)


def build(bottle, liquid_ml, density=1.0):
    return build_scene(scene_of(bottle, liquid_ml, density), SO101 / MJCF)


def run(model, data, seconds):
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def forward(model):
    """The arm held at HOME by its actuators, the bottle where the scene puts it."""
    data = mujoco.MjData(model)
    data.qpos[:6] = [*HOME, GRIPPER_HOLD]
    data.ctrl[:6] = [*HOME, GRIPPER_HOLD]
    mujoco.mj_forward(model, data)
    return data


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("fill", ["empty", "payload"])
def test_the_model_weighs_the_empty_bottle_plus_the_liquid(size, fill):
    bottle = bottle_of(size)
    ml = 0.0 if fill == "empty" else payload_ml(bottle)
    model = build(bottle, ml)
    mass = model.body_subtreemass[model.body("container").id]
    assert mass == pytest.approx(bottle.empty_mass_kg + ml * 1e-3, rel=1e-9)


@pytest.mark.parametrize("size", SIZES)
def test_the_liquid_sits_at_its_closed_form_centroid(size):
    bottle = bottle_of(size)
    ml = payload_ml(bottle)
    empty, full = forward(build(bottle, 0.0)), forward(build(bottle, ml))
    body = build(bottle, 0.0).body("container").id
    m_liquid = ml * 1e-3
    m_empty = bottle.empty_mass_kg
    moment_full = (m_empty + m_liquid) * full.subtree_com[body][2]
    moment_empty = m_empty * empty.subtree_com[body][2]
    liquid_height = (moment_full - moment_empty) / m_liquid
    assert liquid_height == pytest.approx(liquid_centroid_m(bottle.shape, ml), abs=1e-6)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("fill", ["empty", "payload"])
def test_the_bottle_settles_upright_on_the_table_and_does_not_creep(size, fill):
    bottle = bottle_of(size)
    ml = 0.0 if fill == "empty" else payload_ml(bottle)
    model = build(bottle, ml)
    data = forward(model)
    body = data.body("container")
    run(model, data, SETTLE_SECONDS)
    before = body.xpos.copy()
    run(model, data, WATCH_SECONDS)
    assert body.xmat.reshape(3, 3)[2, 2] > 0.999
    assert np.linalg.norm(body.xpos - before) < CREEP
    assert body.xpos[2] == pytest.approx(
        scene_of(bottle, ml).container.height / 2, abs=LEVEL_OFF_FLOOR
    )


def test_the_bottle_parts_collide_and_the_drawn_liquid_level_does_not():
    bottle = bottle_of("pet_500ml")
    model = build(bottle, 100.0)
    shape = bottle.shape
    for name in ("container", "container_shoulder", "container_neck", "container_ring"):
        assert model.geom(name).contype[0] > 0, name
    assert model.geom("container_level").contype[0] == 0
    assert model.geom("container_level").conaffinity[0] == 0
    ring = model.geom("container_ring")
    assert ring.size[0] == pytest.approx(shape.support_ring_diameter / 2)


def test_the_ring_stands_at_its_drop_below_the_top_of_the_neck():
    bottle = bottle_of("pet_500ml")
    model = build(bottle, 0.0)
    data = forward(model)
    ring = data.geom("container_ring")
    shape = bottle.shape
    total = shape.body_height + shape.shoulder_height + shape.neck_height
    underside = ring.xpos[2] - model.geom("container_ring").size[1]
    # The bottle starts standing on the table, so world height is height above its base.
    assert underside == pytest.approx(total - shape.support_ring_drop, abs=1e-6)


def test_a_scene_without_a_bottle_still_holds_one_can_cylinder():
    model = build_scene(SCENE, SO101 / MJCF)
    assert model.geom("container").type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert "container_shoulder" not in [model.geom(i).name for i in range(model.ngeom)]


def test_the_shoulder_spans_the_height_between_the_body_and_the_neck():
    bottle = bottle_of("pet_1000ml")
    model = build(bottle, 0.0)
    data = forward(model)
    mesh = model.mesh("container_shoulder")
    first = mesh.vertadr[0]
    local = model.mesh_vert[first : first + mesh.vertnum[0]]
    geom = data.geom("container_shoulder")
    world = local @ geom.xmat.reshape(3, 3).T + geom.xpos
    shape = bottle.shape
    assert world[:, 2].min() == pytest.approx(shape.body_height, abs=1e-6)
    assert world[:, 2].max() == pytest.approx(shape.body_height + shape.shoulder_height, abs=1e-6)
    radius = np.hypot(world[:, 0] - FAR_X, world[:, 1])
    assert radius.max() == pytest.approx(shape.body_diameter / 2, abs=1e-6)


@pytest.mark.parametrize("fill", ["part", "brim"])
def test_a_denser_drink_weighs_more_but_stands_at_the_same_height(fill):
    bottle = bottle_of("pet_500ml")
    ml = 100.0 if fill == "part" else capacity_ml(bottle.shape)
    density = 1.05
    model = build(bottle, ml, density)
    body = model.body("container").id
    assert model.body_subtreemass[body] == pytest.approx(
        bottle.empty_mass_kg + ml * density * 1e-3, rel=1e-9
    )
    empty, full = forward(build(bottle, 0.0)), forward(model)
    moment_full = model.body_subtreemass[body] * full.subtree_com[body][2]
    moment_empty = bottle.empty_mass_kg * empty.subtree_com[body][2]
    liquid_height = (moment_full - moment_empty) / (ml * density * 1e-3)
    assert liquid_height == pytest.approx(liquid_centroid_m(bottle.shape, ml), abs=1e-6)
