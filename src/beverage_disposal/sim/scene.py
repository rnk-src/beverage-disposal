"""The simulated world: the arm on a table with one container standing beside it."""

from pathlib import Path

import mujoco

from beverage_disposal.core.config import BinConfig, PadConfig, SceneConfig, SetAsideConfig
from beverage_disposal.core.containers import total_mass
from beverage_disposal.sim.arm import arm_spec
from beverage_disposal.sim.bottle import add_bottle
from beverage_disposal.sim.pads import add_gripper_pads

# MuJoCo's default torsional and rolling friction; only the sliding coefficient is configured.
# Two touching geoms use the larger of their sliding coefficients, so the table's counts too.
TORSIONAL_FRICTION = 0.005
ROLLING_FRICTION = 0.0001
# MuJoCo's default pyramidal cone with impratio 1 lets a resting can creep 1.4 mm under a push that
# is a seventh of its friction limit (measured, 2.1a); impratio 10 stiffens the friction and the
# elliptic cone makes it isotropic. These are solver settings, not physical ones.
CONE = mujoco.mjtCone.mjCONE_ELLIPTIC
IMPRATIO = 10
# Contact dimension 4 gives the container friction against twisting on the table and the pads.
CONTAINER_CONDIM = 4
# The set-aside marker is paint: thin and without collisions.
MARKER_THICKNESS = 0.001
MARKER_COLOUR = [0.9, 0.6, 0.1, 1.0]


def build_scene(config: SceneConfig, mjcf: str | Path) -> mujoco.MjModel:
    """Compile the arm at `mjcf` together with the table and container described by `config`."""
    spec = arm_spec(mjcf)
    spec.option.cone = CONE
    spec.option.impratio = IMPRATIO
    spec.option.timestep = config.physics.timestep
    add_gripper_pads(spec, config, mjcf)
    spec.actuator("gripper").forcerange = [-config.grasp.force_limit, config.grasp.force_limit]
    world = spec.worldbody
    table = config.table
    # The table's top is z = 0 of the arm's base frame, where the base body sits.
    world.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[table.half_x, table.half_y, table.thickness / 2],
        pos=[table.center_x, 0.0, -table.thickness / 2],
        friction=[table.friction, TORSIONAL_FRICTION, ROLLING_FRICTION],
    )
    _add_bin(world, config.bin)
    _add_pad(world, config.pad)
    _add_set_aside_marker(world, config.set_aside)
    can = config.container
    friction = [can.friction, can.torsional_friction, ROLLING_FRICTION]
    if config.bottle is not None:
        add_bottle(
            spec,
            world,
            config.bottle,
            config.fill_ml,
            x=config.container_x,
            y=config.container_y,
            friction=friction,
            condim=CONTAINER_CONDIM,
        )
    else:
        body = world.add_body(
            name="container", pos=[config.container_x, config.container_y, can.height / 2]
        )
        body.add_freejoint(name="container")
        body.add_geom(
            name="container",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[can.diameter / 2, can.height / 2, 0.0],
            mass=total_mass(can, config.fill_ml),
            friction=friction,
            condim=CONTAINER_CONDIM,
        )
    _soften(spec, config)
    return spec.compile()


def _friction(coefficient):
    return [coefficient, TORSIONAL_FRICTION, ROLLING_FRICTION]


def _add_box(world, name, center, half, friction=None):
    box = world.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half,
        pos=center,
    )
    if friction is not None:
        box.friction = _friction(friction)
    return box


def _add_bin(world, bin_: BinConfig) -> None:
    """A floor and four walls; `bin_.height` is the walls' top above the table."""
    outer_x = bin_.inner_x / 2 + bin_.wall
    outer_y = bin_.inner_y / 2 + bin_.wall
    _add_box(
        world,
        "bin_floor",
        [bin_.center_x, bin_.center_y, bin_.floor / 2],
        [outer_x, outer_y, bin_.floor / 2],
        bin_.friction,
    )
    middle = bin_.height / 2
    for side, sign in [("east", 1), ("west", -1)]:
        _add_box(
            world,
            f"bin_wall_{side}",
            [bin_.center_x + sign * (bin_.inner_x / 2 + bin_.wall / 2), bin_.center_y, middle],
            [bin_.wall / 2, outer_y, middle],
            bin_.friction,
        )
    for side, sign in [("north", 1), ("south", -1)]:
        _add_box(
            world,
            f"bin_wall_{side}",
            [bin_.center_x, bin_.center_y + sign * (bin_.inner_y / 2 + bin_.wall / 2), middle],
            [bin_.inner_x / 2, bin_.wall / 2, middle],
            bin_.friction,
        )


def _add_pad(world, pad: PadConfig) -> None:
    _add_box(
        world,
        "pad",
        [pad.center_x, pad.center_y, pad.height / 2],
        [pad.half_x, pad.half_y, pad.height / 2],
        pad.friction,
    )


def _add_set_aside_marker(world, area: SetAsideConfig) -> None:
    """A thin painted rectangle that is seen but never touched: contype and conaffinity are 0."""
    marker = _add_box(
        world,
        "set_aside_marker",
        [area.center_x, area.center_y, MARKER_THICKNESS / 2],
        [area.half_x, area.half_y, MARKER_THICKNESS / 2],
    )
    marker.contype = 0
    marker.conaffinity = 0
    marker.rgba = MARKER_COLOUR


def _soften(spec, config: SceneConfig) -> None:
    """The configured contact softness on the scene's own geoms (the pads get it in sim/pads.py)."""
    solref = [config.physics.contact_time_constant, config.physics.contact_damping_ratio]
    scene_geoms = {
        "table",
        "container",
        "container_shoulder",
        "container_neck",
        "container_ring",
        "pad",
        "bin_floor",
    } | {f"bin_wall_{side}" for side in ("north", "south", "east", "west")}
    for geom in spec.geoms:
        if geom.name in scene_geoms:
            geom.solref = solref
