"""The simulated world: the arm on a table with one container standing beside it."""

import mujoco

from beverage_disposal.core.containers import total_mass
from beverage_disposal.sim.arm import arm_spec

# MuJoCo's default torsional and rolling friction; only the sliding coefficient is configured.
# Two touching geoms use the larger of their sliding coefficients, so the table's counts too.
TORSIONAL_FRICTION = 0.005
ROLLING_FRICTION = 0.0001
# MuJoCo's default pyramidal cone with impratio 1 lets a resting can creep 1.4 mm under a push that
# is a seventh of its friction limit (measured, 2.1a); impratio 10 stiffens the friction and the
# elliptic cone makes it isotropic. These are solver settings, not physical ones.
CONE = mujoco.mjtCone.mjCONE_ELLIPTIC
IMPRATIO = 10


def build_scene(config, mjcf):
    """Compile the arm at `mjcf` together with the table and container described by `config`."""
    spec = arm_spec(mjcf)
    spec.option.cone = CONE
    spec.option.impratio = IMPRATIO
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
    can = config.container
    body = world.add_body(
        name="container", pos=[config.container_x, config.container_y, can.height / 2]
    )
    body.add_freejoint()
    body.add_geom(
        name="container",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[can.diameter / 2, can.height / 2, 0.0],
        mass=total_mass(can, config.fill_ml),
        friction=[can.friction, TORSIONAL_FRICTION, ROLLING_FRICTION],
    )
    return spec.compile()
