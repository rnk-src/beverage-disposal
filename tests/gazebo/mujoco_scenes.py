"""The same scenes as the Gazebo worlds, built in MuJoCo, so the two engines can be compared."""

import math

import mujoco

DT = 0.001
INCLINE = math.radians(30)


def run(model, steps):
    data = mujoco.MjData(model)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data


def ball_height_on_ground(steps):
    """Height of a 1 kg, 0.1 m radius ball released 0.1 m above a ground whose top is at z = 0."""
    model = mujoco.MjModel.from_xml_string(f"""
        <mujoco><option timestep="{DT}"/><worldbody>
          <geom type="box" size="5 5 0.1" pos="0 0 -0.1"/>
          <body pos="0 0 0.2"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
        </worldbody></mujoco>""")
    return run(model, steps).qpos[2]


def block_travel_on_incline(friction, steps, slab_friction=None):
    """Distance a 1 kg block, starting at rest on the slope, has moved after `steps`.

    The slab gets `slab_friction` when given, else the same coefficient as the block.
    """
    s, c = math.sin(INCLINE), math.cos(INCLINE)
    tilt = f"{math.cos(INCLINE / 2)} 0 {math.sin(INCLINE / 2)} 0"
    block_surface = f'friction="{friction} 0.005 0.0001"'
    slab_surface = f'friction="{slab_friction or friction} 0.005 0.0001"'
    model = mujoco.MjModel.from_xml_string(f"""
        <mujoco><option timestep="{DT}"/><worldbody>
          <geom type="box" size="5 5 0.1" pos="{-0.1 * s} 0 {-0.1 * c}" quat="{tilt}"
                {slab_surface}/>
          <body pos="{0.05 * s} 0 {0.05 * c}" quat="{tilt}"><freejoint/>
            <geom type="box" size="0.1 0.1 0.05" mass="1" {block_surface}/></body>
        </worldbody></mujoco>""")
    start = mujoco.MjData(model)
    mujoco.mj_forward(model, start)
    end = run(model, steps)
    return math.dist(start.qpos[[0, 2]], end.qpos[[0, 2]])
