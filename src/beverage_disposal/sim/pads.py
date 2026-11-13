"""Two flat pads stand in for the gripper's collision geometry.

The vendored jaw meshes collide by their convex hulls, and the hull of the fixed jaw bulges into
the grasp gap (found in 1.1). The moving jaw is also tilted by about 50 degrees at the opening a
66 mm can needs, so it pinches at a different height from the fixed jaw and tips the can. So the
jaw meshes and the servo housing stop colliding here and two boxes take their place: the fixed one
on the fixed finger's inner face, the moving one attached to the moving jaw and placed so that its
face is parallel to the fixed one at `design_angle` and one container diameter away.
"""

import mujoco
import numpy as np

from beverage_disposal.core.config import SceneConfig
from beverage_disposal.sim.arm import load_arm

FIXED_JAW = "gripper"
MOVING_JAW = "moving_jaw_so101_v1"
# The fixed finger's inner face in the gripper frame, metres (7.9 mm off the roll axis, D-77).
FIXED_FACE_X = -0.0079
# MuJoCo's default rolling friction; sliding and torsional friction are configured.
ROLLING_FRICTION = 0.0001
# Contact dimension 4 adds friction against twisting about the contact normal (3 has none).
PAD_CONDIM = 4


def _moving_jaw_in_gripper_frame(arm, angle):
    """Rotation and position of the moving jaw body in the gripper frame at gripper `angle`."""
    data = mujoco.MjData(arm)
    data.qpos[arm.joint("gripper").qposadr[0]] = angle
    mujoco.mj_kinematics(arm, data)
    gripper, jaw = data.body(FIXED_JAW), data.body(MOVING_JAW)
    rotation = gripper.xmat.reshape(3, 3).T @ jaw.xmat.reshape(3, 3)
    return rotation, gripper.xmat.reshape(3, 3).T @ (jaw.xpos - gripper.xpos)


def add_gripper_pads(spec, config: SceneConfig, mjcf) -> None:
    """Switch the jaw collisions off and add the two pads to `spec`."""
    pads = config.gripper_pads
    half = [pads.thickness / 2, pads.half_width, pads.half_height]
    friction = [pads.friction, pads.torsional_friction, ROLLING_FRICTION]
    solref = [config.physics.contact_time_constant, config.physics.contact_damping_ratio]

    rotation, jaw_origin = _moving_jaw_in_gripper_frame(load_arm(mjcf), pads.design_angle)
    # Where the moving pad's centre must be, in the gripper frame, so its face is one diameter
    # from the fixed face; then the same point in the jaw's own frame.
    centre = np.array(
        [FIXED_FACE_X + config.container.diameter + pads.thickness / 2, 0.0, pads.depth]
    )
    in_jaw = rotation.T @ (centre - jaw_origin)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.T.flatten())

    for geom in spec.geoms:
        if geom.parent.name in (FIXED_JAW, MOVING_JAW):
            geom.contype = 0
            geom.conaffinity = 0
    spec.body(FIXED_JAW).add_geom(
        name="fixed_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half,
        pos=[FIXED_FACE_X - pads.thickness / 2, 0.0, pads.depth],
        friction=friction,
        solref=solref,
        condim=PAD_CONDIM,
    )
    spec.body(MOVING_JAW).add_geom(
        name="moving_pad",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half,
        pos=in_jaw.tolist(),
        quat=quat.tolist(),
        friction=friction,
        solref=solref,
        condim=PAD_CONDIM,
    )
