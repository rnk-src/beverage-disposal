"""Our arm model: the vendored SO-101 plus a tool center point (TCP) placed from the jaws."""

import mujoco
import numpy as np
import pytest
from support.so101 import MJCF, SO101

from beverage_disposal.sim.arm import load_arm


@pytest.fixture(scope="module")
def model():
    return load_arm(SO101 / MJCF)


def test_the_arm_model_has_a_tcp_site_on_the_gripper_body(model):
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    assert site >= 0
    assert model.site_bodyid[site] == model.body("gripper").id


# Half the thickness of the slice of jaw vertices taken at one depth along the fingers, in metres.
SLICE = 0.001


def in_gripper_frame(data, world_points):
    gripper = data.body("gripper")
    return (world_points - gripper.xpos) @ gripper.xmat.reshape(3, 3)


def jaw_vertices(model, data, body, mesh_prefix):
    """World positions of the vertices of the colliding mesh on `body` whose mesh name matches."""
    for geom in range(model.ngeom):
        mesh = model.geom_dataid[geom]
        if (
            model.geom_bodyid[geom] == model.body(body).id
            and model.geom_contype[geom] > 0
            and model.mesh(mesh).name.startswith(mesh_prefix)
        ):
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            local = model.mesh_vert[start : start + count]
            return data.geom_xpos[geom] + local @ data.geom_xmat[geom].reshape(3, 3).T
    raise AssertionError(f"no colliding {mesh_prefix} mesh on {body}")


def jaw_faces(model, data, depth):
    """Inner faces of both jaws at one depth, in the gripper frame: (fixed x, moving x, pads' y).

    The jaws close along the gripper's x axis: the fixed jaw (part of the wrist-roll follower mesh)
    on the negative side, the moving jaw on the positive side.
    """
    fixed = in_gripper_frame(data, jaw_vertices(model, data, "gripper", "wrist_roll_follower"))
    moving = in_gripper_frame(data, jaw_vertices(model, data, "moving_jaw_so101_v1", "moving_jaw"))
    fixed = fixed[np.abs(fixed[:, 2] - depth) < SLICE]
    moving = moving[np.abs(moving[:, 2] - depth) < SLICE]
    assert len(fixed) and len(moving), "the slice misses a jaw: the depth is not on the fingers"
    pads_y = (fixed[:, 1].min() + fixed[:, 1].max() + moving[:, 1].min() + moving[:, 1].max()) / 4
    return fixed[:, 0].max(), moving[:, 0].min(), pads_y


def at_rest(model):
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    return data


def test_at_gripper_zero_the_tcp_is_halfway_between_the_jaws_at_the_depth_of_gripperframe(model):
    # D-15 and D-77: the TCP is the midpoint between the finger pads at gripper angle 0. Upstream's
    # gripperframe marks a point on the fixed jaw's pad, so it gives the depth along the fingers.
    data = at_rest(model)
    tcp = in_gripper_frame(data, data.site("tcp").xpos)
    depth = in_gripper_frame(data, data.site("gripperframe").xpos)[2]
    fixed_x, moving_x, pads_y = jaw_faces(model, data, depth)
    assert tcp[2] == pytest.approx(depth, abs=1e-6)
    assert tcp[0] == pytest.approx((fixed_x + moving_x) / 2, abs=1e-4)
    assert tcp[1] == pytest.approx(pads_y, abs=3e-4)


def unit(vector):
    return vector / np.linalg.norm(vector)


def test_the_tcp_z_axis_points_to_the_fingertips_and_x_towards_the_moving_jaw(model):
    # D-77: the approach-vector convention. 1.3 reads the grasp pitch off this z axis.
    data = at_rest(model)
    axes = data.site("tcp").xmat.reshape(3, 3)
    towards_fingertips = unit(data.site("tcp").xpos - data.body("gripper").xpos)
    depth = in_gripper_frame(data, data.site("tcp").xpos)[2]
    fixed_x, moving_x, _ = jaw_faces(model, data, depth)
    closing = data.body("gripper").xmat.reshape(3, 3)[:, 0] * np.sign(moving_x - fixed_x)
    assert axes[:, 2] @ towards_fingertips > 0.999
    assert axes[:, 0] @ closing > 0.999


def random_poses(model, count, seed):
    return np.random.default_rng(seed).uniform(*model.jnt_range.T, size=(count, model.nq))


def largest_tcp_move_while_sweeping(model, joint, poses, steps=9):
    """How far the TCP moves while `joint` sweeps its whole range from each pose, in metres."""
    data = mujoco.MjData(model)
    address = model.joint(joint).qposadr[0]
    largest = 0.0
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        start = data.site("tcp").xpos.copy()
        for angle in np.linspace(*model.joint(joint).range, steps):
            data.qpos[address] = angle
            mujoco.mj_kinematics(model, data)
            largest = max(largest, np.linalg.norm(data.site("tcp").xpos - start))
    return largest


def test_rolling_the_wrist_does_not_move_the_tcp(model):
    # A point on the roll axis stays put; upstream's gripperframe, 7.9 mm off it, moves 15.8 mm.
    # 1.2 and 1.3 rely on this: with the roll locked, the tip depends on four joints only.
    assert largest_tcp_move_while_sweeping(model, "wrist_roll", random_poses(model, 20, 1)) < 1e-6


def test_opening_the_jaws_does_not_move_the_tcp(model):
    # D-77 fixes the TCP to the gripper body at gripper angle 0, so the jaw hinge must not carry it.
    assert largest_tcp_move_while_sweeping(model, "gripper", random_poses(model, 20, 2)) < 1e-9


def test_the_arm_model_is_the_vendored_model_plus_one_site(model):
    vendored = mujoco.MjModel.from_xml_path(str(SO101 / MJCF))
    for count in ["nbody", "njnt", "nq", "nu", "ngeom", "nmesh"]:
        assert getattr(model, count) == getattr(vendored, count), count
    for field in [
        "body_mass",
        "body_inertia",
        "jnt_range",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "actuator_ctrlrange",
        "geom_friction",
    ]:
        np.testing.assert_array_equal(getattr(model, field), getattr(vendored, field), field)
    assert model.opt.timestep == vendored.opt.timestep
    np.testing.assert_array_equal(model.opt.gravity, vendored.opt.gravity)
    sites = [model.site(i).name for i in range(model.nsite)]
    assert sites == [vendored.site(i).name for i in range(vendored.nsite)] + ["tcp"]
