"""Where the two jaws are, read off the colliding meshes in the gripper frame."""

import numpy as np

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


# Half the depth band along the fingers that counts as "at the can", in metres. The moving jaw
# swings through it on the way open and leaves it entirely at about 1 rad.
FINGER_BAND = 0.02


def jaw_gap(model, data, depth):
    """Room between the fixed jaw and the nearest part of the moving jaw within the finger band.

    Infinite when the moving jaw has swung clear of the band, so nothing in it blocks a container.
    """
    fixed = in_gripper_frame(data, jaw_vertices(model, data, "gripper", "wrist_roll_follower"))
    moving = in_gripper_frame(data, jaw_vertices(model, data, "moving_jaw_so101_v1", "moving_jaw"))
    fixed = fixed[np.abs(fixed[:, 2] - depth) < FINGER_BAND]
    moving = moving[np.abs(moving[:, 2] - depth) < FINGER_BAND]
    if not len(moving):
        return float("inf")
    return moving[:, 0].min() - fixed[:, 0].max()
