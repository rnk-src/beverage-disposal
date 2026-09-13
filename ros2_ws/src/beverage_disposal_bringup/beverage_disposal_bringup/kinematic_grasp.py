"""Pure math for kinematic snap-grasp: once a real grip is detected, track
the grasped object's world pose as a fixed offset from the gripper's own
world pose, instead of relying on friction/contact physics to keep it in
hand through a lift.

Why this exists (see commit-notes/10-iteration-3-pick-and-lift.md's
2026-09-13 research entry): no comparable SO-101 simulation environment
found in that research -- not LeRobot's own official MuJoCo env, not
sim-engine, not pick-101 -- actually keeps an object in a gripper via real
friction-contact physics either. sim-engine's own docs say it plainly:
"Real friction-based grasping in simulation is unreliable... SimEngine
uses kinematic snap-grasp, the same approach used by Robosuite, MetaWorld,
and other standard benchmarks." This module implements that same
approach for this project -- triggered off this project's own
contact-detection signal (a real stalled-under-effort readout from
gripper_control.py, not sim-engine's simpler proximity+angle heuristic),
not a wholesale reinvention of the technique.

Quaternion convention throughout: (x, y, z, w), matching
geometry_msgs/Quaternion's field order, since that's what every pose this
module touches (bridged from Gazebo, or set via ros_gz_interfaces/srv/
SetEntityPose) already uses.
"""


def _quat_conjugate(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_rotate_vector(q, v):
    qv = (v[0], v[1], v[2], 0.0)
    rotated = _quat_multiply(_quat_multiply(q, qv), _quat_conjugate(q))
    return (rotated[0], rotated[1], rotated[2])


def pose_relative_to(reference_position, reference_orientation,
                      target_position, target_orientation):
    """The target's pose expressed in the reference's own local frame --
    call this once, at the moment a grasp is detected, with the gripper as
    the reference and the grasped object as the target, to get the fixed
    offset to track for the rest of the hold."""
    inverse_reference_orientation = _quat_conjugate(reference_orientation)
    delta = tuple(t - r for t, r in zip(target_position, reference_position))
    offset_position = _quat_rotate_vector(inverse_reference_orientation, delta)
    offset_orientation = _quat_multiply(inverse_reference_orientation, target_orientation)
    return offset_position, offset_orientation


def pose_from_relative(reference_position, reference_orientation,
                        offset_position, offset_orientation):
    """Inverse of pose_relative_to: composes a fixed relative offset with
    the reference's CURRENT pose to get the target's tracked world pose.
    Call this every control tick while a kinematic grasp is held, passing
    the gripper's live pose as the reference and the offset recorded at
    grasp time -- this is what makes the held object move rigidly with the
    gripper through a lift, a rotation, or both."""
    rotated_offset = _quat_rotate_vector(reference_orientation, offset_position)
    target_position = tuple(r + o for r, o in zip(reference_position, rotated_offset))
    target_orientation = _quat_multiply(reference_orientation, offset_orientation)
    return target_position, target_orientation
