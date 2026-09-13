"""Unit tests for kinematic_grasp.py's pure pose-offset math.

No ROS/Gazebo dependency -- same pattern as kinematics.py/grasp_geometry.py.
Quaternion convention throughout: (x, y, z, w), matching
geometry_msgs/Quaternion field order.
"""
import math

import pytest

from beverage_disposal_bringup.kinematic_grasp import (
    pose_from_relative,
    pose_relative_to,
)

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def quat_from_axis_angle(axis, angle):
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / norm, ay / norm, az / norm
    s = math.sin(angle / 2.0)
    return (ax * s, ay * s, az * s, math.cos(angle / 2.0))


def assert_positions_close(a, b, tol=1e-9):
    for av, bv in zip(a, b):
        assert abs(av - bv) < tol, f'{a} != {b}'


def assert_quats_close(a, b, tol=1e-9):
    # q and -q represent the same rotation.
    same_sign = all(abs(av - bv) < tol for av, bv in zip(a, b))
    flipped_sign = all(abs(av + bv) < tol for av, bv in zip(a, b))
    assert same_sign or flipped_sign, f'{a} != {b} (as rotations)'


def test_relative_pose_with_identity_reference_is_the_target_itself():
    offset_pos, offset_quat = pose_relative_to(
        (0.0, 0.0, 0.0), IDENTITY_QUAT, (1.0, 2.0, 3.0), IDENTITY_QUAT)
    assert_positions_close(offset_pos, (1.0, 2.0, 3.0))
    assert_quats_close(offset_quat, IDENTITY_QUAT)


def test_relative_pose_accounts_for_reference_position():
    # Target directly "in front of" a reference that's been translated --
    # the offset should be relative to the reference, not absolute.
    offset_pos, _ = pose_relative_to(
        (1.0, 0.0, 0.0), IDENTITY_QUAT, (1.0, 0.0, 0.5), IDENTITY_QUAT)
    assert_positions_close(offset_pos, (0.0, 0.0, 0.5))


def test_relative_pose_accounts_for_reference_orientation():
    # Reference rotated 90deg about Z: what was "ahead" in world X is now
    # "to the side" in the reference's own local frame.
    ref_quat = quat_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2)
    offset_pos, _ = pose_relative_to(
        (0.0, 0.0, 0.0), ref_quat, (1.0, 0.0, 0.0), IDENTITY_QUAT)
    assert_positions_close(offset_pos, (0.0, -1.0, 0.0))


def test_round_trip_relative_then_from_relative_recovers_target():
    ref_pos = (0.3, -0.1, 0.2)
    ref_quat = quat_from_axis_angle((0.2, 0.7, 0.1), 1.1)
    # Normalize the arbitrary axis/angle-derived quat isn't needed since
    # quat_from_axis_angle already normalizes the axis.
    target_pos = (0.35, 0.05, 0.25)
    target_quat = quat_from_axis_angle((0.0, 1.0, 0.0), 0.3)

    offset_pos, offset_quat = pose_relative_to(ref_pos, ref_quat, target_pos, target_quat)
    recovered_pos, recovered_quat = pose_from_relative(ref_pos, ref_quat, offset_pos, offset_quat)

    assert_positions_close(recovered_pos, target_pos, tol=1e-6)
    assert_quats_close(recovered_quat, target_quat, tol=1e-6)


def test_tracking_through_a_moved_reference_moves_the_target_by_the_same_delta():
    """The actual grasp-tracking use case: record the offset once at grasp
    time, then re-derive the target's pose at a LATER reference pose (the
    gripper having moved, e.g. during a lift) -- the target should move by
    exactly the same rigid-body delta as the reference did, not snap back
    to its original world position."""
    grasp_ref_pos = (0.3, 0.0, 0.2)
    grasp_ref_quat = IDENTITY_QUAT
    can_pos_at_grasp = (0.3, 0.0, 0.21)
    can_quat_at_grasp = IDENTITY_QUAT

    offset_pos, offset_quat = pose_relative_to(
        grasp_ref_pos, grasp_ref_quat, can_pos_at_grasp, can_quat_at_grasp)

    # Gripper lifts straight up by 5cm, no rotation.
    lifted_ref_pos = (0.3, 0.0, 0.25)
    lifted_ref_quat = IDENTITY_QUAT

    tracked_pos, _ = pose_from_relative(lifted_ref_pos, lifted_ref_quat, offset_pos, offset_quat)
    assert_positions_close(tracked_pos, (0.3, 0.0, 0.26), tol=1e-9)


def test_tracking_through_a_rotated_reference_carries_the_offset_along():
    """If the gripper also rotates (e.g. the arm's base rotates while
    holding the object), the tracked pose must rotate the offset with it,
    not just translate it -- otherwise a held object would appear to slide
    sideways out of the gripper as the arm turns."""
    grasp_ref_pos = (0.3, 0.0, 0.2)
    grasp_ref_quat = IDENTITY_QUAT
    can_pos_at_grasp = (0.35, 0.0, 0.2)  # 5cm "ahead" of the gripper in +X

    offset_pos, offset_quat = pose_relative_to(
        grasp_ref_pos, grasp_ref_quat, can_pos_at_grasp, IDENTITY_QUAT)

    # Gripper's base rotates 90deg about Z, staying at the same position.
    rotated_ref_quat = quat_from_axis_angle((0.0, 0.0, 1.0), math.pi / 2)
    tracked_pos, _ = pose_from_relative(grasp_ref_pos, rotated_ref_quat, offset_pos, offset_quat)

    # What was "ahead" (+X) should now be "to the side" (+Y) in world frame.
    assert_positions_close(tracked_pos, (0.3, 0.05, 0.2), tol=1e-6)
