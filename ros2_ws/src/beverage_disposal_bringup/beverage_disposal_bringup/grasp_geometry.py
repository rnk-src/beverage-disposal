"""Derives grasp/hover Cartesian targets in the arm's vertical plane from
the can's own *live measured* position, instead of the fixed joint-angle
constants this project used before.

Why this exists: the pipeline's pre-grasp retraction (HOME->SAFE_RAISE)
causes a small, real, run-to-run-variable disturbance to the can's
position (see commit-notes/10-iteration-3-pick-and-lift.md). The pipeline
already recomputed the *azimuth* to the can's actual post-disturbance
position (shoulder_pan = atan2(can_y, can_x)), but the *reach* -- the
radius and height the gripper opens/closes at -- was a fixed joint-angle
constant tuned against one specific can spawn position. Any radial or
vertical shift in the can was therefore never compensated for: the gripper
would open/close at the original calibrated radius regardless of where the
can actually ended up, which is a real, uncompensated aim error, not just
measurement noise.

This module makes the reach adaptive too: grasp_pose_for_can and
hover_pose_for_can take the can's live-measured (radius, height) and
return a Cartesian (radius, height, pitch) target that tracks it, via a
fixed offset calibrated once from this iteration's own proven, live-sim-
verified grasp -- so the *gripper's reach relative to the can* is what's
preserved, not the gripper's reach relative to a specific point in the
world the can happened to spawn at.
"""
import math

from .kinematics import forward_kinematics

WRIST_ROLL = 1.5708

# This iteration's own empirically-verified, live-sim-proven joint tuples
# (see commit-notes/10-iteration-3-pick-and-lift.md) -- the source of
# truth the offsets below are derived from, not independently guessed.
_KNOWN_GOOD_GRASP_JOINTS = (-0.35, -0.35, 1.6)
_KNOWN_GOOD_HOVER_JOINTS = (-0.55, -0.55, 1.6)

# The can's own true position (radius from the shoulder_pan axis, height)
# at the moment those joint tuples were calibrated against it -- read from
# live TF during that verification (see commit-notes/10). This is the
# reference point the offsets below are relative to; it is not used again
# after being validated by test_grasp_geometry.py's calibration-position
# round-trip tests, since a live grasp always measures the can's *current*
# position instead.
CALIBRATION_CAN_RADIUS = 0.26977
CALIBRATION_CAN_HEIGHT = 0.21112

_grasp_radius, _grasp_height, _GRASP_PITCH = forward_kinematics(*_KNOWN_GOOD_GRASP_JOINTS, WRIST_ROLL)
_hover_radius, _hover_height, _HOVER_PITCH = forward_kinematics(*_KNOWN_GOOD_HOVER_JOINTS, WRIST_ROLL)

# Offset of the gripper's target (radius, height) from the can's own
# (radius, height), holding across any can position because it represents
# a fixed physical relationship -- how far into/around the can's own body
# the gripper needs to reach to grip it -- not a property of any specific
# point in the world.
_GRASP_RADIUS_OFFSET = _grasp_radius - CALIBRATION_CAN_RADIUS
_GRASP_HEIGHT_OFFSET = _grasp_height - CALIBRATION_CAN_HEIGHT
_HOVER_RADIUS_OFFSET = _hover_radius - CALIBRATION_CAN_RADIUS
_HOVER_HEIGHT_OFFSET = _hover_height - CALIBRATION_CAN_HEIGHT


def grasp_pose_for_can(can_radius, can_height):
    """(radius, height, pitch) the gripper should close at to grip a can
    whose own live-measured position is (can_radius, can_height) -- radius
    being the can's distance from the shoulder_pan axis in the arm's
    vertical plane (i.e. after the caller has already picked that plane
    via the can's azimuth), height being its height above base_link."""
    return can_radius + _GRASP_RADIUS_OFFSET, can_height + _GRASP_HEIGHT_OFFSET, _GRASP_PITCH


def hover_pose_for_can(can_radius, can_height):
    """(radius, height, pitch) the gripper should be at, gripper open,
    immediately before descending to grasp_pose_for_can's target."""
    return can_radius + _HOVER_RADIUS_OFFSET, can_height + _HOVER_HEIGHT_OFFSET, _HOVER_PITCH
