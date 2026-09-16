"""Fixed Cartesian/joint targets for the bin, in the same (radius, height,
pitch) plane grasp_geometry.py already uses for the can.

Unlike the can, the bin doesn't move -- there's no live pose to adapt to,
just one fixed target derived once from the bin's own world-file <pose>.
The azimuth uses the same negated-atan2 convention pick_and_lift.py already
uses for the can's azimuth (verified live there: this URDF's shoulder_pan
positive direction is inverted relative to plain world-frame atan2), reused
deliberately rather than re-derived, to avoid reintroducing that exact
sign bug for a second target.

BIN_PITCH=0.0 (gripper level) was chosen, not guessed: sweeping
solve_arm_ik over pitch/height at the bin's radius found this is the value
that reaches both a safe hover height above the bin's walls and a release
height inside it, with real margin from every joint limit -- unlike the
can's own grasp/hover targets, which sit exactly at wrist_flex's hard
limit. See CLAUDE.md's Iteration 5 entry for the full numbers.

What this module does NOT solve: the transit path connecting a post-lift
pose to BIN_HOVER. Both targets here are individually collision-free by
construction (they target the bin's exact horizontal center), but nothing
here certifies the base-rotation sweep between them is collision-free with
the table, the bin's own walls, or the arm's other links -- that's a live-
verification question, not a code one.
"""
import math

from .grasp_geometry import WRIST_ROLL
from .kinematics import solve_arm_ik

# Tied by comment (not by any shared code -- kept in sync by hand, same
# documented gap class as can_mass_kg between spawn_arm.launch.py and
# pick_and_lift.py) to the bin's real <pose> in
# worlds/beverage_disposal_world.sdf.
BIN_MODEL_X = -0.07
BIN_MODEL_Y = -0.32

BIN_AZIMUTH = -math.atan2(BIN_MODEL_Y, BIN_MODEL_X)
BIN_RADIUS = math.hypot(BIN_MODEL_X, BIN_MODEL_Y)

# Tied by comment to the bin's 5-box shell geometry in the world file (four
# 0.3m-tall walls).
BIN_WALL_TOP_Z = 0.30

BIN_PITCH = 0.0

# BIN_WALL_TOP_Z plus a 3cm margin -- deliberately larger than the lift's
# own 1.5cm success margin, since this pose has to survive a much longer
# base-rotation transit with more accumulated positioning uncertainty than
# the already-proven, short grasp reach.
BIN_HOVER_HEIGHT_M = BIN_WALL_TOP_Z + 0.03

# Inside the bin's interior height band (floor top at 0.01m, wall top at
# BIN_WALL_TOP_Z), confirmed reachable with real slack rather than sitting
# at a limit.
BIN_RELEASE_HEIGHT_M = 0.17


def bin_hover_joint_targets(wrist_roll=WRIST_ROLL, elbow_up=False, clamp_near_limits=0.0):
    """(shoulder_lift, elbow_flex, wrist_flex) for the hover pose above the
    bin, gripper still closed on the can -- reached before descending to
    bin_release_joint_targets."""
    return solve_arm_ik(BIN_RADIUS, BIN_HOVER_HEIGHT_M, BIN_PITCH, wrist_roll,
                         elbow_up=elbow_up, clamp_near_limits=clamp_near_limits)


def bin_release_joint_targets(wrist_roll=WRIST_ROLL, elbow_up=False, clamp_near_limits=0.0):
    """(shoulder_lift, elbow_flex, wrist_flex) for the release pose inside
    the bin -- where the kinematic lock is stopped and the gripper reopens."""
    return solve_arm_ik(BIN_RADIUS, BIN_RELEASE_HEIGHT_M, BIN_PITCH, wrist_roll,
                         elbow_up=elbow_up, clamp_near_limits=clamp_near_limits)
