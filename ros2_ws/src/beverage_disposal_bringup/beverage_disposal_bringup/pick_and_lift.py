"""Real, committed SO-101 pick-and-lift pipeline (Iteration 3).

Replaces the old, stale pick_and_place_demo.py (a pre-SO-101 relic still
using OpenManipulator-X joint names/values -- never wired to this arm at
all). Every prior SO-101 grasp trial in this project's history was run
through throwaway /tmp scripts on the VM that no longer exist; this is the
first time the pipeline exists as real, tested, committed code.

Built from this iteration's own hard-won, live-verified findings (see
commit-notes/10-iteration-3-pick-and-lift.md for the full investigation):

  - grasp_geometry.py's grasp_joint_targets()/hover_joint_targets() give a
    joint target that tracks the can's *live measured* position, not a
    fixed constant calibrated to one spawn.
  - The approach still goes through a HOVER waypoint above the can before
    descending -- moving there directly from HOME, low and fast, risks
    sweeping an arm link through the can before the gripper ever gets
    near it, the same "joint-space interpolation between two clean
    endpoints doesn't guarantee a clean path between them" lesson this
    project has hit before (commit-notes/10). But the final HOVER->GRASP
    descent is a single, direct FollowJointTrajectory move -- not a
    multi-step interpolated Cartesian descent, and not a separate
    last-mile correction move right before closing. A real A/B comparison
    (2026-09-11 session) found that interpolated descent and correction
    step were themselves the cause of degraded closure quality, not a fix
    for it: one direct hop from hover to the grasp target produced
    consistently better grip (9 of 11 trials in the proven 0.31-0.52 rad /
    real sustained-effort range) than the more "careful" multi-step
    approach.
  - Closing the gripper now goes through gripper_action_server.py's
    contact-detection redesign (see gripper_control.py): it commands a
    genuine fully-closed target and lets the controller itself distinguish
    "reached the number" from "stopped early against the can," rather than
    always driving to one hand-picked "closed enough for this can" position
    the way earlier throwaway scripts did.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

from .grasp_geometry import WRIST_ROLL, grasp_joint_targets, hover_joint_targets
from .gripper import send_gripper_goal
from .motion import move_to_joint_position

GRIPPER_JOINT_NAME = 'gripper_joint'

ARM_JOINTS = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_flex_joint',
    'wrist_flex_joint',
    'wrist_roll_joint',
]

HOME = (0.0, 0.0, 0.0, 0.0, 0.0)
GRIPPER_OPEN = 1.5
GRIPPER_CLOSED = 0.0
GRIPPER_CLOSE_MAX_EFFORT = 10.0

# How high the can needs to have risen, and stayed risen, to count as a
# real lift rather than a transient bounce or a grip that let go under
# load. See commit-notes/10: several past configurations showed a real
# transient rise (up to ~0.83cm) that still fell short of a lift a caller
# should trust -- 1.5cm was this iteration's own established bar for "real
# margin, not noise."
LIFT_SUCCESS_HEIGHT_MARGIN = 0.015
LIFT_HOLD_CHECK_SEC = 2.0


# How long to let the gripper actually move before reading its position/
# effort back out. gripper_action_server.py's own action result isn't
# usable here for the close goal: when the node is launched with
# hold_after_reaching:=true (needed so the grip survives the following
# lift move), the goal doesn't produce a terminal result until
# hold_timeout elapses, and the whole point of holding is to keep
# squeezing *while* the arm lifts, not to finish holding before the lift
# starts. So this reads /joint_states directly instead of waiting on the
# action -- see gripper.py's send_gripper_goal docstring.
GRIPPER_SETTLE_SEC = 1.5


class PickAndLiftDemo(Node):

    def __init__(self):
        # Every duration in this node (gripper settle waits, the lift-hold
        # check) is measured via self.get_clock().now(). use_sim_time is
        # already auto-declared by rclpy.Node itself (declaring it again
        # raises ParameterAlreadyDeclaredException), so it's set here via
        # parameter_overrides instead. Without this, the clock defaults to
        # the wall clock, not /clock -- harmless when the sim runs at
        # roughly real speed, but under real VM load (e.g. a trial batch
        # that relaunches Gazebo back-to-back many times) the sim can run
        # noticeably slower than real time, so a wall-clock wait can elapse
        # before the thing it's supposed to be waiting for (the gripper
        # settling, the arm finishing its trajectory) has actually happened
        # in sim time -- producing readings taken mid-motion, not at rest.
        super().__init__(
            'pick_and_lift_demo',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.can_pose = None
        self.gripper_position = 0.0
        self.gripper_effort = 0.0
        self.create_subscription(TFMessage, 'world_pose', self._pose_cb, 10)
        self.create_subscription(JointState, 'joint_states', self._joint_state_cb, 10)

    def _pose_cb(self, msg):
        for t in msg.transforms:
            if t.child_frame_id == 'can':
                tl = t.transform.translation
                self.can_pose = (tl.x, tl.y, tl.z)

    def _joint_state_cb(self, msg):
        if GRIPPER_JOINT_NAME not in msg.name:
            return
        index = msg.name.index(GRIPPER_JOINT_NAME)
        self.gripper_position = msg.position[index]
        if msg.effort:
            self.gripper_effort = msg.effort[index]

    def wait_for_can_pose(self, timeout_sec=10.0):
        end_time = self.get_clock().now().nanoseconds + int(timeout_sec * 1e9)
        while self.can_pose is None and self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.2)
        return self.can_pose

    def _spin_for(self, duration_sec):
        end_time = self.get_clock().now().nanoseconds + int(duration_sec * 1e9)
        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)

    def move_arm(self, joint_positions, duration_sec=3.0):
        return move_to_joint_position(
            self, ARM_JOINTS, list(joint_positions), duration_sec=duration_sec)

    def _cancel_gripper_goal(self, goal_handle, timeout_sec=3.0):
        cancel_future = goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=timeout_sec)

    def run(self):
        """Runs one full pick-and-lift attempt. Returns a result dict --
        not just logs -- so a caller (a live trial batch, a future disposal
        pipeline stage, a launch_testing test) can act on the outcome
        instead of re-deriving it from log output."""
        pose = self.wait_for_can_pose()
        if pose is None:
            self.get_logger().error('Never got a can pose on /world_pose, aborting.')
            return {'success': False, 'reason': 'no_can_pose'}

        can_x, can_y, can_height_before = pose
        # Negated: verified live (2026-09-12 diagnostic) that commanding
        # shoulder_pan_joint to plain atan2(can_y, can_x) rotates the arm
        # to the *wrong* side -- shoulder_pan's positive direction is
        # inverted relative to world-frame azimuth for this URDF (likely
        # from shoulder_link's ~180-degree-flipped mounting rpy onto
        # base_link, see kinematics.py's _J_SHOULDER_PAN comment). Confirmed
        # by commanding both signs and reading the arm's actual world pose
        # back via /world_pose: only the negated sign lands the gripper on
        # the same side of the arm as the can.
        azimuth = -math.atan2(can_y, can_x)
        can_radius = math.hypot(can_x, can_y)
        self.get_logger().info(
            f'Can at x={can_x:.3f} y={can_y:.3f} z={can_height_before:.3f}, '
            f'azimuth={azimuth:.3f} radius={can_radius:.3f}')

        try:
            hover_lift, hover_elbow, hover_wrist = hover_joint_targets(
                can_radius, can_height_before)
            shoulder_lift, elbow_flex, wrist_flex = grasp_joint_targets(
                can_radius, can_height_before)
        except ValueError as exc:
            self.get_logger().error(f'Can position is unreachable: {exc}')
            return {'success': False, 'reason': 'unreachable'}

        hover_joints = (azimuth, hover_lift, hover_elbow, hover_wrist, WRIST_ROLL)
        grasp_joints = (azimuth, shoulder_lift, elbow_flex, wrist_flex, WRIST_ROLL)

        open_goal_handle = send_gripper_goal(self, GRIPPER_OPEN)
        self._spin_for(GRIPPER_SETTLE_SEC)

        # HOME -> HOVER first, well above the can, so this transit move (a
        # bigger joint-space jump, including the full azimuth rotation)
        # never comes near the can's height. Then one single, direct
        # HOVER -> GRASP hop -- see this module's docstring for why a
        # multi-step interpolated descent is deliberately not used there.
        self.move_arm(hover_joints, duration_sec=3.0)
        self.move_arm(grasp_joints, duration_sec=2.0)

        # The open goal is still alive and actively holding (the node is
        # launched with hold_after_reaching:=true, see this module's
        # docstring), so it has to be explicitly canceled before sending
        # the close goal -- otherwise both goals run concurrently
        # (gripper_action_server.py uses a MultiThreadedExecutor) and fight
        # over the same effort topic, one still servoing toward open while
        # the other drives toward closed. That produces exactly the
        # symptom a first pass at this trial batch showed: near-zero net
        # effort and an arbitrary resting position, neither goal ever
        # actually winning.
        self._cancel_gripper_goal(open_goal_handle)

        # Left running (not canceled): this pipeline never re-opens after
        # closing, so there's nothing that needs it released, and letting
        # it keep holding is exactly what the following lift step needs.
        send_gripper_goal(self, GRIPPER_CLOSED, max_effort=GRIPPER_CLOSE_MAX_EFFORT)
        self._spin_for(GRIPPER_SETTLE_SEC)
        close_position = self.gripper_position
        close_effort = self.gripper_effort
        self.get_logger().info(
            f'Gripper close: position={close_position:.3f} effort={close_effort:.3f}')

        lift_joints = (azimuth, shoulder_lift - 0.4, elbow_flex, wrist_flex, WRIST_ROLL)
        self.move_arm(lift_joints, duration_sec=2.0)
        self._spin_for(LIFT_HOLD_CHECK_SEC)

        can_height_after = self.can_pose[2] if self.can_pose is not None else can_height_before
        height_gain = can_height_after - can_height_before
        success = height_gain >= LIFT_SUCCESS_HEIGHT_MARGIN

        self.get_logger().info(
            f'Lift check: height_before={can_height_before:.4f} '
            f'height_after={can_height_after:.4f} gain={height_gain:.4f} '
            f'success={success}')

        return {
            'success': success,
            'reason': 'ok' if success else 'lift_not_sustained',
            'close_position': close_position,
            'close_effort': close_effort,
            'azimuth': azimuth,
            'can_height_before': can_height_before,
            'can_height_after': can_height_after,
            'height_gain': height_gain,
        }


def main():
    rclpy.init()
    node = PickAndLiftDemo()
    try:
        result = node.run()
        node.get_logger().info(f'Pick-and-lift result: {result}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
