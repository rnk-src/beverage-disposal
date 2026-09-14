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
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

from .grasp_geometry import WRIST_ROLL, grasp_joint_targets, hover_joint_targets
from .gripper import send_gripper_goal
from .kinematic_grasp import pose_from_relative, pose_relative_to
from .motion import move_to_joint_position

GRIPPER_JOINT_NAME = 'gripper_joint'
GRIPPER_LINK_FRAME = 'gripper_link'
CAN_MODEL_NAME = 'can'

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

# Kinematic snap-grasp (see kinematic_grasp.py's module docstring for the
# full reasoning): once gripper_action_server.py's live feedback confirms
# real contact, the can's world pose is overridden every tick to track a
# fixed offset from the gripper, instead of trusting friction/contact
# physics to hold it through a lift. WORLD_NAME/SET_POSE_SERVICE match the
# bridge added in spawn_arm.launch.py.
WORLD_NAME = 'beverage_disposal_world'
SET_POSE_SERVICE = f'/world/{WORLD_NAME}/set_pose'
# 20Hz: comfortably faster than one control tick needs to be (a persistent
# client measured ~0.7ms round-trip against this same service, see
# commit-notes/10), and frequent enough that the one sim-tick of free-fall
# between updates (set_pose only overrides position/orientation, not
# velocity -- confirmed empirically) is imperceptible.
KINEMATIC_LOCK_PERIOD_SEC = 0.05

# How high the can needs to have risen, and stayed risen, to count as a
# real lift rather than a transient bounce or a grip that let go under
# load. See commit-notes/10: several past configurations showed a real
# transient rise (up to ~0.83cm) that still fell short of a lift a caller
# should trust -- 1.5cm was this iteration's own established bar for "real
# margin, not noise."
LIFT_SUCCESS_HEIGHT_MARGIN = 0.015
LIFT_HOLD_CHECK_SEC = 2.0

# Applied to the arm's *actual current* shoulder_lift position at lift
# time (see run()'s comment on why -- not a fixed absolute target).
# Verified directly via live TF in isolation: from a clean grasp pose,
# this raises gripper_link's world Z by ~3.6cm.
LIFT_SHOULDER_LIFT_DELTA = -0.4


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
        self.can_orientation = None
        # Distinct from gripper_position/gripper_effort below (those are
        # the gripper_joint's own angle/torque) -- this is gripper_link's
        # world pose, needed to compute and track the kinematic-lock
        # offset. (position tuple, orientation quaternion tuple) or None.
        self.gripper_link_pose = None
        self.gripper_position = 0.0
        self.gripper_effort = 0.0
        self._actual_shoulder_lift = None
        self._actual_elbow_flex = None
        self._actual_wrist_flex = None
        self._grasp_locked = False
        self._grasp_offset = None
        self.create_subscription(TFMessage, 'world_pose', self._pose_cb, 10)
        self.create_subscription(JointState, 'joint_states', self._joint_state_cb, 10)
        self._set_pose_client = self.create_client(SetEntityPose, SET_POSE_SERVICE)
        self.create_timer(KINEMATIC_LOCK_PERIOD_SEC, self._kinematic_lock_tick)

    def _pose_cb(self, msg):
        for t in msg.transforms:
            if t.child_frame_id == CAN_MODEL_NAME:
                tl = t.transform.translation
                rot = t.transform.rotation
                self.can_pose = (tl.x, tl.y, tl.z)
                self.can_orientation = (rot.x, rot.y, rot.z, rot.w)
            elif t.child_frame_id == GRIPPER_LINK_FRAME:
                tl = t.transform.translation
                rot = t.transform.rotation
                self.gripper_link_pose = ((tl.x, tl.y, tl.z), (rot.x, rot.y, rot.z, rot.w))

    def _kinematic_lock_tick(self):
        """Runs at KINEMATIC_LOCK_PERIOD_SEC regardless of what else this
        node is doing (a rclpy timer fires during any spin_once/
        spin_until_future_complete call on this node, including the ones
        inside move_arm's blocking trajectory waits) -- this is what makes
        the can track the gripper *through* a lift move, not just at the
        one instant the grasp was confirmed. No-ops unless a grasp is
        currently locked."""
        if not self._grasp_locked or self.gripper_link_pose is None or self._grasp_offset is None:
            return
        if not self._set_pose_client.service_is_ready():
            return

        gripper_pos, gripper_quat = self.gripper_link_pose
        offset_pos, offset_quat = self._grasp_offset
        target_pos, target_quat = pose_from_relative(gripper_pos, gripper_quat, offset_pos, offset_quat)

        req = SetEntityPose.Request()
        req.entity.name = CAN_MODEL_NAME
        req.entity.type = Entity.MODEL
        req.pose.position.x, req.pose.position.y, req.pose.position.z = target_pos
        (req.pose.orientation.x, req.pose.orientation.y,
         req.pose.orientation.z, req.pose.orientation.w) = target_quat
        # Fire-and-forget: waiting for the response inside a timer callback
        # would need a nested spin to ever resolve, and at ~0.7ms measured
        # round-trip against this same service (see commit-notes/10) there's
        # no real risk of requests piling up at a 20Hz tick rate.
        self._set_pose_client.call_async(req)
        self._lock_tick_count = getattr(self, '_lock_tick_count', 0) + 1
        if self._lock_tick_count % 5 == 0:
            print(f'DEBUG lock tick #{self._lock_tick_count}: gripper_z={gripper_pos[2]:.4f} '
                  f'target_can_z={target_pos[2]:.4f} actual_can_z={self.can_pose[2]:.4f}', flush=True)

    def _start_kinematic_lock(self):
        """Records the can's current pose relative to the gripper's
        current pose as a fixed offset, and turns on _kinematic_lock_tick's
        per-tick override from now on. Call this once, right after a real
        grasp is confirmed -- see run()."""
        if self.gripper_link_pose is None or self.can_pose is None or self.can_orientation is None:
            return False
        gripper_pos, gripper_quat = self.gripper_link_pose
        self._grasp_offset = pose_relative_to(
            gripper_pos, gripper_quat, self.can_pose, self.can_orientation)
        self._grasp_locked = True
        return True

    def _stop_kinematic_lock(self):
        self._grasp_locked = False
        self._grasp_offset = None

    def _joint_state_cb(self, msg):
        # Tracks the arm's own pitch-joint positions too (not just the
        # gripper's), needed by run()'s lift step -- see its comment on
        # why the lift target is computed from these *actual* live values
        # rather than re-deriving a theoretical grasp target.
        for joint_name, attr in (
            ('shoulder_lift_joint', '_actual_shoulder_lift'),
            ('elbow_flex_joint', '_actual_elbow_flex'),
            ('wrist_flex_joint', '_actual_wrist_flex'),
        ):
            if joint_name in msg.name:
                setattr(self, attr, msg.position[msg.name.index(joint_name)])

        if GRIPPER_JOINT_NAME not in msg.name:
            return
        index = msg.name.index(GRIPPER_JOINT_NAME)
        self.gripper_position = msg.position[index]
        if msg.effort:
            self.gripper_effort = msg.effort[index]

    def wait_for_can_pose(self, timeout_sec=10.0):
        # Real bug, found live: self.get_clock().now() reads 0 until this
        # node's own /clock subscription has processed its first message
        # (use_sim_time is set, see __init__). If this deadline were
        # computed from that initial 0 reading, the *next* real clock
        # sample can jump straight to Gazebo's actual already-elapsed sim
        # time (e.g. tens of seconds, if this node started well after
        # Gazebo did) -- which can already exceed a 0-based 10s deadline
        # on the very first real tick, timing this out before it ever gets
        # a real chance to wait. Deferring end_time until the clock has
        # actually started (nonzero) avoids computing a deadline against a
        # placeholder value.
        end_time = None
        while self.can_pose is None:
            rclpy.spin_once(self, timeout_sec=0.2)
            now = self.get_clock().now().nanoseconds
            if now > 0 and end_time is None:
                end_time = now + int(timeout_sec * 1e9)
            if end_time is not None and now >= end_time:
                break
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

        # grip_confirmed reuses gripper_action_server.py's own live
        # contact-detection signal (feedback.stalled) as-is, rather than
        # re-deriving "was this a real grip" from position/effort
        # thresholds here -- that controller's CONTACT_DETECTED status is
        # already sticky (see gripper_control.py), so once True it stays
        # True for the rest of this goal.
        #
        # Left running (not canceled): this pipeline never re-opens after
        # closing, so there's nothing that needs it released, and letting
        # it keep holding is exactly what the following lift step needs.
        grip_state = {'confirmed': False}

        def _on_close_feedback(feedback):
            if feedback.stalled:
                grip_state['confirmed'] = True

        send_gripper_goal(
            self, GRIPPER_CLOSED, max_effort=GRIPPER_CLOSE_MAX_EFFORT,
            feedback_callback=_on_close_feedback)
        self._spin_for(GRIPPER_SETTLE_SEC)
        close_position = self.gripper_position
        close_effort = self.gripper_effort
        grip_confirmed = grip_state['confirmed']
        self.get_logger().info(
            f'Gripper close: position={close_position:.3f} effort={close_effort:.3f} '
            f'grip_confirmed={grip_confirmed}')

        # Kinematic snap-grasp (see kinematic_grasp.py's module docstring
        # for why): only lock the can to the gripper if a real grip was
        # actually confirmed -- a miss (gripper closed on nothing) should
        # still show up as a miss in the lift-survival numbers, not be
        # masked by kinematically locking an ungrasped can to the gripper
        # anyway.
        kinematic_locked = grip_confirmed and self._start_kinematic_lock()

        # Real bug found live (2026-09-13, see commit-notes/10): this used
        # to be (shoulder_lift - LIFT_SHOULDER_LIFT_DELTA, elbow_flex,
        # wrist_flex) -- an ABSOLUTE target computed from the theoretical
        # pre-close grasp_joint_targets(). Verified directly (with the now-
        # working kinematic lock exposing the gripper's true trajectory
        # with near-zero tracking error) that this offset genuinely does
        # raise the gripper when tested in isolation from a clean start
        # (+3.6cm, confirmed via live TF) -- but in the full pipeline, the
        # arm's *actual* position by this point (after settling through
        # hover, grasp, and a real gripper closure) can already differ
        # from that theoretical target by several cm, for reasons not
        # fully isolated (possibly closure reaction effects). Since the
        # old code commanded an ABSOLUTE target computed from the stale
        # theoretical value, a real starting-position drift could put the
        # arm *above* that absolute target, turning an intended lift into
        # a net lowering. Anchoring the delta to the arm's actual current
        # joint positions (tracked live in _joint_state_cb) removes this
        # confound entirely, regardless of its exact cause.
        actual_shoulder_lift = (
            self._actual_shoulder_lift if self._actual_shoulder_lift is not None else shoulder_lift)
        actual_elbow_flex = (
            self._actual_elbow_flex if self._actual_elbow_flex is not None else elbow_flex)
        actual_wrist_flex = (
            self._actual_wrist_flex if self._actual_wrist_flex is not None else wrist_flex)
        lift_joints = (
            azimuth, actual_shoulder_lift + LIFT_SHOULDER_LIFT_DELTA,
            actual_elbow_flex, actual_wrist_flex, WRIST_ROLL)
        self.move_arm(lift_joints, duration_sec=2.0)
        self._spin_for(LIFT_HOLD_CHECK_SEC)

        can_height_after = self.can_pose[2] if self.can_pose is not None else can_height_before
        height_gain = can_height_after - can_height_before
        success = height_gain >= LIFT_SUCCESS_HEIGHT_MARGIN

        # Release hook: this pipeline doesn't carry the can to the bin yet
        # (out of scope for this task -- see kinematic_grasp.py's module
        # docstring / commit-notes/10), so this is deliberately minimal,
        # just enough that a future caller reopening the gripper isn't
        # left fighting a still-active lock.
        self._stop_kinematic_lock()

        self.get_logger().info(
            f'Lift check: height_before={can_height_before:.4f} '
            f'height_after={can_height_after:.4f} gain={height_gain:.4f} '
            f'success={success} kinematic_locked={kinematic_locked}')

        return {
            'success': success,
            'reason': 'ok' if success else 'lift_not_sustained',
            'close_position': close_position,
            'close_effort': close_effort,
            'grip_confirmed': grip_confirmed,
            'kinematic_locked': kinematic_locked,
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
