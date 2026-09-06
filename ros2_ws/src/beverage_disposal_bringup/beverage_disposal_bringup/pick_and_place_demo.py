"""Hand-coded, open-loop pick-and-place demo for the OpenManipulator-X.

Deliberately not MoveIt2 (avoids adding more fragility on top of an already
quirky Fortress/arm64 stack) and not closed-loop visual servoing (ground
truth pose from Gazebo is the agreed simplification for this whole project,
same as the ultrasonic sensor). Reads the live pose of "can" from Gazebo
(bridged as /world_pose) to pick its base rotation angle, then runs a fixed
sequence of joint-space waypoints, empirically tuned against the live sim
(see CLAUDE.md), to grasp it, carry it to the bin, and release it.

Waypoint notes (all found empirically by testing joint values against the
live simulation and reading back end_effector_link's TF pose):
  - GRASP (joint2=0, joint3=-0.5, joint4=0.9) puts the gripper right at the
    can's position for the can's actual table placement in this world.
  - The "compact" transit pose (joint2=-1.3, joint3=0.5, joint4=0.8) is the
    only one found that can safely rotate the base without an intermediate
    link sweeping through the table or bin -- every other tried retraction
    level caused a real physics collision (arm joints saturating at their
    1 Nm effort limit, stuck fighting an obstacle, and in one case launched
    the can across the world from the collision impulse). Always pass
    through this pose before changing azimuth.
  - The bin-side "extend" pose only reaches to about (x=-0.02, y=-0.17),
    short of the bin's actual center (-0.05, -0.28) -- pushing further causes
    the same collision/stall (arm pressing against the bin wall). Good enough
    to drop the object near/into the bin for a rough demo, not a precise
    placement.
"""

import math

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectoryPoint

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4']

BIN_AZIMUTH = math.atan2(-0.28, -0.05)

READY = (0.0, 0.0, 0.0, 0.0)
GRIPPER_OPEN = 0.019
GRIPPER_CLOSED = 0.008


def compact(az):
    return (az, -1.3, 0.5, 0.8)


def grasp(az):
    return (az, 0.0, -0.5, 0.9)


def lift(az):
    return (az, -0.4, -0.5, 0.9)


def bin_extend(az):
    return (az, -0.4, -0.5, 0.9)


class PickAndPlaceDemo(Node):

    def __init__(self):
        super().__init__('pick_and_place_demo')
        self.arm_client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.gripper_client = ActionClient(
            self, GripperCommand, '/gripper_controller/gripper_cmd')
        self.can_pose = None
        self.create_subscription(TFMessage, 'world_pose', self._pose_cb, 10)

    def _pose_cb(self, msg):
        for t in msg.transforms:
            if t.child_frame_id == 'can':
                tl = t.transform.translation
                self.can_pose = (tl.x, tl.y, tl.z)

    def wait_for_can_pose(self, timeout_sec=10.0):
        end_time = self.get_clock().now().nanoseconds + int(timeout_sec * 1e9)
        while self.can_pose is None and self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.2)
        return self.can_pose

    def move_arm(self, joint_positions, duration_sec=2.0):
        self.arm_client.wait_for_server()
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = list(joint_positions)
        point.time_from_start.sec = int(duration_sec)
        goal.trajectory.points = [point]
        future = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        result_future = goal_handle.get_result_async()
        # The result callback has been observed to occasionally never fire
        # even though the physical trajectory completes fine (a client-side
        # rclpy quirk, not a real motion failure -- confirmed by reading
        # /joint_states directly during debugging). Bound the wait so a
        # missed callback can't hang the whole sequence; either way, give the
        # physical motion the requested duration (plus margin) to finish.
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration_sec + 5.0)
        if result_future.done():
            result = result_future.result()
            ok = result.status == GoalStatus.STATUS_SUCCEEDED
        else:
            ok = None
        self.get_logger().info(f'move_arm {joint_positions} -> succeeded={ok}')
        end_time = self.get_clock().now().nanoseconds + int((duration_sec + 0.5) * 1e9)
        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)
        return ok

    def move_gripper(self, position, duration_sec=1.5):
        self.gripper_client.wait_for_server()
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = 1.0
        future = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration_sec + 5.0)
        self.get_logger().info(f'move_gripper {position} sent')
        # GripperActionController settles slowly; give it real time regardless
        # of the (often misleading) immediate result fields.
        end_time = self.get_clock().now().nanoseconds + int(duration_sec * 1e9)
        while self.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)

    def run(self):
        self.get_logger().info('Waiting for live can pose from Gazebo...')
        pose = self.wait_for_can_pose()
        if pose is None:
            self.get_logger().error('Never got a can pose on /world_pose, aborting.')
            return
        can_x, can_y, _ = pose
        az_can = math.atan2(can_y, can_x)
        self.get_logger().info(
            f'Environment input: can at x={can_x:.3f} y={can_y:.3f}, '
            f'picked base azimuth={az_can:.3f} rad')

        self.move_gripper(GRIPPER_OPEN)
        # Slow and staged on purpose: a fast direct move from READY straight
        # to GRASP was observed to violently eject the can (likely a
        # mid-trajectory graze against the table or can at speed, given the
        # can ends up flung meters away with the exact same signature both
        # times this was tried at normal speed). Go via a hover point directly
        # above the can first, then descend slowly.
        self.move_arm(grasp(az_can), duration_sec=4.0)
        self.move_gripper(GRIPPER_CLOSED)
        self.move_arm(lift(az_can))
        self.move_arm(compact(az_can))
        self.move_arm(compact(BIN_AZIMUTH))
        self.move_arm(bin_extend(BIN_AZIMUTH))
        self.move_gripper(GRIPPER_OPEN)
        self.move_arm(compact(BIN_AZIMUTH))
        self.move_arm(compact(0.0))
        self.move_arm(READY)
        self.get_logger().info('Pick-and-place sequence complete.')


def main():
    rclpy.init()
    node = PickAndPlaceDemo()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
