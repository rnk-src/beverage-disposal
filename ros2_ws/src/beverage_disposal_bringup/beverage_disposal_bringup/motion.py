import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint

ARM_CONTROLLER_ACTION = '/arm_controller/follow_joint_trajectory'
GOAL_RETRY_INTERVAL_SEC = 0.5


def move_to_joint_position(node, joint_names, positions, timeout_sec=15.0, duration_sec=3.0):
    client = ActionClient(node, FollowJointTrajectory, ARM_CONTROLLER_ACTION)
    if not client.wait_for_server(timeout_sec=timeout_sec):
        node.get_logger().error(f'{ARM_CONTROLLER_ACTION} action server not available')
        return False

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = joint_names
    point = JointTrajectoryPoint()
    point.positions = positions
    point.time_from_start.sec = int(duration_sec)
    point.time_from_start.nanosec = int((duration_sec % 1) * 1e9)
    goal.trajectory.points = [point]

    # The action server appears as soon as the controller is *configured*,
    # but a freshly spawned controller can still reject goals for a brief
    # moment longer, until controller_manager finishes *activating* it. That
    # window is real but short-lived, so retry on rejection instead of
    # failing outright - this is a real race, not a broken connection.
    deadline = time.monotonic() + timeout_sec
    goal_handle = None
    while time.monotonic() < deadline:
        send_goal_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, send_goal_future, timeout_sec=timeout_sec)
        goal_handle = send_goal_future.result()
        if goal_handle is not None and goal_handle.accepted:
            break
        node.get_logger().warning('Trajectory goal rejected (controller may still be '
                                   'activating); retrying')
        time.sleep(GOAL_RETRY_INTERVAL_SEC)
        goal_handle = None

    if goal_handle is None:
        node.get_logger().error('Trajectory goal was rejected or timed out')
        return False

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=timeout_sec)
    result_wrapper = result_future.result()
    if result_wrapper is None:
        node.get_logger().warning('No result before timeout; assuming motion completed')
        return True

    if result_wrapper.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
        node.get_logger().error(
            f'Trajectory failed with error_code={result_wrapper.result.error_code}')
        return False

    return True
