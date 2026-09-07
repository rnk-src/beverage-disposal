import rclpy
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient

GRIPPER_ACTION_NAME = '/gripper_controller/gripper_cmd'


def move_gripper(node, position, max_effort=10.0, timeout_sec=10.0):
    client = ActionClient(node, GripperCommand, GRIPPER_ACTION_NAME)
    if not client.wait_for_server(timeout_sec=timeout_sec):
        raise TimeoutError(f'{GRIPPER_ACTION_NAME} action server not available')

    goal = GripperCommand.Goal()
    goal.command.position = position
    goal.command.max_effort = max_effort

    send_goal_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_goal_future, timeout_sec=timeout_sec)
    goal_handle = send_goal_future.result()
    if goal_handle is None or not goal_handle.accepted:
        raise RuntimeError('gripper goal was rejected')

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=timeout_sec)
    wrapped_result = result_future.result()
    if wrapped_result is None:
        raise TimeoutError('gripper action did not complete in time')

    return wrapped_result.result
