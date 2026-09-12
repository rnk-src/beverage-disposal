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


def send_gripper_goal(node, position, max_effort=10.0, timeout_sec=10.0):
    """Sends a GripperCommand goal and returns as soon as it's accepted,
    without waiting for a terminal result.

    move_gripper() blocks until the goal finishes -- fine normally, but
    gripper_action_server.py's hold_after_reaching mode (needed so a grip
    survives a subsequent arm move, e.g. pick_and_lift.py's lift step)
    means the goal doesn't produce a terminal result until hold_timeout
    elapses or someone cancels it. A caller that just needs the gripper to
    start moving toward a target -- not to sit blocked for the whole hold
    window before it can do anything else -- should use this instead, then
    poll /joint_states directly for the actual position/effort it cares
    about. The hold loop keeps running server-side regardless of whether
    the client is waiting on it, so this doesn't cut the hold short.
    """
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

    return goal_handle
