import time

import rclpy
from control_msgs.action import GripperCommand
from rclpy.action import ActionServer
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

GRIPPER_JOINT_NAME = 'gripper_joint'
EFFORT_COMMAND_TOPIC = '/gripper_controller/commands'
ACTION_NAME = '/gripper_controller/gripper_cmd'
CONTROL_PERIOD_SEC = 0.02


class GripperActionServer(Node):
    """Drives gripper_joint to a target position using an effort command.

    gz_ros2_control's built-in position control only ever commands a target
    velocity for a position-controlled joint (see commit-notes/09 for the
    full investigation). That mechanism does not move gripper_joint at all,
    while a direct effort (torque) command does. This node is a small,
    from-scratch PID controller that reads the joint's current position,
    computes an effort command toward the goal, and exposes the same
    GripperCommand action interface a standard position-based gripper
    controller would — so callers like gripper.py don't need to know the
    difference.
    """

    def __init__(self):
        super().__init__('gripper_action_server')

        self.declare_parameter('p_gain', 40.0)
        self.declare_parameter('i_gain', 0.0)
        self.declare_parameter('d_gain', 2.0)
        self.declare_parameter('goal_tolerance', 0.05)
        self.declare_parameter('stall_velocity_threshold', 0.001)
        self.declare_parameter('stall_timeout', 1.0)
        self.declare_parameter('default_max_effort', 5.0)

        self._position = 0.0
        self._velocity = 0.0
        self._have_joint_state = False

        self._effort_pub = self.create_publisher(
            Float64MultiArray, EFFORT_COMMAND_TOPIC, 10)
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10)

        self._action_server = ActionServer(
            self, GripperCommand, ACTION_NAME, self._execute_goal)

    def _on_joint_state(self, msg):
        if GRIPPER_JOINT_NAME not in msg.name:
            return
        index = msg.name.index(GRIPPER_JOINT_NAME)
        self._position = msg.position[index]
        if msg.velocity:
            self._velocity = msg.velocity[index]
        self._have_joint_state = True

    def _stop_effort(self):
        self._effort_pub.publish(Float64MultiArray(data=[0.0]))

    def _execute_goal(self, goal_handle):
        target = goal_handle.request.command.position
        max_effort = goal_handle.request.command.max_effort
        if max_effort <= 0.0:
            max_effort = self.get_parameter('default_max_effort').value

        p_gain = self.get_parameter('p_gain').value
        i_gain = self.get_parameter('i_gain').value
        d_gain = self.get_parameter('d_gain').value
        goal_tolerance = self.get_parameter('goal_tolerance').value
        stall_velocity_threshold = self.get_parameter('stall_velocity_threshold').value
        stall_timeout = self.get_parameter('stall_timeout').value

        integral = 0.0
        previous_error = target - self._position
        previous_time = self.get_clock().now()
        stall_started_at = None
        reached_goal = False

        # Timing here uses the simulation clock (self.get_clock(), backed by
        # /clock once use_sim_time is set), not the wall clock. Gazebo can
        # run much slower than real time when the host is under load; a
        # wall-clock stall timeout would then trip after one real second
        # even though barely any simulated time — and therefore barely any
        # chance for the gripper to actually move — had passed.
        while rclpy.ok():
            now = self.get_clock().now()
            dt = (now - previous_time).nanoseconds / 1e9
            if dt <= 0.0:
                dt = CONTROL_PERIOD_SEC
            previous_time = now

            error = target - self._position
            integral += error * dt
            derivative = (error - previous_error) / dt
            previous_error = error

            effort = p_gain * error + i_gain * integral + d_gain * derivative
            effort = max(-max_effort, min(max_effort, effort))
            self._effort_pub.publish(Float64MultiArray(data=[effort]))

            if abs(error) <= goal_tolerance:
                reached_goal = True
                break

            if abs(self._velocity) < stall_velocity_threshold:
                if stall_started_at is None:
                    stall_started_at = now
                elif (now - stall_started_at).nanoseconds / 1e9 >= stall_timeout:
                    reached_goal = False
                    break
            else:
                stall_started_at = None

            time.sleep(CONTROL_PERIOD_SEC)

        self._stop_effort()
        goal_handle.succeed()

        result = GripperCommand.Result()
        result.position = self._position
        result.effort = effort
        result.stalled = not reached_goal
        result.reached_goal = reached_goal
        return result


def main(args=None):
    rclpy.init(args=args)
    node = GripperActionServer()
    # A goal's execute callback blocks (it polls position in a loop) for as
    # long as the gripper is moving. A MultiThreadedExecutor lets the
    # joint_state subscription keep processing new positions in a separate
    # thread while a goal is in progress, instead of freezing on one goal.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
