import time

import rclpy
from control_msgs.action import GripperCommand
from rclpy.action import ActionServer, CancelResponse
from rclpy.duration import Duration
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

        # jaw_link (the part this joint actually moves) has a tiny mass
        # relative to the arm's other links - see commit-notes/09. The
        # original gains (p=40, d=2) combined with the default 10 N*m
        # max_effort produced enough torque to fling the joint straight
        # through its target and into the hard 1.70 rad limit before the
        # (comparatively weak) derivative term could slow it down. These
        # lower, more damped gains were retuned empirically against the
        # live simulation to settle near the target instead of overshooting
        # into the limit.
        self.declare_parameter('p_gain', 10.0)
        self.declare_parameter('i_gain', 0.0)
        self.declare_parameter('d_gain', 1.2)
        self.declare_parameter('goal_tolerance', 0.05)
        self.declare_parameter('stall_velocity_threshold', 0.001)
        self.declare_parameter('stall_timeout', 1.0)
        self.declare_parameter('default_max_effort', 5.0)
        self.declare_parameter('hold_timeout', 20.0)
        self.declare_parameter('hold_stable_effort', 0.3)
        self.declare_parameter('hold_stable_duration', 0.3)

        self._position = 0.0
        self._velocity = 0.0
        self._have_joint_state = False

        self._effort_pub = self.create_publisher(
            Float64MultiArray, EFFORT_COMMAND_TOPIC, 10)
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10)

        self._action_server = ActionServer(
            self, GripperCommand, ACTION_NAME, self._execute_goal,
            # rclpy rejects every cancel request by default (its own
            # default_cancel_callback always returns REJECT) -- a caller
            # needs to be able to cancel a still-holding goal (see the hold
            # loop above) to get the gripper moving again, so accept every
            # cancel request here.
            cancel_callback=lambda cancel_request: CancelResponse.ACCEPT)

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
        effort = 0.0

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

        # gripper_joint does not stay where it's put once effort commanding
        # stops (see commit-notes/10): its very low inertia -- the same
        # property that caused the position-vs-effort bug documented above
        # -- means even a small unbalanced torque (most likely gravity,
        # since the joint's axis isn't vertical) accelerates it back toward
        # 0 within a fraction of a second. That's harmless when the target
        # itself is at or near the closed/stalled rest position (nothing to
        # hold against), but it means a caller that opens the gripper and
        # then goes on to do something else (like moving the arm) would
        # find the jaws already closed again by the time it matters.
        #
        # So instead of cutting effort and returning the instant the target
        # is reached, keep running the same PID for as long as it's still
        # doing real work (i.e. still commanding a non-negligible corrective
        # effort to stay there), which lets a caller keep this goal active
        # -- and the gripper physically open -- while it does something
        # else, then explicitly cancel this goal via the action client
        # right when it's actually ready for the gripper to move again.
        # Bounded by hold_timeout as a safety net in case nothing ever
        # cancels it, and cut short early once holding here has stopped
        # taking any real effort (already-closed and stalled-against-an-
        # object cases both land here almost immediately, so this doesn't
        # meaningfully slow down a plain open-then-close sequence).
        hold_timeout = Duration(seconds=self.get_parameter('hold_timeout').value)
        stable_effort = self.get_parameter('hold_stable_effort').value
        stable_duration = self.get_parameter('hold_stable_duration').value
        hold_started_at = self.get_clock().now()
        stable_since = None
        canceled = False

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                canceled = True
                break

            now = self.get_clock().now()
            if now - hold_started_at >= hold_timeout:
                break

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

            if abs(effort) < stable_effort:
                if stable_since is None:
                    stable_since = now
                elif (now - stable_since) >= Duration(seconds=stable_duration):
                    break
            else:
                stable_since = None

            time.sleep(CONTROL_PERIOD_SEC)

        self._stop_effort()

        result = GripperCommand.Result()
        result.position = self._position
        result.effort = effort
        result.stalled = not reached_goal
        result.reached_goal = reached_goal
        if canceled:
            goal_handle.canceled()
        else:
            goal_handle.succeed()
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
