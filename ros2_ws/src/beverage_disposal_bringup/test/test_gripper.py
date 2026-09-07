import os
import unittest

import launch
import launch_ros.actions
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from sensor_msgs.msg import JointState

from beverage_disposal_bringup.gripper import move_gripper

GRIPPER_OPEN = 1.5
GRIPPER_CLOSED = 0.0
GRIPPER_TOLERANCE = 0.05


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    # See test_arm_motion.py for why this is needed and why it must be set
    # here rather than at module level: an isolated ROS_DOMAIN_ID per test
    # file prevents a leftover Gazebo/ROS process from a previous test
    # (Gazebo doesn't always exit cleanly on SIGTERM) from cross-talking on
    # /joint_states with this file's own, fresh instance. GZ_PARTITION does
    # the same for Ignition Transport, Gazebo's own internal pub/sub that
    # ROS_DOMAIN_ID has no effect on - both layers need a unique name here.
    os.environ['ROS_DOMAIN_ID'] = '35'
    os.environ['GZ_PARTITION'] = 'test_gripper'

    robot_description = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution(
            [FindPackageShare('so_arm101_description'), 'urdf', 'so_arm101.urdf.xacro']),
        ' ',
        'ros2_control_hardware_type:=gazebo',
        ' ',
        'simulation_controllers:=',
        PathJoinSubstitution(
            [FindPackageShare('beverage_disposal_bringup'), 'config',
             'so_arm101_controllers.yaml']),
    ])

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])
        ),
        launch_arguments={
            'gz_args': [
                '-s -r ',
                PathJoinSubstitution(
                    [FindPackageShare('beverage_disposal_bringup'), 'worlds',
                     'beverage_disposal_world.sdf']),
            ],
        }.items(),
    )

    robot_state_publisher = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen',
    )

    spawn_robot = launch_ros.actions.Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'so_arm101', '-z', '0.01'],
        output='screen',
    )

    joint_state_broadcaster_spawner = launch_ros.actions.Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )

    gripper_controller_spawner = launch_ros.actions.Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller'],
        output='screen',
    )

    gripper_action_server = launch_ros.actions.Node(
        package='beverage_disposal_bringup',
        executable='gripper_action_server',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    clock_bridge = launch_ros.actions.Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
        output='screen',
    )

    return launch.LaunchDescription([
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        joint_state_broadcaster_spawner,
        gripper_controller_spawner,
        gripper_action_server,
        clock_bridge,
        launch_testing.actions.ReadyToTest(),
    ]), {}


class TestGripperOpenClose(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_gripper_node')
        self.latest_joint_state = None
        self.node.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10)

    def tearDown(self):
        self.node.destroy_node()

    def _on_joint_state(self, msg):
        self.latest_joint_state = msg

    def _wait_for_gripper_position(self, target, timeout_sec=30.0):
        end_time = self.node.get_clock().now().nanoseconds + int(timeout_sec * 1e9)
        while self.node.get_clock().now().nanoseconds < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.5)
            if self.latest_joint_state is None:
                continue
            if 'gripper_joint' not in self.latest_joint_state.name:
                continue
            index = self.latest_joint_state.name.index('gripper_joint')
            position = self.latest_joint_state.position[index]
            if abs(position - target) <= GRIPPER_TOLERANCE:
                return position
        self.fail(
            f'gripper_joint did not reach {target} within {timeout_sec}s '
            f'(last seen: {self.latest_joint_state})')

    def test_gripper_opens_and_closes(self):
        move_gripper(self.node, GRIPPER_OPEN, timeout_sec=25.0)
        self._wait_for_gripper_position(GRIPPER_OPEN)

        move_gripper(self.node, GRIPPER_CLOSED, timeout_sec=25.0)
        self._wait_for_gripper_position(GRIPPER_CLOSED)
