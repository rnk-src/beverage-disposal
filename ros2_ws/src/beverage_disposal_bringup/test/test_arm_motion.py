import time
import unittest

import launch
import launch_testing.actions
import pytest
import rclpy
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from sensor_msgs.msg import JointState

from beverage_disposal_bringup.motion import move_to_joint_position

ARM_JOINTS = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_flex_joint',
    'wrist_flex_joint',
    'wrist_roll_joint',
]
TARGET_POSITIONS = [0.3, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.launch_test
def generate_test_description():
    robot_description = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution(
            [FindPackageShare('so_arm101_description'), 'urdf', 'so_arm101.urdf.xacro']
        ),
        ' ',
        'ros2_control_hardware_type:=gazebo',
        ' ',
        'simulation_controllers:=',
        PathJoinSubstitution(
            [FindPackageShare('beverage_disposal_bringup'), 'config', 'so_arm101_controllers.yaml']
        ),
    ])

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py'])
        ),
        launch_arguments={'gz_args': '-s -r empty.sdf'}.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen',
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'so_arm101', '-z', '0.01'],
        output='screen',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen',
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller'],
        output='screen',
    )

    return launch.LaunchDescription([
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        launch_testing.actions.ReadyToTest(),
    ])


class TestArmMovesToCommandedPosition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_arm_motion')

    def tearDown(self):
        self.node.destroy_node()

    def test_arm_reaches_commanded_position(self):
        succeeded = move_to_joint_position(
            self.node, ARM_JOINTS, TARGET_POSITIONS, timeout_sec=20.0)
        self.assertTrue(succeeded, 'move_to_joint_position reported failure')

        joint_state = self._latest_joint_state(timeout_sec=10.0)
        for name, target in zip(ARM_JOINTS, TARGET_POSITIONS):
            index = joint_state.name.index(name)
            self.assertAlmostEqual(
                joint_state.position[index], target, delta=0.05,
                msg=f'{name} did not reach {target}')

    def _latest_joint_state(self, timeout_sec):
        received = {}

        def on_joint_state(msg):
            received['msg'] = msg

        subscription = self.node.create_subscription(
            JointState, '/joint_states', on_joint_state, 10)
        deadline = time.monotonic() + timeout_sec
        while 'msg' not in received and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.5)
        self.node.destroy_subscription(subscription)

        if 'msg' not in received:
            self.fail('No /joint_states message received before timeout')
        return received['msg']
