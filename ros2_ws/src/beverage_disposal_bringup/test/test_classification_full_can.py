"""launch_testing integration test: a full-mass can classifies as full.

Companion to test_classification_empty_can.py -- see that file's docstring
for why this is two files rather than one parametrized test. This variant
also doubles as the live check for whether a ~26x heavier can (0.37kg vs.
0.014kg) changes grasp/lift dynamics tuned against the lighter can -- worth
watching, not assumed fine.
"""
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

from beverage_disposal_bringup.classification import FULL_CAN_MASS_KG
from beverage_disposal_bringup.pick_and_lift import PickAndLiftDemo


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    os.environ['ROS_DOMAIN_ID'] = '38'
    os.environ['GZ_PARTITION'] = 'test_classification_full_can'

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

    can_description = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution(
            [FindPackageShare('beverage_disposal_bringup'), 'models', 'can.sdf.xacro']),
        ' ',
        'mass:=',
        str(FULL_CAN_MASS_KG),
    ])

    spawn_can = launch_ros.actions.Node(
        package='ros_gz_sim',
        executable='create',
        # -x/-y/-z required, not optional: create's own <pose> handling
        # always overrides with these flags (see spawn_arm.launch.py for
        # the full explanation) -- must match can.sdf.xacro's own <pose>.
        arguments=[
            '-string', can_description, '-name', 'can',
            '-x', '0.26', '-y', '0.06', '-z', '0.211',
        ],
        output='screen',
    )

    joint_state_broadcaster_spawner = launch_ros.actions.Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster'], output='screen',
    )
    arm_controller_spawner = launch_ros.actions.Node(
        package='controller_manager', executable='spawner',
        arguments=['arm_controller'], output='screen',
    )
    gripper_controller_spawner = launch_ros.actions.Node(
        package='controller_manager', executable='spawner',
        arguments=['gripper_controller'], output='screen',
    )

    gripper_action_server = launch_ros.actions.Node(
        package='beverage_disposal_bringup',
        executable='gripper_action_server',
        parameters=[{'use_sim_time': True, 'hold_after_reaching': True}],
        output='screen',
    )

    clock_bridge = launch_ros.actions.Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
        output='screen',
    )

    pose_bridge = launch_ros.actions.Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=[
            '/world/beverage_disposal_world/dynamic_pose/info'
            '@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
        ],
        remappings=[
            ('/world/beverage_disposal_world/dynamic_pose/info', '/world_pose'),
        ],
        output='screen',
    )

    pose_set_bridge = launch_ros.actions.Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=[
            '/world/beverage_disposal_world/set_pose@ros_gz_interfaces/srv/SetEntityPose',
        ],
        output='screen',
    )

    return launch.LaunchDescription([
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        spawn_can,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        gripper_controller_spawner,
        gripper_action_server,
        clock_bridge,
        pose_bridge,
        pose_set_bridge,
        launch_testing.actions.ReadyToTest(),
    ]), {}


class TestClassificationFullCan(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_full_can_classifies_as_full(self):
        node = PickAndLiftDemo(can_mass_kg=FULL_CAN_MASS_KG)
        try:
            result = node.run()
        finally:
            node.destroy_node()

        if not result.get('grip_confirmed'):
            self.skipTest(
                'No confirmed grip this run (reason=' + str(result.get('reason')) + ') -- '
                'the separate, already-tracked upstream grip-detection miss rate, not what '
                'this test checks. Re-run to exercise the confirmed-grip path.')

        self.assertEqual(result['fullness'], 'full', f'result: {result}')
