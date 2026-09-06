"""Bring up the OpenManipulator-X in Gazebo Fortress (gz-sim), replacing the
upstream Gazebo Classic launch path which doesn't run on this arm64 setup.

Combines:
  - open_manipulator_x_bringup/base.launch.py (robot_state_publisher +
    ros2_control spawners; simulator-agnostic, reused unmodified)
  - ros_gz_sim/gz_sim.launch.py (launches gz-sim/Fortress itself; built from
    source, see CLAUDE.md)
  - ros_gz_sim's `create` tool to spawn the robot into the running world
  - a ros_gz_bridge clock bridge so use_sim_time works
  - a ros_gz_bridge pose bridge (object poses -> tf2_msgs/TFMessage) feeding
    our own ultrasonic_range_node, which computes a simulated ultrasonic
    reading from pose data since Gazebo's native sensor pipeline was
    unreliable on this VM (see CLAUDE.md)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    start_rviz = LaunchConfiguration('start_rviz')
    world = LaunchConfiguration('world')

    declared_arguments = [
        DeclareLaunchArgument(
            'start_rviz', default_value='false',
            description='Whether to launch rviz2'),
        DeclareLaunchArgument(
            'world',
            default_value=PathJoinSubstitution(
                [FindPackageShare('beverage_disposal_bringup'), 'worlds', 'beverage_disposal_world.sdf']
            ),
            description='World file to load (bundled gz-sim world name or path)'),
    ]

    arm_base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('open_manipulator_x_bringup'), 'launch', 'base.launch.py']
            )
        ),
        launch_arguments={
            'start_rviz': start_rviz,
            'use_sim': 'true',
        }.items(),
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('ros_gz_sim'), 'launch', 'gz_sim.launch.py']
            )
        ),
        launch_arguments={
            'gz_args': ['-r --render-engine ogre ', world],
        }.items(),
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'open_manipulator_x',
            '-z', '0.01',
        ],
        output='screen',
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
        output='screen',
    )

    pose_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/world/beverage_disposal_world/dynamic_pose/info'
            '@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
        ],
        remappings=[
            ('/world/beverage_disposal_world/dynamic_pose/info', '/world_pose'),
        ],
        output='screen',
    )

    ultrasonic_node = Node(
        package='beverage_disposal_bringup',
        executable='ultrasonic_range_node',
        output='screen',
    )

    return LaunchDescription(declared_arguments + [
        gz_sim,
        arm_base,
        spawn_robot,
        clock_bridge,
        pose_bridge,
        ultrasonic_node,
    ])
