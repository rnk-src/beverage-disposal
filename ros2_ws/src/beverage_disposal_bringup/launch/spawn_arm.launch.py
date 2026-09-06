"""Bring up the SO-101 arm in Gazebo Fortress (gz-sim).

Combines:
  - robot_state_publisher, built from so_arm101_description's URDF with
    ros2_control_hardware_type:=gazebo (this project's own bringup piece;
    the upstream package's controllers_bringup.launch.py doesn't support
    the gazebo hardware type, only mock_components/real, so there's no
    upstream launch file to reuse for this path)
  - ros_gz_sim/gz_sim.launch.py (launches gz-sim/Fortress itself; built from
    source, see CLAUDE.md)
  - ros_gz_sim's `create` tool to spawn the robot into the running world
  - controller spawners (joint_state_broadcaster, arm_controller,
    gripper_controller) — no standalone ros2_control_node, because in
    Gazebo the gz_ros2_control plugin loaded from the URDF's <gazebo> tag
    runs the controller_manager itself
  - a ros_gz_bridge clock bridge so use_sim_time works
  - a ros_gz_bridge pose bridge (object poses -> tf2_msgs/TFMessage), read
    by later pipeline stages to find the can/bottle in the world
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    world = LaunchConfiguration('world')

    declared_arguments = [
        DeclareLaunchArgument(
            'world',
            default_value=PathJoinSubstitution(
                [FindPackageShare('beverage_disposal_bringup'), 'worlds', 'beverage_disposal_world.sdf']
            ),
            description='World file to load (bundled gz-sim world name or path)'),
    ]

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

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
        output='screen',
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
            '-name', 'so_arm101',
            '-z', '0.01',
        ],
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

    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller'],
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

    return LaunchDescription(declared_arguments + [
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        gripper_controller_spawner,
        clock_bridge,
        pose_bridge,
    ])
