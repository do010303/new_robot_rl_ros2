#!/usr/bin/env python3
"""
Launch file to spawn the HOP drone in Gazebo Harmonic (gz-sim8) with ROS 2 Humble.

This launch file works with or without ros_gz packages:
- If ros_gz_sim is installed: uses its 'create' node to spawn
- Falls back to gz CLI spawn if ros_gz_sim is not available

Usage: ros2 launch hop_description gazebo.launch.py
       ros2 launch hop_description gazebo.launch.py use_rviz:=false
       ros2 launch hop_description gazebo.launch.py world:=/path/to/world.sdf
"""

import os
import subprocess

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
    AppendEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def _ros_gz_available():
    """Check if ros_gz_sim is installed."""
    try:
        from ament_index_python.packages import get_package_share_directory as _get
        _get('ros_gz_sim')
        return True
    except Exception:
        return False


def generate_launch_description():
    pkg_hop = get_package_share_directory('hop_description')
    
    # Path to the workspace's install/share directory (parent of pkg_hop)
    install_share_dir = os.path.dirname(pkg_hop)

    urdf_path = os.path.join(pkg_hop, 'urdf', 'hop.xacro')
    sdf_path = os.path.join(pkg_hop, 'sdf', 'hop.sdf')

    # ── Declare launch arguments ──────────────────────────────────────────
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='false',
        description='Whether to start RViz2',
    )

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(pkg_hop, 'worlds', 'empty.sdf'),
        description='Path to the Gazebo world SDF file',
    )

    spawn_x_arg = DeclareLaunchArgument('x', default_value='0.0')
    spawn_y_arg = DeclareLaunchArgument('y', default_value='0.0')
    spawn_z_arg = DeclareLaunchArgument('z', default_value='0.5')

    # ── Robot description (xacro → URDF) ──────────────────────────────────
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution([
            FindPackageShare('hop_description'),
            'urdf',
            'hop.xacro',
        ]),
    ])
    robot_description = {'robot_description': ParameterValue(robot_description_content, value_type=str)}

    # ── Robot State Publisher ─────────────────────────────────────────────
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    # ── Start Gazebo Harmonic ─────────────────────────────────────────────
    gz_sim = ExecuteProcess(
        cmd=[
            'gz', 'sim', '-r',
            LaunchConfiguration('world'),
        ],
        output='screen',
    )

    # ── Spawn the robot in Gazebo ─────────────────────────────────────────
    # Use gz service CLI to spawn the SDF model (works without ros_gz_sim)
    spawn_entity = TimerAction(
        period=3.0,  # Wait for Gazebo to start
        actions=[
            ExecuteProcess(
                cmd=[
                    'gz', 'service',
                    '-s', '/world/hop_world/create',
                    '--reqtype', 'gz.msgs.EntityFactory',
                    '--reptype', 'gz.msgs.Boolean',
                    '--timeout', '5000',
                    '--req',
                    'sdf_filename: "' + sdf_path + '", '
                    'name: "hop", '
                    'pose: {position: {x: 0.0, y: 0.0, z: 0.5}}',
                ],
                output='screen',
            ),
        ],
    )

    # ── Optional: ros_gz_sim spawn (if ros_gz is installed) ───────────────
    actions = [
        AppendEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH', install_share_dir
        ),
        use_rviz_arg,
        world_arg,
        spawn_x_arg,
        spawn_y_arg,
        spawn_z_arg,
        robot_state_publisher_node,
        gz_sim,
    ]

    if _ros_gz_available():
        # Use ros_gz_sim 'create' node for spawning
        spawn_entity_ros = TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='ros_gz_sim',
                    executable='create',
                    name='spawn_hop',
                    output='screen',
                    arguments=[
                        '-name', 'hop',
                        '-topic', '/robot_description',
                        '-x', LaunchConfiguration('x'),
                        '-y', LaunchConfiguration('y'),
                        '-z', LaunchConfiguration('z'),
                    ],
                ),
            ],
        )

        # Bridge joint states from Gazebo to ROS 2
        gz_bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_bridge',
            output='screen',
            arguments=[
                # Joint states: Gazebo → ROS 2
                '/world/hop_world/model/hop/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model',
                # Clock: Gazebo → ROS 2
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            ],
            remappings=[
                ('/world/hop_world/model/hop/joint_state', '/joint_states'),
            ],
        )

        actions.append(spawn_entity_ros)
        actions.append(gz_bridge)
    else:
        # Fallback: use gz CLI to spawn
        actions.append(spawn_entity)

    # ── RViz2 (optional) ──────────────────────────────────────────────────
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        arguments=['-d', os.path.join(pkg_hop, 'rviz', 'display.rviz')],
    )
    actions.append(rviz_node)

    return LaunchDescription(actions)
