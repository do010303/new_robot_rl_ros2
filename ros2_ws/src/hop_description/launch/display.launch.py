#!/usr/bin/env python3
"""
Launch file to display the HOP drone model in RViz2 only.
Usage: ros2 launch hop_description display.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Declare arguments
    use_gui = DeclareLaunchArgument(
        'use_gui',
        default_value='true',
        description='Whether to start joint_state_publisher_gui'
    )

    # Process xacro to URDF
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

    # Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    # Joint State Publisher GUI (optional)
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=IfCondition(LaunchConfiguration('use_gui')),
    )

    # RViz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('hop_description'),
            'rviz',
            'display.rviz',
        ])],
    )

    return LaunchDescription([
        use_gui,
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node,
    ])
