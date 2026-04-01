import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_robot_arm2 = get_package_share_directory('robot_arm2')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Load the custom Twin controllers.yaml
    controllers_file = os.path.join(pkg_robot_arm2, 'config', 'controllers_twin.yaml')

    # Re-use your existing empty world launch or just spawn bare Gazebo
    # Assuming there's a world file or we just load an empty one.
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': 'empty.sdf -r -v 4'}.items()
    )

    # Robot State Publisher
    urdf_file = os.path.join(pkg_robot_arm2, 'urdf', 'robot_arm2.xacro')
    # Use xacro to process the URDF. For simplicity, we assume we just pass the raw file, 
    # but ideally we use Command(['xacro ', urdf_file]).
    from launch.substitutions import Command
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command(['xacro ', urdf_file]),
            'use_sim_time': True
        }]
    )

    # Spawn the robot in Gazebo
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'robot_arm2',
            '-allow_renaming', 'true'
        ]
    )

    # Bridge the clock and joint states
    # Map ROS 2 <-> Gazebo transport
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
        ],
        output='screen'
    )

    # Start the Controller Manager with the TWIN settings
    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {'robot_description': Command(['xacro ', urdf_file]), 'use_sim_time': True},
            controllers_file
        ],
        output="screen"
    )

    # Spawner for Joint State Broadcaster
    jsb_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    # Spawner for Arm Controller
    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
    )

    # ── DIGITAL TWIN MIRROR NODE ──
    # Starts the python script that forwards Pi /joint_states to Gazebo /joint_trajectory
    mirror_node = Node(
        package="robot_arm2",
        executable="gazebo_state_mirror.py",
        name="gazebo_state_mirror",
        output="screen"
    )

    return LaunchDescription([
        gazebo_launch,
        robot_state_publisher,
        gz_spawn_entity,
        clock_bridge,
        controller_manager,
        jsb_spawner,
        arm_controller_spawner,
        mirror_node
    ])
