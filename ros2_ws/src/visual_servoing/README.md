# Visual Servoing Package

Visual servoing system for ceiling-mounted 6-DOF robot arm with ArUco marker detection.

## Package Contents

### Vision System
- **ArUco Detection**: 4-marker board detection using OpenCV
- **Camera Integration**: Gazebo camera simulation and ROS2 bridge
- **Camera Viewer**: Real-time camera feed visualization

### Simulation
- **Flipped Robot URDF**: Ceiling-mounted robot configuration (`New_flipped.xacro`)
- **Gazebo World**: Visual servoing training environment
- **ArUco Marker Models**: 4 individual 20mm ArUco markers (IDs 0-3)

### Control Nodes
- **Vision ArUco Detector**: Detects markers and publishes board pose
- **Drawing Executor**: Executes drawing trajectories
- **Shape Generator**: Generates geometric shapes for drawing

### Core Modules
- **Kinematics**: Forward/inverse kinematics utilities
- **Filters**: Signal filtering for smooth control
- **Profiler**: Performance profiling tools

### Drivers
- **I2C Manager**: I2C communication for hardware
- **Sensor Driver**: Sensor interface utilities

## Quick Start

### 1. Build the Package
```bash
cd ~/new_rl_ros2/ros2_ws
colcon build --packages-select visual_servoing
source install/setup.bash
```

### 2. Launch Visual Servoing Test
```bash
ros2 launch visual_servoing visual_servoing_test.launch.py
```

This launches:
- Gazebo with visual servoing world
- Ceiling-mounted robot (flipped configuration)
- ArUco marker detection node
- Camera bridge

### 3. View Camera Feed
In another terminal:
```bash
source ~/new_rl_ros2/ros2_ws/install/setup.bash
ros2 run visual_servoing camera_viewer
```

## Available Executables

- `vision_aruco_detector` - ArUco marker detection node
- `camera_viewer` - Camera image viewer
- `drawing_executor` - Drawing trajectory executor
- `shape_generator` - Shape generation utility
- `vision_node` - General vision processing node

## Topics

### Subscribed
- `/camera/image_raw` (sensor_msgs/Image) - Camera image feed
- `/camera/camera_info` (sensor_msgs/CameraInfo) - Camera calibration

### Published
- `/vision/board_pose` (geometry_msgs/PoseStamped) - ArUco board pose
- `/vision/debug_image` (sensor_msgs/Image) - Annotated image with markers

## Configuration

Camera parameters are configured in the URDF (`urdf/New_flipped.xacro`):
- **Position**: Y=0.15m from base
- **Orientation**: RPY=(0, -1.9, -1.5708) for optimal marker viewing
- **FOV**: 80° horizontal
- **Resolution**: 640x480 @ 5Hz
- **Clip planes**: 0.05m - 2.0m

ArUco marker positions defined in world file (`worlds/visual_servoing_training.world`):
- 4 markers in 96mm grid spacing
- Mounted on white 200mm x 200mm drawing surface
- Located at Y=-0.27m, Z=0.304-0.400m

## Dependencies

- ROS 2 Humble
- Gazebo Fortress
- OpenCV (cv_bridge)
- Python 3.10+
- NumPy

## License

MIT
