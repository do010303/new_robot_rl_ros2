# Visual Servoing RL — 6-DOF Robot Arm

Deep Reinforcement Learning for precise drawing operations on a 6-DOF robotic arm, using visual servoing with ArUco marker detection.

## Overview

This project trains a robot arm to perform precise reaching and drawing tasks on a vertical board, using:
- **SAC (Soft Actor-Critic)** with **HER (Hindsight Experience Replay)**
- **ArUco-based visual servoing** for dynamic workspace detection
- **Neural Inverse Kinematics** for 3D position control
- **Gazebo Fortress** simulation with `ros2_control`

### Architecture

```
Camera → ArUco Detection → Board Transform → RL Agent → Joint Commands → Robot
                ↓                                           ↑
         Board-relative                              Neural IK (optional)
         target generation                           Position → Joints
```

## Project Structure

```
ros2_ws/src/visual_servoing/
├── launch/
│   └── visual_servoing_test.launch.py    # Main launch file
├── urdf/new_arm/
│   └── new_arm.xacro                     # 6-DOF robot URDF (180° yaw, faces world +X)
├── worlds/
│   └── visual_servoing_training.world    # Gazebo world with ArUco board
├── models/
│   └── aruco_marker_*/                   # ArUco marker models (4 corners)
├── scripts/
│   ├── train_visual_servoing.py          # Main training script (6 modes)
│   ├── rl/
│   │   ├── rl_environment.py             # Base RL environment (reaching)
│   │   ├── drawing_environment.py        # Drawing environment (waypoint following)
│   │   ├── fk_ik_utils.py               # Pure Python FK (exact URDF chain)
│   │   ├── neural_ik.py                  # Neural IK with position-based loss
│   │   └── board_transform.py           # ArUco board → base_link transform
│   ├── drawing/
│   │   ├── gazebo_visualizer.py          # Spawns shapes & pen lines in Gazebo
│   │   ├── shape_generator.py            # Triangle/square/line waypoint generator
│   │   └── drawing_config.py            # Drawing configuration
│   └── agents/
│       └── sac_agent.py                  # SAC agent implementation
├── vs_lib/vision/
│   ├── vision_aruco_detector.py          # ArUco board detection node
│   └── camera_viewer.py                  # Camera overlay with RL visualization
└── config/
    └── controllers.yaml                  # ros2_control joint trajectory controller
```

## Coordinate Frames

The robot is mounted with a **180° yaw flip** relative to the world:

| Frame | Forward (toward board) | Left/Right | Up/Down |
|-------|----------------------|------------|---------|
| **World** | +X | +Y | +Z |
| **base_link** | −X | −Y | +Z |

- Board at **world X=0.50** → **base_link X≈−0.50**
- FK and all RL targets operate in **base_link** frame
- Camera is fixed to base_link, pitched down toward the board

## Training Modes

Run `python3 train_visual_servoing.py` to access the training menu:

| Option | Mode | Description |
|--------|------|-------------|
| 1 | Manual Test | Verify environment, FK, and pen drawing |
| 2 | SAC 6D Direct | Direct joint angle control for reaching |
| 3 | SAC + Neural IK | 3D position control via Neural IK |
| 4 | Train Neural IK | Generate FK data and train IK network |
| 5 | Drawing (6D Direct) | Multi-waypoint triangle drawing |
| 6 | Drawing (Neural IK) | Drawing with Neural IK position control |

> **Note**: Run option 4 first before using options 3 or 6.

## Quick Start

### Prerequisites

- ROS 2 Humble
- Gazebo Fortress (Ignition Fortress / gz-sim 6)
- Python 3.10+, PyTorch, NumPy, OpenCV

### Build

```bash
cd ros2_ws
colcon build --packages-select visual_servoing
source install/setup.bash
```

### Launch Simulation

```bash
ros2 launch visual_servoing visual_servoing_test.launch.py
```

This starts: Gazebo world, robot with controllers, ArUco detector, camera viewer, and drawing visualizer.

### Train

In a separate terminal (with the simulation running):

```bash
cd ros2_ws/src/visual_servoing/scripts
python3 train_visual_servoing.py
```

### Digital Twin (Sim-to-Real)

To mirror the simulation movements to the physical 4-DOF Raspberry Pi robot without Wi-Fi multicast errors, use the **FastDDS Discovery Server**.

**1. On the Raspberry Pi:**
```bash
# Start the discovery server in the background
fastdds discovery -i 0 -l 192.168.50.1 -p 11811 &

# Point ROS to this server and launch the hardware listeners
export ROS_DISCOVERY_SERVER="192.168.50.1:11811"
cd ~/ros2_ws
source install/setup.bash
ros2 launch wicom_roboarm wicom_roboarm.launch.py
```

**2. On the Laptop (Gazebo Terminal):**
```bash
cd ~/new_rl_ros2/ros2_ws
source install/setup.bash
unset RMW_IMPLEMENTATION
unset FASTRTPS_DEFAULT_PROFILES_FILE
export ROS_DISCOVERY_SERVER="192.168.50.1:11811"

ros2 launch visual_servoing visual_servoing_test.launch.py digital_twin_mode:=sim_to_real
```

**3. On the Laptop (Python Control Terminal):**
```bash
cd ~/new_rl_ros2/ros2_ws
source install/setup.bash
unset RMW_IMPLEMENTATION
unset FASTRTPS_DEFAULT_PROFILES_FILE
export ROS_DISCOVERY_SERVER="192.168.50.1:11811"

cd src/visual_servoing/scripts
python3 train_visual_servoing.py
```

## Key Features

### ArUco Board Detection
- 4 ArUco markers define a 12×12cm drawing workspace
- Board pose is detected once and locked for the training session
- All targets are generated relative to the detected board position

### Neural Inverse Kinematics
- Position-based loss: `FK(predicted_joints) vs target_position`
- Trained on 500k FK samples from the exact URDF kinematic chain
- Jacobian refinement for sub-millimeter accuracy
- Reduces action space from 6D (joints) to 3D (XYZ position)

### Drawing Task
- Configurable shapes: triangle, square, line
- 7 waypoints per edge (22 total for triangle)
- Sparse reward: 0 on waypoint reached, −1 otherwise
- Board-relative waypoint generation via ArUco detection

## Hardware

- **Robot**: 6-DOF arm (custom URDF from Onshape CAD)
- **Joints**: 4× Rz (yaw), 2× Ry (pitch) — all ±90° (joint 2: -60° to +90°)
- **End-effector**: Pen tip (`bibut_1` link) with ArUco marker (ID 4) for camera tracking
- **Camera**: Fixed overhead, pitched 33° toward board
- **Target deployment**: Raspberry Pi 4 via ONNX Runtime

## End-Effector ArUco Marker

A small ArUco marker (ID 4, 20mm) is attached next to the pen for camera-based end-effector detection. To adjust its position, edit the `ee_aruco_joint` in `urdf/new_arm/new_arm.xacro`:

```xml
<joint name="ee_aruco_joint" type="fixed">
  <origin xyz="0.015 0.025 -0.015" rpy="1.5708 0 0"/>
  <parent link="but_1"/>
  <child link="ee_aruco_marker"/>
</joint>
```

| Parameter | Effect |
|-----------|--------|
| `xyz X` (0.015) | Forward/back relative to pen (increase = further from pen axis) |
| `xyz Y` (0.025) | Left/right offset from pen center |
| `xyz Z` (-0.015) | Up/down along pen shaft |
| `rpy R` (1.5708) | Rotation so marker face is visible to overhead camera |

The marker texture is generated from `DICT_4X4_1000` ID 4 and stored in `models/aruco_marker_4/materials/textures/marker_4.png`. The Gazebo material (PBR albedo map) is configured in `urdf/new_arm/new_arm.gazebo.xacro`.

