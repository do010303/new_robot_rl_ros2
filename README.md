# 6-DOF Robot Arm — Deep Reinforcement Learning for Precise Operations

A ROS2-based system for training a **6-DOF robotic manipulator** to perform precise operations (target reaching & shape drawing) using **Soft Actor-Critic (SAC)** reinforcement learning, with an optional **Neural Predictive Inverse Kinematics** module. Includes a full **Sim-to-Real pipeline** for deploying trained models onto a Raspberry Pi 4.

![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)
![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-orange)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)

<p align="center">
  <img src="docs/robot_gazebo.png" alt="Robot in Gazebo Simulation" width="600"/>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
  - [Simulation Training (robot\_arm2)](#1-simulation-training-robot_arm2)
  - [Visual Servoing Training (visual\_servoing)](#2-visual-servoing-training-visual_servoing)
- [Training Menu Options](#training-menu-options)
- [Neural IK Module](#neural-ik-module)
- [Sim-to-Real Deployment](#sim-to-real-deployment)
- [Training Results](#training-results)
- [Troubleshooting](#troubleshooting)
- [Authors](#authors)

---

## Overview

This project implements:

1. **SAC (Soft Actor-Critic)** agent with **Hindsight Experience Replay (HER)** for sample-efficient training
2. **Neural Predictive IK** — a learned inverse kinematics module that maps 3D Cartesian targets to 6-DOF joint angles, allowing the RL agent to operate in a simpler 3D action space instead of the full 6D joint space
3. **Two training scenarios**:
   - **Target Reaching** — the robot learns to touch randomly spawned target spheres
   - **Shape Drawing** — the robot learns to trace geometric shapes (triangle, square, circle, etc.) with a pen end-effector
4. **Visual Servoing** — camera-based ArUco marker detection for closed-loop control using real-time vision feedback
5. **Sim-to-Real Pipeline** — ONNX export with INT8 quantization for real-time inference on Raspberry Pi 4

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   Gazebo Harmonic Sim                       │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │ 6-DOF Arm│  │ Target Sphere│  │ Drawing Surface +     │ │
│  │ (URDF)   │  │ / Shape      │  │ ArUco Markers + Camera│ │
│  └────┬─────┘  └──────────────┘  └───────────────────────┘ │
│       │ ros2_control (position controllers)                 │
└───────┼────────────────────────────────────────────────────┘
        │
┌───────▼────────────────────────────────────────────────────┐
│                  ROS2 Humble Middleware                      │
│  /joint_states, /arm_controller/commands, /camera/image_raw │
└───────┬────────────────────────────────────────────────────┘
        │
┌───────▼────────────────────────────────────────────────────┐
│               RL Training / Visual Servoing                 │
│  ┌─────────────┐  ┌───────────┐  ┌───────────────────────┐ │
│  │ SAC Agent   │  │ Neural IK │  │ Vision (ArUco Detect) │ │
│  │ + HER       │  │ Module    │  │ + Camera Feedback     │ │
│  └─────────────┘  └───────────┘  └───────────────────────┘ │
└────────────────────────────────────────────────────────────┘
        │
┌───────▼────────────────────────────────────────────────────┐
│              Deployment (Raspberry Pi 4)                    │
│  ONNX Runtime + INT8 Quantized Models + PCA9685 Servos     │
└────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Ubuntu | 22.04 LTS |
| ROS2 | Humble Hawksbill |
| Gazebo | **Harmonic** (gz-sim 8) |
| Python | 3.10+ |
| PyTorch | 2.0+ |

> **Note**: This project was upgraded from Gazebo Fortress to **Gazebo Harmonic**. ROS2 Humble does not ship pre-built packages for Harmonic, so `ros_gz` and `gz_ros2_control` must be built from source (see Step 2 below).

---

## Installation

### 1. Install ROS2 Humble & Gazebo Harmonic

```bash
# ROS2 Humble
sudo apt update && sudo apt install ros-humble-desktop-full

# ROS2 control framework
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers
sudo apt install ros-humble-xacro python3-colcon-common-extensions

# Install Gazebo Harmonic
sudo apt-get install curl lsb-release gnupg
sudo curl https://packages.osrfoundation.org/gazebo.gpg \
  --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
  http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install gz-harmonic
```

### 2. Build `ros_gz` and `gz_ros2_control` from Source

Since ROS2 Humble has no pre-built packages for Gazebo Harmonic, these must be built from source:

```bash
# Remove any existing Fortress-era packages
sudo apt remove -y ros-humble-ros-gz* ros-humble-ros-ign* 2>/dev/null
sudo apt remove -y ros-humble-ign-ros2-control ros-humble-gz-ros2-control 2>/dev/null

cd ~/new_rl_ros2/ros2_ws/src

# Clone ros_gz (Humble branch)
git clone -b humble https://github.com/gazebosim/ros_gz.git

# Clone gz_ros2_control (Humble branch)
git clone -b humble https://github.com/ros-controls/gz_ros2_control.git

# Install rosdep dependencies (some may fail for gz-transport13 — that's OK)
rosdep install --from-paths . --ignore-src -r -y 2>/dev/null || true

# Build with GZ_VERSION=harmonic (use sequential build to avoid OOM on low-RAM systems)
cd ~/new_rl_ros2/ros2_ws
source /opt/ros/humble/setup.bash
export GZ_VERSION=harmonic
MAKEFLAGS="-j2" colcon build --executor sequential
source install/setup.bash
```

> ⚠️ **Low RAM (≤8GB)?** Use `MAKEFLAGS="-j1"` to avoid out-of-memory crashes during build.

### 3. Install Python Dependencies

```bash
pip install torch numpy matplotlib pandas opencv-contrib-python scipy
```

### 4. Clone & Build Project Packages

```bash
git clone https://github.com/EmNetLab411/new_rl_ros2.git
cd new_rl_ros2/ros2_ws

source /opt/ros/humble/setup.bash
colcon build --packages-select robot_arm2 visual_servoing
source install/setup.bash
```

> **Note**: You must run `source /opt/ros/humble/setup.bash && source install/setup.bash` every time you open a new terminal.

---

## Project Structure

```
new_rl_ros2/
├── README.md
├── .gitignore
├── docs/                              # Documentation images
│   └── robot_gazebo.png
├── ref/                               # Reference code & configs
│
└── ros2_ws/src/
    ├── robot_arm2/                    # PRIMARY PACKAGE — Simulation RL Training
    │   ├── CMakeLists.txt
    │   ├── package.xml
    │   ├── config/                    # ros2_control controller configs
    │   ├── launch/
    │   │   ├── rl_training.launch.py          # Reaching task
    │   │   └── drawing_training.launch.py     # Drawing task
    │   ├── meshes/                    # Robot STL meshes
    │   ├── models/                    # Gazebo models (ArUco markers, targets)
    │   ├── urdf/                      # Robot XACRO/URDF descriptions
    │   ├── worlds/                    # Gazebo world files
    │   └── scripts/
    │       ├── train_robot.py                 # ★ Main training entry point
    │       ├── target_manager.py              # Random target spawning
    │       ├── agents/
    │       │   └── sac_agent.py               # SAC + HER implementation
    │       ├── rl/
    │       │   ├── rl_environment.py          # Reaching RL environment
    │       │   ├── drawing_environment.py     # Drawing RL environment
    │       │   ├── neural_ik.py               # Neural IK model definition
    │       │   └── fk_ik_utils.py             # FK/IK utilities
    │       ├── drawing/
    │       │   ├── drawing_config.py          # Shape configs
    │       │   ├── shape_generator.py         # Target shape generation
    │       │   ├── gazebo_visualizer.py       # Gazebo marker visualization
    │       │   └── line_visualizer.py         # Line drawing visualization
    │       ├── utils/
    │       │   └── her.py                     # Hindsight Experience Replay
    │       └── deployment/                    # Sim-to-Real deployment
    │           ├── deploy_drawing_on_pi.py    # Pi deployment script
    │           ├── deploy_to_pi.sh            # SCP deploy helper
    │           ├── export_onnx_quantized.py   # ONNX export + INT8 quant
    │           └── wicom_roboarm/             # Pi-side ROS2 package
    │
    └── visual_servoing/               # VISUAL SERVOING PACKAGE — Camera-based
        ├── setup.py / setup.cfg
        ├── package.xml
        ├── aruco_markers/             # ArUco marker images & SVGs
        ├── config/                    # Camera calibration & robot config
        ├── launch/
        │   └── visual_servoing_test.launch.py
        ├── meshes/                    # Robot STL meshes
        ├── models/                    # Gazebo ArUco marker models
        ├── urdf/                      # Robot XACRO (with camera)
        ├── worlds/                    # Visual servoing Gazebo world
        ├── scripts/
        │   ├── train_visual_servoing.py       # ★ VS training entry point
        │   ├── agents/
        │   │   └── sac_agent.py               # SAC agent (VS variant)
        │   ├── rl/
        │   │   ├── rl_environment.py          # VS RL environment
        │   │   ├── drawing_environment.py     # VS drawing environment
        │   │   ├── neural_ik.py               # Neural IK for VS
        │   │   └── fk_ik_utils.py
        │   ├── drawing/                       # Shape generation
        │   └── utils/
        │       └── her.py                     # HER for VS
        └── vs_lib/                    # Visual Servoing Library
            ├── core/
            │   ├── kinematics.py              # FK/IK computations
            │   ├── filters.py                 # Signal filtering
            │   └── profiler.py                # Performance profiling
            ├── drivers/
            │   ├── i2c_manager.py             # I2C comm (Pi hardware)
            │   └── sensor_driver.py           # Sensor interface
            ├── nodes/
            │   ├── vision_node_ros2.py        # ROS2 vision node
            │   ├── drawing_executor_ros2.py   # Drawing execution node
            │   └── shape_generator.py         # Shape target generator
            └── vision/
                ├── vision_aruco_detector.py   # ArUco detection pipeline
                └── camera_viewer.py           # Camera debug viewer
```

---

## How to Run

### 1. Simulation Training (`robot_arm2`)

Open **2 terminals**.

#### Terminal 1 — Launch Simulation

```bash
cd ~/new_rl_ros2/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash

# For reaching (target) training:
ros2 launch robot_arm2 rl_training.launch.py

# OR for drawing training:
ros2 launch robot_arm2 drawing_training.launch.py
```

Wait ~10 seconds for Gazebo to fully load.

#### Terminal 2 — Start Training

```bash
cd ~/new_rl_ros2/ros2_ws/src/robot_arm2/scripts
source /opt/ros/humble/setup.bash
python3 train_robot.py
```

### 2. Visual Servoing Training (`visual_servoing`)

Open **3 terminals**.

#### Terminal 1 — Launch Visual Servoing Simulation

```bash
cd ~/new_rl_ros2/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch visual_servoing visual_servoing_test.launch.py
```

#### Terminal 2 — Start VS Training

```bash
cd ~/new_rl_ros2/ros2_ws/src/visual_servoing/scripts
source /opt/ros/humble/setup.bash
python3 train_visual_servoing.py
```

#### Terminal 3 — Open Camera Viewer (with RL Overlay)

```bash
cd ~/new_rl_ros2/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 run visual_servoing camera_viewer
```

This opens a live camera feed showing ArUco marker detection and RL agent overlay in real time.

---

## Training Menu Options

When you run `train_robot.py`, you will see:

```
======================================================================
🎮 TRAINING MENU
======================================================================
1. 🎮 Manual Test Mode          — Verify environment works
2. 🤖 SAC Training (6D Direct)  — Direct joint control
3. 🧠 SAC + Neural IK (3D)      — Cartesian position control
4. 🧠 Train Neural IK Model     — Pre-train the IK module
5. 🖋️ Drawing Task (6D Direct)   — Draw shapes with joint control
6. 🖋️ Drawing Task + Neural IK  — Draw shapes with position control
======================================================================
```

### Quick Start Guide

| Goal | Launch File | Menu Option |
|---|---|---|
| Verify setup | `rl_training.launch.py` | **1** |
| Train reaching (basic) | `rl_training.launch.py` | **2** |
| Train reaching (Neural IK) | `rl_training.launch.py` | **3** ★ |
| Train drawing (basic) | `drawing_training.launch.py` | **5** |
| Train drawing (Neural IK) | `drawing_training.launch.py` | **6** ★ |

> ★ Options 3, 6 require a pre-trained Neural IK model. Train it first with option **4**.

---

## Neural IK Module

The **Neural Predictive Inverse Kinematics** module learns to map 3D Cartesian positions (x, y, z) → 6-DOF joint angles. This simplifies the RL action space from 6D joint space to 3D position space, dramatically improving training efficiency.

```
Training flow:
  Step 1: Run option 4 → Generates neural_ik.pth (in checkpoints/)
  Step 2: Use option 3 or 6 → Agent controls (x, y, z), Neural IK converts to joints
```

---

## Sim-to-Real Deployment

Deploy trained models to a physical robot running on **Raspberry Pi 4**:

1. **Export models** to ONNX with INT8 quantization:
   ```bash
   cd ros2_ws/src/robot_arm2/scripts/deployment
   python3 export_onnx_quantized.py
   ```

2. **Deploy to Pi**:
   ```bash
   bash deploy_to_pi.sh <PI_IP_ADDRESS>
   ```

3. **Run on Pi**:
   ```bash
   python3 deploy_drawing_on_pi.py --episodes 10 --steps 200
   ```

The deployment uses **ONNX Runtime** for fast inference and **PCA9685** servo driver for motor control.

---

## Training Results

Training outputs are saved locally (not tracked in git):

```
scripts/training_results/
├── png/        # Training reward/loss curves
├── csv/        # Tabular training data
├── pkl/        # Replay buffers & model checkpoints
└── step_logs/  # Per-step JSONL logs
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| **Gazebo won't open** | Run both `source /opt/ros/humble/setup.bash` and `source install/setup.bash` |
| **`gz_ros2_control` plugin not found** | Rebuild from source with `GZ_VERSION=harmonic` (see Installation Step 2) |
| **Build crashes (OOM)** | Use `MAKEFLAGS="-j1" colcon build --executor sequential` |
| **Clock skew warnings** | Fix with `sudo timedatectl set-local-rtc 1 --adjust-system-clock` (common on dual-boot) |
| **Robot not moving** | Check controllers: `ros2 control list_controllers` — should show `joint_state_broadcaster [active]` and `arm_controller [active]` |
| **"Neural IK not found"** | Train Neural IK first using menu option **4** |
| **Camera not showing** | Rebuild visual_servoing: `colcon build --packages-select visual_servoing` |
| **Slow training FPS** | Close Gazebo GUI: add `--headless` to launch, or reduce physics step size |

---

## Authors

**EmNetLab411** — [github.com/EmNetLab411](https://github.com/EmNetLab411)

**ducanh** — [github.com/do010303](https://github.com/do010303)
