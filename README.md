# Visual Servoing RL — 6-DOF Robot Arm

Deep Reinforcement Learning for precise reaching/drawing on a 6-DOF robotic arm, using visual servoing with ArUco marker detection in Gazebo.

---

# 🚀 8. Option 8 - Digital Twin Realtime Mirror

Chạy training + mirroring cùng lúc trong 1 terminal. Mirror chạy background bắt `/joint_states` từ Gazebo, training chạy foreground di chuyển robot trong sim. Khi robot sim cử động, robot thật trên Pi bám theo realtime.

```text
┌──────────┐   /joint_states   ┌──────────────────┐  /pca9685_servo/command  ┌─────┐
│  Gazebo  │ ───────────────→  │ sim_to_pi_mirror │ ───────────────────────→ │ Pi  │
│  + Train │   (radian, 50Hz)  │  (background)    │   (degree, 10Hz)        │     │
└──────────┘                   │  deadband 0.5°   │                         └─────┘
                               └──────────────────┘
```

### Environment Variables (BẮT BUỘC trên cả 2 máy)

```bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=~/new_rl_ros2/ros2_ws/src/visual_servoing/config/fastdds_twin.xml
```

### Bước 1: Pi — Khởi chạy node servo

```bash
ssh piros2@192.168.50.1
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch wicom_roboarm wicom_roboarm.launch.py
```

### Bước 2: Laptop Terminal 1 — Khởi chạy Gazebo

```bash
cd ~/new_rl_ros2/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=~/new_rl_ros2/ros2_ws/src/visual_servoing/config/fastdds_twin.xml
ros2 launch visual_servoing visual_servoing_test.launch.py
```

### Bước 3: Laptop Terminal 2 — Chạy Option 8 (mirror + training)

```bash
cd ~/new_rl_ros2/ros2_ws/src/visual_servoing/scripts
source /opt/ros/humble/setup.bash
source ~/new_rl_ros2/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=~/new_rl_ros2/ros2_ws/src/visual_servoing/config/fastdds_twin.xml
python3 train_visual_servoing.py
```

Chọn **8** từ menu. Script sẽ hỏi:

```text
Mirror publish rate Hz (default 10):     ← Enter để dùng 10Hz
Mirror deadband degrees (default 0.5):   ← Enter để dùng 0.5°
```

---


## Repo layout

- `ros2_ws/` — Gazebo + `visual_servoing` stack (laptop)
- `wicom_roboarm/` — Raspberry Pi hardware package (PCA9685 servo driver)
- `docs/` — documentation
- `ref/` — reference snapshots / notes (not meant for deployment)

## Overview

This project trains a robot arm to perform precise reaching and drawing tasks on a vertical board, using:
- SAC (Soft Actor-Critic) with HER (Hindsight Experience Replay)
- ArUco-based visual servoing for dynamic workspace detection
- Neural Inverse Kinematics (optional) for XYZ control
- Gazebo + `ros2_control` joint trajectory controller

### Architecture

```
Camera → ArUco Detection → Board Transform → RL Agent → Joint Commands → Robot
                ↓                                           ↑
         Board-relative                              Neural IK (optional)
         target generation                           Position → Joints
```

## Project structure (main package)

```
ros2_ws/src/visual_servoing/
├── launch/                       # main launch files
├── urdf/new_arm/                 # robot URDF / xacros
├── worlds/                       # Gazebo world(s)
├── models/                       # ArUco marker models
├── scripts/
│   ├── train_visual_servoing.py  # interactive training entrypoint
│   ├── rl/                       # environments + FK/IK utilities
│   └── drawing/                  # Gazebo drawing visualization
└── vs_lib/vision/                # camera viewer + ArUco detector
```

## Quick start (simulation)

Prepare Python first. The RL/drawing stack uses SciPy, Gymnasium, Torch, Matplotlib, and OpenCV ArUco on top of ROS 2:

```bash
cd /home/ducanh/new_rl_ros2
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -U pip
pip install -r ros2_ws/src/visual_servoing/requirements.txt
```

Build:

```bash
cd ros2_ws
colcon build --packages-select visual_servoing
source install/setup.bash
```

Launch simulation:

```bash
ros2 launch visual_servoing visual_servoing_test.launch.py
```

Run training menu (separate terminal):

```bash
cd /home/ducanh/new_rl_ros2
source .venv/bin/activate
cd ros2_ws/src/visual_servoing/scripts
python3 train_visual_servoing.py
```

## Square PID training

The most reliable square-learning path in the current codebase is:

1. Launch Gazebo with `ros2 launch visual_servoing visual_servoing_test.launch.py`
2. Run `python3 train_visual_servoing.py`
3. Choose `7` for PID tuning
4. Choose `b` for drawing
5. Choose backend `a` for `sim`
6. Answer `n` to live board detection unless your camera pipeline is already publishing `/vision/board_pose`

Without live board detection, the drawing environment now uses a deterministic fallback board plane in `base_link` at roughly `[-0.50, 0.0, 0.60]` instead of leaving square waypoints in raw board-local coordinates near the origin.

## Digital Twin (Sim-to-Real)

Sim-to-Real is implemented as **timed command mirroring** for *single-command* moves:
- Gazebo: timed `FollowJointTrajectory`
- Pi: `/pca9685_servo/trajectory` (`trajectory_msgs/JointTrajectory`) using `time_from_start`

Docs + tools:
- Full walkthrough: `docs/digital_twin_sim_to_real.md`
- Pi command cheatsheet: `docs/pi_robot_control_commands.md`
- Pi deploy helper: `scripts/deploy_pi_wicom_roboarm.sh`
- FastDDS SUPER_CLIENT template: `scripts/super_client.xml`
