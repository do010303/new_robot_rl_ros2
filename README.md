# Visual Servoing RL — 6-DOF Robot Arm

Deep Reinforcement Learning for precise reaching/drawing on a 6-DOF robotic arm, using visual servoing with ArUco marker detection in Gazebo.

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
cd ros2_ws/src/visual_servoing/scripts
python3 train_visual_servoing.py
```

## Digital Twin (Sim-to-Real)

Sim-to-Real is implemented as **timed command mirroring** for *single-command* moves:
- Gazebo: timed `FollowJointTrajectory`
- Pi: `/pca9685_servo/trajectory` (`trajectory_msgs/JointTrajectory`) using `time_from_start`

Docs + tools:
- Full walkthrough: `docs/digital_twin_sim_to_real.md`
- Pi command cheatsheet: `docs/pi_robot_control_commands.md`
- Pi deploy helper: `scripts/deploy_pi_wicom_roboarm.sh`
- FastDDS SUPER_CLIENT template: `scripts/super_client.xml`
