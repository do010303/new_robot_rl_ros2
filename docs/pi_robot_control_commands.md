# Pi Robot Arm Control Commands (wicom_roboarm)

This is the tracked (non-`ref/`) copy of the Pi command cheatsheet.

## Launch (Pi)

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch wicom_roboarm wicom_roboarm.launch.py
```

## Topics

- Immediate setpoint (old): `/pca9685_servo/command` — `sensor_msgs/msg/JointState` (degrees)
- Timed move (new, Digital Twin): `/pca9685_servo/trajectory` — `trajectory_msgs/msg/JointTrajectory` (degrees + `time_from_start`)

Sending `/pca9685_servo/command` cancels any active timed trajectory.

## Immediate command examples

```bash
# Base → 135°
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['base'], position:[135.0]}"

# All joints → 90°
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState \
  "{name:['base','shoulder','elbow','wrist_roll','wrist_pitch','pen'], position:[90.0, 90.0, 90.0, 90.0, 90.0, 90.0]}"
```

## Timed trajectory examples

```bash
# Base → 135° in 1s
ros2 topic pub --once -w 1 /pca9685_servo/trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['base'], points: [{positions: [135.0], time_from_start: {sec: 1, nanosec: 0}}]}"

# All joints in 1s
ros2 topic pub --once -w 1 /pca9685_servo/trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['base','shoulder','elbow','wrist_roll','wrist_pitch','pen'],
    points: [{positions: [135.0, 90.0, 90.0, 90.0, 90.0, 90.0], time_from_start: {sec: 1, nanosec: 0}}]}"
```

## Services

```bash
ros2 service call /pca9685_servo/enable std_srvs/srv/Trigger
ros2 service call /pca9685_servo/disable std_srvs/srv/Trigger
ros2 service call /pca9685_servo/home std_srvs/srv/Trigger
```

