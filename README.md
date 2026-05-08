# new_robot_rl_ros2

ROS 2 (Humble) workspace for visual servoing + RL training in Gazebo, with an optional **Digital Twin Sim-to-Real** path that mirrors *single* joint-target commands to a physical Raspberry Pi servo arm.

## Repo layout

- `ros2_ws/` — laptop Gazebo + `visual_servoing` stack
- `wicom_roboarm/` — Raspberry Pi hardware package (PCA9685 servo driver + topics)
- `docs/` — documentation
- `ref/` — reference snapshots / notes

## Digital Twin (Sim-to-Real) — timed command mirroring

Goal: when you issue a single move from the laptop control script, **Gazebo and the physical arm start together and share the same motion window**.

This uses:
- Gazebo: `FollowJointTrajectory` with an explicit duration
- Pi: `/pca9685_servo/trajectory` (`trajectory_msgs/JointTrajectory`) with `time_from_start` to ramp the servos over the same duration

### 0) Deploy `wicom_roboarm` to the Pi

Recommended (avoids the “nested folder” mistake):

```bash
cd /path/to/new_robot_rl_ros2
rsync -av --delete ./wicom_roboarm/ piros2@192.168.50.1:~/ros2_ws/src/wicom_roboarm/
```

If you prefer `scp`, copy into `~/ros2_ws/src/` (NOT into `~/ros2_ws/src/wicom_roboarm`):

```bash
scp -r ./wicom_roboarm piros2@192.168.50.1:~/ros2_ws/src/
```

### 1) Pi: start Fast DDS Discovery Server

```bash
fastdds discovery -i 0 -l 192.168.50.1 -p 11811 &
```

### 2) Pi: configure ROS 2 CLI introspection (SUPER_CLIENT)

With Discovery Server v2, `ros2 topic list/info` can show only `/rosout` + `/parameter_events` unless the CLI runs as a SUPER_CLIENT.

Create `~/super_client.xml` (or copy `scripts/super_client.xml` from this repo onto the Pi):

```bash
cat > ~/super_client.xml <<'EOF'
<?xml version="1.0" encoding="UTF-8" ?>
<dds>
  <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <participant profile_name="super_client_profile" is_default_profile="true">
      <rtps>
        <builtin>
          <discovery_config>
            <discoveryProtocol>SUPER_CLIENT</discoveryProtocol>
            <discoveryServersList>
              <RemoteServer prefix="44.53.00.5f.45.50.52.4f.53.49.4d.41">
                <metatrafficUnicastLocatorList>
                  <locator>
                    <udpv4>
                      <address>192.168.50.1</address>
                      <port>11811</port>
                    </udpv4>
                  </locator>
                </metatrafficUnicastLocatorList>
              </RemoteServer>
            </discoveryServersList>
          </discovery_config>
        </builtin>
      </rtps>
    </participant>
  </profiles>
</dds>
EOF
```

Then, in every Pi terminal where you run `ros2 ...`:

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISCOVERY_SERVER="192.168.50.1:11811"
export FASTRTPS_DEFAULT_PROFILES_FILE=~/super_client.xml
export FASTDDS_DEFAULT_PROFILES_FILE=~/super_client.xml

ros2 daemon stop
ros2 daemon start
```

### 3) Pi: build + launch hardware node

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISCOVERY_SERVER="192.168.50.1:11811"
export FASTRTPS_DEFAULT_PROFILES_FILE=~/super_client.xml

cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select wicom_roboarm
source install/setup.bash

ros2 launch wicom_roboarm wicom_roboarm.launch.py
```

### 4) Pi: sanity test motion locally

Old (immediate) command topic:

```bash
ros2 topic pub --once /pca9685_servo/command sensor_msgs/msg/JointState \
"{name: ['base'], position: [135.0]}"
```

New (timed) trajectory topic:

```bash
ros2 topic pub --once -w 1 /pca9685_servo/trajectory trajectory_msgs/msg/JointTrajectory \
"{joint_names: ['base'], points: [{positions: [135.0], time_from_start: {sec: 1, nanosec: 0}}]}"
```

### 5) Laptop: launch Gazebo stack

```bash
cd ~/new_rl_ros2/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISCOVERY_SERVER="192.168.50.1:11811"

ros2 launch visual_servoing visual_servoing_test.launch.py digital_twin_mode:=sim_to_real
```

### 6) Laptop: run the control script (Terminal C)

```bash
cd ~/new_rl_ros2/ros2_ws
source install/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISCOVERY_SERVER="192.168.50.1:11811"
export VISUAL_SERVOING_DIGITAL_TWIN_MODE=sim_to_real

cd src/visual_servoing/scripts
python3 train_visual_servoing.py
```

## Troubleshooting

- If you see `Unknown topic '/pca9685_servo/trajectory'` but the robot still moves, it’s usually the ROS 2 CLI introspection (SUPER_CLIENT) issue, not the node.
- If `ros2 topic pub --once /pca9685_servo/trajectory ...` hangs at “Waiting for subscription(s)”, check that the installed launch file includes the remap:

```bash
PREFIX=$(ros2 pkg prefix wicom_roboarm)
grep -n trajectory $PREFIX/share/wicom_roboarm/launch/wicom_roboarm.launch.py
```

- If you accidentally copied into `~/ros2_ws/src/wicom_roboarm` and created a nested folder (`.../wicom_roboarm/wicom_roboarm/...`), colcon may build the wrong one.

See `docs/digital_twin_sim_to_real.md` for the full walkthrough.
See `docs/pi_robot_control_commands.md` for Pi command cheatsheet.
