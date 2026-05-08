# Digital Twin (Sim-to-Real) — command-sync (single move)

This doc captures the working sequence to mirror **single-command** joint moves so Gazebo and the physical Pi arm move in sync *by time window*, without requiring encoder feedback.

## Architecture (why timing was off before)

- Gazebo is driven by a timed `FollowJointTrajectory` action (smooth trajectory + explicit duration).
- The old “mirror” approach sampled `/joint_states` at low rate and forwarded staircase targets to the Pi.
- The Pi arm has no true joint feedback, so state mirroring can’t lock timing.

The fix is **command mirroring**:

1) send the *same target* to Gazebo and the Pi at nearly the same time  
2) include one shared duration  
3) do not issue the next command until the motion window ends

## Topics

- Immediate setpoint (old, still supported): `/pca9685_servo/command` (`sensor_msgs/JointState`)
- Timed move (new): `/pca9685_servo/trajectory` (`trajectory_msgs/JointTrajectory`)

Sending `/pca9685_servo/command` cancels any active timed trajectory on the Pi.

## Deploy to Pi (avoid nested folder)

Bad (creates nested `wicom_roboarm/wicom_roboarm/...` if the folder already exists):

```bash
scp -r ./wicom_roboarm/ piros2@192.168.50.1:~/ros2_ws/src/wicom_roboarm
```

Good (copy into `src/`):

```bash
scp -r ./wicom_roboarm piros2@192.168.50.1:~/ros2_ws/src/
```

Best (keeps Pi folder in sync):

```bash
rsync -av --delete ./wicom_roboarm/ piros2@192.168.50.1:~/ros2_ws/src/wicom_roboarm/
```

If you already nested it, a safe fix is:

```bash
cd ~/ros2_ws/src
mv wicom_roboarm wicom_roboarm.bak
mv wicom_roboarm.bak/wicom_roboarm wicom_roboarm
```

## Discovery Server v2 and “Unknown topic”

When using Fast DDS Discovery Server v2, `ros2 topic list/info` may show only `/rosout` and `/parameter_events` unless the ROS 2 CLI is configured as a SUPER_CLIENT.

### 1) Start server (Pi)

```bash
fastdds discovery -i 0 -l 192.168.50.1 -p 11811 &
```

### 2) Create SUPER_CLIENT profile (Pi)

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

### 3) Export env (each terminal that uses `ros2 ...`)

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISCOVERY_SERVER="192.168.50.1:11811"
export FASTRTPS_DEFAULT_PROFILES_FILE=~/super_client.xml
export FASTDDS_DEFAULT_PROFILES_FILE=~/super_client.xml

ros2 daemon stop
ros2 daemon start
```

## Pi: build + run

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select wicom_roboarm
source install/setup.bash
ros2 launch wicom_roboarm wicom_roboarm.launch.py
```

## Pi: local motion tests

Immediate:

```bash
ros2 topic pub --once /pca9685_servo/command sensor_msgs/msg/JointState \
"{name: ['base'], position: [135.0]}"
```

Timed:

```bash
ros2 topic pub --once -w 1 /pca9685_servo/trajectory trajectory_msgs/msg/JointTrajectory \
"{joint_names: ['base'], points: [{positions: [135.0], time_from_start: {sec: 1, nanosec: 0}}]}"
```

## Laptop: run sim-to-real command-sync

Gazebo terminal:

```bash
cd ~/new_rl_ros2/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISCOVERY_SERVER="192.168.50.1:11811"
ros2 launch visual_servoing visual_servoing_test.launch.py digital_twin_mode:=sim_to_real
```

Control terminal:

```bash
cd ~/new_rl_ros2/ros2_ws
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DISCOVERY_SERVER="192.168.50.1:11811"
export VISUAL_SERVOING_DIGITAL_TWIN_MODE=sim_to_real
cd src/visual_servoing/scripts
python3 train_visual_servoing.py
```

