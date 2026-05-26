# Digital Twin (Sim-to-Real) — Unified Pipeline

This doc captures the complete working sequence to train in Gazebo and deploy on the physical Pi robot arm.

## Architecture

- **Gazebo** (laptop): Runs simulation, scores RL episodes via `FollowJointTrajectory` action
- **Pi** (Raspberry Pi 4): Runs `wicom_roboarm_unified_node.py` — direct PCA9685 I2C control of 6 servos
- **Communication**: Laptop publishes to `/pca9685_servo/trajectory`, Pi publishes `/pca9685_servo/joint_states`

### Topic Contract

| Direction | Topic | Type | Units |
|-----------|-------|------|-------|
| Laptop → Pi | `/pca9685_servo/trajectory` | `JointTrajectory` | degrees |
| Pi → Laptop | `/pca9685_servo/joint_states` | `JointState` | radians |
| Laptop → Pi | `/pca9685_servo/command` | `JointState` | degrees |
| Laptop → Pi | `/pca9685_servo/home` | `Trigger` service | — |

### Joint Mapping (Gazebo ↔ Pi)

| Gazebo Joint | Pi Joint | Home (deg) | Inverted | Servo | Pi Channel |
|-------------|----------|-----------|----------|-------|------------|
| Revolute 20 | base | 90 | No | TD-8120MG | CH0 |
| Revolute 22 | shoulder | 90 | No | TD-8120MG | CH1 |
| Revolute 23 | elbow | 90 | No | MG996R | CH4 |
| Revolute 26 | wrist_roll | 90 | Yes | MG90S | CH8 |
| Revolute 28 | wrist_pitch | 90 | No | MG90S | CH9 |
| Revolute 30 | pen | 90 | No | MG90S | CH12 |

---

## Deploy to Pi (sync code)

```bash
# Best (keeps Pi folder in sync):
rsync -av --delete ./wicom_roboarm/ piros2@192.168.50.1:~/ros2_ws/src/wicom_roboarm/
```

---

## Full Test Sequence

### Step 0: Network Setup

Connect your laptop to the Pi's Wi-Fi hotspot (IP: `192.168.50.1`).

### Step 1: Start Pi Robot Node (Pi SSH terminal)

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select wicom_roboarm
source install/setup.bash
ros2 launch wicom_roboarm wicom_roboarm.launch.py
```

You should see the node log all 6 joints and their channel assignments.

### Step 2: Verify Connection (Laptop terminal 1)

```bash
cd ~/new_rl_ros2/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=~/new_rl_ros2/ros2_ws/src/visual_servoing/config/fastdds_twin.xml
ros2 run visual_servoing verify_connection
```

**Must print `✅ CONNECTION SUCCESSFUL!`** before proceeding. If it times out:
1. Check both machines are on the same network
2. Verify `fastdds_twin.xml` has the Pi IP (`192.168.50.1`)
3. Verify the Pi node is running and publishing

### Step 3: Launch Gazebo (Laptop terminal 2)

```bash
cd ~/new_rl_ros2/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
# IMPORTANT: unset stale Discovery Server env vars from old sessions
unset ROS_DISCOVERY_SERVER FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE
ros2 launch visual_servoing visual_servoing_test.launch.py
```

Wait ~15 seconds for the robot arm to appear in Gazebo.

### Step 4: Shadow Training (Laptop terminal 3)

```bash
cd ~/new_rl_ros2/ros2_ws/src/visual_servoing/scripts
source /opt/ros/humble/setup.bash
source ~/new_rl_ros2/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=~/new_rl_ros2/ros2_ws/src/visual_servoing/config/fastdds_twin.xml
export PI_SHADOW_REPLAY_HZ=5.0
python3 train_visual_servoing.py
```

At the menu:
1. Choose **`7`** (PID Tuning)
2. Submode: **`a`** (Reaching) or **`b`** (Drawing)
3. Backend: **`b`** (sim_to_real_shadow)
4. Board detection: **`n`**

Each episode:
- RL agent tunes PID gains in Gazebo (fast, ~50Hz)
- At episode end, the best trajectory is downsampled to the configured rate (e.g. `5.0` Hz) and replayed on the physical Pi robot.
- **Segment Command Monitoring**: The terminal displays real-time segment-by-segment statuses:
  `[SEG 1/15] Cmd: [base=90.0°, shoulder=90.0°, ...] | Actual: [base=89.8°, shoulder=90.2°, ...] | Status: OK | dur=0.20s`
- **Telemetry Logs**: Detailed command-vs-actual joint data for every segment is logged directly on the computer to:
  `training_results/logs/shadow_pid_episode_log_[timestamp].txt`
- **Replay Telemetry Summary**: Replay finishes with a summary showing how many segments received fresh Pi joint-state feedback.
- **Auto-Homing**: The physical robot automatically calls `/pca9685_servo/home` Trigger service after replay to reset state between episodes.

### Step 5: Multi-Episode Deploy to Pi — Option 8 (Laptop terminal 3)

After training, start the deployment workflow:

```bash
python3 train_visual_servoing.py
```

At the menu:
1. Choose **`8`** (Deploy to Pi)
2. Submode: **`a`** (Reaching) or **`b`** (Drawing)
3. Select artifact: press **Enter** to use the latest training result
4. Select gains: press **Enter** to use the latest gains
5. **Parameters**:
   - Number of episodes: input target run count (default `5`)
   - Replay rate: type `5.0` or press **Enter** for default
 
The script will run a structured multi-episode validation:
- **Episode Loop**: For each run, it homes the arm, moves it to the starting pose, and executes the trajectory.
- **Per-segment Printouts**: Displays live commanded vs actual angles and status for each step.
- **Session Telemetry Logs**: Appends all segment metrics, packet loss indicators, and final Cartesian targets to:
  `training_results/logs/deploy_replay_log_[timestamp].txt`
- **Data & Plot Generation**:
  - Saves the full session data structure to `training_results/pkl/deploy_results_*.pkl`.
  - Saves a multi-panel comparison plot comparing the Commanded trajectory with the Actual trajectories of all runs overlaid to visualize repeatability:
    `training_results/png/deploy_comparison_*.png`
- **Clean Exit**: Automatically homes the arm on complete execution or Ctrl+C interruption.

---

## Simple Discovery Configuration

The `fastdds_twin.xml` file (in `ros2_ws/src/visual_servoing/config/`) configures unicast peer discovery:

```xml
<initialPeersList>
    <locator><udpv4><address>192.168.50.1</address></udpv4></locator>   <!-- Pi -->
    <locator><udpv4><address>127.0.0.1</address></udpv4></locator>      <!-- localhost -->
</initialPeersList>
```

Both machines must export:
```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=<path_to_fastdds_twin.xml>
```

> **Note**: The Gazebo terminal (Step 3) must NOT have these DDS variables set, otherwise Gazebo's internal transport breaks. Only the training terminal (Step 4/5) and verify_connection terminal (Step 2) need them.
