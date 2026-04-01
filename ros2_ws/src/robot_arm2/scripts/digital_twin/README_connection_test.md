# ROS 2 Communication Test (Laptop ↔ Raspberry Pi)

## Communication Protocol: Native ROS 2 DDS

We use **native ROS 2 DDS** — no custom sockets, no bridges needed.  
Both machines just need to be on the same Wi-Fi network with the same `ROS_DOMAIN_ID`.

## Prerequisites

### On BOTH machines:
```bash
export ROS_DOMAIN_ID=0          # Same ID on both!
export ROS_LOCALHOST_ONLY=0     # Allow network discovery
source /opt/ros/humble/setup.bash
```

### Network: Connect laptop to Pi's Wi-Fi hotspot

## Step-by-Step Test

### Step 1: On the Raspberry Pi

```bash
# Terminal 1 — Launch the robot arm
ros2 launch wicom_roboarm wicom_roboarm.launch.py

# Terminal 2 — Run the ping/pong test node
python3 test_ros2_connection_pi.py
```

### Step 2: On the Laptop

```bash
# Run the laptop test
python3 test_ros2_connection.py
```

### Step 3: Expected Output

**On the Laptop:**
```
✅ PONG received! seq=1  RTT=12.3 ms
🦾 Joint States #1: {'base': '90.0°', 'shoulder': '180.0°', ...}
```

**On the Pi:**
```
📡 Ping #1 received, pong sent back!
```

## What the test checks

| Check | Topic | Type | Direction |
|---|---|---|---|
| Heartbeat | `/twin/ping` → `/twin/pong` | `std_msgs/String` | Laptop → Pi → Laptop |
| Arm telemetry | `/pca9685_servo/joint_states` | `sensor_msgs/JointState` | Pi → Laptop |

## Troubleshooting

If you see **no pongs**:
1. `ros2 topic list` — do you see /twin/ping on *both* machines?
2. Check `ROS_DOMAIN_ID` matches on both
3. Check `ROS_LOCALHOST_ONLY=0`
4. Try: `ros2 multicast receive` on both to test DDS multicast

If pings work but **no joint_states**:
1. Is `wicom_roboarm.launch.py` running on the Pi?
2. On the Pi, run: `ros2 topic echo /pca9685_servo/joint_states`
