# Robot Arm Control Commands

## 🔧 Hardware Wiring (PCA9685 → RPi4 direct, no mux)

| Channel | Joint        | Servo      |
|:--------|:-------------|:-----------|
| CH 0    | Base         | TD-8120MG  |
| CH 1    | Shoulder     | TD-8120MG  |
| CH 2    | Elbow        | MG996R     |
| CH 3    | Wrist Roll   | MG90S      |
| CH 4    | Wrist Pitch  | MG90S      |
| CH 5    | Pen/Gripper  | MG90S      |

---

## Khởi taoh chương trình điều khiển
#On raspberry pi (Trên pi)
cd ros2_ws/
source install/setup.bash
ros2 launch wicom_roboarm wicom_roboarm.launch.py

## 📌 Điều khiển từng Joint riêng lẻ (Physical Robot trên Pi)

> Topic: `/pca9685_servo/command` — Msg: `sensor_msgs/msg/JointState`
> Các joint: `base`, `shoulder`, `elbow`, `wrist_roll`, `wrist_pitch`, `pen`
> Góc: **0.0° → 180.0°** (neutral = 90°)

## ⏱️ Timed move (Digital Twin / New)

> Topic: `/pca9685_servo/trajectory` — Msg: `trajectory_msgs/msg/JointTrajectory`  
> Positions: **degree** (same as servo command)  
> Duration: `time_from_start` (seconds)  
>
> Note:
> - `/pca9685_servo/command` is still supported (old behavior, immediate setpoint).
> - Sending `/pca9685_servo/command` will cancel any active timed trajectory.

### Di chuyển timed (1 lệnh, có duration)

```bash
# Base → 135° trong 1 giây
ros2 topic pub --once -w 0 /pca9685_servo/trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['base'],
    points: [{positions: [135.0], time_from_start: {sec: 1, nanosec: 0}}]}"

# Di chuyển toàn bộ 6 joint trong 1 giây
ros2 topic pub --once -w 0 /pca9685_servo/trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['base','shoulder','elbow','wrist_roll','wrist_pitch','pen'],
    points: [{positions: [135.0, 90.0, 90.0, 90.0, 90.0, 90.0], time_from_start: {sec: 1, nanosec: 0}}]}"
```

### Di chuyển từng joint

```bash
# Base (xoay đế) → quay sang trái 45° (Góc 1)
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['base'], position:[45.0]}"

# Base → quay giữa (home) (Góc 1)
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['base'], position:[90.0]}"

# Base → quay sang phải 135° (Góc 1)
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['base'], position:[135.0]}"

# Shoulder (vai) → nâng lên (Góc 2)
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['shoulder'], position:[120.0]}"

# Shoulder → hạ xuống (Góc 2)
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['shoulder'], position:[60.0]}"

# Elbow (khuỷu tay) → gập vào (Góc 3)
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['elbow'], position:[150.0]}"

# Elbow → duỗi ra (Góc 3)
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['elbow'], position:[30.0]}"

# Wrist Roll (xoay cổ tay) → xoay trái (Góc 4)
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['wrist_roll'], position:[45.0]}"

# Wrist Roll → xoay phải (Góc 4)
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['wrist_roll'], position:[135.0]}"

# Wrist Pitch (ngửa/cúi cổ tay) → ngửa lên (Góc 5)
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['wrist_pitch'], position:[130.0]}"

# Wrist Pitch → cúi xuống (Góc 5)
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['wrist_pitch'], position:[50.0]}"

# Pen (gripper/bút) → đóng / kẹp (Góc 6)
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['pen'], position:[30.0]}"

# Pen → mở / thả (Góc 6)
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['pen'], position:[120.0]}"
```

### Di chuyển nhiều joint cùng lúc

```bash
# Di chuyển base + shoulder + elbow cùng lúc
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState \
  "{name:['base','shoulder','elbow'], position:[90.0, 120.0, 60.0]}"

# Di chuyển 4 joint (base, shoulder, elbow, wrist_pitch)
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState \
  "{name:['base','shoulder','elbow','wrist_pitch'], position:[90.0, 100.0, 45.0, 80.0]}"

# Di chuyển toàn bộ 6 joint cùng lúc
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState \
  "{name:['base','shoulder','elbow','wrist_roll','wrist_pitch','pen'], position:[90.0, 90.0, 90.0, 90.0, 90.0, 90.0]}"
```

---

## 🏠 Các Pose đặc biệt (Preset Poses)

```bash
# === HOME — Tất cả về 90° (giữa) ===
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState \
  "{name:['base','shoulder','elbow','wrist_roll','wrist_pitch','pen'], position:[90.0, 90.0, 90.0, 90.0, 90.0, 90.0]}"

# === FOLDED — Gập cánh tay lại (compact) ===
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState \
  "{name:['base','shoulder','elbow','wrist_roll','wrist_pitch','pen'], position:[90.0, 180.0, 170.0, 90.0, 90.0, 90.0]}"

# === READY — Tư thế sẵn sàng thao tác ===
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState \
  "{name:['base','shoulder','elbow','wrist_roll','wrist_pitch','pen'], position:[90.0, 135.0, 45.0, 90.0, 90.0, 90.0]}"

# === DRAW — Tư thế sẵn sàng vẽ (bút hướng xuống) ===
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState \
  "{name:['base','shoulder','elbow','wrist_roll','wrist_pitch','pen'], position:[90.0, 120.0, 60.0, 90.0, 45.0, 30.0]}"

# === LOOK UP — Ngửa đầu cánh tay lên trên ===
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState \
  "{name:['base','shoulder','elbow','wrist_roll','wrist_pitch','pen'], position:[90.0, 90.0, 90.0, 90.0, 150.0, 90.0]}"

# === REACH FORWARD — Duỗi ra phía trước ===
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState \
  "{name:['base','shoulder','elbow','wrist_roll','wrist_pitch','pen'], position:[90.0, 110.0, 20.0, 90.0, 70.0, 90.0]}"

# === PICK UP — Tư thế nhặt vật (gripper mở, tay hạ thấp) ===
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState \
  "{name:['base','shoulder','elbow','wrist_roll','wrist_pitch','pen'], position:[90.0, 100.0, 30.0, 90.0, 50.0, 120.0]}"

# === PLACE — Đặt vật xuống (gripper đóng, di chuyển sang vị trí) ===
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState \
  "{name:['base','shoulder','elbow','wrist_roll','wrist_pitch','pen'], position:[45.0, 100.0, 30.0, 90.0, 50.0, 30.0]}"
```

---

## 🔧 Services (Bật/Tắt/Home - gọi qua lệnh service)

```bash
# Bật servo (enable outputs)
ros2 service call /pca9685_servo/enable std_srvs/srv/Trigger

# Tắt servo (disable outputs - tắt PWM, servo mềm)
ros2 service call /pca9685_servo/disable std_srvs/srv/Trigger

# Về Home (tất cả joint về neutral 90°)
ros2 service call /pca9685_servo/home std_srvs/srv/Trigger
```

---

## ⚡ Emergency Stop (kill all PWM)

```bash
# Tắt tất cả PCA9685 channels (direct, no mux)
python3 pca9685_kill_all.py

# Nếu dùng mux (cấu hình cũ)
python3 pca9685_kill_all.py --use-mux --mux-addr 0x70 --mux-channel 2
```

---

## 🎯 Điều khiển theo tọa độ XYZ (qua IK Node)

> Cần chạy IK node trước! Đơn vị: **cm**

**Terminal 1** — Khởi động robot:
```bash
ros2 launch wicom_roboarm wicom_roboarm.launch.py
```

**Terminal 2** — Chạy IK node:
```bash
ros2 run wicom_roboarm wicom_roboarm_drawing_ik_node.py
```

**Terminal 3** — Gửi tọa độ XYZ:

```bash
# Di chuyển đến tọa độ (x=20cm, y=0cm, z=15cm) — phía trước, giữa, cao 15cm
ros2 topic pub --once /target_xyz_cm geometry_msgs/msg/Point "{x: 20.0, y: 0.0, z: 15.0}"

# Cao hơn (z=25cm)
ros2 topic pub --once /target_xyz_cm geometry_msgs/msg/Point "{x: 20.0, y: 0.0, z: 25.0}"

# Thấp xuống gần mặt bàn (z=5cm)
ros2 topic pub --once /target_xyz_cm geometry_msgs/msg/Point "{x: 20.0, y: 0.0, z: 5.0}"

# Sang trái (y=10cm)
ros2 topic pub --once /target_xyz_cm geometry_msgs/msg/Point "{x: 20.0, y: 10.0, z: 15.0}"

# Sang phải (y=-10cm)
ros2 topic pub --once /target_xyz_cm geometry_msgs/msg/Point "{x: 20.0, y: -10.0, z: 15.0}"

# Gần hơn (x=12cm)
ros2 topic pub --once /target_xyz_cm geometry_msgs/msg/Point "{x: 12.0, y: 0.0, z: 15.0}"

# Xa hơn (x=28cm)
ros2 topic pub --once /target_xyz_cm geometry_msgs/msg/Point "{x: 28.0, y: 0.0, z: 15.0}"

# Vẽ hình vuông — 4 góc liên tiếp
ros2 topic pub --once /target_xyz_cm geometry_msgs/msg/Point "{x: 18.0, y: -5.0, z: 10.0}"
ros2 topic pub --once /target_xyz_cm geometry_msgs/msg/Point "{x: 18.0, y: 5.0, z: 10.0}"
ros2 topic pub --once /target_xyz_cm geometry_msgs/msg/Point "{x: 18.0, y: 5.0, z: 20.0}"
ros2 topic pub --once /target_xyz_cm geometry_msgs/msg/Point "{x: 18.0, y: -5.0, z: 20.0}"
```

---

## 🖥️ Điều khiển Joint trong Gazebo Simulation

> Controller: `arm_controller` (JointTrajectoryController)
> Joints: `Joint 1` → `Joint 6`
> Đơn vị: **radian** (khác servo thật dùng degree!)

```bash
# Di chuyển Joint 1 (base) sang 0.5 rad (~28.6°) trong 2 giây
ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['Joint 1','Joint 2','Joint 3','Joint 4','Joint 5','Joint 6'],
    points: [{positions: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 2, nanosec: 0}}]}"

# Di chuyển tất cả joints trong Gazebo
ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['Joint 1','Joint 2','Joint 3','Joint 4','Joint 5','Joint 6'],
    points: [{positions: [0.0, -0.5, 1.0, 0.0, 0.5, 0.0], time_from_start: {sec: 2, nanosec: 0}}]}"

# Multi-waypoint — đi qua 2 tư thế liên tiếp
ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['Joint 1','Joint 2','Joint 3','Joint 4','Joint 5','Joint 6'],
    points: [
      {positions: [0.0, -0.3, 0.6, 0.0, 0.3, 0.0], time_from_start: {sec: 2, nanosec: 0}},
      {positions: [0.5, -0.5, 1.0, 0.0, 0.5, 0.0], time_from_start: {sec: 4, nanosec: 0}}
    ]}"

# Quét 180° base (Joint 1) qua 3 điểm
ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['Joint 1','Joint 2','Joint 3','Joint 4','Joint 5','Joint 6'],
    points: [
      {positions: [-1.57, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 2, nanosec: 0}},
      {positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 4, nanosec: 0}},
      {positions: [1.57, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 6, nanosec: 0}}
    ]}"

# Home position trong Gazebo (tất cả về 0)
ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['Joint 1','Joint 2','Joint 3','Joint 4','Joint 5','Joint 6'],
    points: [{positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 3, nanosec: 0}}]}"
```

---

## 📡 Monitor & Debug

```bash
# Xem trạng thái joint hiện tại (servo thật)
ros2 topic echo /pca9685_servo/joint_states

# Xem tất cả topics đang hoạt động
ros2 topic list

# Xem joint states trong Gazebo
ros2 topic echo /joint_states

# Kiểm tra info topic
ros2 topic info /pca9685_servo/command
ros2 topic info /arm_controller/joint_trajectory

# Xem tần suất publish
ros2 topic hz /pca9685_servo/joint_states

# Xem tất cả services
ros2 service list

# Xem nodes đang chạy
ros2 node list

# Kiểm tra PCA9685 trên I2C bus (should show 0x40)
i2cdetect -y 1
```

## Auto square (failed)
python3 ~/ros2_ws/src/wicom_roboarm/src/wicom_roboarm_4dof_standalone.py --ros-args \
  -p use_mux:=false \
  -p ch_base:=0 \
  -p ch_shoulder:=1 -p shoulder_mirror_enabled:=false \
  -p ch_elbow:=2 \
  -p ch_wrist_pitch:=4 \
  -p sign_base:=1.0 \
  -p sign_shoulder:=1.0 \
  -p sign_elbow:=-1.0 \
  -p offset_base_deg:=0.0 \
  -p offset_shoulder_deg:=0.0 \
  -p offset_elbow_deg:=0.0 \
  -p offset_wrist_deg:=0.0 \
  -p fixed_channels:="[3,5]" -p fixed_degs:="[90.0,90.0]" \
  -p auto_draw:=true -p auto_loop:=true
