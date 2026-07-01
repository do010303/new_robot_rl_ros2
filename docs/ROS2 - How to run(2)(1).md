# ROS2 Robot Runbook

Đây là tài liệu chạy chính cho pipeline hiện tại. Folder `docs/` chỉ giữ một file này để tránh trùng lặp.

## 0. Luồng nên dùng

```text
Option 8: Digital Twin Realtime Mirror (Laptop điều khiển Pi chạy song song với Gazebo)
- Giúp nhìn trực tiếp robot mô phỏng và robot thật để so sánh độ trễ/sai lệch
Option 7: Mô phỏng/Training thuần trên laptop (để sinh artifact, pkl)
JSON Replay: Chạy offline trên Pi (dùng làm fallback/debug sau)
```

## 1. Topic và đơn vị

| Hướng | Topic/service | Type | Đơn vị |
| --- | --- | --- | --- |
| Pi command nhanh | `/pca9685_servo/command` | `sensor_msgs/msg/JointState` | degree |
| Pi trajectory timed | `/pca9685_servo/trajectory` | `trajectory_msgs/msg/JointTrajectory` | degree |
| Pi feedback | `/pca9685_servo/joint_states` | `sensor_msgs/msg/JointState` | radian |
| Pi home | `/pca9685_servo/home` | `std_srvs/srv/Trigger` | - |
| Pi enable/disable | `/pca9685_servo/enable`, `/pca9685_servo/disable` | `std_srvs/srv/Trigger` | - |

Joint Pi:

```text
base, shoulder, elbow, wrist_roll, wrist_pitch, pen
```

Pin map đang dùng theo log launch:

| Joint | PCA9685 |
| --- | --- |
| base | CH0 |
| shoulder | CH1 |
| elbow | CH4 |
| wrist_roll | CH7 |
| wrist_pitch | CH9 |
| pen | CH15 |

Nếu terminal launch in pin map khác, ưu tiên pin map trong terminal.

## 2. Cập nhật code lên Pi

Laptop:

```bash
cd ~/new_rl_ros2
./scripts/deploy_pi_wicom_roboarm.sh
```

Pi:

```bash
ssh piros2@192.168.50.1
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select wicom_roboarm
source install/setup.bash
```

## 3. Chạy node tay robot trên Pi

Pi terminal 1:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch wicom_roboarm wicom_roboarm.launch.py
```

Kiểm tra:

```bash
ros2 topic echo /pca9685_servo/joint_states
ros2 topic hz /pca9685_servo/joint_states
ros2 service call /pca9685_servo/home std_srvs/srv/Trigger
```

## 4. Điều khiển Pi nhanh

Enable/home/disable:

```bash
ros2 service call /pca9685_servo/enable std_srvs/srv/Trigger
ros2 service call /pca9685_servo/home std_srvs/srv/Trigger
ros2 service call /pca9685_servo/disable std_srvs/srv/Trigger
```

Set một joint:

```bash
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState "{name:['base'], position:[135.0]}"
```

Set tất cả joint về 90 độ:

```bash
ros2 topic pub -r 10 -t 2 /pca9685_servo/command sensor_msgs/msg/JointState \
  "{name:['base','shoulder','elbow','wrist_roll','wrist_pitch','pen'], position:[90.0,90.0,90.0,90.0,90.0,90.0]}"
```

Timed move 1 giây:

```bash
ros2 topic pub --once -w 1 /pca9685_servo/trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: ['base'], points: [{positions: [135.0], time_from_start: {sec: 1, nanosec: 0}}]}"
```

## 5. Option 7 - train hiện tại

Laptop terminal 1, chạy Gazebo:

```bash
cd ~/new_rl_ros2/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
unset ROS_DISCOVERY_SERVER FASTRTPS_DEFAULT_PROFILES_FILE FASTDDS_DEFAULT_PROFILES_FILE
ros2 launch visual_servoing visual_servoing_test.launch.py
```

Laptop terminal 2, chạy trainer:

```bash
cd ~/new_rl_ros2/ros2_ws/src/visual_servoing/scripts
source /opt/ros/humble/setup.bash
source ~/new_rl_ros2/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=~/new_rl_ros2/ros2_ws/src/visual_servoing/config/fastdds_twin.xml
python3 train_visual_servoing.py
```

Menu:

```text
7  -> PID Tuning
a/b -> reaching hoặc drawing
sim -> train trong sim
episodes -> số tập muốn train, ví dụ 30
```

Ghi chú:

```text
Mỗi episode chỉ chạy 1 lần.
Không tự replay cuối episode.
Reset/home vẫn giúp episode sau bắt đầu sạch.
Artifact tốt nhất được lưu trong training_results/pkl.
```

Tùy chọn legacy nếu cố tình muốn replay phần cứng tự động:

```bash
export PID_SHADOW_AUTO_REPLAY=1
export PID_SHADOW_REPLAY_BEST=1
export PID_AUTO_HOME_ON_EXIT=0
```

## 6. Export replay JSON từ artifact

Tìm artifact mới nhất:

```bash
cd ~/new_rl_ros2/ros2_ws/src/visual_servoing/scripts
ls -lt training_results/pkl/pid_best_artifact_*.pkl | head
```

Export JSON, ví dụ drawing 3 Hz:

```bash
python3 digital_twin/export_replay_plan.py \
  --artifact training_results/pkl/pid_best_artifact_sac_pid_tuning_drawing_sim_YYYYMMDD_HHMMSS.pkl \
  --mode drawing \
  --rate 3.0 \
  --tolerance-deg 2.0 \
  --output /tmp/pi_replay_plan_drawing.json
```

Với artifact drawing, exporter sẽ ưu tiên `target_metadata.shape_joint_waypoints` để dựng lại hình vuông nominal đúng một lần. Không dùng `replay_trajectory_rad` cũ nếu nó chứa tail lặp, và không dùng `commanded_trajectory_rad` nếu PID correction làm méo hình. JSON mới sẽ có:

```text
"trajectory_source": "target_metadata.shape_joint_waypoints"
"keyframe_count": số điểm chính để Pi nội suy S-curve
"keyframes_deg": các keyframe Pi-degree
```

Gợi ý tốc độ:

```text
1 Hz -> test pin/nguồn yếu
2 Hz -> an toàn để test lần đầu
3 Hz -> mặc định nên dùng
5 Hz -> chỉ thử khi log ổn định, ít LAG/LOST/I2C_ERROR
```

Inspect JSON:

```bash
python3 -m json.tool /tmp/pi_replay_plan_drawing.json | less
```

Copy sang Pi:

```bash
scp /tmp/pi_replay_plan_drawing.json piros2@192.168.50.1:~/ros2_ws/pi_replay_plan_drawing.json
```

Không cần train lại nếu đã có artifact `.pkl`. Chỉ cần export lại JSON từ artifact đó.

## 7. Chạy JSON replay offline trên Pi

Pi terminal 2, dry-run trước:

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run wicom_roboarm pi_replay_executor_node.py \
  --plan ~/ros2_ws/pi_replay_plan_drawing.json \
  --episodes 1 \
  --publish-mode keyframe-scurve \
  --stream-hz 10.0 \
  --move-time-sec 1.2 \
  --deadband-deg 0.5 \
  --tolerance-deg 2.0 \
  --dry-run \
  --print-segments
```

Chạy thật:

```bash
ros2 run wicom_roboarm pi_replay_executor_node.py \
  --plan ~/ros2_ws/pi_replay_plan_drawing.json \
  --episodes 1 \
  --publish-mode keyframe-scurve \
  --stream-hz 10.0 \
  --move-time-sec 1.2 \
  --deadband-deg 0.5 \
  --tolerance-deg 2.0 \
  --print-segments
```

Chỉnh số tập và độ mượt ở đây:

```text
--episodes 5
--move-time-sec 1.5   chậm hơn, mượt hơn
--move-time-sec 1.2   mặc định test đầu tiên
--move-time-sec 1.0   nhanh hơn sau khi ổn
--stream-hz 10.0      giống project S-curve mẫu, ít spam I2C
--stream-hz 20.0      thử sau nếu 10Hz ổn
--deadband-deg 0.5    giảm lệnh nhỏ gây jitter
--deadband-deg 0.3    mịn hơn nhưng gửi nhiều lệnh hơn
```

Mặc định executor dùng `--publish-mode keyframe-scurve`, tức là lấy `keyframes_deg` trong JSON, sinh S-curve giữa các điểm chính, rồi publish `/pca9685_servo/command`. Đây là flow gần nhất với project `com_arm_uav-main`: quỹ đạo được làm mượt trước, Pi driver chỉ bám theo command.

So sánh với kiểu cũ nếu cần debug:

```bash
--publish-mode stream
--publish-mode trajectory
--publish-mode scurve
```

`trajectory` là legacy, dễ reset motion theo segment. `stream` và `scurve` vẫn dùng danh sách segment, nên không phải đường khuyến nghị cho servo thật nếu đã có `keyframes_deg`.

Mặc định node home trước mỗi episode. Bỏ home nếu cần:

```bash
--no-home
```

Theo dõi:

```bash
ros2 topic echo /pca9685_servo/replay_status
tail -f ~/ros2_ws/replay_logs/*.jsonl
```

Status:

```text
OK         segment đã chạy trong tolerance
LAG        servo/Pi/I2C không theo kịp replay_rate_hz
LOST       mất feedback joint_states hoặc feedback quá cũ
I2C_ERROR  lỗi PCA9685/I2C
```

Lấy log vừa chạy trên Pi:

```bash
cd ~/ros2_ws/replay_logs
latest=$(ls -t pi_replay_executor_log_*.jsonl | head -1)
echo "$latest"
tail -n 5 "$latest"
```

Nếu tên file log nhìn "cũ" như `20260609` hoặc `20260612`, nguyên nhân thường là clock trên Raspberry Pi chưa sync đúng ngày. Tên file chỉ lấy từ giờ hệ thống của Pi, không phải ngày thật của laptop.

Copy log về laptop:

```bash
scp piros2@192.168.50.1:~/ros2_ws/replay_logs/pi_replay_executor_log_YYYYMMDD_HHMMSS.jsonl /tmp/
```

Log replay hiện bắt tốt các lỗi cấp segment như `LAG`, `LOST`, `I2C_ERROR`, `cmd_deg`, `actual_deg`, `cmd_delta_deg`, `cmd_speed_deg_s`. Nó chưa đo được rung/giật rất nhanh bên trong một segment; muốn đo kiểu đó cần log joint_states/PWM ở tần số cao hơn.

Ghi chú làm mượt servo:

```text
wicom_roboarm hiện dùng trajectory_profile=min_jerk.
Profile này là S-curve bậc 5: 10t^3 - 15t^4 + 6t^5.
trajectory_update_rate_hz=50 giúp mỗi segment có nhiều điểm nội suy hơn.
Nếu cần so sánh, đổi trajectory_profile thành linear trong wicom_roboarm/config/servos.yaml.
```

Ctrl-C launch:

```text
wicom_roboarm_unified_node.py gọi set_all_off() khi shutdown.
shutdown_behavior=off để Ctrl-C tắt PWM, không tự chạy về neutral trước khi tắt.
Nếu I2C bị treo hoặc process bị kill cứng, vẫn nên cắt nguồn servo/PCA9685.
```

## 8. Option 8 - Digital Twin Realtime Mirror

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

Select training mode to run WITH mirror:
  a. 📍 PID Reaching
  b. 🖋️  PID Drawing      ← khuyến nghị test đầu tiên
  c. 🎮 Manual Control
Select (a/b/c, default=b): b

Require live board detection? (y/N):   ← Nhập 'y' nếu muốn nhận diện board / 'n' để dùng fallback
Drawing segment steps (default 20, lower = faster):   ← Enter để dùng 20 steps, nhập số nhỏ hơn (vd: 10, 5) để robot di chuyển nhanh hơn
```

Sau khi chọn mode, script tự động:

```text
1. Start sim_to_pi_mirror.py ở background (subscribe /joint_states, publish /pca9685_servo/command)
2. Chạy training/manual ở foreground (di chuyển robot trong Gazebo)
3. Mỗi khi Gazebo joint thay đổi → mirror forward sang Pi → robot thật bám theo
4. Khi training xong hoặc Ctrl+C → tự kill mirror
```

### Kiểm tra trước khi chạy (optional)

Nếu không chắc Pi có online không, chạy riêng:

```bash
python3 digital_twin/verify_connection.py
```

### Tham số mirror

```text
--rate-hz 10    mặc định, an toàn cho I2C trên Pi
--rate-hz 5     giảm nếu bị nghẽn I2C
--rate-hz 20    thử nếu 10Hz ổn và muốn mượt hơn

--deadband-deg 0.5   mặc định, lọc thay đổi nhỏ
--deadband-deg 1.0   giảm spam nếu cần
--deadband-deg 0.2   mịn hơn, nhiều lệnh hơn
```

## 9. Debug nguồn pin và I2C

Nếu chạy bằng nguồn cắm ổn nhưng pin cell lỗi, kiểm tra các dấu hiệu này:

```text
I2C error [Errno 121] Remote I/O error
LAG liên tục với max_err lớn
LOST và feedback_age tăng đều
Robot vẫn giữ/chạy sau khi kill node vì PCA9685 còn giữ PWM cuối
```

Lệnh kiểm tra:

```bash
vcgencmd get_throttled
dmesg -T | grep -i -E "voltage|under|thrott|i2c|error"
i2cdetect -y 1
ros2 topic hz /pca9685_servo/joint_states
```

Khuyến nghị phần cứng:

```text
Pin cell -> buck/BEC 5-6V dòng cao -> V+ servo trên PCA9685
Pi dùng nguồn 5V riêng ổn định
GND Pi, GND PCA9685, GND nguồn servo nối chung
SDA/SCL ngắn, chắc, tránh đi song song dây nguồn servo
Thêm tụ 1000uF-2200uF gần PCA9685 V+ servo
```

Khi cần dừng thật:

```bash
ros2 service call /pca9685_servo/disable std_srvs/srv/Trigger
```

Nếu service không phản hồi, cắt nguồn servo/PCA9685.

## 10. Camera và UAV

Camera Pi:

```bash
ros2 run web_video_server web_video_server
```

```bash
ros2 run usb_cam usb_cam_node_exe --ros-args \
  -p video_device:="/dev/video0" \
  -p image_width:=640 \
  -p image_height:=480 \
  -p pixel_format:="yuyv"
```

Mở trên laptop:

```text
http://192.168.50.1:8080/
```

PX4:

```bash
MicroXRCEAgent serial --dev /dev/ttyS0 -b 921600
sudo systemctl restart mavlink-router
```

Offboard teleop:

```bash
cd ros2_px4_teleop_example
source install/setup.bash
ros2 run teleop teleop
```
