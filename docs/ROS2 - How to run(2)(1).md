# ROS2 Robot Runbook

Đây là tài liệu chạy chính cho pipeline hiện tại. Folder `docs/` chỉ giữ một file này để tránh trùng lặp.

## 0. Luồng nên dùng

```text
Option 7 train trên laptop
-> export replay plan JSON
-> copy JSON sang Pi
-> Pi chạy replay offline bằng wicom_roboarm
```

Vẫn giữ:

```text
Option 8: deploy cũ, laptop điều khiển trực tiếp Pi
```

Khuyến nghị hiện tại: dùng JSON replay trên Pi. Option 8 chỉ dùng để so sánh/debug.

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
  --replay-rate-hz 3.0 \
  --tolerance-deg 2.0 \
  --dry-run \
  --print-segments
```

Chạy thật:

```bash
ros2 run wicom_roboarm pi_replay_executor_node.py \
  --plan ~/ros2_ws/pi_replay_plan_drawing.json \
  --episodes 3 \
  --replay-rate-hz 3.0 \
  --tolerance-deg 2.0 \
  --print-segments
```

Chỉnh số tập và tốc độ ở đây:

```text
--episodes 5
--replay-rate-hz 1.0
--replay-rate-hz 2.0
--replay-rate-hz 3.0
--replay-rate-hz 5.0
```

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

## 8. Option 8 - deploy cũ từ laptop

Option 8 vẫn giữ để so sánh/debug, nhưng không phải đường khuyến nghị.

Laptop:

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
8  -> Deploy to Pi
a/b -> reaching hoặc drawing
artifact -> Enter để lấy artifact mới nhất, hoặc chọn file
gains -> Enter để lấy gains mới nhất
episodes -> số lần chạy
replay rate -> ví dụ 3.0 hoặc 5.0
```

Option 8 sẽ:

```text
home trước mỗi run
gửi trajectory từ laptop sang Pi
in commanded/actual/status từng segment
lưu log trong training_results/logs/deploy_replay_log_*.txt
lưu pkl trong training_results/pkl/deploy_results_*.pkl
lưu plot trong training_results/png/deploy_comparison_*.png
```

Nếu network/DDS không ổn, chuyển sang JSON replay trên Pi ở mục 7.

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
