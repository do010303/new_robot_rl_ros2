```bash
export ROS_SIMULATION=true
```

### Chạy Vision Node (Giả lập): Nếu không có Camera, node này sẽ đợi. Nếu có Webcam laptop, nó sẽ dùng Webcam
```bash
ros2 run vision_node_ros2.py --ros-args -p show_gui:=true
```

### Chạy Executor (Vẽ): Mở terminal mới:
```bash
export ROS_SIMULATION=true
source install/setup.bash
```

### Lệnh vẽ hình vuông mặc định
```bash
ros2 run drawing_executor_ros2.py --mode square
```
