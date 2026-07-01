#!/usr/bin/env bash
set -euo pipefail

# Deploy the hardware package to the Raspberry Pi ROS 2 workspace.
# Uses rsync to avoid the common nested-folder mistake with scp.

PI_HOST="${PI_HOST:-192.168.50.1}"
PI_USER="${PI_USER:-piros2}"
PI_WS="${PI_WS:-/home/${PI_USER}/ros2_ws}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${REPO_ROOT}/wicom_roboarm/"
DEST_DIR="${PI_USER}@${PI_HOST}:${PI_WS}/src/wicom_roboarm/"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Missing source dir: ${SRC_DIR}" >&2
  exit 1
fi

echo "Deploying ${SRC_DIR} -> ${DEST_DIR}"
rsync -av --delete "${SRC_DIR}" "${DEST_DIR}"

cat <<EOF

Next on the Pi:
  cd ~/ros2_ws
  source /opt/ros/humble/setup.bash
  colcon build --packages-select wicom_roboarm
  source install/setup.bash
  ros2 launch wicom_roboarm wicom_roboarm.launch.py

Pi-local replay plan runner:
  # Copy a quantized pi_replay_plan_v1 JSON to the Pi first, e.g.
  #   scp /tmp/pi_replay_plan_drawing.json ${PI_USER}@${PI_HOST}:${PI_WS}/pi_replay_plan_drawing.json

  # Option A: run beside the existing robot launch
  ros2 run wicom_roboarm pi_replay_executor_node.py \\
    --plan ${PI_WS}/pi_replay_plan_drawing.json \\
    --episodes 1 \\
    --publish-mode keyframe-scurve \\
    --stream-hz 10.0 \\
    --move-time-sec 1.2 \\
    --deadband-deg 0.5 \\
    --print-segments \\
    --log-dir ${PI_WS}/replay_logs

  # Option B: launch robot node + replay executor together
  ros2 launch wicom_roboarm wicom_roboarm_replay.launch.py \\
    replay_plan:=${PI_WS}/pi_replay_plan_drawing.json \\
    episodes:=1 \\
    replay_rate_hz:=3.0 \\
    tolerance_deg:=2.0 \\
    publish_mode:=keyframe-scurve \\
    stream_hz:=10.0 \\
    move_time_sec:=1.2 \\
    deadband_deg:=0.5
EOF
