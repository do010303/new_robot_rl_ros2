#!/usr/bin/env bash
# ===========================================================================
# setup_workspace.sh — One-command setup for new_rl_ros2 on a fresh machine
#
# Tested on: Ubuntu 22.04 LTS + ROS 2 Humble + Gazebo Harmonic
#
# Usage:
#   chmod +x setup_workspace.sh
#   ./setup_workspace.sh
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$SCRIPT_DIR/ros2_ws"
VENV_DIR="$SCRIPT_DIR/.venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# ── Step 0: Check prerequisites ──────────────────────────────────────────
info "Checking prerequisites..."

if [ ! -f /opt/ros/humble/setup.bash ]; then
    fail "ROS 2 Humble not found at /opt/ros/humble/setup.bash.
    Install it first: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html"
fi
ok "ROS 2 Humble found"

# Source ROS 2
source /opt/ros/humble/setup.bash

# ── Step 1: Install system (apt) dependencies ────────────────────────────
info "Installing ROS 2 and Gazebo system packages (requires sudo)..."

# Gazebo Harmonic bridge for ROS 2 Humble
# See: https://gazebosim.org/docs/harmonic/ros_installation
SYSTEM_PKGS=(
    # Gazebo Harmonic ↔ ROS 2 bridge
    ros-humble-ros-gzharmonic

    # ros2_control framework
    ros-humble-ros2-control
    ros-humble-ros2-controllers
    ros-humble-controller-manager

    # Robot description & transforms
    ros-humble-robot-state-publisher
    ros-humble-joint-state-publisher
    ros-humble-joint-state-publisher-gui
    ros-humble-xacro

    # Vision
    ros-humble-cv-bridge

    # Visualization
    ros-humble-rviz2

    # vcstool for cloning external repos
    python3-vcstool

    # Python venv
    python3-venv
)

sudo apt-get update -qq
sudo apt-get install -y -qq "${SYSTEM_PKGS[@]}"
ok "System packages installed"

# ── Step 2: Clone external source dependencies via vcstool ────────────────
info "Cloning external source dependencies (gz_ros2_control)..."

DEPS_FILE="$WS_DIR/src/deps.repos"
if [ ! -f "$DEPS_FILE" ]; then
    fail "deps.repos not found at $DEPS_FILE"
fi

# Only clone if not already present
if [ -d "$WS_DIR/src/gz_ros2_control/.git" ]; then
    ok "gz_ros2_control already cloned — skipping"
else
    cd "$WS_DIR/src"
    vcs import --input deps.repos .
    ok "External dependencies cloned"
fi

# Clean up empty ros_gz directory if it exists (leftover from older setups)
if [ -d "$WS_DIR/src/ros_gz" ] && [ -z "$(ls -A "$WS_DIR/src/ros_gz")" ]; then
    rmdir "$WS_DIR/src/ros_gz"
    info "Removed empty ros_gz directory (using apt ros-humble-ros-gzharmonic instead)"
fi

# ── Step 3: Create Python virtual environment ────────────────────────────
info "Setting up Python virtual environment..."

if [ -d "$VENV_DIR" ]; then
    warn "Existing venv found at $VENV_DIR — reusing it"
else
    python3 -m venv --system-site-packages "$VENV_DIR"
    ok "Created venv at $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install -U pip -q

info "Installing Python dependencies from requirements.txt..."
pip install -r "$WS_DIR/src/visual_servoing/requirements.txt" -q
ok "Python dependencies installed"

# ── Step 4: Build the workspace ──────────────────────────────────────────
info "Building ROS 2 workspace with colcon..."

cd "$WS_DIR"
source /opt/ros/humble/setup.bash

colcon build --symlink-install 2>&1 | tail -5
ok "Workspace built successfully"

# ── Step 5: Verify ───────────────────────────────────────────────────────
info "Running quick verification..."

# shellcheck disable=SC1091
source "$WS_DIR/install/setup.bash"

# Check critical Python imports
python3 -c "
import torch, numpy, scipy, gymnasium, cv2, yaml, onnxruntime
print(f'  torch       {torch.__version__}')
print(f'  numpy       {numpy.__version__}')
print(f'  scipy       {scipy.__version__}')
print(f'  gymnasium   {gymnasium.__version__}')
print(f'  opencv      {cv2.__version__}')
print(f'  onnxruntime {onnxruntime.__version__}')
" && ok "All Python imports verified" || fail "Python import check failed"

# Check ROS 2 packages are findable
ros2 pkg list 2>/dev/null | grep -q "visual_servoing" && ok "visual_servoing package found" || fail "visual_servoing package not found"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ Setup complete! To use the workspace:${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "  # Terminal 1 — Launch Gazebo simulation:"
echo "  source /opt/ros/humble/setup.bash"
echo "  source $VENV_DIR/bin/activate"
echo "  source $WS_DIR/install/setup.bash"
echo "  ros2 launch visual_servoing visual_servoing_test.launch.py"
echo ""
echo "  # Terminal 2 — Run training:"
echo "  source /opt/ros/humble/setup.bash"
echo "  source $VENV_DIR/bin/activate"
echo "  source $WS_DIR/install/setup.bash"
echo "  cd $WS_DIR/src/visual_servoing/scripts"
echo "  python3 train_visual_servoing.py"
echo ""
