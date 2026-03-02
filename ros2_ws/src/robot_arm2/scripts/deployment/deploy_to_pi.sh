#!/usr/bin/bash
set -e

echo "========================================================================"
echo "🚀 Deploying RL Model to Raspberry Pi (ROS2)"
echo "========================================================================"

# Configuration
PI_USER="piros2"
DEFAULT_PI_IP="192.168.50.1"
PI_RL_DIR="/home/piros2/rl_deployment"

# Prompt for IP address
read -p "Enter Raspberry Pi IP address [${DEFAULT_PI_IP}]: " PI_HOST
PI_HOST=${PI_HOST:-$DEFAULT_PI_IP}

# Model configuration
# We expect the model to be in the checkpoints directory relative to the workspace
# Script is usually run from ros2_ws/src/robot_arm2/scripts/deployment or ros2_ws/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
CHECKPOINT_DIR="${WORKSPACE_DIR}/ros2_ws/src/robot_arm2/scripts/deployment/onnx_models"
ONNX_ACTOR="${CHECKPOINT_DIR}/actor_drawing_quant.onnx"
ONNX_NIK="${CHECKPOINT_DIR}/neural_ik_quant.onnx"

echo ""
echo "📋 Deployment Configuration:"
echo "   Target: ${PI_USER}@${PI_HOST}"
echo "   RL Directory: ${PI_RL_DIR}"
echo "   Model: ${ONNX_ACTOR}"
echo ""

# Check if ONNX model exists
if [ ! -f "${ONNX_ACTOR}" ]; then
    echo "❌ ERROR: ONNX model not found at ${ONNX_ACTOR}"
    echo ""
    echo "📝 Please export your trained model first:"
    echo "   cd ros2_ws/src/robot_arm2/scripts/deployment"
    echo "   python3 export_onnx_quantized.py"
    echo ""
    exit 1
fi

echo "✅ ONNX model found ($(du -h ${ONNX_ACTOR} | cut -f1))"

# ========================================================================
# Part 1: Deploy RL Model and Scripts
# ========================================================================
echo ""
echo "========================================================================"
echo "📦 Part 1: Deploying RL Model and Scripts"
echo "========================================================================"

# Test SSH connection
echo ""
echo "🔌 Testing SSH connection to ${PI_USER}@${PI_HOST}..."
if ! ssh -o ConnectTimeout=5 ${PI_USER}@${PI_HOST} "echo 'SSH OK'" > /dev/null 2>&1; then
    echo "❌ ERROR: Cannot connect to ${PI_USER}@${PI_HOST}"
    echo "   Please check:"
    echo "     1. Pi is powered on and connected to network"
    echo "     2. SSH is enabled on Pi"
    echo "     3. IP address is correct (edit PI_HOST in this script)"
    exit 1
fi
echo "✅ SSH connection successful"

# Create target directory on Pi
echo ""
echo "📁 Creating RL deployment directory on Pi..."
ssh ${PI_USER}@${PI_HOST} "mkdir -p ${PI_RL_DIR}"

# Copy ONNX models
echo ""
echo "📦 Copying ONNX models..."
scp "${ONNX_ACTOR}" ${PI_USER}@${PI_HOST}:${PI_RL_DIR}/
if [ -f "${ONNX_NIK}" ]; then
    scp "${ONNX_NIK}" ${PI_USER}@${PI_HOST}:${PI_RL_DIR}/
fi

# Copy deployment script
echo ""
echo "📦 Copying deployment script..."
scp ${SCRIPT_DIR}/deploy_drawing_on_pi.py ${PI_USER}@${PI_HOST}:${PI_RL_DIR}/

# Make script executable on Pi
echo ""
echo "🔧 Making script executable on Pi..."
ssh ${PI_USER}@${PI_HOST} "chmod +x ${PI_RL_DIR}/deploy_drawing_on_pi.py"

echo ""
echo "✅ Part 1 Complete: RL model and scripts deployed"

# ========================================================================
# Part 2: Install Dependencies on Pi
# ========================================================================
echo ""
echo "========================================================================"
echo "📦 Part 2: Installing Dependencies on Pi"
echo "========================================================================"

echo ""
echo "📦 Installing onnxruntime on Pi (if not present)..."
ssh ${PI_USER}@${PI_HOST} "pip3 install --quiet onnxruntime 2>/dev/null || echo 'onnxruntime may already be installed'"

echo ""
echo "📦 Installing servo libraries on Pi (if not present)..."
ssh ${PI_USER}@${PI_HOST} "pip3 install --quiet adafruit-circuitpython-servokit 2>/dev/null || echo 'servokit may already be installed'"

echo ""
echo "✅ Part 2 Complete: Dependencies installed"

# ========================================================================
# Deployment Complete
# ========================================================================
echo ""
echo "========================================================================"
echo "✅ Deployment Complete!"
echo "========================================================================"
echo ""
echo "📝 Files deployed to Pi (${PI_RL_DIR}):"
echo "   ✓ $(basename ${ONNX_ACTOR})"
echo "   ✓ $(basename ${ONNX_NIK})"
echo "   ✓ deploy_drawing_on_pi.py"
echo ""
echo "========================================================================"
echo "🚀 Next Steps on Raspberry Pi"
echo "========================================================================"
echo ""
echo "  Terminal 1 (wicom_roboarm servo interface):"
echo "    cd ~/wicom_roboarm"
echo "    ros2 launch wicom_roboarm wicom_roboarm.launch.py"
echo ""
echo "  Terminal 2 (RL drawing deployment):"
echo "    cd ~/rl_deployment"
echo "    python3 deploy_drawing_on_pi.py --model $(basename ${ONNX_ACTOR}) --ik $(basename ${ONNX_NIK})"
echo ""
echo "  Options:"
echo "    --no-ros                  # Run without ROS (simulation mode)"
echo "    --steps 100               # Max steps per episode"
echo ""
echo "========================================================================"
