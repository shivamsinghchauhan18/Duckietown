#!/bin/bash

# SUPER SIMPLE - RUN DIRECTLY ON PINKDUCKIE
# Since you already have the repo on the robot, just run this!

echo "🤖 STARTING RL ON PINKDUCKIE (REPO ALREADY ON ROBOT)"
echo "=================================================="

# Set robot name
export ROBOT_NAME=pinkduckie

echo "Robot: $ROBOT_NAME"
echo "Using existing repo on robot"

# Kill any existing RL processes
echo "🛑 Stopping any existing RL processes..."
pkill -f rl_inference_node.py || true
pkill -f champion_model || true

# Setup ROS environment
echo "🔧 Setting up ROS environment..."
source /opt/ros/noetic/setup.bash

# Check if model file exists
if [ -f "champion_model.pth" ]; then
    echo "✅ Found champion_model.pth"
    MODEL_PATH="$(pwd)/champion_model.pth"
elif [ -f "Final-Deployment/champion_model.pth" ]; then
    echo "✅ Found Final-Deployment/champion_model.pth"
    MODEL_PATH="$(pwd)/Final-Deployment/champion_model.pth"
else
    echo "❌ Cannot find champion_model.pth"
    echo "Please run this script from the repo directory"
    exit 1
fi

# Check if config exists
if [ -f "enhanced_config.yml" ]; then
    CONFIG_PATH="$(pwd)/enhanced_config.yml"
elif [ -f "Final-Deployment/enhanced_config.yml" ]; then
    CONFIG_PATH="$(pwd)/Final-Deployment/enhanced_config.yml"
else
    echo "⚠️  No config file found, using defaults"
    CONFIG_PATH=""
fi

echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_PATH"

# Check if inference node exists
if [ -f "Final-Deployment/rl_inference_node.py" ]; then
    INFERENCE_NODE="$(pwd)/Final-Deployment/rl_inference_node.py"
elif [ -f "duckiebot_deployment_dts/src/duckiebot_rl_inference_node.py" ]; then
    INFERENCE_NODE="$(pwd)/duckiebot_deployment_dts/src/duckiebot_rl_inference_node.py"
else
    echo "❌ Cannot find RL inference node"
    exit 1
fi

echo "Inference node: $INFERENCE_NODE"

# Install dependencies if needed
echo "📦 Checking dependencies..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch..."
    pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
fi

if ! python3 -c "import cv2" 2>/dev/null; then
    echo "Installing OpenCV..."
    pip3 install opencv-python-headless==4.5.3.56
fi

if ! python3 -c "import yaml" 2>/dev/null; then
    echo "Installing PyYAML..."
    pip3 install pyyaml
fi

echo "✅ Dependencies ready"

# Test model loading
echo "🧪 Testing model loading..."
python3 -c "
import torch
import sys
try:
    model = torch.load('$MODEL_PATH', map_location='cpu', weights_only=False)
    print('✅ Model loads successfully')
    if isinstance(model, dict) and 'model_state_dict' in model:
        params = sum(p.numel() for p in model['model_state_dict'].values())
        print(f'   Parameters: {params:,}')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
    sys.exit(1)
"

echo ""
echo "🚀 STARTING RL INFERENCE..."
echo "Press Ctrl+C to stop"
echo ""

# Start the RL inference node
python3 $INFERENCE_NODE _robot_name:=$ROBOT_NAME