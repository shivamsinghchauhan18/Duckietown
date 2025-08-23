#!/bin/bash

# Manual Deployment Script (Fallback if Docker fails)
# Usage: ./deploy_manual.sh ROBOT_NAME

set -e

ROBOT_NAME=${1:-"duckiebot"}

echo "🔧 MANUAL DEPLOYMENT TO DUCKIEBOT"
echo "================================="
echo "Robot Name: $ROBOT_NAME"
echo ""

# Step 1: Verify files
echo "📋 Step 1: Verifying files..."
if [ ! -f "champion_model.pth" ]; then
    echo "❌ champion_model.pth not found!"
    exit 1
fi

if [ ! -f "rl_inference_node.py" ]; then
    echo "❌ rl_inference_node.py not found!"
    exit 1
fi

echo "✅ Files verified"

# Step 2: Check robot connectivity
echo ""
echo "🔍 Step 2: Checking robot connectivity..."
if ! ping -c 1 $ROBOT_NAME.local &> /dev/null; then
    echo "❌ Cannot reach $ROBOT_NAME.local"
    exit 1
fi

echo "✅ Robot is reachable"

# Step 3: Create directories on robot
echo ""
echo "📁 Step 3: Creating directories on robot..."
ssh duckie@$ROBOT_NAME.local 'mkdir -p /data/models /data/config /code'

# Step 4: Copy files to robot
echo ""
echo "📤 Step 4: Copying files to robot..."
scp champion_model.pth duckie@$ROBOT_NAME.local:/data/models/
scp enhanced_config.yml duckie@$ROBOT_NAME.local:/data/config/
scp rl_inference_node.py duckie@$ROBOT_NAME.local:/code/

echo "✅ Files copied"

# Step 5: Install dependencies on robot
echo ""
echo "📦 Step 5: Installing dependencies on robot..."
ssh duckie@$ROBOT_NAME.local << 'EOF'
    echo "Installing PyTorch CPU version..."
    pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    
    echo "Installing other dependencies..."
    pip3 install numpy==1.21.6 opencv-python-headless==4.5.3.56 pyyaml rospkg
    
    echo "Dependencies installed successfully"
EOF

echo "✅ Dependencies installed"

# Step 6: Create startup script on robot
echo ""
echo "📝 Step 6: Creating startup script on robot..."
ssh duckie@$ROBOT_NAME.local << 'EOF'
cat > /code/start_rl.sh << 'SCRIPT_EOF'
#!/bin/bash

# Kill any existing RL processes
pkill -f rl_inference_node.py || true

# Setup ROS environment
source /opt/ros/noetic/setup.bash

# Set robot name
export ROBOT_NAME=$(hostname)

echo "Starting RL Inference for robot: $ROBOT_NAME"

# Launch the node
cd /code
python3 rl_inference_node.py _robot_name:=$ROBOT_NAME
SCRIPT_EOF

chmod +x /code/start_rl.sh
EOF

echo "✅ Startup script created"

# Step 7: Test the deployment
echo ""
echo "🧪 Step 7: Testing deployment..."
ssh duckie@$ROBOT_NAME.local << 'EOF'
    echo "Testing model loading..."
    cd /code
    python3 -c "
import torch
import sys
try:
    model = torch.load('/data/models/champion_model.pth', map_location='cpu', weights_only=False)
    print('✅ Model loads successfully')
    print(f'Model keys: {list(model.keys()) if isinstance(model, dict) else \"Complete model\"}')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
    sys.exit(1)
"
EOF

echo "✅ Deployment test passed"

# Step 8: Instructions
echo ""
echo "🎉 MANUAL DEPLOYMENT COMPLETED!"
echo ""
echo "To start the RL system:"
echo "  ssh duckie@$ROBOT_NAME.local"
echo "  /code/start_rl.sh"
echo ""
echo "To monitor:"
echo "  rostopic echo /$ROBOT_NAME/rl_inference/status"
echo ""
echo "To stop:"
echo "  pkill -f rl_inference_node.py"
echo ""
echo "⚠️  SAFETY REMINDER:"
echo "  - Test in a safe, open area"
echo "  - Have emergency stop ready"
echo "  - Monitor robot behavior closely"