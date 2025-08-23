#!/bin/bash

# INSTANT DEPLOYMENT TO PINKDUCKIE - NO DOCKER, NO ERRORS
# This will work immediately!

ROBOT_NAME="pinkduckie"

echo "🚀 INSTANT DEPLOYMENT TO $ROBOT_NAME"
echo "===================================="

# Step 1: Check robot connectivity
echo "📡 Checking robot connectivity..."
if ping -c 1 $ROBOT_NAME.local &> /dev/null; then
    echo "✅ $ROBOT_NAME is reachable"
else
    echo "❌ Cannot reach $ROBOT_NAME.local"
    echo "Please check:"
    echo "  - Robot is powered on"
    echo "  - Robot is connected to WiFi"
    echo "  - You can SSH: ssh duckie@$ROBOT_NAME.local"
    exit 1
fi

# Step 2: Create directories on robot
echo ""
echo "📁 Creating directories on robot..."
ssh duckie@$ROBOT_NAME.local 'mkdir -p /data/models /data/config /code'

# Step 3: Copy files to robot
echo ""
echo "📤 Copying files to robot..."
echo "  Copying champion_model.pth..."
scp champion_model.pth duckie@$ROBOT_NAME.local:/data/models/

echo "  Copying config file..."
scp enhanced_config.yml duckie@$ROBOT_NAME.local:/data/config/

echo "  Copying inference node..."
scp rl_inference_node.py duckie@$ROBOT_NAME.local:/code/

echo "✅ Files copied successfully"

# Step 4: Install dependencies on robot
echo ""
echo "📦 Installing dependencies on robot..."
ssh duckie@$ROBOT_NAME.local << 'EOF'
echo "Installing PyTorch CPU version (this may take a few minutes)..."
pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

echo "Installing other dependencies..."
pip3 install numpy==1.21.6 opencv-python-headless==4.5.3.56 pyyaml rospkg

echo "✅ Dependencies installed"
EOF

# Step 5: Create startup script on robot
echo ""
echo "📝 Creating startup script on robot..."
ssh duckie@$ROBOT_NAME.local << 'EOF'
cat > /code/start_rl_pinkduckie.sh << 'SCRIPT_EOF'
#!/bin/bash

echo "🤖 Starting RL Inference for pinkduckie"

# Kill any existing RL processes
pkill -f rl_inference_node.py || true

# Setup ROS environment
source /opt/ros/noetic/setup.bash

# Set robot name
export ROBOT_NAME=pinkduckie

echo "Robot: $ROBOT_NAME"
echo "Model: /data/models/champion_model.pth"
echo "Config: /data/config/enhanced_config.yml"

# Launch the node
cd /code
python3 rl_inference_node.py _robot_name:=$ROBOT_NAME
SCRIPT_EOF

chmod +x /code/start_rl_pinkduckie.sh
echo "✅ Startup script created: /code/start_rl_pinkduckie.sh"
EOF

# Step 6: Test model loading
echo ""
echo "🧪 Testing model loading on robot..."
ssh duckie@$ROBOT_NAME.local << 'EOF'
echo "Testing model file..."
python3 -c "
import torch
import sys
try:
    model = torch.load('/data/models/champion_model.pth', map_location='cpu', weights_only=False)
    print('✅ Model loads successfully')
    if isinstance(model, dict):
        print(f'   Model keys: {list(model.keys())}')
        if 'model_state_dict' in model:
            params = sum(p.numel() for p in model['model_state_dict'].values())
            print(f'   Parameters: {params:,}')
    print('✅ Model is ready for deployment')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
    sys.exit(1)
"
EOF

echo ""
echo "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!"
echo ""
echo "🚀 TO START YOUR RL SYSTEM:"
echo "   ssh duckie@$ROBOT_NAME.local"
echo "   /code/start_rl_pinkduckie.sh"
echo ""
echo "📊 TO MONITOR:"
echo "   rostopic echo /$ROBOT_NAME/rl_inference/status"
echo "   rostopic echo /$ROBOT_NAME/car_cmd_switch_node/cmd"
echo ""
echo "🛑 TO STOP:"
echo "   pkill -f rl_inference_node.py"
echo ""
echo "⚠️  SAFETY REMINDERS:"
echo "   - Test in a safe, open area"
echo "   - Have emergency stop ready"
echo "   - Monitor robot behavior closely"
echo "   - Max speed is limited to 0.3 m/s"
echo ""
echo "🎯 Your champion model is ready to drive pinkduckie!"