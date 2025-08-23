#!/bin/bash

# Monitor pinkduckie RL system
ROBOT_NAME="pinkduckie"

echo "📊 MONITORING PINKDUCKIE RL SYSTEM"
echo "================================="

# Check if robot is reachable
if ! ping -c 1 $ROBOT_NAME.local &> /dev/null; then
    echo "❌ Cannot reach $ROBOT_NAME.local"
    exit 1
fi

echo "✅ Robot is reachable"
echo ""

# Check if RL process is running
echo "🔍 Checking RL process..."
ssh duckie@$ROBOT_NAME.local << 'EOF'
if pgrep -f rl_inference_node.py > /dev/null; then
    echo "✅ RL inference process is running"
    echo "   PID: $(pgrep -f rl_inference_node.py)"
else
    echo "❌ RL inference process is NOT running"
    echo ""
    echo "To start it:"
    echo "  ssh duckie@pinkduckie.local"
    echo "  /code/start_rl_pinkduckie.sh"
    exit 1
fi
EOF

echo ""
echo "🤖 Checking ROS topics..."
ssh duckie@$ROBOT_NAME.local << 'EOF'
source /opt/ros/noetic/setup.bash

echo "Available topics:"
timeout 5 rostopic list | grep -E "(camera|rl_inference|car_cmd)" || echo "No relevant topics found"

echo ""
echo "Checking RL status topic:"
if timeout 3 rostopic echo /pinkduckie/rl_inference/status -n 1 &> /dev/null; then
    echo "✅ RL status topic is active"
    echo ""
    echo "Latest status:"
    timeout 3 rostopic echo /pinkduckie/rl_inference/status -n 1 2>/dev/null || echo "No status received"
else
    echo "❌ RL status topic is not active"
fi

echo ""
echo "Checking camera feed:"
if timeout 3 rostopic echo /pinkduckie/camera_node/image/compressed -n 1 &> /dev/null; then
    echo "✅ Camera feed is active"
else
    echo "❌ Camera feed is not active"
fi
EOF

echo ""
echo "💻 System health:"
ssh duckie@$ROBOT_NAME.local << 'EOF'
echo "CPU usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $5}') used"

if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
    temp=$(cat /sys/class/thermal/thermal_zone0/temp)
    temp_c=$((temp/1000))
    echo "Temperature: ${temp_c}°C"
fi
EOF

echo ""
echo "🎯 MONITORING COMPLETE"
echo ""
echo "Useful commands:"
echo "  Watch status: ssh duckie@$ROBOT_NAME.local 'rostopic echo /pinkduckie/rl_inference/status'"
echo "  Watch commands: ssh duckie@$ROBOT_NAME.local 'rostopic echo /pinkduckie/car_cmd_switch_node/cmd'"
echo "  Stop RL: ssh duckie@$ROBOT_NAME.local 'pkill -f rl_inference_node.py'"