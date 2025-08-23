#!/bin/bash

# Deployment Verification Script
# Usage: ./verify_deployment.sh ROBOT_NAME

ROBOT_NAME=${1:-"duckiebot"}

echo "🔍 DEPLOYMENT VERIFICATION"
echo "========================="
echo "Robot Name: $ROBOT_NAME"
echo ""

# Check robot connectivity
echo "📡 Checking robot connectivity..."
if ping -c 1 $ROBOT_NAME.local &> /dev/null; then
    echo "✅ Robot is reachable"
else
    echo "❌ Robot is not reachable"
    exit 1
fi

# Check container status
echo ""
echo "🐳 Checking Docker container..."
ssh duckie@$ROBOT_NAME.local << 'EOF'
if docker ps | grep -q rl-inference; then
    echo "✅ RL container is running"
    
    echo ""
    echo "📊 Container details:"
    docker ps --filter name=rl-inference --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo ""
    echo "📝 Recent logs:"
    docker logs rl-inference --tail 5
    
else
    echo "❌ RL container is not running"
    echo ""
    echo "Available containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}"
    
    echo ""
    echo "Checking if container exists but stopped:"
    if docker ps -a | grep -q rl-inference; then
        echo "⚠️  Container exists but is stopped"
        echo "Logs from stopped container:"
        docker logs rl-inference --tail 10
    else
        echo "❌ No rl-inference container found"
    fi
fi
EOF

# Check ROS topics
echo ""
echo "🤖 Checking ROS topics..."
ssh duckie@$ROBOT_NAME.local << EOF
source /opt/ros/noetic/setup.bash

echo "Available ROS topics:"
timeout 5 rostopic list | grep -E "(camera|rl_inference|car_cmd)" || echo "No relevant topics found"

echo ""
echo "Checking RL inference status topic:"
if timeout 3 rostopic echo /$ROBOT_NAME/rl_inference/status -n 1 &> /dev/null; then
    echo "✅ RL inference status topic is active"
    echo "Latest status:"
    timeout 3 rostopic echo /$ROBOT_NAME/rl_inference/status -n 1
else
    echo "❌ RL inference status topic is not active"
fi

echo ""
echo "Checking camera feed:"
if timeout 3 rostopic echo /$ROBOT_NAME/camera_node/image/compressed -n 1 &> /dev/null; then
    echo "✅ Camera feed is active"
else
    echo "❌ Camera feed is not active"
fi
EOF

# System health check
echo ""
echo "💻 System health check..."
ssh duckie@$ROBOT_NAME.local << 'EOF'
echo "CPU usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1

echo ""
echo "Memory usage:"
free -h | grep Mem

echo ""
echo "Disk usage:"
df -h / | tail -1

echo ""
echo "Temperature:"
if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
    temp=$(cat /sys/class/thermal/thermal_zone0/temp)
    temp_c=$((temp/1000))
    echo "${temp_c}°C"
else
    echo "Temperature sensor not available"
fi
EOF

echo ""
echo "🎯 VERIFICATION COMPLETE"
echo ""
echo "If all checks passed, your deployment is working!"
echo ""
echo "Useful monitoring commands:"
echo "  ssh duckie@$ROBOT_NAME.local 'docker logs rl-inference -f'"
echo "  ssh duckie@$ROBOT_NAME.local 'rostopic echo /$ROBOT_NAME/rl_inference/status'"
echo "  ssh duckie@$ROBOT_NAME.local 'rostopic echo /$ROBOT_NAME/car_cmd_switch_node/cmd'"