#!/bin/bash

# Emergency Stop Script
# Usage: ./emergency_stop.sh ROBOT_NAME

ROBOT_NAME=${1:-"duckiebot"}

echo "🚨 EMERGENCY STOP ACTIVATED"
echo "=========================="
echo "Robot Name: $ROBOT_NAME"
echo ""

# Stop the container immediately
echo "🛑 Stopping RL container..."
ssh duckie@$ROBOT_NAME.local 'docker stop rl-inference' &

# Send stop command via ROS
echo "📡 Sending ROS stop command..."
ssh duckie@$ROBOT_NAME.local << EOF &
source /opt/ros/noetic/setup.bash
rostopic pub /$ROBOT_NAME/car_cmd_switch_node/cmd duckietown_msgs/Twist2DStamped "header: {stamp: now, frame_id: ''}, v: 0.0, omega: 0.0" -1
EOF

# Wait for both commands to complete
wait

echo "✅ Emergency stop commands sent"
echo ""
echo "Robot should now be stopped."
echo ""
echo "To restart:"
echo "  ./deploy.sh $ROBOT_NAME"
echo ""
echo "To check status:"
echo "  ./verify_deployment.sh $ROBOT_NAME"