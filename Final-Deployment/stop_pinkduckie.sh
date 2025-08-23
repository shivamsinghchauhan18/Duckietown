#!/bin/bash

# Emergency stop for pinkduckie
ROBOT_NAME="pinkduckie"

echo "🚨 EMERGENCY STOP FOR $ROBOT_NAME"
echo "================================"

# Stop the RL process
echo "🛑 Stopping RL process..."
ssh duckie@$ROBOT_NAME.local 'pkill -f rl_inference_node.py' &

# Send ROS stop command
echo "📡 Sending ROS stop command..."
ssh duckie@$ROBOT_NAME.local << 'EOF' &
source /opt/ros/noetic/setup.bash
rostopic pub /pinkduckie/car_cmd_switch_node/cmd duckietown_msgs/Twist2DStamped "header: {stamp: now, frame_id: ''}, v: 0.0, omega: 0.0" -1
EOF

# Wait for commands to complete
wait

echo "✅ Emergency stop commands sent"
echo ""
echo "Robot should now be stopped."
echo ""
echo "To restart:"
echo "  ssh duckie@$ROBOT_NAME.local"
echo "  /code/start_rl_pinkduckie.sh"