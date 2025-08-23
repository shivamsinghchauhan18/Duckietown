#!/bin/bash

# Setup ROS environment
source /opt/ros/noetic/setup.bash

# Get robot name
export ROBOT_NAME=${ROBOT_NAME:-$(hostname)}

echo "Starting RL Inference for robot: $ROBOT_NAME"

# Launch the node
python3 /code/rl_inference_node.py _robot_name:=$ROBOT_NAME