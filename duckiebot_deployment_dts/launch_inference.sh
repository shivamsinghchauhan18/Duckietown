#!/bin/bash

# DTS-compatible launch script for RL inference
source /opt/ros/noetic/setup.bash
source /code/devel/setup.bash 2>/dev/null || true

# Set ROS environment
export ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}
export ROS_IP=$(hostname -I | awk '{print $1}')
export ROBOT_NAME=${ROBOT_NAME:-$(hostname)}

echo "Starting RL Inference System for ${ROBOT_NAME}"
echo "ROS_MASTER_URI: ${ROS_MASTER_URI}"
echo "ROS_IP: ${ROS_IP}"

# Launch ROS nodes using roslaunch (DTS standard)
exec roslaunch duckiebot_rl_inference rl_inference.launch \
    robot_name:=${ROBOT_NAME} \
    model_path:=${MODEL_PATH:-/models/champion_model.pth} \
    config_path:=${CONFIG_PATH:-/config/enhanced_config.yml}