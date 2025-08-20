#!/bin/bash
# Quick deployment script for Duckiebot RL system

set -e

# Configuration
ROBOT_NAME=${1:-"duckiebot"}
ROBOT_IP=${2:-"192.168.1.100"}
MODEL_PATH=${3:-"models/champion_model.pth"}
CONFIG_PATH=${4:-"config/enhanced_config.yml"}

echo "🤖 Deploying RL system to $ROBOT_NAME at $ROBOT_IP"

# Run tests first
echo "🧪 Running tests..."
python tests/run_comprehensive_tests.py --quick

# Deploy to robot
echo "🚀 Deploying to robot..."
python duckiebot_deployment/deploy_to_duckiebot.py \
    --robot-name "$ROBOT_NAME" \
    --robot-ip "$ROBOT_IP" \
    --model-path "$MODEL_PATH" \
    --config-path "$CONFIG_PATH"

# Health check
echo "🔍 Checking deployment health..."
python duckiebot_deployment/deploy_to_duckiebot.py \
    --robot-name "$ROBOT_NAME" \
    --robot-ip "$ROBOT_IP" \
    --health-check

echo "✅ Deployment complete!"
echo "Monitor with: rostopic echo /$ROBOT_NAME/rl_commands"