#!/bin/bash
set -e

echo "🚀 Deploying Enhanced RL Model with DTS"

# Check if DTS is available
if ! command -v dts &> /dev/null; then
    echo "❌ DTS not found. Please install Duckietown Shell first."
    exit 1
fi

# Build and deploy
echo "Building DTS image..."
dts devel build -f

echo "Deploying to robot..."
dts devel run -R $ROBOT_NAME

echo "✅ DTS deployment completed!"
