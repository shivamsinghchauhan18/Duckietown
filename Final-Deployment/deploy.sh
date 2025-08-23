#!/bin/bash

# Bulletproof Duckiebot Deployment Script
# Usage: ./deploy.sh ROBOT_NAME [DOCKER_USERNAME]

set -e  # Exit on any error

ROBOT_NAME=${1:-"duckiebot"}
DOCKER_USERNAME=${2:-"shivamsinghchauhan"}
IMAGE_NAME="$DOCKER_USERNAME/duckiebot-rl:latest"

echo "🚀 BULLETPROOF DUCKIEBOT DEPLOYMENT"
echo "=================================="
echo "Robot Name: $ROBOT_NAME"
echo "Docker Image: $IMAGE_NAME"
echo ""

# Step 1: Verify files
echo "📋 Step 1: Verifying files..."
if [ ! -f "champion_model.pth" ]; then
    echo "❌ champion_model.pth not found!"
    exit 1
fi

if [ ! -f "enhanced_config.yml" ]; then
    echo "❌ enhanced_config.yml not found!"
    exit 1
fi

echo "✅ Files verified"

# Step 2: Build Docker image
echo ""
echo "🔨 Step 2: Building Docker image..."
if command -v docker buildx &> /dev/null; then
    echo "Using buildx for ARM64 build..."
    docker buildx build --platform linux/arm64 -t $IMAGE_NAME .
else
    echo "Using standard docker build..."
    docker build -t $IMAGE_NAME .
fi

echo "✅ Docker image built"

# Step 3: Push to registry (optional)
echo ""
echo "📤 Step 3: Pushing to Docker registry..."
read -p "Push to Docker Hub? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker push $IMAGE_NAME
    echo "✅ Image pushed to registry"
else
    echo "⏭️  Skipping push to registry"
fi

# Step 4: Deploy to robot
echo ""
echo "🤖 Step 4: Deploying to robot..."
echo "Connecting to $ROBOT_NAME.local..."

# Check if robot is reachable
if ! ping -c 1 $ROBOT_NAME.local &> /dev/null; then
    echo "❌ Cannot reach $ROBOT_NAME.local"
    echo "Please check:"
    echo "  - Robot is powered on"
    echo "  - Robot is connected to WiFi"
    echo "  - Robot name is correct"
    exit 1
fi

echo "✅ Robot is reachable"

# Deploy via SSH
echo "Deploying container..."
ssh duckie@$ROBOT_NAME.local << EOF
    # Stop existing container
    docker stop rl-inference 2>/dev/null || true
    docker rm rl-inference 2>/dev/null || true
    
    # Pull latest image
    docker pull $IMAGE_NAME
    
    # Run new container
    docker run -d \\
        --name rl-inference \\
        --network host \\
        --privileged \\
        -e ROBOT_NAME=\$(hostname) \\
        -e ROS_MASTER_URI=http://localhost:11311 \\
        $IMAGE_NAME
    
    echo "Container started successfully"
EOF

echo "✅ Deployment completed"

# Step 5: Verify deployment
echo ""
echo "🔍 Step 5: Verifying deployment..."
sleep 5  # Give container time to start

ssh duckie@$ROBOT_NAME.local << 'EOF'
    echo "Checking container status..."
    if docker ps | grep -q rl-inference; then
        echo "✅ Container is running"
        
        echo "Checking logs..."
        docker logs rl-inference --tail 10
        
        echo ""
        echo "🎉 DEPLOYMENT SUCCESSFUL!"
        echo ""
        echo "Monitor with:"
        echo "  docker logs rl-inference -f"
        echo "  rostopic echo /$HOSTNAME/rl_inference/status"
        echo ""
        echo "Emergency stop:"
        echo "  docker stop rl-inference"
        
    else
        echo "❌ Container is not running"
        echo "Check logs with: docker logs rl-inference"
        exit 1
    fi
EOF

echo ""
echo "🎉 BULLETPROOF DEPLOYMENT COMPLETED!"
echo ""
echo "Next steps:"
echo "1. Monitor robot behavior in a safe area"
echo "2. Check ROS topics: rostopic list | grep rl_inference"
echo "3. Emergency stop: ssh duckie@$ROBOT_NAME.local 'docker stop rl-inference'"