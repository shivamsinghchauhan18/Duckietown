#!/bin/bash
set -e

echo "🚀 Deploying Enhanced RL Model with Docker"

# Build Docker image
echo "Building Docker image..."
docker build -t duckietown-rl-enhanced .

# Run container
echo "Starting inference server..."
docker-compose up -d

echo "✅ Deployment completed!"
echo "Inference server available at: http://localhost:8000"
echo "Health check: curl http://localhost:8000/health"
