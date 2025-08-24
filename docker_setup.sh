#!/bin/bash
set -e

echo "🐳 Enhanced Duckietown RL - Docker Setup"
echo "========================================"
echo "One-command Docker setup for WSL/Ubuntu"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed!"
    echo ""
    echo "Install Docker with:"
    echo "curl -fsSL https://get.docker.com -o get-docker.sh"
    echo "sudo sh get-docker.sh"
    echo "sudo usermod -aG docker \$USER"
    echo ""
    echo "Then log out and back in, and run this script again."
    exit 1
fi

print_success "Docker found: $(docker --version)"

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    print_error "Docker Compose is not available!"
    echo ""
    echo "Install Docker Compose with:"
    echo "sudo apt install docker-compose-plugin"
    exit 1
fi

print_success "Docker Compose available"

# Check for NVIDIA Docker (optional)
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected, checking for nvidia-docker..."
    if docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        print_success "NVIDIA Docker support available"
        GPU_SUPPORT=true
    else
        print_warning "NVIDIA Docker not properly configured"
        print_warning "GPU training will not be available"
        GPU_SUPPORT=false
    fi
else
    print_warning "No NVIDIA GPU detected - using CPU mode"
    GPU_SUPPORT=false
fi

# Build Docker image
print_status "Building Docker image (this may take 10-15 minutes)..."
if docker build -f Dockerfile.wsl -t duckietown-rl-enhanced:latest .; then
    print_success "Docker image built successfully!"
else
    print_error "Docker image build failed!"
    exit 1
fi

# Create run script
print_status "Creating run script..."
if [ "$GPU_SUPPORT" = true ]; then
    # GPU version
    cat > run_docker.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Enhanced Duckietown RL Docker Container (GPU)"

# Allow X11 forwarding (for WSL)
xhost +local:docker 2>/dev/null || true

# Run with GPU support
docker run -it --rm \
    --gpus all \
    --name duckietown-rl \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,display \
    -v $(pwd):/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --network host \
    duckietown-rl-enhanced:latest
EOF
else
    # CPU version
    cat > run_docker.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Enhanced Duckietown RL Docker Container (CPU)"

# Allow X11 forwarding (for WSL)
xhost +local:docker 2>/dev/null || true

# Run CPU-only version
docker run -it --rm \
    --name duckietown-rl \
    -e DISPLAY=$DISPLAY \
    -v $(pwd):/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --network host \
    duckietown-rl-enhanced:latest
EOF
fi

chmod +x run_docker.sh
print_success "Run script created: run_docker.sh"

# Create test script
print_status "Creating test script..."
cat > test_docker.sh << 'EOF'
#!/bin/bash
echo "🧪 Testing Docker Environment"

docker run --rm \
    $([ "$GPU_SUPPORT" = true ] && echo "--gpus all") \
    -v $(pwd):/workspace \
    duckietown-rl-enhanced:latest \
    python -c "
import torch
import numpy as np
import gymnasium as gym
from ultralytics import YOLO
import gym_duckietown

print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ NumPy:', np.__version__)
print('✅ Gymnasium:', gym.__version__)

# Test YOLO
try:
    model = YOLO('yolov5s.pt')
    print('✅ YOLO: Model loaded')
except Exception as e:
    print('⚠️  YOLO:', e)

# Test gym-duckietown
try:
    env = gym.make('Duckietown-loop_empty-v0')
    obs = env.reset()
    print(f'✅ gym-duckietown: Environment created, obs shape: {obs.shape}')
    env.close()
except Exception as e:
    print('⚠️  gym-duckietown:', e)

print('🎉 Docker environment test completed!')
"
EOF

chmod +x test_docker.sh
print_success "Test script created: test_docker.sh"

# Create training scripts
print_status "Creating training scripts..."

# Simple training script
cat > train_simple_docker.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Simple RL Training in Docker"

docker run -it --rm \
    $(command -v nvidia-smi &> /dev/null && echo "--gpus all") \
    -v $(pwd):/workspace \
    -v duckietown-models:/workspace/models \
    -v duckietown-logs:/workspace/logs \
    duckietown-rl-enhanced:latest \
    python train_enhanced_rl_simple.py
EOF

# Complete pipeline script
cat > train_pipeline_docker.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Complete RL Pipeline in Docker"

docker run -it --rm \
    $(command -v nvidia-smi &> /dev/null && echo "--gpus all") \
    -v $(pwd):/workspace \
    -v duckietown-models:/workspace/models \
    -v duckietown-logs:/workspace/logs \
    -v duckietown-results:/workspace/pipeline_results \
    duckietown-rl-enhanced:latest \
    python complete_enhanced_rl_pipeline.py --mode full
EOF

chmod +x train_simple_docker.sh train_pipeline_docker.sh
print_success "Training scripts created"

# Test the Docker environment
print_status "Testing Docker environment..."
if ./test_docker.sh; then
    print_success "Docker environment test passed!"
else
    print_warning "Docker environment test had issues (may still work)"
fi

# Final summary
echo ""
echo "🎉 DOCKER SETUP COMPLETED!"
echo "=========================="
echo ""
echo "🐳 Available commands:"
echo "  ./run_docker.sh                    # Interactive Docker shell"
echo "  ./test_docker.sh                   # Test environment"
echo "  ./train_simple_docker.sh           # Simple RL training"
echo "  ./train_pipeline_docker.sh         # Complete pipeline"
echo ""
echo "🎯 Quick start:"
echo "1. ./test_docker.sh                  # Verify everything works"
echo "2. ./train_simple_docker.sh          # Start with simple training"
echo ""
echo "💡 Docker advantages:"
echo "  ✅ No conda conflicts"
echo "  ✅ Consistent environment"
echo "  ✅ Easy to share and reproduce"
echo "  ✅ Isolated from host system"
if [ "$GPU_SUPPORT" = true ]; then
    echo "  ✅ GPU acceleration enabled"
else
    echo "  ⚠️  CPU-only mode (no GPU detected)"
fi
echo ""
print_success "Ready for enhanced RL training with Docker! 🚀"