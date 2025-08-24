#!/bin/bash
set -e

echo "🔧 INTERACTIVE ENHANCED DUCKIETOWN RL - DOCKER"
echo "=============================================="
echo "Interactive development environment with all tools"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
DOCKER_IMAGE="duckietown-rl-enhanced:latest"
CONTAINER_NAME="duckietown-rl-interactive"

# GPU support detection
GPU_ARGS=""
if command -v nvidia-smi &> /dev/null && docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi &> /dev/null 2>&1; then
    GPU_ARGS="--gpus all"
    print_success "GPU support detected and enabled"
else
    print_warning "No GPU support - using CPU mode"
fi

# Parse command line arguments
JUPYTER_MODE=false
PORT_JUPYTER=8888
PORT_TENSORBOARD=6006
MOUNT_EXTRA=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --jupyter)
            JUPYTER_MODE=true
            shift
            ;;
        --port-jupyter)
            PORT_JUPYTER="$2"
            shift 2
            ;;
        --port-tensorboard)
            PORT_TENSORBOARD="$2"
            shift 2
            ;;
        --mount)
            MOUNT_EXTRA="-v $2"
            shift 2
            ;;
        --help)
            echo "Interactive Docker Environment Options:"
            echo "  --jupyter              Start Jupyter Lab server"
            echo "  --port-jupyter PORT    Jupyter Lab port (default: 8888)"
            echo "  --port-tensorboard PORT TensorBoard port (default: 6006)"
            echo "  --mount PATH           Mount additional path"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for available options"
            exit 1
            ;;
    esac
done

# Function to start interactive shell
start_interactive_shell() {
    print_status "Starting interactive Docker environment..."
    
    # Allow X11 forwarding (for WSL/Linux)
    xhost +local:docker 2>/dev/null || true
    
    docker run -it --rm \
        $GPU_ARGS \
        --name "$CONTAINER_NAME" \
        -v "$(pwd):/workspace" \
        -v "duckietown-models:/workspace/models" \
        -v "duckietown-logs:/workspace/logs" \
        -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        $MOUNT_EXTRA \
        -e "DISPLAY=${DISPLAY:-:0}" \
        -e "PYTHONUNBUFFERED=1" \
        -e "LIBGL_ALWAYS_INDIRECT=1" \
        -p "$PORT_JUPYTER:8888" \
        -p "$PORT_TENSORBOARD:6006" \
        --network host \
        "$DOCKER_IMAGE" \
        bash -c "
            echo '🔧 Enhanced Duckietown RL - Interactive Environment'
            echo '=================================================='
            echo 'Container: $CONTAINER_NAME'
            echo 'Working directory: /workspace'
            echo 'GPU available: \$(python -c \"import torch; print(torch.cuda.is_available())\")'
            echo ''
            echo '🎯 Available commands:'
            echo '  python train_enhanced_rl_simple.py           # Simple RL training'
            echo '  python enhanced_rl_training_system.py        # Enhanced RL training'
            echo '  python complete_enhanced_rl_pipeline.py      # Complete pipeline'
            echo '  jupyter lab --ip=0.0.0.0 --allow-root       # Start Jupyter Lab'
            echo '  tensorboard --logdir=logs --host=0.0.0.0     # Start TensorBoard'
            echo '  python docker_test_comprehensive.py          # Run comprehensive tests'
            echo ''
            echo '📁 Mounted volumes:'
            echo '  /workspace          -> $(pwd)'
            echo '  /workspace/models   -> duckietown-models volume'
            echo '  /workspace/logs     -> duckietown-logs volume'
            echo ''
            echo '🌐 Port mappings:'
            echo '  Jupyter Lab:   http://localhost:$PORT_JUPYTER'
            echo '  TensorBoard:   http://localhost:$PORT_TENSORBOARD'
            echo ''
            echo 'Type \"exit\" to leave the container'
            echo ''
            
            # Set up shell prompt
            export PS1='🐳 duckietown-rl:\w\$ '
            
            # Start bash
            /bin/bash
        "
}

# Function to start Jupyter Lab
start_jupyter() {
    print_status "Starting Jupyter Lab server..."
    
    docker run -it --rm \
        $GPU_ARGS \
        --name "${CONTAINER_NAME}-jupyter" \
        -v "$(pwd):/workspace" \
        -v "duckietown-models:/workspace/models" \
        -v "duckietown-logs:/workspace/logs" \
        -e "PYTHONUNBUFFERED=1" \
        -p "$PORT_JUPYTER:8888" \
        -p "$PORT_TENSORBOARD:6006" \
        "$DOCKER_IMAGE" \
        bash -c "
            echo '🚀 Starting Jupyter Lab Server'
            echo '============================='
            echo 'Jupyter Lab will be available at: http://localhost:$PORT_JUPYTER'
            echo 'TensorBoard will be available at: http://localhost:$PORT_TENSORBOARD'
            echo ''
            
            cd /workspace
            
            # Start TensorBoard in background
            tensorboard --logdir=logs --host=0.0.0.0 --port=6006 &
            
            # Start Jupyter Lab
            jupyter lab \
                --ip=0.0.0.0 \
                --port=8888 \
                --allow-root \
                --no-browser \
                --NotebookApp.token='' \
                --NotebookApp.password=''
        "
}

# Function to show interactive menu
show_interactive_menu() {
    echo ""
    echo "🔧 INTERACTIVE ENVIRONMENT OPTIONS:"
    echo "=================================="
    echo "1. 🖥️  Interactive Shell (Full access)"
    echo "2. 📓 Jupyter Lab Server (Web interface)"
    echo "3. 🧪 Run Comprehensive Tests"
    echo "4. 🚀 Quick Training Session"
    echo "5. 📊 Start TensorBoard"
    echo "6. 🔍 Environment Information"
    echo "7. 🧹 Cleanup Docker Resources"
    echo "0. ❌ Exit"
    echo ""
    read -p "Enter your choice (0-7): " choice
    
    case $choice in
        1)
            start_interactive_shell
            ;;
        2)
            start_jupyter
            ;;
        3)
            run_comprehensive_tests
            ;;
        4)
            quick_training_session
            ;;
        5)
            start_tensorboard
            ;;
        6)
            show_environment_info
            ;;
        7)
            cleanup_docker_resources
            ;;
        0)
            print_status "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please select 0-7."
            show_interactive_menu
            ;;
    esac
}

# Function to run comprehensive tests
run_comprehensive_tests() {
    print_status "Running comprehensive tests..."
    
    docker run --rm \
        $GPU_ARGS \
        -v "$(pwd):/workspace" \
        "$DOCKER_IMAGE" \
        bash -c "
            cd /workspace
            echo '🧪 Running Comprehensive Tests'
            echo '============================='
            
            # Run the comprehensive test script
            if [ -f 'docker_test_comprehensive.sh' ]; then
                chmod +x docker_test_comprehensive.sh
                ./docker_test_comprehensive.sh
            else
                echo 'Test script not found, running basic tests...'
                python -c '
import torch
import numpy as np
from ultralytics import YOLO
import gym
import gym_duckietown

print(\"✅ PyTorch:\", torch.__version__)
print(\"✅ CUDA available:\", torch.cuda.is_available())
print(\"✅ YOLO: Loading model...\")
model = YOLO(\"yolov5s.pt\")
print(\"✅ YOLO: Model loaded\")
print(\"✅ Duckietown: Creating environment...\")
env = gym.make(\"Duckietown-loop_empty-v0\")
obs = env.reset()
print(f\"✅ Duckietown: Environment created, obs shape: {obs.shape}\")
env.close()
print(\"🎉 All basic tests passed!\")
'
            fi
        "
}

# Function for quick training session
quick_training_session() {
    print_status "Starting quick training session..."
    
    docker run -it --rm \
        $GPU_ARGS \
        --name "${CONTAINER_NAME}-training" \
        -v "$(pwd):/workspace" \
        -v "duckietown-models:/workspace/models" \
        -v "duckietown-logs:/workspace/logs" \
        -e "PYTHONUNBUFFERED=1" \
        "$DOCKER_IMAGE" \
        bash -c "
            cd /workspace
            echo '🚀 Quick Training Session'
            echo '======================='
            echo 'Running simple RL training for demonstration...'
            echo ''
            
            python train_enhanced_rl_simple.py
            
            echo ''
            echo '🎉 Quick training completed!'
            echo 'Check the logs directory for training results.'
        "
}

# Function to start TensorBoard
start_tensorboard() {
    print_status "Starting TensorBoard server..."
    
    docker run -it --rm \
        --name "${CONTAINER_NAME}-tensorboard" \
        -v "$(pwd):/workspace" \
        -v "duckietown-logs:/workspace/logs" \
        -p "$PORT_TENSORBOARD:6006" \
        "$DOCKER_IMAGE" \
        bash -c "
            cd /workspace
            echo '📊 Starting TensorBoard Server'
            echo '============================='
            echo 'TensorBoard will be available at: http://localhost:$PORT_TENSORBOARD'
            echo 'Press Ctrl+C to stop'
            echo ''
            
            tensorboard --logdir=logs --host=0.0.0.0 --port=6006
        "
}

# Function to show environment information
show_environment_info() {
    print_status "Gathering environment information..."
    
    docker run --rm \
        $GPU_ARGS \
        -v "$(pwd):/workspace" \
        "$DOCKER_IMAGE" \
        bash -c "
            echo '🔍 ENVIRONMENT INFORMATION'
            echo '========================='
            echo ''
            
            echo '🐳 Docker Image:'
            echo '  Image: $DOCKER_IMAGE'
            echo '  Container: $CONTAINER_NAME'
            echo ''
            
            echo '🖥️  System Information:'
            echo '  OS: \$(cat /etc/os-release | grep PRETTY_NAME | cut -d'\"' -f2)'
            echo '  Python: \$(python --version)'
            echo '  Working Directory: \$(pwd)'
            echo ''
            
            echo '🚀 GPU Information:'
            python -c '
import torch
print(f\"  PyTorch: {torch.__version__}\")
print(f\"  CUDA Available: {torch.cuda.is_available()}\")
if torch.cuda.is_available():
    print(f\"  CUDA Version: {torch.version.cuda}\")
    print(f\"  GPU Count: {torch.cuda.device_count()}\")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f\"  GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)\")
else:
    print(\"  No GPU available\")
'
            echo ''
            
            echo '📦 Key Packages:'
            python -c '
packages = [
    (\"numpy\", \"np\"),
    (\"opencv-python\", \"cv2\"),
    (\"gymnasium\", \"gym\"),
    (\"ultralytics\", None),
    (\"stable_baselines3\", None),
    (\"ray\", None)
]

for pkg_name, import_name in packages:
    try:
        if import_name:
            module = __import__(import_name)
        else:
            module = __import__(pkg_name.replace(\"-\", \"_\"))
        version = getattr(module, \"__version__\", \"unknown\")
        print(f\"  {pkg_name}: {version}\")
    except ImportError:
        print(f\"  {pkg_name}: Not installed\")
'
            echo ''
            
            echo '📁 Mounted Volumes:'
            echo '  Workspace: /workspace'
            echo '  Models: /workspace/models'
            echo '  Logs: /workspace/logs'
            echo ''
            
            echo '🌐 Available Ports:'
            echo '  Jupyter Lab: $PORT_JUPYTER'
            echo '  TensorBoard: $PORT_TENSORBOARD'
            echo ''
            
            echo '📋 Available Scripts:'
            ls -la /workspace/*.py 2>/dev/null | head -10 || echo '  No Python scripts found in workspace'
        "
}

# Function to cleanup Docker resources
cleanup_docker_resources() {
    print_status "Cleaning up Docker resources..."
    
    # Stop running containers
    containers=$(docker ps --filter "name=duckietown-rl" --format "{{.Names}}")
    if [ -n "$containers" ]; then
        echo "Stopping containers: $containers"
        echo "$containers" | xargs docker stop
    fi
    
    # Remove stopped containers
    stopped_containers=$(docker ps -a --filter "name=duckietown-rl" --format "{{.Names}}")
    if [ -n "$stopped_containers" ]; then
        echo "Removing containers: $stopped_containers"
        echo "$stopped_containers" | xargs docker rm
    fi
    
    # Optional: Clean up volumes
    read -p "Remove Docker volumes (models, logs)? (y/N): " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        docker volume rm duckietown-models duckietown-logs 2>/dev/null || true
        print_success "Volumes removed"
    fi
    
    print_success "Docker cleanup completed"
}

# Main execution
main() {
    # Check if Docker image exists
    if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
        print_error "Docker image '$DOCKER_IMAGE' not found!"
        echo "Please run './docker_setup.sh' first to build the image."
        exit 1
    fi
    
    # If Jupyter mode requested, start directly
    if [ "$JUPYTER_MODE" = true ]; then
        start_jupyter
        return
    fi
    
    # If no arguments provided, show menu
    if [ $# -eq 0 ]; then
        show_interactive_menu
    else
        # Start interactive shell by default
        start_interactive_shell
    fi
}

# Handle script execution
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi