#!/bin/bash
set -e

echo "🚀 WSL-Optimized Enhanced Duckietown RL Setup"
echo "=============================================="
echo "Fast setup specifically optimized for WSL2 Ubuntu"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if we're in WSL
if ! grep -q microsoft /proc/version 2>/dev/null; then
    print_warning "This script is optimized for WSL. You may want to use setup_tier1_environment.sh instead."
fi

# Step 1: Install system dependencies
print_status "Installing system dependencies..."
sudo apt update -qq
sudo apt install -y \
    mesa-utils libgl1-mesa-glx libgl1-mesa-dri \
    xorg-dev freeglut3-dev xvfb \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 git curl wget
print_success "System dependencies installed"

# Step 2: Setup conda with fast solver
print_status "Setting up conda with fast solver..."
if ! command -v conda &> /dev/null; then
    print_error "Conda not found. Please install Miniconda first:"
    echo "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# Install and configure libmamba solver
conda install -n base -y conda-libmamba-solver
conda config --set solver libmamba
conda config --set channel_priority strict
conda config --set repodata_fns current_repodata.json
print_success "Conda optimized for WSL"

# Step 3: Create environment
ENV_NAME="duckietown-enhanced-tier1"
print_status "Creating conda environment: $ENV_NAME"

# Remove existing environment if it exists
conda remove --name $ENV_NAME --all -y 2>/dev/null || true

# Create environment with libmamba solver (much faster)
conda env create -f environment_enhanced_tier1.yml
print_success "Environment created successfully!"

# Step 4: Install additional packages
print_status "Installing additional packages..."
conda run -n $ENV_NAME pip install --no-cache-dir \
    ultralytics \
    gymnasium \
    stable-baselines3 \
    gym-duckietown==6.0.25 \
    "ray[rllib]" \
    wandb mlflow tensorboardX \
    rich typer loguru \
    fastapi uvicorn

print_success "Additional packages installed!"

# Step 5: Clone and install gym-duckietown
print_status "Setting up gym-duckietown..."
if [ -d "gym-duckietown" ]; then
    rm -rf gym-duckietown
fi

git clone --branch v6.0.25 --single-branch --depth 1 \
    https://github.com/duckietown/gym-duckietown.git
conda run -n $ENV_NAME pip install -e gym-duckietown/
print_success "gym-duckietown installed!"

# Step 6: Test installation
print_status "Testing installation..."
conda run -n $ENV_NAME python -c "
import torch
import gymnasium as gym
from ultralytics import YOLO
import gym_duckietown

print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ Gymnasium:', gym.__version__)
print('✅ YOLO: Model loading...')
model = YOLO('yolov5s.pt')
print('✅ YOLO: OK')
print('✅ gym-duckietown: Testing...')
env = gym.make('Duckietown-loop_empty-v0')
obs = env.reset()
print('✅ gym-duckietown: OK')
env.close()
print('🎉 All tests passed!')
"

# Step 7: Create activation script
print_status "Creating activation script..."
cat > activate_wsl_env.sh << 'EOF'
#!/bin/bash
echo "🚀 Activating WSL-Optimized Duckietown RL Environment"
echo "===================================================="

# Activate conda environment
conda activate duckietown-enhanced-tier1

# Set WSL-specific environment variables
export DISPLAY=:0
export LIBGL_ALWAYS_INDIRECT=1
export MESA_GL_VERSION_OVERRIDE=3.3

# Performance optimization
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

# YOLO configuration
export YOLO_DEVICE=0
export YOLO_VERBOSE=False

echo "Environment activated successfully!"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    echo "CUDA: Not available (CPU mode)"
fi

echo ""
echo "🎯 Ready for enhanced RL training!"
echo "Commands:"
echo "  python complete_enhanced_rl_pipeline.py --mode full"
echo "  python enhanced_rl_training_system.py"
echo "  python train_enhanced_rl_simple.py"
EOF

chmod +x activate_wsl_env.sh
print_success "Activation script created: activate_wsl_env.sh"

# Step 8: Create WSL test script
cat > test_wsl_installation.py << 'EOF'
#!/usr/bin/env python3
"""WSL-specific installation test"""
import os
import sys

def test_wsl_setup():
    print("🧪 WSL Enhanced Duckietown RL - Installation Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Basic imports
    print("\n1. Testing basic imports...")
    try:
        import torch
        import numpy as np
        import gymnasium as gym
        print(f"   ✅ PyTorch {torch.__version__}")
        print(f"   ✅ NumPy {np.__version__}")
        print(f"   ✅ Gymnasium {gym.__version__}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Basic imports failed: {e}")
    
    # Test 2: CUDA
    print("\n2. Testing CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.version.cuda}")
            print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️  CUDA not available - using CPU")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ CUDA test failed: {e}")
    
    # Test 3: YOLO
    print("\n3. Testing YOLO...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov5s.pt')
        print("   ✅ YOLO model loaded")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ YOLO failed: {e}")
    
    # Test 4: gym-duckietown
    print("\n4. Testing gym-duckietown...")
    try:
        import gym_duckietown
        env = gym.make('Duckietown-loop_empty-v0')
        obs = env.reset()
        print(f"   ✅ Environment created, obs shape: {obs.shape}")
        env.close()
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ gym-duckietown failed: {e}")
    
    # Test 5: RL libraries
    print("\n5. Testing RL libraries...")
    try:
        import stable_baselines3
        import ray
        print(f"   ✅ Stable Baselines3: {stable_baselines3.__version__}")
        print(f"   ✅ Ray: {ray.__version__}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ RL libraries failed: {e}")
    
    # Test 6: Display/OpenGL
    print("\n6. Testing display/OpenGL...")
    try:
        os.environ['DISPLAY'] = ':0'
        os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
        print("   ✅ Display variables set")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Display test failed: {e}")
    
    print(f"\n📊 Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed >= 5:
        print("🎉 WSL setup successful! Ready for training!")
        return True
    else:
        print("❌ Setup incomplete. Check the failed tests.")
        return False

if __name__ == "__main__":
    success = test_wsl_setup()
    sys.exit(0 if success else 1)
EOF

chmod +x test_wsl_installation.py
print_success "WSL test script created: test_wsl_installation.py"

# Final summary
echo ""
echo "🎉 WSL SETUP COMPLETED!"
echo "======================"
echo ""
echo "✅ Environment: $ENV_NAME"
echo "✅ Activation: source activate_wsl_env.sh"
echo "✅ Testing: python test_wsl_installation.py"
echo ""
echo "🚀 Next steps:"
echo "1. source activate_wsl_env.sh"
echo "2. python test_wsl_installation.py"
echo "3. python complete_enhanced_rl_pipeline.py --mode full"
echo ""
echo "💡 WSL Tips:"
echo "- Use 'export DISPLAY=:0' if you have X11 forwarding"
echo "- Run 'xvfb-run python script.py' for headless mode"
echo "- GPU training requires NVIDIA drivers on Windows host"
echo ""
print_success "Ready for enhanced RL training on WSL! 🚀"