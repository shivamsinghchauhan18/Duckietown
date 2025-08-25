#!/bin/bash
# WSL + RTX 3060 Enhanced Duckietown RL Setup Script

set -e

echo "🚀 Setting up Enhanced Duckietown RL for WSL + RTX 3060"
echo "=================================================="

# Check if running in WSL
if grep -qi microsoft /proc/version; then
    echo "✅ WSL environment detected"
    export WSL_ENV=1
else
    echo "⚠️ Not running in WSL - some optimizations may not apply"
    export WSL_ENV=0
fi

# Check for NVIDIA GPU
if nvidia-smi > /dev/null 2>&1; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader
    export GPU_AVAILABLE=1
else
    echo "⚠️ NVIDIA GPU not detected or drivers not installed"
    export GPU_AVAILABLE=0
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt update
sudo apt install -y python3-pip python3-dev build-essential

# Install WSL-specific packages for graphics
if [ "$WSL_ENV" = "1" ]; then
    echo "🐧 Installing WSL graphics support..."
    sudo apt install -y mesa-utils libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
    
    # Set up X11 forwarding for WSL
    export DISPLAY=:0
    export LIBGL_ALWAYS_INDIRECT=1
fi

# Create virtual environment
echo "🐍 Setting up Python environment..."
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv

# Create project virtual environment
if [ ! -d "venv_enhanced_rl" ]; then
    python3 -m virtualenv venv_enhanced_rl
fi

source venv_enhanced_rl/bin/activate

# Install PyTorch with CUDA support for RTX 3060
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install compatible NumPy (avoid 2.0 issues)
echo "🔢 Installing compatible NumPy..."
pip install "numpy<2.0"

# Install Gym/Gymnasium
echo "🏃 Installing Gym environment..."
pip install gymnasium[classic_control,box2d,atari,accept-rom-license]

# Try to install gym-duckietown with fallback
echo "🦆 Installing Duckietown environment..."
pip install gym-duckietown || echo "⚠️ gym-duckietown installation failed - will use fallback mode"

# Install YOLO dependencies
echo "👁️ Installing YOLO dependencies..."
pip install ultralytics opencv-python Pillow

# Install TensorBoard with compatibility fixes
echo "📊 Installing TensorBoard..."
pip install tensorboard "protobuf<4.0" || echo "⚠️ TensorBoard installation issues - will use fallback"

# Install other ML dependencies
echo "🧠 Installing ML dependencies..."
pip install matplotlib seaborn pandas scikit-learn

# Install Pyglet with WSL compatibility
echo "🎮 Installing Pyglet..."
if [ "$WSL_ENV" = "1" ]; then
    pip install "pyglet==1.5.27"  # Stable version for WSL
else
    pip install pyglet
fi

# Set up CUDA environment variables
if [ "$GPU_AVAILABLE" = "1" ]; then
    echo "⚡ Setting up CUDA environment..."
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    
    # Add to bashrc for persistence
    echo "export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
    echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512" >> ~/.bashrc
fi

# Test the installation
echo "🧪 Testing installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

try:
    import numpy as np
    print(f'NumPy version: {np.__version__}')
except ImportError:
    print('❌ NumPy not available')

try:
    import gymnasium as gym
    print('✅ Gymnasium available')
except ImportError:
    try:
        import gym
        print('✅ Legacy Gym available')
    except ImportError:
        print('❌ No Gym environment available')

try:
    from ultralytics import YOLO
    print('✅ YOLO available')
except ImportError:
    print('❌ YOLO not available')

try:
    import gym_duckietown
    print('✅ gym-duckietown available')
except ImportError:
    print('⚠️ gym-duckietown not available - fallback mode will be used')
"

echo ""
echo "🎉 Setup completed!"
echo ""
echo "To activate the environment:"
echo "  source venv_enhanced_rl/bin/activate"
echo ""
echo "To run enhanced training:"
echo "  python3 complete_enhanced_rl_pipeline.py --mode full --timesteps 1000000"
echo ""
echo "📝 Notes:"
echo "  - If gym-duckietown fails, the system will use intelligent fallback mode"
echo "  - YOLO will work with synthetic test images if real objects aren't detected"
echo "  - All enhanced features are now enabled by default (no headless bypass)"
echo ""