#!/bin/bash
set -e

echo "🚀 WSL Minimal Enhanced Duckietown RL Setup"
echo "==========================================="
echo "Minimal setup that handles package conflicts gracefully"
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

# Step 1: Install only essential system packages
print_status "Installing essential system packages..."
sudo apt update -qq

# Install packages one by one to handle failures gracefully
ESSENTIAL_PACKAGES=(
    "git"
    "curl" 
    "wget"
    "build-essential"
    "python3-dev"
    "python3-pip"
)

OPTIONAL_PACKAGES=(
    "mesa-utils"
    "xvfb"
    "libgomp1"
)

# Install essential packages (must succeed)
for pkg in "${ESSENTIAL_PACKAGES[@]}"; do
    if sudo apt install -y "$pkg" 2>/dev/null; then
        print_success "Installed: $pkg"
    else
        print_error "Failed to install essential package: $pkg"
        exit 1
    fi
done

# Install optional packages (can fail)
for pkg in "${OPTIONAL_PACKAGES[@]}"; do
    if sudo apt install -y "$pkg" 2>/dev/null; then
        print_success "Installed: $pkg"
    else
        print_warning "Skipped: $pkg (not available)"
    fi
done

# Try to install OpenGL packages (version-agnostic)
print_status "Attempting to install OpenGL packages..."
OPENGL_PACKAGES=(
    "libgl1-mesa-dev"
    "libgl1-mesa-glx" 
    "libgl1-mesa-dri"
    "libegl1-mesa-dev"
    "libgles2-mesa-dev"
)

for pkg in "${OPENGL_PACKAGES[@]}"; do
    if sudo apt install -y "$pkg" 2>/dev/null; then
        print_success "Installed OpenGL: $pkg"
    else
        print_warning "Skipped OpenGL: $pkg (not available)"
    fi
done

print_success "System packages installation completed"

# Step 2: Setup conda with fast solver
print_status "Setting up conda..."
if ! command -v conda &> /dev/null; then
    print_error "Conda not found. Install with:"
    echo "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# Configure conda for speed
print_status "Configuring conda for optimal performance..."
conda install -n base -y conda-libmamba-solver 2>/dev/null || print_warning "Could not install libmamba solver"
conda config --set solver libmamba 2>/dev/null || true
conda config --set channel_priority strict 2>/dev/null || true
conda config --set repodata_fns current_repodata.json 2>/dev/null || true
print_success "Conda configured"

# Step 3: Create environment
ENV_NAME="duckietown-enhanced-tier1"
print_status "Creating conda environment: $ENV_NAME"

# Remove existing environment
conda remove --name $ENV_NAME --all -y 2>/dev/null || true

# Create environment
if conda env create -f environment_enhanced_tier1.yml; then
    print_success "Environment created successfully!"
else
    print_error "Environment creation failed. Trying fallback method..."
    
    # Fallback: create minimal environment and install packages manually
    print_status "Creating minimal environment..."
    conda create -n $ENV_NAME python=3.10 -y
    
    print_status "Installing packages manually..."
    conda run -n $ENV_NAME conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    conda run -n $ENV_NAME conda install -y numpy scipy matplotlib pandas scikit-learn pillow opencv -c conda-forge
    conda run -n $ENV_NAME pip install --no-cache-dir \
        ultralytics gymnasium stable-baselines3 \
        gym-duckietown==6.0.25 "ray[rllib]" \
        wandb tensorboard rich typer
    
    print_success "Manual installation completed!"
fi

# Step 4: Test basic functionality
print_status "Testing basic functionality..."
conda run -n $ENV_NAME python -c "
try:
    import torch
    import numpy as np
    print('✅ PyTorch:', torch.__version__)
    print('✅ CUDA available:', torch.cuda.is_available())
    print('✅ NumPy:', np.__version__)
    print('🎉 Basic test passed!')
except Exception as e:
    print('❌ Basic test failed:', e)
    exit(1)
"

# Step 5: Create simple activation script
print_status "Creating activation script..."
cat > activate_wsl_minimal.sh << 'EOF'
#!/bin/bash
echo "🚀 Activating WSL Minimal Duckietown RL Environment"

# Activate conda environment
conda activate duckietown-enhanced-tier1

# Set WSL-friendly environment variables
export DISPLAY=:0.0
export LIBGL_ALWAYS_INDIRECT=1
export MESA_GL_VERSION_OVERRIDE=3.3
export PYTHONUNBUFFERED=1

# Performance settings
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "Environment activated!"
echo "Python: $(python --version)"

# Test PyTorch
if python -c 'import torch; print("PyTorch:", torch.__version__)' 2>/dev/null; then
    if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        echo "CUDA: Available"
    else
        echo "CUDA: Not available (CPU mode)"
    fi
else
    echo "PyTorch: Not available"
fi

echo ""
echo "🎯 Ready for training!"
echo "Commands:"
echo "  python complete_enhanced_rl_pipeline.py --mode full"
echo "  python train_enhanced_rl_simple.py"
EOF

chmod +x activate_wsl_minimal.sh
print_success "Activation script created: activate_wsl_minimal.sh"

# Step 6: Create simple test
cat > test_wsl_minimal.py << 'EOF'
#!/usr/bin/env python3
"""Minimal WSL test"""
import sys

def test_minimal():
    print("🧪 WSL Minimal Test")
    print("=" * 30)
    
    tests = []
    
    # Test PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda}")
        else:
            print("⚠️  CUDA: Not available")
        tests.append(True)
    except Exception as e:
        print(f"❌ PyTorch: {e}")
        tests.append(False)
    
    # Test basic packages
    try:
        import numpy as np
        import matplotlib
        print(f"✅ NumPy: {np.__version__}")
        print(f"✅ Matplotlib: {matplotlib.__version__}")
        tests.append(True)
    except Exception as e:
        print(f"❌ Basic packages: {e}")
        tests.append(False)
    
    # Test YOLO (optional)
    try:
        from ultralytics import YOLO
        print("✅ YOLO: Available")
        tests.append(True)
    except Exception as e:
        print(f"⚠️  YOLO: {e}")
        tests.append(False)
    
    passed = sum(tests)
    total = len(tests)
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed >= 2:
        print("🎉 Minimal setup successful!")
        return True
    else:
        print("❌ Setup needs attention")
        return False

if __name__ == "__main__":
    success = test_minimal()
    sys.exit(0 if success else 1)
EOF

chmod +x test_wsl_minimal.py

# Final summary
echo ""
echo "🎉 WSL MINIMAL SETUP COMPLETED!"
echo "==============================="
echo ""
echo "✅ Environment: $ENV_NAME"
echo "✅ Activation: source activate_wsl_minimal.sh"
echo "✅ Testing: python test_wsl_minimal.py"
echo ""
echo "🚀 Next steps:"
echo "1. source activate_wsl_minimal.sh"
echo "2. python test_wsl_minimal.py"
echo "3. python train_enhanced_rl_simple.py  # Start with simple version"
echo ""
echo "💡 If packages are missing:"
echo "   conda activate duckietown-enhanced-tier1"
echo "   pip install <missing-package>"
echo ""
print_success "Ready for enhanced RL training! 🚀"