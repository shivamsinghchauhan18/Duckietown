#!/bin/bash
set -e

echo "🚀 Enhanced Duckietown RL - Tier 1 Setup Script"
echo "=================================================="
echo "This script sets up the complete environment for the enhanced RL system"
echo "Optimized for Tier 1 hardware (RTX 3080+, Ubuntu 20.04+, CUDA 11.8+)"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Ubuntu
if [[ ! -f /etc/lsb-release ]] || ! grep -q "Ubuntu" /etc/lsb-release; then
    print_warning "This script is optimized for Ubuntu. Proceeding anyway..."
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_success "Conda found: $(conda --version)"

# Check NVIDIA GPU and CUDA
print_status "Checking NVIDIA GPU and CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    
    if command -v nvcc &> /dev/null; then
        print_success "CUDA toolkit found: $(nvcc --version | grep release | awk '{print $6}')"
    else
        print_warning "CUDA toolkit not found in PATH. Please ensure CUDA 11.8+ is installed."
    fi
else
    print_warning "NVIDIA GPU not detected. The system will fall back to CPU training."
fi

# Remove existing environment if it exists
ENV_NAME="duckietown-enhanced-tier1"
print_status "Checking for existing environment: $ENV_NAME"
if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment $ENV_NAME already exists. Removing it..."
    conda remove --name $ENV_NAME --all -y
    print_success "Existing environment removed"
fi

# Create new environment
print_status "Creating new conda environment: $ENV_NAME"
print_status "This may take 10-15 minutes depending on your internet connection..."
conda env create -f environment_enhanced_tier1.yml

print_success "Conda environment created successfully!"

# Activate environment and install additional components
print_status "Activating environment and installing additional components..."

# Function to run commands in the conda environment
run_in_env() {
    conda run -n $ENV_NAME "$@"
}

# Clone and install gym-duckietown
print_status "Cloning and installing gym-duckietown..."
if [ -d "gym-duckietown" ]; then
    print_warning "gym-duckietown directory exists. Removing it..."
    rm -rf gym-duckietown
fi

git clone --branch v6.0.25 --single-branch --depth 1 https://github.com/duckietown/gym-duckietown.git
run_in_env pip install -e gym-duckietown/
print_success "gym-duckietown installed successfully!"

# Install additional YOLO models and dependencies
print_status "Setting up YOLO models and dependencies..."
run_in_env python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Download YOLO models
print_status "Downloading YOLO models (this may take a few minutes)..."
run_in_env python -c "
try:
    from ultralytics import YOLO
    print('Downloading YOLOv5 models...')
    
    # Download different model sizes
    models = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt']
    for model in models:
        print(f'Downloading {model}...')
        yolo = YOLO(model)
        print(f'{model} downloaded successfully!')
    
    print('All YOLO models downloaded successfully!')
except Exception as e:
    print(f'Warning: Could not download YOLO models: {e}')
    print('Models will be downloaded automatically on first use.')
"

# Copy custom maps if they exist
if [ -d "maps" ]; then
    print_status "Copying custom maps to gym-duckietown..."
    run_in_env python -c "
try:
    import sys
    sys.path.insert(0, '.')
    from maps.copy_custom_maps_to_duckietown_libs import main
    main()
    print('Custom maps copied successfully!')
except Exception as e:
    print(f'Warning: Could not copy custom maps: {e}')
"
else
    print_warning "No custom maps directory found. Skipping map installation."
fi

# Test the installation
print_status "Testing the installation..."

# Test basic imports
run_in_env python -c "
import sys
import traceback

def test_import(module_name, description):
    try:
        __import__(module_name)
        print(f'✅ {description}: OK')
        return True
    except ImportError as e:
        print(f'❌ {description}: FAILED - {e}')
        return False
    except Exception as e:
        print(f'⚠️  {description}: WARNING - {e}')
        return True

print('Testing core dependencies...')
success_count = 0
total_tests = 0

tests = [
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'),
    ('tensorflow', 'TensorFlow'),
    ('numpy', 'NumPy'),
    ('gym', 'OpenAI Gym'),
    ('cv2', 'OpenCV'),
    ('matplotlib', 'Matplotlib'),
    ('sklearn', 'Scikit-learn'),
    ('ray', 'Ray'),
    ('ultralytics', 'Ultralytics YOLO'),
    ('stable_baselines3', 'Stable Baselines3'),
]

for module, desc in tests:
    if test_import(module, desc):
        success_count += 1
    total_tests += 1

print(f'\\nCore dependencies test: {success_count}/{total_tests} passed')

# Test gym-duckietown
print('\\nTesting gym-duckietown...')
try:
    import gym_duckietown
    print('✅ gym-duckietown: Import OK')
    
    # Try to create an environment
    import gym
    env = gym.make('Duckietown-loop_empty-v0')
    print('✅ gym-duckietown: Environment creation OK')
    env.close()
    
except Exception as e:
    print(f'❌ gym-duckietown: FAILED - {e}')
    traceback.print_exc()

# Test YOLO integration
print('\\nTesting YOLO integration...')
try:
    from ultralytics import YOLO
    model = YOLO('yolov5s.pt')
    print('✅ YOLO: Model loading OK')
    
    import numpy as np
    test_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    results = model(test_image, verbose=False)
    print('✅ YOLO: Inference OK')
    
except Exception as e:
    print(f'❌ YOLO: FAILED - {e}')

print('\\nInstallation test completed!')
"

# Test project-specific imports
print_status "Testing project-specific components..."
run_in_env python -c "
import sys
sys.path.insert(0, '.')

def test_project_import(module_path, description):
    try:
        parts = module_path.split('.')
        module = __import__(module_path, fromlist=[parts[-1]])
        print(f'✅ {description}: OK')
        return True
    except ImportError as e:
        print(f'❌ {description}: FAILED - {e}')
        return False
    except Exception as e:
        print(f'⚠️  {description}: WARNING - {e}')
        return True

print('Testing project components...')

# Test if project files exist and can be imported
project_tests = []

# Check if key files exist
import os
if os.path.exists('duckietown_utils'):
    project_tests.extend([
        ('duckietown_utils.env', 'Environment utilities'),
        ('duckietown_utils.yolo_utils', 'YOLO utilities'),
        ('config.enhanced_config', 'Enhanced configuration'),
    ])

if os.path.exists('enhanced_rl_training_system.py'):
    print('✅ Enhanced RL training system: File exists')
else:
    print('❌ Enhanced RL training system: File not found')

for module_path, desc in project_tests:
    test_project_import(module_path, desc)

print('\\nProject components test completed!')
"

# Create activation script
print_status "Creating activation script..."
cat > activate_enhanced_env.sh << 'EOF'
#!/bin/bash
echo "🚀 Activating Enhanced Duckietown RL Environment"
echo "Environment: duckietown-enhanced-tier1"
echo "================================================"

# Activate conda environment
conda activate duckietown-enhanced-tier1

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export OMP_NUM_THREADS=8       # Optimize CPU threads
export MKL_NUM_THREADS=8       # Optimize MKL threads

# YOLO configuration
export YOLO_DEVICE=0           # Use GPU 0 for YOLO
export YOLO_VERBOSE=False      # Reduce YOLO output

# Display environment info
echo "Environment activated successfully!"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

echo ""
echo "Ready to run enhanced RL training!"
echo "Try: python enhanced_rl_training_system.py --help"
EOF

chmod +x activate_enhanced_env.sh
print_success "Activation script created: activate_enhanced_env.sh"

# Create quick test script
print_status "Creating quick test script..."
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Quick installation test for Enhanced Duckietown RL
"""
import sys
import traceback

def main():
    print("🧪 Enhanced Duckietown RL - Installation Test")
    print("=" * 50)
    
    # Test 1: Core ML libraries
    print("\n1. Testing core ML libraries...")
    try:
        import torch
        import torchvision
        import tensorflow as tf
        print(f"   ✅ PyTorch {torch.__version__}")
        print(f"   ✅ TorchVision {torchvision.__version__}")
        print(f"   ✅ TensorFlow {tf.__version__}")
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA {torch.version.cuda} - {torch.cuda.device_count()} GPU(s)")
            print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️  CUDA not available - will use CPU")
            
    except Exception as e:
        print(f"   ❌ Core ML libraries failed: {e}")
        return False
    
    # Test 2: YOLO integration
    print("\n2. Testing YOLO integration...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov5s.pt')
        print("   ✅ YOLO model loaded successfully")
        
        import numpy as np
        test_img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        results = model(test_img, verbose=False)
        print("   ✅ YOLO inference working")
        
    except Exception as e:
        print(f"   ❌ YOLO integration failed: {e}")
        return False
    
    # Test 3: gym-duckietown
    print("\n3. Testing gym-duckietown...")
    try:
        import gym
        import gym_duckietown
        env = gym.make('Duckietown-loop_empty-v0')
        obs = env.reset()
        print(f"   ✅ Environment created - obs shape: {obs.shape}")
        env.close()
        
    except Exception as e:
        print(f"   ❌ gym-duckietown failed: {e}")
        return False
    
    # Test 4: Project components (if available)
    print("\n4. Testing project components...")
    try:
        sys.path.insert(0, '.')
        
        # Test enhanced config
        from config.enhanced_config import load_enhanced_config
        config = load_enhanced_config()
        print("   ✅ Enhanced configuration loaded")
        
        # Test YOLO utilities
        from duckietown_utils.yolo_utils import create_yolo_inference_system
        yolo_system = create_yolo_inference_system('yolov5s.pt')
        print("   ✅ YOLO utilities working")
        
    except Exception as e:
        print(f"   ⚠️  Project components: {e}")
        print("   (This is expected if project files are not present)")
    
    print("\n" + "=" * 50)
    print("🎉 Installation test completed successfully!")
    print("\nYou're ready to run the enhanced RL system!")
    print("\nNext steps:")
    print("1. Activate environment: source activate_enhanced_env.sh")
    print("2. Run training: python enhanced_rl_training_system.py")
    print("3. Or run simple version: python train_enhanced_rl_simple.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

chmod +x test_installation.py
print_success "Test script created: test_installation.py"

# Final summary
echo ""
echo "🎉 SETUP COMPLETED SUCCESSFULLY!"
echo "================================="
echo ""
echo "Environment name: $ENV_NAME"
echo "Activation script: ./activate_enhanced_env.sh"
echo "Test script: ./test_installation.py"
echo ""
echo "Next steps:"
echo "1. Activate the environment:"
echo "   source ./activate_enhanced_env.sh"
echo ""
echo "2. Test the installation:"
echo "   python test_installation.py"
echo ""
echo "3. Run the enhanced RL training:"
echo "   python enhanced_rl_training_system.py"
echo ""
echo "4. Or start with the simple version:"
echo "   python train_enhanced_rl_simple.py"
echo ""
echo "📚 Documentation:"
echo "   - Setup guide: SETUP_GUIDE.md"
echo "   - Enhanced system: ENHANCED_RL_SYSTEM_README.md"
echo "   - Deployment: DUCKIETOWN_RL_DEPLOYMENT_GUIDE.md"
echo ""
print_success "Happy training! 🚀"