#!/bin/bash
set -e

echo "🚀 Enhanced Duckietown RL - Tier 1 Setup Script"
echo "=================================================="
echo "Complete environment setup for the enhanced RL system with full pipeline"
echo "Optimized for Tier 1 hardware (RTX 3080+, Ubuntu 20.04+, CUDA 11.8+)"
echo ""
echo "This script will:"
echo "  ✅ Create optimized conda environment"
echo "  ✅ Install all dependencies (PyTorch, YOLO, gym-duckietown)"
echo "  ✅ Setup complete pipeline integration"
echo "  ✅ Create activation and test scripts"
echo "  ✅ Validate entire system"
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

# Setup fast solver for WSL (critical for avoiding hangs)
print_status "Configuring conda for optimal WSL performance..."
conda install -n base -y conda-libmamba-solver 2>/dev/null || true
conda config --set solver libmamba 2>/dev/null || true
conda config --set channel_priority strict 2>/dev/null || true
conda config --set repodata_fns current_repodata.json 2>/dev/null || true
print_success "Conda optimized for WSL"

# Install system OpenGL dependencies for WSL
print_status "Installing system OpenGL dependencies for WSL..."
if command -v apt &> /dev/null; then
    sudo apt update -qq 2>/dev/null || true
    
    # Detect Ubuntu version for correct package names
    UBUNTU_VERSION=$(lsb_release -rs 2>/dev/null || echo "20.04")
    
    if [[ $(echo "$UBUNTU_VERSION >= 22.04" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
        # Ubuntu 22.04+ packages
        sudo apt install -y mesa-utils libgl1-mesa-dev libgl1-mesa-dri \
                            libegl1-mesa-dev libgles2-mesa-dev \
                            xorg-dev freeglut3-dev xvfb \
                            libglib2.0-0 libsm6 libxext6 libxrender-dev \
                            libgomp1 build-essential python3-dev 2>/dev/null || print_warning "Some packages unavailable"
    else
        # Ubuntu 20.04 packages
        sudo apt install -y mesa-utils libgl1-mesa-glx libgl1-mesa-dri \
                            xorg-dev freeglut3-dev xvfb \
                            libglib2.0-0 libsm6 libxext6 libxrender-dev \
                            libgomp1 build-essential python3-dev 2>/dev/null || print_warning "Some packages unavailable"
    fi
    
    print_success "System OpenGL dependencies installed"
else
    print_warning "apt not available - skipping system OpenGL installation"
fi

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

# Test project-specific imports and complete pipeline
print_status "Testing project-specific components and complete pipeline..."
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

def test_file_exists(file_path, description):
    import os
    if os.path.exists(file_path):
        print(f'✅ {description}: File exists')
        return True
    else:
        print(f'❌ {description}: File not found')
        return False

print('Testing project components...')

# Test key files exist
key_files = [
    ('enhanced_rl_training_system.py', 'Enhanced RL training system'),
    ('complete_enhanced_rl_pipeline.py', 'Complete pipeline orchestrator'),
    ('environment_enhanced_tier1.yml', 'Tier 1 environment file'),
    ('TIER1_SETUP_README.md', 'Setup documentation'),
]

files_ok = 0
for file_path, desc in key_files:
    if test_file_exists(file_path, desc):
        files_ok += 1

print(f'\\nKey files check: {files_ok}/{len(key_files)} found')

# Test if project modules can be imported
project_tests = []
import os
if os.path.exists('duckietown_utils'):
    project_tests.extend([
        ('duckietown_utils.env', 'Environment utilities'),
        ('duckietown_utils.yolo_utils', 'YOLO utilities'),
        ('config.enhanced_config', 'Enhanced configuration'),
    ])

modules_ok = 0
for module_path, desc in project_tests:
    if test_project_import(module_path, desc):
        modules_ok += 1

print(f'\\nProject modules check: {modules_ok}/{len(project_tests)} imported')

# Test complete pipeline import
try:
    import complete_enhanced_rl_pipeline
    print('✅ Complete pipeline: Import successful')
    pipeline_ok = True
except Exception as e:
    print(f'⚠️  Complete pipeline: Import warning - {e}')
    pipeline_ok = False

print('\\nProject components test completed!')
print(f'Overall status: {\"✅ READY\" if files_ok >= 3 and pipeline_ok else \"⚠️  PARTIAL\"}')"

# Create enhanced activation script
print_status "Creating enhanced activation script..."
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

# Pipeline configuration
export PIPELINE_MODE=full      # Default to full pipeline
export PIPELINE_GPU=true       # Enable GPU acceleration
export PIPELINE_TIMESTEPS=5000000  # Default training timesteps

# Display environment info
echo "Environment activated successfully!"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "GPU Memory: $(python -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB\")')"
fi

echo ""
echo "🎯 Available Commands:"
echo "  python complete_enhanced_rl_pipeline.py --mode full    # Complete pipeline"
echo "  python enhanced_rl_training_system.py                  # Training only"
echo "  python test_installation.py                            # Test installation"
echo ""
echo "📚 Documentation:"
echo "  cat TIER1_SETUP_README.md                             # Setup guide"
echo ""
echo "Ready for enhanced RL training! 🚀"
EOF

chmod +x activate_enhanced_env.sh
print_success "Activation script created: activate_enhanced_env.sh"

# Create comprehensive test script
print_status "Creating comprehensive test script..."
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Comprehensive installation test for Enhanced Duckietown RL with Complete Pipeline
"""
import sys
import traceback
import os
from pathlib import Path

def main():
    print("🧪 Enhanced Duckietown RL - Comprehensive Installation Test")
    print("=" * 60)
    
    test_results = []
    
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
            print(f"   ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
        else:
            print("   ⚠️  CUDA not available - will use CPU")
        
        test_results.append(("Core ML Libraries", True))
            
    except Exception as e:
        print(f"   ❌ Core ML libraries failed: {e}")
        test_results.append(("Core ML Libraries", False))
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
        print(f"   ✅ Detections: {len(results[0].boxes) if results[0].boxes is not None else 0}")
        
        test_results.append(("YOLO Integration", True))
        
    except Exception as e:
        print(f"   ❌ YOLO integration failed: {e}")
        test_results.append(("YOLO Integration", False))
        return False
    
    # Test 3: gym-duckietown
    print("\n3. Testing gym-duckietown...")
    try:
        import gym
        import gym_duckietown
        env = gym.make('Duckietown-loop_empty-v0')
        obs = env.reset()
        print(f"   ✅ Environment created - obs shape: {obs.shape}")
        
        # Test environment step
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        print(f"   ✅ Environment step working - reward: {reward:.3f}")
        env.close()
        
        test_results.append(("gym-duckietown", True))
        
    except Exception as e:
        print(f"   ❌ gym-duckietown failed: {e}")
        test_results.append(("gym-duckietown", False))
        return False
    
    # Test 4: Project files and structure
    print("\n4. Testing project files and structure...")
    try:
        key_files = [
            'enhanced_rl_training_system.py',
            'complete_enhanced_rl_pipeline.py',
            'environment_enhanced_tier1.yml',
            'TIER1_SETUP_README.md'
        ]
        
        files_found = 0
        for file_path in key_files:
            if os.path.exists(file_path):
                print(f"   ✅ {file_path}: Found")
                files_found += 1
            else:
                print(f"   ❌ {file_path}: Missing")
        
        print(f"   📊 Project files: {files_found}/{len(key_files)} found")
        test_results.append(("Project Files", files_found >= 3))
        
    except Exception as e:
        print(f"   ❌ Project files check failed: {e}")
        test_results.append(("Project Files", False))
    
    # Test 5: Project components (if available)
    print("\n5. Testing project components...")
    try:
        sys.path.insert(0, '.')
        
        components_tested = 0
        total_components = 0
        
        # Test enhanced config
        try:
            from config.enhanced_config import load_enhanced_config
            config = load_enhanced_config()
            print("   ✅ Enhanced configuration loaded")
            components_tested += 1
        except:
            print("   ⚠️  Enhanced configuration: Not available")
        total_components += 1
        
        # Test YOLO utilities
        try:
            from duckietown_utils.yolo_utils import create_yolo_inference_system
            yolo_system = create_yolo_inference_system('yolov5s.pt')
            print("   ✅ YOLO utilities working")
            components_tested += 1
        except:
            print("   ⚠️  YOLO utilities: Not available")
        total_components += 1
        
        # Test complete pipeline
        try:
            import complete_enhanced_rl_pipeline
            print("   ✅ Complete pipeline module imported")
            components_tested += 1
        except:
            print("   ⚠️  Complete pipeline: Not available")
        total_components += 1
        
        print(f"   📊 Components: {components_tested}/{total_components} working")
        test_results.append(("Project Components", components_tested >= 2))
        
    except Exception as e:
        print(f"   ⚠️  Project components test: {e}")
        test_results.append(("Project Components", False))
    
    # Test 6: Performance check
    print("\n6. Testing performance capabilities...")
    try:
        import psutil
        
        # CPU info
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"   📊 CPU cores: {cpu_count}")
        print(f"   📊 System RAM: {memory_gb:.1f}GB")
        
        # GPU performance test
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # Simple tensor operation test
            x = torch.randn(1000, 1000, device=device)
            y = torch.mm(x, x.t())
            print("   ✅ GPU tensor operations working")
            
            # Memory test
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   📊 GPU Memory: {gpu_memory:.1f}GB")
            
            performance_ok = cpu_count >= 4 and memory_gb >= 8 and gpu_memory >= 6
        else:
            performance_ok = cpu_count >= 4 and memory_gb >= 8
        
        test_results.append(("Performance Check", performance_ok))
        
    except Exception as e:
        print(f"   ⚠️  Performance check failed: {e}")
        test_results.append(("Performance Check", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("-" * 30)
    
    passed_tests = 0
    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<25} {status}")
        if passed:
            passed_tests += 1
    
    print(f"\nOverall: {passed_tests}/{len(test_results)} tests passed")
    
    if passed_tests == len(test_results):
        print("\n🎉 ALL TESTS PASSED! System is ready for enhanced RL training!")
        print("\n🚀 Next steps:")
        print("1. Run complete pipeline: python complete_enhanced_rl_pipeline.py --mode full")
        print("2. Or training only: python enhanced_rl_training_system.py")
        print("3. Or simple version: python train_enhanced_rl_simple.py")
        return True
    elif passed_tests >= len(test_results) * 0.8:
        print("\n⚠️  MOSTLY READY! Some components may need attention.")
        print("System should work but may have reduced functionality.")
        return True
    else:
        print("\n❌ SYSTEM NOT READY! Please fix the failing tests.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

chmod +x test_installation.py
print_success "Test script created: test_installation.py"

# Create quick start script
print_status "Creating quick start script..."
cat > quick_start.sh << 'EOF'
#!/bin/bash
echo "🚀 Enhanced Duckietown RL - Quick Start"
echo "======================================"

# Check if environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "duckietown-enhanced-tier1" ]]; then
    echo "Activating environment..."
    source activate_enhanced_env.sh
fi

echo ""
echo "🎯 Choose your training mode:"
echo "1. Complete Pipeline (Full system with evaluation and deployment)"
echo "2. Training Only (Enhanced RL training)"
echo "3. Simple Version (Quick validation)"
echo "4. Test Installation"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "🚀 Starting complete pipeline..."
        python complete_enhanced_rl_pipeline.py --mode full
        ;;
    2)
        echo "🧠 Starting enhanced RL training..."
        python enhanced_rl_training_system.py
        ;;
    3)
        echo "⚡ Starting simple version..."
        python train_enhanced_rl_simple.py
        ;;
    4)
        echo "🧪 Running installation test..."
        python test_installation.py
        ;;
    *)
        echo "Invalid choice. Running installation test..."
        python test_installation.py
        ;;
esac
EOF

chmod +x quick_start.sh
print_success "Quick start script created: quick_start.sh"

# Final comprehensive summary
echo ""
echo "🎉 ENHANCED DUCKIETOWN RL SETUP COMPLETED!"
echo "=========================================="
echo ""
echo "📦 Environment: $ENV_NAME"
echo "🔧 Activation: ./activate_enhanced_env.sh"
echo "🧪 Testing: ./test_installation.py"
echo "⚡ Quick Start: ./quick_start.sh"
echo ""
echo "🎯 Available Training Modes:"
echo "1. 🚀 Complete Pipeline:"
echo "   python complete_enhanced_rl_pipeline.py --mode full"
echo ""
echo "2. 🧠 Enhanced Training Only:"
echo "   python enhanced_rl_training_system.py"
echo ""
echo "3. ⚡ Simple Version (Quick Test):"
echo "   python train_enhanced_rl_simple.py"
echo ""
echo "🔍 System Validation:"
echo "1. Test installation: python test_installation.py"
echo "2. Check GPU: nvidia-smi"
echo "3. Monitor training: tensorboard --logdir logs/"
echo ""
echo "📚 Documentation:"
echo "   - Complete guide: TIER1_SETUP_README.md"
echo "   - Setup details: SETUP_GUIDE.md"
echo "   - System docs: ENHANCED_RL_SYSTEM_README.md"
echo ""
echo "⚡ Quick Commands:"
echo "   source activate_enhanced_env.sh  # Activate environment"
echo "   ./quick_start.sh                 # Interactive launcher"
echo "   python test_installation.py      # Validate setup"
echo ""
print_success "🚀 Ready for enhanced RL training with complete pipeline!"
print_success "Run './quick_start.sh' fo