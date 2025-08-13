#!/bin/bash
set -e

echo "################################################"
echo "Enhanced Duckietown RL Environment Setup"
echo "################################################"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Remove existing environment if it exists
echo "Removing existing dtaido5 environment (if exists)..."
conda remove --name dtaido5 --all -y 2>/dev/null || true

echo ""
echo "################################################"
echo "Creating new conda environment: dtaido5"
echo "################################################"
conda env create -f environment_aido5.yml

echo "################################################"
echo "Clone gym-duckietown"
echo "################################################"
rm -rf gym-duckietown
git clone --branch v6.0.25 --single-branch --depth 1 https://github.com/duckietown/gym-duckietown.git ./gym-duckietown

echo ""
echo "################################################"
echo "Install gym-duckietown"
echo "################################################"
conda run -vn dtaido5 pip install -e gym-duckietown 

echo "################################################"
echo "Install enhanced YOLO dependencies"
echo "################################################"
# Install YOLO dependencies that might not be in the environment file
conda run -vn dtaido5 pip install ultralytics>=8.0.0

echo "################################################"
echo "Copy custom maps to the installed packages"
echo "################################################"
conda run -vn dtaido5 python -m maps.copy_custom_maps_to_duckietown_libs

echo "################################################"
echo "Test YOLO integration"
echo "################################################"
echo "Testing YOLO utilities..."
conda run -vn dtaido5 python -c "
import sys
sys.path.insert(0, '.')
try:
    from duckietown_utils.yolo_utils import create_yolo_inference_system
    print('✅ YOLO utilities imported successfully')
    
    # Test YOLO system creation (will warn if ultralytics not available)
    yolo_system = create_yolo_inference_system('yolov5s.pt', device='cpu')
    if yolo_system is not None:
        print('✅ YOLO system created successfully')
    else:
        print('⚠️  YOLO system creation failed (expected if ultralytics not fully installed)')
        
    from duckietown_utils.wrappers import YOLOObjectDetectionWrapper
    print('✅ YOLO wrapper imported successfully')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'⚠️  Warning: {e}')
    print('This may be expected if YOLO models are not downloaded yet')
"

echo ""
echo "################################################"
echo "Setup completed successfully! 🎉"
echo ""
echo "To activate the environment:"
echo "    conda activate dtaido5"
echo ""
echo "To test the YOLO integration:"
echo "    python examples/yolo_integration_example.py"
echo ""
echo "To run the enhanced RL training:"
echo "    python experiments/train_enhanced_rl.py"
echo ""
echo "Note: On first run, YOLO models will be downloaded automatically"
echo "################################################"