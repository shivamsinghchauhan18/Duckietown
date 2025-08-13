# Enhanced Duckietown RL Setup Guide

This guide explains how to set up the Enhanced Duckietown RL environment with YOLO v5 object detection integration.

## Prerequisites

- **For Conda Setup**: Miniconda or Anaconda installed
- **For Docker Setup**: Docker and nvidia-docker (for GPU support)
- **Hardware**: NVIDIA GPU recommended for optimal performance

## Option 1: Conda Environment Setup (Recommended)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Duckietown-RL
```

### 2. Run the Enhanced Setup Script

```bash
./setup_enhanced_environment.sh
```

This script will:
- Create a new conda environment `dtaido5` with all dependencies
- Install gym-duckietown v6.0.25
- Install YOLO v5 dependencies (ultralytics, torch, torchvision)
- Copy custom maps to the appropriate locations
- Test the YOLO integration

### 3. Activate the Environment

```bash
conda activate dtaido5
```

### 4. Test the Installation

```bash
# Test YOLO integration
python examples/yolo_integration_example.py

# Run unit tests
python -m pytest tests/ -v

# Test YOLO wrapper specifically
python -m unittest tests.test_yolo_detection_wrapper -v
```

## Option 2: Docker Setup

### 1. Build the Enhanced Docker Image

```bash
docker build -f Dockerfile.enhanced -t duckietown-rl-enhanced .
```

### 2. Run the Container

**With GPU support (recommended):**
```bash
docker run -it --gpus all \
  -p 2222:22 \
  -p 7000:7000 \
  -p 7001:7001 \
  -p 8888:8888 \
  --name duckietown-enhanced \
  duckietown-rl-enhanced
```

**CPU only:**
```bash
docker run -it \
  -p 2222:22 \
  -p 7000:7000 \
  -p 7001:7001 \
  -p 8888:8888 \
  --name duckietown-enhanced \
  duckietown-rl-enhanced
```

### 3. Connect to the Container

```bash
# SSH into the container
ssh duckie@localhost -p 2222
# Password: dt2020

# Or use docker exec
docker exec -it duckietown-enhanced bash
```

## Verifying the Installation

### 1. Check YOLO Integration

```python
from duckietown_utils.yolo_utils import create_yolo_inference_system
from duckietown_utils.wrappers import YOLOObjectDetectionWrapper

# Test YOLO system creation
yolo_system = create_yolo_inference_system('yolov5s.pt')
print(f"YOLO system created: {yolo_system is not None}")
```

### 2. Test Object Detection Wrapper

```python
import gym
import numpy as np
from duckietown_utils.wrappers import YOLOObjectDetectionWrapper

# Create a mock environment for testing
class MockEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
    
    def reset(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def step(self, action):
        obs = np.zeros((480, 640, 3), dtype=np.uint8)
        return obs, 0.0, False, {}

# Test wrapper
env = MockEnv()
wrapped_env = YOLOObjectDetectionWrapper(env)
obs = wrapped_env.reset()
print(f"Wrapped observation keys: {obs.keys()}")
```

## Environment Details

### Conda Environment (`dtaido5`)

The conda environment includes:
- **Python 3.6.12**
- **TensorFlow 2.0.0** with GPU support
- **PyTorch** (latest compatible version)
- **Ultralytics YOLOv5** for object detection
- **gym-duckietown v6.0.25**
- **Ray RLLib 0.8.2** for RL training
- All original dependencies from the base environment

### Key Dependencies for YOLO Integration

```yaml
# Added to environment_aido5.yml
- torch>=1.7.0
- torchvision>=0.8.0  
- ultralytics>=8.0.0
```

## Usage Examples

### 1. Basic YOLO Integration

```bash
python examples/yolo_integration_example.py
```

### 2. Enhanced RL Training

```bash
python experiments/train_enhanced_rl.py --config config/enhanced_config.yaml
```

### 3. Running Tests

```bash
# All tests
python -m pytest tests/ -v

# YOLO-specific tests
python -m unittest tests.test_yolo_detection_wrapper -v
python -m unittest tests.test_yolo_utils -v
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU for YOLO inference
   export YOLO_DEVICE=cpu
   ```

2. **ultralytics Import Error**
   ```bash
   conda activate dtaido5
   pip install ultralytics>=8.0.0
   ```

3. **gym-duckietown Import Error**
   ```bash
   # Reinstall gym-duckietown
   pip install -e gym-duckietown
   ```

4. **Missing Custom Maps**
   ```bash
   python -m maps.copy_custom_maps_to_duckietown_libs
   ```

### Performance Optimization

1. **GPU Usage**: Ensure CUDA is properly installed and accessible
2. **Model Selection**: Use `yolov5s.pt` for faster inference, `yolov5x.pt` for better accuracy
3. **Image Resolution**: Reduce input image size if needed for real-time performance
4. **Batch Processing**: Process multiple images together when possible

### Environment Variables

```bash
# Force CPU usage for YOLO
export YOLO_DEVICE=cpu

# Set YOLO model path
export YOLO_MODEL_PATH=/path/to/custom/model.pt

# Enable debug logging
export YOLO_DEBUG=1
```

## Development Workflow

### 1. Making Changes to YOLO Integration

```bash
# Edit the wrapper
vim duckietown_utils/wrappers/yolo_detection_wrapper.py

# Run tests
python -m unittest tests.test_yolo_detection_wrapper -v

# Test with example
python examples/yolo_integration_example.py
```

### 2. Adding New Features

1. Update the wrapper implementation
2. Add corresponding unit tests
3. Update documentation
4. Test with both conda and Docker environments

### 3. Performance Profiling

```python
# Profile YOLO inference
from duckietown_utils.yolo_utils import create_yolo_inference_system
import time

yolo_system = create_yolo_inference_system('yolov5s.pt')
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

start_time = time.time()
results = yolo_system.detect_objects(image)
inference_time = time.time() - start_time

print(f"Inference time: {inference_time:.3f}s")
print(f"FPS: {1/inference_time:.1f}")
```

## Next Steps

After successful setup:

1. **Explore Examples**: Run the provided examples to understand the integration
2. **Read Documentation**: Check `docs/YOLO_Integration.md` for detailed API documentation
3. **Run Training**: Start enhanced RL training with object detection capabilities
4. **Customize**: Modify the YOLO wrapper for your specific use case

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the test files for usage examples
3. Consult the API documentation in `docs/`
4. Check the original Duckietown documentation for base environment issues