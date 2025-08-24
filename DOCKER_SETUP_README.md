# 🐳 Enhanced Duckietown RL - Docker Setup

**Zero-hassle Docker setup that eliminates all conda conflicts!**

## 🎯 Why Docker?

- **No conda issues**: Pre-built environment with all dependencies
- **One-command setup**: Works on any system with Docker
- **GPU support**: Automatic NVIDIA Docker integration
- **Consistent results**: Same environment everywhere
- **Easy sharing**: Your friend just runs one script

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Docker installed on your system
- NVIDIA Docker (optional, for GPU support)

### Setup Commands
```bash
# 1. Make setup script executable
chmod +x docker_setup.sh

# 2. Run setup (builds Docker image)
./docker_setup.sh

# 3. Test the environment
./test_docker.sh

# 4. Start training!
./train_simple_docker.sh
```

That's it! No conda, no conflicts, no hassle! 🎉

## 📋 What Gets Created

After running `./docker_setup.sh`, you'll have:

```
├── Dockerfile.wsl              # Docker image definition
├── docker-compose.wsl.yml     # Docker Compose configuration
├── run_docker.sh              # Interactive Docker shell
├── test_docker.sh             # Test environment
├── train_simple_docker.sh     # Simple RL training
└── train_pipeline_docker.sh   # Complete pipeline
```

## 🎮 Usage Examples

### Interactive Development
```bash
# Start interactive Docker shell
./run_docker.sh

# Inside container:
python train_enhanced_rl_simple.py
python complete_enhanced_rl_pipeline.py --mode full
```

### Direct Training
```bash
# Simple training (recommended first)
./train_simple_docker.sh

# Complete pipeline
./train_pipeline_docker.sh
```

### Testing and Validation
```bash
# Test all components
./test_docker.sh

# Check GPU support
docker run --rm --gpus all duckietown-rl-enhanced:latest nvidia-smi
```

## 🔧 Docker Image Details

### Base Image
- **PyTorch 2.0.1** with CUDA 11.7 support
- **Ubuntu 20.04** base system
- **Pre-installed dependencies**: All Python packages ready

### Included Packages
- **YOLO**: Ultralytics YOLOv5/v8 for object detection
- **RL Stack**: Gymnasium + Stable Baselines3 + Ray RLLib
- **Duckietown**: gym-duckietown v6.0.25
- **ML Tools**: PyTorch, NumPy, OpenCV, Matplotlib
- **Development**: Jupyter, TensorBoard, Wandb, MLflow

### GPU Support
- **Automatic detection**: Script detects NVIDIA GPUs
- **CUDA acceleration**: Full GPU training support
- **Fallback**: CPU-only mode if no GPU available

## 🎯 Performance Expectations

### With GPU (RTX 3080+)
- **Training Speed**: ~2000 FPS
- **YOLO Inference**: ~50 FPS
- **Memory Usage**: ~8-12GB GPU VRAM
- **Training Time**: 3-6 hours for 5M timesteps

### CPU Only
- **Training Speed**: ~200 FPS
- **YOLO Inference**: ~5 FPS
- **Memory Usage**: ~8GB RAM
- **Training Time**: 30-60 hours for 5M timesteps

## 🔍 Troubleshooting

### Docker Issues
```bash
# If Docker daemon not running
sudo systemctl start docker

# If permission denied
sudo usermod -aG docker $USER
# Then log out and back in

# If build fails
docker system prune -a  # Clean up
./docker_setup.sh       # Try again
```

### GPU Issues
```bash
# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi

# Install NVIDIA Docker (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Display Issues (WSL)
```bash
# Install X11 server on Windows (VcXsrv or Xming)
# Then in WSL:
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0

# Or use headless mode:
xvfb-run -a ./train_simple_docker.sh
```

## 🎯 Advantages Over Conda

| Aspect | Conda | Docker |
|--------|-------|--------|
| **Setup Time** | 30+ minutes (often fails) | 5-10 minutes (reliable) |
| **Conflicts** | Frequent dependency issues | Zero conflicts |
| **Reproducibility** | Environment drift | Identical everywhere |
| **Sharing** | Complex environment files | Single Docker image |
| **GPU Support** | Manual CUDA setup | Automatic detection |
| **Isolation** | Affects host system | Completely isolated |

## 🚀 Advanced Usage

### Custom Training
```bash
# Run with custom parameters
docker run -it --rm --gpus all \
    -v $(pwd):/workspace \
    duckietown-rl-enhanced:latest \
    python complete_enhanced_rl_pipeline.py \
    --mode training-only \
    --timesteps 1000000 \
    --gpu-id 0
```

### Development Mode
```bash
# Mount source code for development
docker run -it --rm --gpus all \
    -v $(pwd):/workspace \
    -v $(pwd)/duckietown_utils:/workspace/duckietown_utils \
    -p 8888:8888 \
    duckietown-rl-enhanced:latest \
    jupyter lab --ip=0.0.0.0 --allow-root
```

### Model Export
```bash
# Export trained models
docker run -it --rm \
    -v $(pwd):/workspace \
    duckietown-rl-enhanced:latest \
    python -c "
from complete_enhanced_rl_pipeline import CompletePipeline
# Export models to ONNX, TensorFlow Lite, etc.
"
```

## 📊 Expected Results

After successful setup and training:

```bash
🧪 Testing Docker Environment
✅ PyTorch: 2.0.1+cu117
✅ CUDA available: True
✅ NumPy: 1.24.3
✅ Gymnasium: 0.29.1
✅ YOLO: Model loaded
✅ gym-duckietown: Environment created, obs shape: (120, 160, 3)
🎉 Docker environment test completed!
```

## 🎉 Success Indicators

You'll know everything is working when:
- ✅ `./test_docker.sh` passes all tests
- ✅ GPU utilization shows ~80-90% during training
- ✅ Training starts without import errors
- ✅ YOLO detections appear in logs
- ✅ Models are saved to `models/` directory

## 💡 Pro Tips

1. **Use volumes**: Models and logs persist between container runs
2. **Monitor resources**: Use `docker stats` to monitor usage
3. **Clean up**: Run `docker system prune` occasionally
4. **Backup models**: Copy important models to host system
5. **Scale up**: Use Docker Swarm for multi-GPU training

**Docker eliminates all the conda headaches and gets you training immediately!** 🚀