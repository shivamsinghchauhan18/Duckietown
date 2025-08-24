# 🚀 Enhanced Duckietown RL - Tier 1 Setup

**One-command setup for the complete enhanced RL system with YOLO integration**

## 🎯 Quick Start

### For WSL2 Ubuntu (Recommended - Fastest Setup)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-name>

# 2. Run WSL-optimized setup (5-10 minutes)
chmod +x wsl_quick_setup.sh
./wsl_quick_setup.sh

# 3. Activate the environment
source activate_wsl_env.sh

# 4. Test everything works
python test_wsl_installation.py

# 5. Start training!
python complete_enhanced_rl_pipeline.py --mode full
```

### For Native Ubuntu (Alternative)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <repo-name>

# 2. Run the setup script (takes 10-15 minutes)
chmod +x setup_tier1_environment.sh
./setup_tier1_environment.sh

# 3. Activate the environment
source activate_enhanced_env.sh

# 4. Test everything works
python test_installation.py

# 5. Start training!
python enhanced_rl_training_system.py
```

That's it! 🎉

## 🖥️ System Requirements

This setup is optimized for **Tier 1 hardware**:
- **GPU**: NVIDIA RTX 3080/4080/4090 or better
- **OS**: Ubuntu 20.04/22.04 LTS
- **CUDA**: 11.8+ with cuDNN 8.7+
- **RAM**: 16GB+ (32GB recommended)
- **Storage**: 50GB+ free space

## 📦 What Gets Installed

The environment includes everything needed:

### Core ML/DL Stack
- **PyTorch 2.0.1** with CUDA 11.8 support
- **TensorFlow 2.10.0** with GPU support
- **Ultralytics YOLO v8** for object detection
- **Ray RLLib 2.5.1** for distributed RL

### Duckietown Ecosystem
- **gym-duckietown 6.0.25** (official simulator)
- **Custom maps** and environments
- **Enhanced wrappers** for YOLO integration

### Development Tools
- **Jupyter Lab** for interactive development
- **TensorBoard** for training visualization
- **pytest** for testing
- **Black/Flake8** for code formatting

### Performance Optimization
- **CUDA-optimized** PyTorch and TensorFlow
- **Multi-GPU** support ready
- **Memory profiling** tools
- **Performance monitoring**

## 🧪 Testing the Installation

After setup, run the test script:

```bash
python test_installation.py
```

This will verify:
- ✅ PyTorch + CUDA working
- ✅ YOLO models downloading and running
- ✅ gym-duckietown environment creation
- ✅ All project components loading

## 🚀 Running the Enhanced System

### Option 1: Full Enhanced System (Recommended)
```bash
# Complete system with YOLO, object avoidance, lane changing
python enhanced_rl_training_system.py

# With custom config
python enhanced_rl_training_system.py --config config/enhanced_rl_champion_config.yml

# Evaluation only
python enhanced_rl_training_system.py --eval-only
```

### Option 2: Simple Version (For Testing)
```bash
# Simplified version for quick validation
python train_enhanced_rl_simple.py
```

### Option 3: Master Orchestrator (Advanced)
```bash
# Full orchestrator with multiple training strategies
python master_rl_orchestrator.py
```

## 📊 Training Performance

On Tier 1 hardware, expect:
- **Training Speed**: ~2000 FPS with GPU acceleration
- **YOLO Inference**: ~50 FPS (20ms per frame)
- **Memory Usage**: ~8-12GB GPU VRAM
- **Training Time**: 3-6 hours for 5M timesteps

## 🔧 Configuration

The system auto-detects your hardware and optimizes settings:

```yaml
# Automatically configured based on your GPU
training:
  use_cuda: true
  device: "cuda:0"
  batch_size: 256  # Optimized for RTX 3080+
  
yolo:
  device: "0"      # Use GPU 0
  model: "yolov5s.pt"  # Fast inference
  
performance:
  num_workers: 8   # Based on CPU cores
  pin_memory: true
```

## 📁 Project Structure

After setup, you'll have:

```
├── enhanced_rl_training_system.py    # Main training system
├── train_enhanced_rl_simple.py       # Simplified version
├── master_rl_orchestrator.py         # Advanced orchestrator
├── config/                           # Configuration files
├── duckietown_utils/                 # Enhanced utilities
├── models/                           # Trained models
├── logs/                            # Training logs
├── gym-duckietown/                  # Simulator (auto-installed)
└── activate_enhanced_env.sh         # Environment activation
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in config
export CUDA_VISIBLE_DEVICES=0
python enhanced_rl_training_system.py --batch-size 128
```

**2. gym-duckietown Display Issues**
```bash
# For headless systems
export DISPLAY=:0
xvfb-run -a python enhanced_rl_training_system.py
```

**3. YOLO Model Download Fails**
```bash
# Manual download
python -c "from ultralytics import YOLO; YOLO('yolov5s.pt')"
```

### Performance Optimization

**For RTX 4090 (24GB VRAM):**
```bash
# Use larger models and batch sizes
python enhanced_rl_training_system.py \
  --batch-size 512 \
  --yolo-model yolov5l.pt
```

**For RTX 3080 (10GB VRAM):**
```bash
# Optimized settings (default)
python enhanced_rl_training_system.py \
  --batch-size 256 \
  --yolo-model yolov5s.pt
```

## 📈 Monitoring Training

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir logs/

# Open browser to http://localhost:6006
```

### Real-time Monitoring
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f logs/enhanced_rl_training.log
```

## 🎯 Expected Results

After successful training, you should see:
- **Success Rate**: 85-95% on loop_empty
- **Lane Following**: Smooth center-line tracking
- **Object Avoidance**: Reliable obstacle detection and avoidance
- **Inference Speed**: <50ms per decision

## 🚀 Next Steps

1. **Train your first model** (3-6 hours)
2. **Evaluate on different maps**
3. **Deploy to real Duckiebot** (see deployment guide)
4. **Experiment with hyperparameters**
5. **Add custom reward functions**

## 📞 Support

If you encounter issues:
1. Check the test script output: `python test_installation.py`
2. Review logs in `logs/` directory
3. Check GPU memory: `nvidia-smi`
4. Verify CUDA installation: `nvcc --version`

## 🎉 Success Indicators

You'll know everything is working when:
- ✅ Test script passes all checks
- ✅ Training starts without errors
- ✅ GPU utilization shows ~80-90%
- ✅ YOLO detections appear in logs
- ✅ Reward increases over time

**Happy training! 🚀**