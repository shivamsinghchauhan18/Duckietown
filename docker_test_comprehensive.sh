#!/bin/bash
set -e

echo "🧪 COMPREHENSIVE DOCKER ENVIRONMENT TESTING"
echo "==========================================="
echo "Complete validation of all system components"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[TEST]${NC} $1"; }
print_success() { echo -e "${GREEN}[PASS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[FAIL]${NC} $1"; }

# Configuration
DOCKER_IMAGE="duckietown-rl-enhanced:latest"
TEST_RESULTS_FILE="docker_test_results_$(date +%Y%m%d_%H%M%S).log"

# GPU support detection
GPU_ARGS=""
if command -v nvidia-smi &> /dev/null && docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi &> /dev/null 2>&1; then
    GPU_ARGS="--gpus all"
    print_success "GPU support detected"
else
    print_warning "No GPU support - testing CPU mode"
fi

# Function to run test in Docker
run_test() {
    local test_name="$1"
    local test_command="$2"
    local description="$3"
    
    print_status "Testing: $description"
    
    if docker run --rm $GPU_ARGS \
        -v "$(pwd):/workspace" \
        "$DOCKER_IMAGE" \
        bash -c "$test_command" >> "$TEST_RESULTS_FILE" 2>&1; then
        print_success "$test_name"
        echo "✅ $test_name: PASSED" >> "$TEST_RESULTS_FILE"
        return 0
    else
        print_error "$test_name"
        echo "❌ $test_name: FAILED" >> "$TEST_RESULTS_FILE"
        return 1
    fi
}

# Initialize test results
echo "🧪 Comprehensive Docker Environment Test" > "$TEST_RESULTS_FILE"
echo "Started at: $(date)" >> "$TEST_RESULTS_FILE"
echo "=========================================" >> "$TEST_RESULTS_FILE"

# Test counter
total_tests=0
passed_tests=0

# Test 1: Basic System Components
print_status "Phase 1: Basic System Components"
echo "" >> "$TEST_RESULTS_FILE"
echo "PHASE 1: BASIC SYSTEM COMPONENTS" >> "$TEST_RESULTS_FILE"
echo "=================================" >> "$TEST_RESULTS_FILE"

((total_tests++))
if run_test "Python Environment" "python --version && python -c 'import sys; print(f\"Python executable: {sys.executable}\")'" "Python installation and version"; then
    ((passed_tests++))
fi

((total_tests++))
if run_test "PyTorch Installation" "python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")'" "PyTorch with CUDA support"; then
    ((passed_tests++))
fi

((total_tests++))
if run_test "GPU Detection" "python -c 'import torch; print(f\"GPU count: {torch.cuda.device_count()}\"); [print(f\"GPU {i}: {torch.cuda.get_device_name(i)}\") for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print(\"No GPU available\")'" "GPU detection and naming"; then
    ((passed_tests++))
fi

# Test 2: Core ML Libraries
print_status "Phase 2: Core ML Libraries"
echo "" >> "$TEST_RESULTS_FILE"
echo "PHASE 2: CORE ML LIBRARIES" >> "$TEST_RESULTS_FILE"
echo "===========================" >> "$TEST_RESULTS_FILE"

((total_tests++))
if run_test "NumPy" "python -c 'import numpy as np; print(f\"NumPy: {np.__version__}\"); a = np.random.rand(1000, 1000); b = np.dot(a, a.T); print(f\"NumPy computation test: {b.shape}\")'" "NumPy installation and computation"; then
    ((passed_tests++))
fi

((total_tests++))
if run_test "OpenCV" "python -c 'import cv2; print(f\"OpenCV: {cv2.__version__}\"); import numpy as np; img = np.zeros((100, 100, 3), dtype=np.uint8); print(f\"OpenCV test image: {img.shape}\")'" "OpenCV installation and basic operations"; then
    ((passed_tests++))
fi

((total_tests++))
if run_test "Matplotlib" "python -c 'import matplotlib; print(f\"Matplotlib: {matplotlib.__version__}\"); import matplotlib.pyplot as plt; plt.figure(); plt.close(); print(\"Matplotlib plotting test: OK\")'" "Matplotlib plotting library"; then
    ((passed_tests++))
fi

# Test 3: YOLO Integration
print_status "Phase 3: YOLO Integration"
echo "" >> "$TEST_RESULTS_FILE"
echo "PHASE 3: YOLO INTEGRATION" >> "$TEST_RESULTS_FILE"
echo "==========================" >> "$TEST_RESULTS_FILE"

((total_tests++))
if run_test "Ultralytics Import" "python -c 'from ultralytics import YOLO; print(\"Ultralytics YOLO imported successfully\")'" "Ultralytics YOLO library import"; then
    ((passed_tests++))
fi

((total_tests++))
if run_test "YOLO Model Loading" "python -c 'from ultralytics import YOLO; model = YOLO(\"yolov5s.pt\"); print(f\"YOLO model loaded: {type(model)}\")'" "YOLO model loading and initialization"; then
    ((passed_tests++))
fi

((total_tests++))
if run_test "YOLO Inference" "python -c 'from ultralytics import YOLO; import numpy as np; model = YOLO(\"yolov5s.pt\"); img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8); results = model(img, verbose=False); print(f\"YOLO inference completed: {len(results)} results\")'" "YOLO inference on test image"; then
    ((passed_tests++))
fi

# Test 4: RL Libraries
print_status "Phase 4: RL Libraries"
echo "" >> "$TEST_RESULTS_FILE"
echo "PHASE 4: RL LIBRARIES" >> "$TEST_RESULTS_FILE"
echo "======================" >> "$TEST_RESULTS_FILE"

((total_tests++))
if run_test "Gymnasium" "python -c 'import gymnasium as gym; print(f\"Gymnasium: {gym.__version__}\"); env = gym.make(\"CartPole-v1\"); obs = env.reset(); print(f\"Gymnasium test: {obs[0].shape if isinstance(obs, tuple) else obs.shape}\"); env.close()'" "Gymnasium RL environment library"; then
    ((passed_tests++))
fi

((total_tests++))
if run_test "Stable Baselines3" "python -c 'import stable_baselines3; print(f\"Stable Baselines3: {stable_baselines3.__version__}\"); from stable_baselines3 import DQN; print(\"SB3 DQN import: OK\")'" "Stable Baselines3 RL algorithms"; then
    ((passed_tests++))
fi

((total_tests++))
if run_test "Ray RLLib" "python -c 'import ray; print(f\"Ray: {ray.__version__}\"); ray.init(local_mode=True, ignore_reinit_error=True); print(\"Ray initialization: OK\"); ray.shutdown()'" "Ray RLLib distributed RL"; then
    ((passed_tests++))
fi

# Test 5: Duckietown Environment
print_status "Phase 5: Duckietown Environment"
echo "" >> "$TEST_RESULTS_FILE"
echo "PHASE 5: DUCKIETOWN ENVIRONMENT" >> "$TEST_RESULTS_FILE"
echo "===============================" >> "$TEST_RESULTS_FILE"

((total_tests++))
if run_test "gym-duckietown Import" "python -c 'import gym_duckietown; print(\"gym-duckietown imported successfully\")'" "gym-duckietown library import"; then
    ((passed_tests++))
fi

((total_tests++))
if run_test "Duckietown Environment Creation" "python -c 'import gym; import gym_duckietown; env = gym.make(\"Duckietown-loop_empty-v0\"); print(f\"Environment created: {type(env)}\"); obs = env.reset(); print(f\"Observation shape: {obs.shape}\"); env.close()'" "Duckietown environment creation and reset"; then
    ((passed_tests++))
fi

((total_tests++))
if run_test "Duckietown Environment Step" "python -c 'import gym; import gym_duckietown; env = gym.make(\"Duckietown-loop_empty-v0\"); obs = env.reset(); action = env.action_space.sample(); next_obs, reward, done, info = env.step(action); print(f\"Step result: obs={next_obs.shape}, reward={reward:.3f}, done={done}\"); env.close()'" "Duckietown environment stepping"; then
    ((passed_tests++))
fi

# Test 6: Project Components
print_status "Phase 6: Project Components"
echo "" >> "$TEST_RESULTS_FILE"
echo "PHASE 6: PROJECT COMPONENTS" >> "$TEST_RESULTS_FILE"
echo "============================" >> "$TEST_RESULTS_FILE"

((total_tests++))
if run_test "Enhanced Config" "cd /workspace && python -c 'from config.enhanced_config import load_enhanced_config; config = load_enhanced_config(); print(f\"Enhanced config loaded: {type(config)}\")'" "Enhanced configuration system"; then
    ((passed_tests++))
fi

((total_tests++))
if run_test "YOLO Utils" "cd /workspace && python -c 'from duckietown_utils.yolo_utils import create_yolo_inference_system; system = create_yolo_inference_system(\"yolov5s.pt\"); print(f\"YOLO utils working: {system is not None}\")'" "YOLO utility functions"; then
    ((passed_tests++))
fi

((total_tests++))
if run_test "Complete Pipeline" "cd /workspace && python -c 'import complete_enhanced_rl_pipeline; print(\"Complete pipeline module imported successfully\")'" "Complete pipeline module"; then
    ((passed_tests++))
fi

# Test 7: Performance Tests
print_status "Phase 7: Performance Tests"
echo "" >> "$TEST_RESULTS_FILE"
echo "PHASE 7: PERFORMANCE TESTS" >> "$TEST_RESULTS_FILE"
echo "===========================" >> "$TEST_RESULTS_FILE"

((total_tests++))
if run_test "GPU Memory Test" "python -c 'import torch; print(f\"GPU available: {torch.cuda.is_available()}\"); [print(f\"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory/1024**3:.1f}GB\") for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print(\"No GPU memory to test\")'" "GPU memory detection"; then
    ((passed_tests++))
fi

((total_tests++))
if run_test "Tensor Operations" "python -c 'import torch; device = \"cuda\" if torch.cuda.is_available() else \"cpu\"; x = torch.randn(1000, 1000, device=device); y = torch.mm(x, x.t()); print(f\"Tensor operations on {device}: {y.shape}\")'" "GPU/CPU tensor operations"; then
    ((passed_tests++))
fi

((total_tests++))
if run_test "YOLO Performance" "python -c 'from ultralytics import YOLO; import numpy as np; import time; model = YOLO(\"yolov5s.pt\"); img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8); start = time.time(); results = model(img, verbose=False); elapsed = time.time() - start; print(f\"YOLO inference time: {elapsed:.3f}s ({1/elapsed:.1f} FPS)\")'" "YOLO inference performance"; then
    ((passed_tests++))
fi

# Test 8: Integration Tests
print_status "Phase 8: Integration Tests"
echo "" >> "$TEST_RESULTS_FILE"
echo "PHASE 8: INTEGRATION TESTS" >> "$TEST_RESULTS_FILE"
echo "===========================" >> "$TEST_RESULTS_FILE"

((total_tests++))
if run_test "YOLO + Duckietown" "cd /workspace && python -c '
import gym
import gym_duckietown
from ultralytics import YOLO
import numpy as np

# Create environment
env = gym.make(\"Duckietown-loop_empty-v0\")
obs = env.reset()
print(f\"Environment observation: {obs.shape}\")

# Load YOLO
model = YOLO(\"yolov5s.pt\")
print(\"YOLO model loaded\")

# Run YOLO on environment observation
results = model(obs, verbose=False)
print(f\"YOLO + Duckietown integration: {len(results)} results\")

env.close()
print(\"Integration test completed successfully\")
'" "YOLO and Duckietown integration"; then
    ((passed_tests++))
fi

((total_tests++))
if run_test "Training Components" "cd /workspace && python -c '
import torch
import torch.nn as nn
import numpy as np

# Test neural network creation
class TestNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.network(x)

# Create and test network
device = \"cuda\" if torch.cuda.is_available() else \"cpu\"
net = TestNetwork().to(device)
test_input = torch.randn(10, 100).to(device)
output = net(test_input)
print(f\"Neural network test on {device}: input {test_input.shape} -> output {output.shape}\")

# Test optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
target = torch.randn(10, 2).to(device)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()
print(f\"Training step test: loss = {loss.item():.4f}\")
'" "Training components and neural networks"; then
    ((passed_tests++))
fi

# Generate final report
echo "" >> "$TEST_RESULTS_FILE"
echo "FINAL TEST SUMMARY" >> "$TEST_RESULTS_FILE"
echo "==================" >> "$TEST_RESULTS_FILE"
echo "Total tests: $total_tests" >> "$TEST_RESULTS_FILE"
echo "Passed: $passed_tests" >> "$TEST_RESULTS_FILE"
echo "Failed: $((total_tests - passed_tests))" >> "$TEST_RESULTS_FILE"
echo "Success rate: $(( passed_tests * 100 / total_tests ))%" >> "$TEST_RESULTS_FILE"
echo "Completed at: $(date)" >> "$TEST_RESULTS_FILE"

# Display results
echo ""
echo "🏁 COMPREHENSIVE TESTING COMPLETED"
echo "=================================="
echo "📊 Results: $passed_tests/$total_tests tests passed ($(( passed_tests * 100 / total_tests ))%)"

if [ $passed_tests -eq $total_tests ]; then
    print_success "🎉 ALL TESTS PASSED! Environment is fully ready for enhanced RL training!"
    echo ""
    echo "🚀 Next steps:"
    echo "  ./docker_complete_pipeline.sh    # Run complete training pipeline"
    echo "  ./docker_train_enhanced.sh       # Start enhanced RL training"
    echo "  ./docker_interactive.sh          # Interactive development"
elif [ $passed_tests -ge $((total_tests * 80 / 100)) ]; then
    print_warning "⚠️  MOSTLY READY: $passed_tests/$total_tests tests passed"
    echo "Some components may have issues but core functionality should work."
    echo ""
    echo "🔧 You can proceed with training but monitor for issues:"
    echo "  ./docker_complete_pipeline.sh"
else
    print_error "❌ SYSTEM NOT READY: Only $passed_tests/$total_tests tests passed"
    echo "Please review the test results and fix failing components."
    echo ""
    echo "🔍 Check the detailed log: $TEST_RESULTS_FILE"
fi

echo ""
echo "📋 Detailed test results saved to: $TEST_RESULTS_FILE"
echo ""

# Exit with appropriate code
if [ $passed_tests -eq $total_tests ]; then
    exit 0
elif [ $passed_tests -ge $((total_tests * 80 / 100)) ]; then
    exit 1
else
    exit 2
fi