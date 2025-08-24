#!/bin/bash
set -e

echo "🏆 ENHANCED DUCKIETOWN RL TRAINING - DOCKER"
echo "==========================================="
echo "Complete enhanced RL training with YOLO integration"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_header() { echo -e "${PURPLE}[TRAINING]${NC} $1"; }
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
DOCKER_IMAGE="duckietown-rl-enhanced:latest"
CONTAINER_NAME="duckietown-rl-training"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TRAINING_DIR="training_results_${TIMESTAMP}"

# Default training parameters
TIMESTEPS=5000000
EVAL_FREQ=50000
SAVE_FREQ=100000
USE_YOLO=true
USE_GPU=true
TRAINING_MODE="enhanced"

# GPU support detection
GPU_ARGS=""
if command -v nvidia-smi &> /dev/null && docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi &> /dev/null 2>&1; then
    GPU_ARGS="--gpus all"
    print_success "GPU support detected and enabled"
else
    print_warning "No GPU support - using CPU mode"
    USE_GPU=false
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --eval-freq)
            EVAL_FREQ="$2"
            shift 2
            ;;
        --save-freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --no-yolo)
            USE_YOLO=false
            shift
            ;;
        --no-gpu)
            USE_GPU=false
            GPU_ARGS=""
            shift
            ;;
        --mode)
            TRAINING_MODE="$2"
            shift 2
            ;;
        --help)
            echo "Enhanced Duckietown RL Training Options:"
            echo "  --timesteps N      Total training timesteps (default: 5000000)"
            echo "  --eval-freq N      Evaluation frequency (default: 50000)"
            echo "  --save-freq N      Model save frequency (default: 100000)"
            echo "  --no-yolo          Disable YOLO object detection"
            echo "  --no-gpu           Force CPU-only training"
            echo "  --mode MODE        Training mode: simple|enhanced|pipeline (default: enhanced)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for available options"
            exit 1
            ;;
    esac
done

# Create training directory
mkdir -p "$TRAINING_DIR"
print_status "Training results will be saved to: $TRAINING_DIR"

# Show training configuration
print_header "TRAINING CONFIGURATION"
echo "======================"
echo "🎯 Mode: $TRAINING_MODE"
echo "⏱️  Timesteps: $(printf "%'d" $TIMESTEPS)"
echo "📊 Evaluation frequency: $(printf "%'d" $EVAL_FREQ)"
echo "💾 Save frequency: $(printf "%'d" $SAVE_FREQ)"
echo "🤖 YOLO detection: $USE_YOLO"
echo "🚀 GPU acceleration: $USE_GPU"
echo "📁 Results directory: $TRAINING_DIR"
echo ""

# Confirm training start
read -p "Start training with these settings? (Y/n): " confirm
if [[ $confirm =~ ^[Nn]$ ]]; then
    print_status "Training cancelled"
    exit 0
fi

# Function to run training
run_training() {
    local mode="$1"
    local script="$2"
    local description="$3"
    
    print_header "STARTING: $description"
    
    # Create training command
    local training_cmd=""
    case $mode in
        "simple")
            training_cmd="python train_enhanced_rl_simple.py"
            ;;
        "enhanced")
            training_cmd="python enhanced_rl_training_system.py --timesteps $TIMESTEPS --eval-freq $EVAL_FREQ --save-freq $SAVE_FREQ"
            if [ "$USE_YOLO" = false ]; then
                training_cmd="$training_cmd --no-yolo"
            fi
            if [ "$USE_GPU" = false ]; then
                training_cmd="$training_cmd --no-gpu"
            fi
            ;;
        "pipeline")
            training_cmd="python complete_enhanced_rl_pipeline.py --mode full --timesteps $TIMESTEPS"
            if [ "$USE_YOLO" = false ]; then
                training_cmd="$training_cmd --no-yolo"
            fi
            if [ "$USE_GPU" = false ]; then
                training_cmd="$training_cmd --no-gpu"
            fi
            ;;
    esac
    
    # Run training in Docker
    print_status "Executing: $training_cmd"
    
    docker run -it --rm \
        $GPU_ARGS \
        --name "$CONTAINER_NAME" \
        -v "$(pwd):/workspace" \
        -v "$(pwd)/$TRAINING_DIR:/workspace/training_output" \
        -v "duckietown-models:/workspace/models" \
        -v "duckietown-logs:/workspace/logs" \
        -e "PYTHONUNBUFFERED=1" \
        -e "TRAINING_MODE=$mode" \
        -e "TIMESTAMP=$TIMESTAMP" \
        -e "CUDA_VISIBLE_DEVICES=0" \
        "$DOCKER_IMAGE" \
        bash -c "
            echo '🚀 Starting Enhanced RL Training in Docker'
            echo '=========================================='
            echo 'Training mode: $mode'
            echo 'Timestamp: $TIMESTAMP'
            echo 'GPU available: \$(python -c \"import torch; print(torch.cuda.is_available())\")'
            echo ''
            
            cd /workspace
            
            # Create training log
            mkdir -p /workspace/training_output/logs
            
            # Set up environment variables
            export DISPLAY=:0
            export LIBGL_ALWAYS_INDIRECT=1
            
            # Run training with logging
            echo 'Executing training command...'
            $training_cmd 2>&1 | tee /workspace/training_output/training_log.txt
            
            # Save training summary
            echo 'Training completed at: \$(date)' >> /workspace/training_output/training_summary.txt
            echo 'Training mode: $mode' >> /workspace/training_output/training_summary.txt
            echo 'Timesteps: $TIMESTEPS' >> /workspace/training_output/training_summary.txt
            echo 'YOLO enabled: $USE_YOLO' >> /workspace/training_output/training_summary.txt
            echo 'GPU enabled: $USE_GPU' >> /workspace/training_output/training_summary.txt
            
            # Copy models to output directory
            if [ -d '/workspace/models' ]; then
                cp -r /workspace/models/* /workspace/training_output/ 2>/dev/null || true
                echo 'Models copied to output directory'
            fi
            
            # Generate training report
            python -c \"
import json
import os
from datetime import datetime

report = {
    'timestamp': datetime.now().isoformat(),
    'training_mode': '$mode',
    'timesteps': $TIMESTEPS,
    'yolo_enabled': $USE_YOLO,
    'gpu_enabled': $USE_GPU,
    'status': 'completed',
    'output_directory': '/workspace/training_output'
}

with open('/workspace/training_output/training_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print('✅ Training report generated')
\"
            
            echo '🎉 Training completed successfully!'
        "
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "Training completed successfully!"
        
        # Show results summary
        echo ""
        print_header "TRAINING RESULTS SUMMARY"
        echo "========================"
        
        if [ -f "$TRAINING_DIR/training_summary.txt" ]; then
            cat "$TRAINING_DIR/training_summary.txt"
        fi
        
        echo ""
        echo "📁 Results location: $TRAINING_DIR"
        echo "📋 Training log: $TRAINING_DIR/training_log.txt"
        echo "📊 Training report: $TRAINING_DIR/training_report.json"
        
        # Check for trained models
        if ls "$TRAINING_DIR"/*.pth 1> /dev/null 2>&1; then
            echo "🏆 Trained models:"
            ls -la "$TRAINING_DIR"/*.pth
        fi
        
        return 0
    else
        print_error "Training failed with exit code $exit_code"
        echo "Check the training log for details: $TRAINING_DIR/training_log.txt"
        return $exit_code
    fi
}

# Function to monitor training (if running in background)
monitor_training() {
    print_status "Monitoring training progress..."
    
    while docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "$CONTAINER_NAME"; do
        echo "Training in progress... ($(date))"
        sleep 30
    done
    
    print_status "Training container has stopped"
}

# Function to show training options menu
show_training_menu() {
    echo ""
    echo "🎯 SELECT TRAINING TYPE:"
    echo "======================="
    echo "1. 🚀 Simple RL Training (Quick validation)"
    echo "2. 🏆 Enhanced RL Training (Full YOLO integration)"
    echo "3. 🌟 Complete Pipeline (Training + Evaluation + Deployment)"
    echo "4. 🔧 Custom Training (Interactive configuration)"
    echo "5. 📊 Monitor Existing Training"
    echo "6. 🧹 Stop All Training Containers"
    echo "0. ❌ Exit"
    echo ""
    read -p "Enter your choice (0-6): " choice
    
    case $choice in
        1)
            TRAINING_MODE="simple"
            run_training "simple" "train_enhanced_rl_simple.py" "Simple RL Training"
            ;;
        2)
            TRAINING_MODE="enhanced"
            run_training "enhanced" "enhanced_rl_training_system.py" "Enhanced RL Training with YOLO"
            ;;
        3)
            TRAINING_MODE="pipeline"
            run_training "pipeline" "complete_enhanced_rl_pipeline.py" "Complete Training Pipeline"
            ;;
        4)
            configure_custom_training
            ;;
        5)
            monitor_training
            ;;
        6)
            stop_training_containers
            ;;
        0)
            print_status "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice. Please select 0-6."
            show_training_menu
            ;;
    esac
}

# Function to configure custom training
configure_custom_training() {
    echo ""
    print_header "CUSTOM TRAINING CONFIGURATION"
    echo "============================="
    
    # Get timesteps
    read -p "Training timesteps (default: $TIMESTEPS): " input_timesteps
    if [[ -n "$input_timesteps" ]]; then
        TIMESTEPS="$input_timesteps"
    fi
    
    # Get evaluation frequency
    read -p "Evaluation frequency (default: $EVAL_FREQ): " input_eval_freq
    if [[ -n "$input_eval_freq" ]]; then
        EVAL_FREQ="$input_eval_freq"
    fi
    
    # YOLO option
    read -p "Enable YOLO detection? (Y/n): " yolo_choice
    if [[ $yolo_choice =~ ^[Nn]$ ]]; then
        USE_YOLO=false
    fi
    
    # Training mode
    echo ""
    echo "Select training mode:"
    echo "1. Simple RL"
    echo "2. Enhanced RL"
    echo "3. Complete Pipeline"
    read -p "Choice (1-3): " mode_choice
    
    case $mode_choice in
        1) TRAINING_MODE="simple" ;;
        2) TRAINING_MODE="enhanced" ;;
        3) TRAINING_MODE="pipeline" ;;
        *) TRAINING_MODE="enhanced" ;;
    esac
    
    # Show configuration and confirm
    echo ""
    echo "Custom configuration:"
    echo "  Timesteps: $(printf "%'d" $TIMESTEPS)"
    echo "  Eval frequency: $(printf "%'d" $EVAL_FREQ)"
    echo "  YOLO: $USE_YOLO"
    echo "  Mode: $TRAINING_MODE"
    echo ""
    
    read -p "Start training with this configuration? (Y/n): " confirm
    if [[ ! $confirm =~ ^[Nn]$ ]]; then
        run_training "$TRAINING_MODE" "" "Custom Training Configuration"
    fi
}

# Function to stop training containers
stop_training_containers() {
    print_status "Stopping all training containers..."
    
    # Find and stop training containers
    containers=$(docker ps --filter "name=duckietown-rl" --format "{{.Names}}")
    
    if [ -n "$containers" ]; then
        echo "$containers" | xargs docker stop
        print_success "Training containers stopped"
    else
        print_status "No training containers found"
    fi
}

# Main execution
main() {
    # Check if Docker image exists
    if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
        print_error "Docker image '$DOCKER_IMAGE' not found!"
        echo "Please run './docker_setup.sh' first to build the image."
        exit 1
    fi
    
    # If no arguments provided, show menu
    if [ $# -eq 0 ]; then
        show_training_menu
    else
        # Run with provided arguments
        run_training "$TRAINING_MODE" "" "Enhanced RL Training"
    fi
}

# Handle script arguments vs interactive mode
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    # Script is being executed directly
    if [ $# -gt 0 ]; then
        # Arguments provided, run directly
        main "$@"
    else
        # No arguments, show interactive menu
        main
    fi
fi