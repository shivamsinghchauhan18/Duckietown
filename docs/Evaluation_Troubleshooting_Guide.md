# Evaluation Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting information for common issues encountered when using the Enhanced Duckietown RL evaluation system. It covers installation problems, configuration errors, runtime issues, and performance optimization.

## Quick Diagnostic Checklist

Before diving into specific issues, run through this quick diagnostic checklist:

```bash
# 1. Check system requirements
python --version  # Should be 3.8+
nvidia-smi       # Check GPU availability (if using CUDA)

# 2. Verify installation
python -c "import duckietown_utils; print('Installation OK')"

# 3. Test basic configuration
python -c "from config.evaluation_config import EvaluationConfig; print('Config OK')"

# 4. Check model files
ls -la models/  # Verify model files exist

# 5. Test evaluation orchestrator
python examples/evaluation_examples.py --test-basic
```

## Installation Issues

### Issue: Import Errors

#### Symptom:
```
ImportError: No module named 'duckietown_utils'
ModuleNotFoundError: No module named 'gym'
```

#### Diagnosis:
```bash
# Check if packages are installed
pip list | grep gym
pip list | grep duckietown
python -c "import sys; print(sys.path)"
```

#### Solutions:

1. **Install missing dependencies**:
```bash
pip install -r requirements.txt
pip install gym[classic_control]
pip install duckietown-gym
```

2. **Fix Python path issues**:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Or add to ~/.bashrc for permanent fix
echo 'export PYTHONPATH="${PYTHONPATH}:$(pwd)"' >> ~/.bashrc
```

3. **Virtual environment issues**:
```bash
# Create new virtual environment
python -m venv evaluation_env
source evaluation_env/bin/activate  # Linux/Mac
# evaluation_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Issue: CUDA/GPU Problems

#### Symptom:
```
RuntimeError: CUDA out of memory
RuntimeError: No CUDA-capable device is detected
```

#### Diagnosis:
```bash
nvidia-smi  # Check GPU status
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.device_count())"
```

#### Solutions:

1. **CUDA out of memory**:
```python
# Reduce batch size in configuration
config = EvaluationConfig(
    seeds_per_map=10,  # Reduce from default 50
    record_videos=False,  # Disable video recording
    export_plots=False   # Disable plot generation
)
```

2. **No CUDA device**:
```python
# Force CPU usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Or in configuration
config = EvaluationConfig(device='cpu')
```

3. **CUDA version mismatch**:
```bash
# Check CUDA version compatibility
python -c "import torch; print(torch.version.cuda)"
nvcc --version
# Reinstall PyTorch with correct CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Configuration Issues

### Issue: Configuration Validation Errors

#### Symptom:
```
ConfigurationError: Composite weights don't sum to 1.0
ValueError: Invalid suite name 'custom_suite'
ValidationError: seeds_per_map must be between 10 and 1000
```

#### Diagnosis:
```python
from config.evaluation_config import EvaluationConfig, validate_config

config = EvaluationConfig.from_yaml('my_config.yml')
validation_result = validate_config(config)
print(validation_result.errors)
```

#### Solutions:

1. **Weight sum errors**:
```yaml
# Problem:
scoring_configuration:
  composite_weights:
    success_rate: 0.5
    mean_reward: 0.3  # Sum = 0.8, not 1.0

# Solution:
scoring_configuration:
  composite_weights:
    success_rate: 0.5
    mean_reward: 0.3
    lateral_deviation: 0.2  # Now sums to 1.0
```

2. **Invalid suite names**:
```yaml
# Problem:
suite_configuration:
  suites: ['base', 'custom_suite']  # 'custom_suite' not recognized

# Solution:
suite_configuration:
  suites: ['base', 'hard', 'ood']  # Use valid suite names
```

3. **Parameter range errors**:
```yaml
# Problem:
suite_configuration:
  seeds_per_map: 5  # Below minimum of 10

# Solution:
suite_configuration:
  seeds_per_map: 25  # Within valid range
```

### Issue: YAML Configuration Loading

#### Symptom:
```
yaml.scanner.ScannerError: mapping values are not allowed here
FileNotFoundError: [Errno 2] No such file or directory: 'config.yml'
```

#### Diagnosis:
```python
import yaml
with open('config.yml', 'r') as f:
    try:
        config = yaml.safe_load(f)
        print("YAML syntax OK")
    except yaml.YAMLError as e:
        print(f"YAML error: {e}")
```

#### Solutions:

1. **YAML syntax errors**:
```yaml
# Problem: Incorrect indentation
suite_configuration:
suites: ['base']  # Missing indentation

# Solution: Correct indentation
suite_configuration:
  suites: ['base']
```

2. **File path issues**:
```python
import os
config_path = 'config/evaluation_config.yml'
if not os.path.exists(config_path):
    print(f"Config file not found: {config_path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Available files: {os.listdir('.')}")
```

## Runtime Issues

### Issue: Model Loading Failures

#### Symptom:
```
FileNotFoundError: Model file not found: models/my_model.pkl
pickle.UnpicklingError: invalid load key
RuntimeError: Error loading model checkpoint
```

#### Diagnosis:
```python
import os
import pickle

model_path = 'models/my_model.pkl'
print(f"File exists: {os.path.exists(model_path)}")
print(f"File size: {os.path.getsize(model_path) if os.path.exists(model_path) else 'N/A'}")

# Test loading
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Loading error: {e}")
```

#### Solutions:

1. **File not found**:
```python
# Check model directory structure
import os
print("Available models:")
for root, dirs, files in os.walk('models'):
    for file in files:
        print(os.path.join(root, file))

# Use correct path
model_path = 'models/trained_models/my_model.pkl'
```

2. **Corrupted model files**:
```python
# Verify model file integrity
import hashlib
def get_file_hash(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

print(f"Model hash: {get_file_hash(model_path)}")
# Compare with known good hash
```

3. **Version compatibility**:
```python
# Check model format compatibility
try:
    import torch
    model = torch.load(model_path, map_location='cpu')
    print("PyTorch model loaded")
except:
    try:
        import pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("Pickle model loaded")
    except Exception as e:
        print(f"Unknown model format: {e}")
```

### Issue: Evaluation Execution Failures

#### Symptom:
```
TimeoutError: Episode timeout after 300 seconds
MemoryError: Unable to allocate memory
RuntimeError: Simulation environment crashed
```

#### Diagnosis:
```python
import psutil
import time

# Monitor system resources
def monitor_resources():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f"CPU: {cpu_percent}%")
    print(f"Memory: {memory.percent}% ({memory.used/1e9:.1f}GB used)")
    
    if psutil.cuda.is_available():
        gpu_memory = psutil.cuda.memory_usage()
        print(f"GPU Memory: {gpu_memory}%")

monitor_resources()
```

#### Solutions:

1. **Timeout issues**:
```python
# Increase timeout or reduce episode complexity
config = EvaluationConfig(
    timeout_seconds=600,  # Increase from default 300
    seeds_per_map=10,     # Reduce number of episodes
)
```

2. **Memory issues**:
```python
# Reduce memory usage
config = EvaluationConfig(
    record_videos=False,      # Disable video recording
    export_plots=False,       # Disable plot generation
    bootstrap_resamples=1000, # Reduce from default 10000
    keep_top_k=3             # Reduce from default 5
)

# Clear memory between evaluations
import gc
gc.collect()
```

3. **Simulation crashes**:
```python
# Add error handling and retry logic
def robust_evaluation(orchestrator, model_path, model_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            return orchestrator.run_single_evaluation(model_path, model_id)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(5)  # Wait before retry
```

### Issue: Statistical Analysis Errors

#### Symptom:
```
ValueError: Not enough data for confidence interval calculation
RuntimeError: Bootstrap resampling failed
StatisticsError: No significant difference detected
```

#### Diagnosis:
```python
# Check data availability
def diagnose_statistical_data(results):
    for suite_name, suite_results in results.suite_results.items():
        episode_count = len(suite_results.episodes)
        print(f"{suite_name}: {episode_count} episodes")
        
        if episode_count < 10:
            print(f"WARNING: {suite_name} has insufficient data")
        
        # Check for data variability
        success_rates = [ep.success for ep in suite_results.episodes]
        if len(set(success_rates)) == 1:
            print(f"WARNING: {suite_name} has no variability in success rates")
```

#### Solutions:

1. **Insufficient data**:
```python
# Increase sample size
config = EvaluationConfig(
    seeds_per_map=50,  # Increase from lower values
    policy_modes=['deterministic', 'stochastic']  # Test both modes
)
```

2. **Bootstrap failures**:
```python
# Reduce bootstrap complexity or disable
config = EvaluationConfig(
    bootstrap_resamples=1000,  # Reduce from 10000
    compute_ci=False          # Disable if problematic
)
```

3. **No significant differences**:
```python
# This may be correct - models might actually be similar
# Check effect sizes instead of just p-values
def check_effect_sizes(model_a_results, model_b_results):
    # Calculate Cohen's d
    mean_diff = model_a_results.mean() - model_b_results.mean()
    pooled_std = np.sqrt((model_a_results.var() + model_b_results.var()) / 2)
    cohens_d = mean_diff / pooled_std
    
    print(f"Cohen's d: {cohens_d:.3f}")
    if abs(cohens_d) < 0.2:
        print("Small effect size - models are genuinely similar")
```

## Performance Issues

### Issue: Slow Evaluation Execution

#### Symptom:
- Evaluation takes hours to complete
- High CPU/GPU usage
- System becomes unresponsive

#### Diagnosis:
```python
import time
import cProfile

# Profile evaluation performance
def profile_evaluation():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run evaluation
    start_time = time.time()
    results = orchestrator.run_single_evaluation(model_path, model_id)
    end_time = time.time()
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')
    
    print(f"Total time: {end_time - start_time:.2f} seconds")
    return results
```

#### Solutions:

1. **Reduce evaluation scope**:
```python
# Quick evaluation configuration
config = EvaluationConfig(
    suites=['base'],          # Test only base suite
    seeds_per_map=10,         # Reduce seeds
    policy_modes=['deterministic'],  # Single mode
    compute_ci=False,         # Skip confidence intervals
    record_videos=False,      # Skip video recording
    export_plots=False        # Skip plot generation
)
```

2. **Parallel execution**:
```python
# Enable parallel processing
config = EvaluationConfig(
    parallel_workers=4,       # Use multiple CPU cores
    gpu_parallel=True        # Parallel GPU execution if available
)
```

3. **Optimize model inference**:
```python
# Model optimization techniques
import torch

# Compile model for faster inference (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    model = torch.compile(model)

# Use half precision for faster GPU inference
if torch.cuda.is_available():
    model = model.half()
```

### Issue: Memory Leaks

#### Symptom:
- Memory usage increases over time
- System runs out of memory during long evaluations
- Evaluation crashes with MemoryError

#### Diagnosis:
```python
import tracemalloc
import gc

# Monitor memory usage
tracemalloc.start()

def check_memory_usage():
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1e6:.1f} MB")
    print(f"Peak memory usage: {peak / 1e6:.1f} MB")
    
    # Check for unreferenced objects
    print(f"Garbage collector objects: {len(gc.get_objects())}")

# Call periodically during evaluation
check_memory_usage()
```

#### Solutions:

1. **Explicit memory cleanup**:
```python
# Add cleanup between evaluations
def cleanup_memory():
    import gc
    import torch
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Use in evaluation loop
for model_path in model_paths:
    results = orchestrator.run_single_evaluation(model_path, model_id)
    cleanup_memory()  # Clean up after each model
```

2. **Reduce memory footprint**:
```python
# Configuration to minimize memory usage
config = EvaluationConfig(
    record_videos=False,      # Videos use significant memory
    save_episode_traces=False, # Traces can be large
    keep_top_k=1,            # Reduce artifact storage
    compression_level=9       # Maximum compression
)
```

## Output and Reporting Issues

### Issue: Missing or Corrupted Reports

#### Symptom:
```
FileNotFoundError: Report file not found
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
PermissionError: [Errno 13] Permission denied: 'evaluation_reports'
```

#### Diagnosis:
```python
import os
import json

# Check output directory permissions
output_dir = 'evaluation_reports'
print(f"Directory exists: {os.path.exists(output_dir)}")
print(f"Directory writable: {os.access(output_dir, os.W_OK)}")

# Check report file integrity
report_path = 'evaluation_reports/report.json'
if os.path.exists(report_path):
    try:
        with open(report_path, 'r') as f:
            data = json.load(f)
        print("Report file is valid JSON")
    except json.JSONDecodeError as e:
        print(f"JSON error: {e}")
```

#### Solutions:

1. **Permission issues**:
```bash
# Fix directory permissions
chmod 755 evaluation_reports
chown $USER:$USER evaluation_reports

# Or use different output directory
mkdir -p ~/evaluation_results
```

2. **Corrupted files**:
```python
# Add error handling for report generation
def safe_report_generation(reporter, results, output_dir):
    try:
        artifacts = reporter.generate_comprehensive_report(
            results, output_dir, 'evaluation_report'
        )
        return artifacts
    except Exception as e:
        print(f"Report generation failed: {e}")
        # Generate minimal report
        return reporter.generate_minimal_report(results, output_dir)
```

3. **Disk space issues**:
```bash
# Check available disk space
df -h .

# Clean up old reports if needed
find evaluation_reports -name "*.mp4" -mtime +7 -delete  # Remove old videos
find evaluation_reports -name "*.png" -mtime +7 -delete  # Remove old plots
```

### Issue: Visualization Problems

#### Symptom:
- Plots not generated or corrupted
- Missing visualization dependencies
- Display issues in headless environments

#### Diagnosis:
```python
# Check visualization dependencies
try:
    import matplotlib
    import seaborn
    import plotly
    print("Visualization libraries available")
    
    # Check backend
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
except ImportError as e:
    print(f"Missing visualization library: {e}")
```

#### Solutions:

1. **Missing dependencies**:
```bash
pip install matplotlib seaborn plotly kaleido
```

2. **Headless environment issues**:
```python
# Configure matplotlib for headless operation
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
```

3. **Display issues**:
```python
# Alternative visualization configuration
config = EvaluationConfig(
    export_plots=True,
    plot_format='png',        # Use PNG instead of interactive plots
    plot_dpi=150,            # Adjust resolution
    headless_mode=True       # Enable headless mode
)
```

## Environment-Specific Issues

### Issue: Docker Container Problems

#### Symptom:
- Container fails to start
- GPU not accessible in container
- Volume mounting issues

#### Solutions:

1. **GPU access in Docker**:
```bash
# Use nvidia-docker runtime
docker run --gpus all -it evaluation_container

# Or with docker-compose
version: '3.8'
services:
  evaluation:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

2. **Volume mounting**:
```bash
# Correct volume mounting
docker run -v $(pwd)/models:/app/models \
           -v $(pwd)/evaluation_reports:/app/evaluation_reports \
           evaluation_container
```

### Issue: Cluster/HPC Environment

#### Symptom:
- Job scheduling issues
- Resource allocation problems
- Network connectivity issues

#### Solutions:

1. **SLURM job script**:
```bash
#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32GB

module load python/3.8
module load cuda/11.8

python evaluation_script.py
```

2. **Resource management**:
```python
# Adapt to available resources
import os

# Get allocated resources
n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
gpu_count = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))

config = EvaluationConfig(
    parallel_workers=n_cpus,
    use_gpu=gpu_count > 0
)
```

## Debugging Tools and Techniques

### Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_debug.log'),
        logging.StreamHandler()
    ]
)

# Enable specific component debugging
logger = logging.getLogger('duckietown_utils.evaluation_orchestrator')
logger.setLevel(logging.DEBUG)
```

### Use Evaluation Test Mode

```python
# Test mode with minimal configuration
config = EvaluationConfig(
    suites=['base'],
    seeds_per_map=1,
    policy_modes=['deterministic'],
    compute_ci=False,
    record_videos=False,
    export_plots=False,
    test_mode=True  # Enable test mode
)
```

### Step-by-Step Debugging

```python
# Debug individual components
def debug_evaluation_pipeline():
    # 1. Test configuration loading
    config = EvaluationConfig.from_yaml('config.yml')
    print("✓ Configuration loaded")
    
    # 2. Test orchestrator initialization
    orchestrator = EvaluationOrchestrator(config)
    print("✓ Orchestrator initialized")
    
    # 3. Test model loading
    model = orchestrator.load_model('models/test_model.pkl')
    print("✓ Model loaded")
    
    # 4. Test single episode
    episode_result = orchestrator.run_single_episode(model, 'base', 'loop_empty', 42)
    print("✓ Single episode completed")
    
    # 5. Test metrics calculation
    metrics = orchestrator.calculate_metrics([episode_result])
    print("✓ Metrics calculated")
    
    print("All components working correctly")
```

## Getting Help

### Log Collection for Support

When reporting issues, collect these logs:

```bash
# System information
python --version > debug_info.txt
pip list >> debug_info.txt
nvidia-smi >> debug_info.txt

# Evaluation logs
python evaluation_script.py --debug 2>&1 | tee evaluation_debug.log

# Configuration
cp config/evaluation_config.yml debug_config.yml

# Create support package
tar -czf evaluation_debug.tar.gz debug_info.txt evaluation_debug.log debug_config.yml
```

### Common Support Channels

1. **GitHub Issues**: For bug reports and feature requests
2. **Documentation**: Check latest documentation for updates
3. **Community Forum**: For usage questions and discussions
4. **Email Support**: For urgent issues or private concerns

### Before Reporting Issues

1. **Check this troubleshooting guide** for known solutions
2. **Update to latest version** to ensure bug fixes are applied
3. **Test with minimal configuration** to isolate the issue
4. **Collect debug information** as described above
5. **Search existing issues** to avoid duplicates

This comprehensive troubleshooting guide should help resolve most common issues encountered when using the evaluation system. For issues not covered here, please refer to the support channels listed above.