# 🏆 Evaluation CLI Documentation

Comprehensive documentation for the Enhanced Duckietown RL Evaluation CLI tools.

## Overview

The evaluation CLI provides a complete suite of command-line tools for:
- Model registration and management
- Batch evaluation execution
- Real-time progress monitoring
- Advanced result analysis and querying
- Automated evaluation orchestration

## Core Components

### 1. Main Evaluation CLI (`evaluation_cli.py`)
Primary interface for evaluation orchestrator operations.

### 2. Batch Evaluation (`batch_evaluation.py`)
Automated batch processing and evaluation campaigns.

### 3. Evaluation Monitor (`evaluation_monitor.py`)
Real-time monitoring and progress reporting.

### 4. Evaluation Analysis (`evaluation_analysis.py`)
Advanced querying, analysis, and insights generation.

---

## Main Evaluation CLI

### Installation and Setup

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Verify CLI is working
python evaluation_cli.py --help
```

### Model Registration

Register models for evaluation:

```bash
# Register a single model
python evaluation_cli.py register \
    --model-path models/champion_v1.pth \
    --model-id champion_v1 \
    --description "Champion model version 1" \
    --tags champion baseline

# Register with custom metadata
python evaluation_cli.py register \
    --model-path models/experimental_v2.onnx \
    --model-type onnx \
    --metadata '{"training_steps": 1000000, "algorithm": "PPO"}'
```

### List Registered Models

```bash
# List all registered models
python evaluation_cli.py list-models
```

Output example:
```
Registered Models (3 total):
================================================================================
ID: champion_v1
  Path: models/champion_v1.pth
  Type: checkpoint
  Registered: 2024-08-16T10:30:00
  Description: Champion model version 1
  Tags: champion, baseline

ID: experimental_v2
  Path: models/experimental_v2.onnx
  Type: onnx
  Registered: 2024-08-16T11:15:00
```

### Running Evaluations

#### Basic Evaluation
```bash
# Evaluate specific models on specific suites
python evaluation_cli.py evaluate \
    --models champion_v1 experimental_v2 \
    --suites base hard law
```

#### Advanced Evaluation Options
```bash
# Evaluate all registered models
python evaluation_cli.py evaluate --all-models

# Evaluate with specific policy modes
python evaluation_cli.py evaluate \
    --models champion_v1 \
    --suites base \
    --deterministic \
    --seeds-per-suite 100

# Evaluate with progress monitoring
python evaluation_cli.py evaluate \
    --models champion_v1 \
    --suites base hard \
    --monitor
```

### Monitoring Progress

#### Real-time Monitoring
```bash
# Follow evaluation progress
python evaluation_cli.py monitor --follow
```

#### One-time Status Check
```bash
# Get current status
python evaluation_cli.py monitor
```

### Querying Results

#### Basic Result Queries
```bash
# Get all results for a model
python evaluation_cli.py results --model champion_v1

# Get results for a specific suite
python evaluation_cli.py results --suite base

# Export results to JSON
python evaluation_cli.py results \
    --model champion_v1 \
    --format json \
    --output champion_results.json
```

#### Result Formats
- `table`: Human-readable table (default)
- `json`: JSON format for programmatic use
- `csv`: CSV format for spreadsheet analysis

### Model Comparison

```bash
# Compare two models
python evaluation_cli.py compare \
    --models champion_v1 experimental_v2

# Generate HTML comparison report
python evaluation_cli.py compare \
    --models champion_v1 experimental_v2 baseline_v1 \
    --output comparison_report.html
```

### Stopping and Cleanup

```bash
# Stop running evaluation
python evaluation_cli.py stop

# Cleanup resources
python evaluation_cli.py cleanup
```

---

## Batch Evaluation

### Batch Model Registration

Register multiple models from a directory:

```bash
# Register all .pth files in models directory
python batch_evaluation.py register-batch \
    --models-dir models/ \
    --patterns "*.pth" "*.onnx" \
    --exclude "*temp*" "*backup*"
```

### Evaluation Campaigns

Run comprehensive evaluation campaigns across multiple configurations:

```bash
# Run campaign with multiple config files
python batch_evaluation.py campaign \
    --config-dir configs/ \
    --config-pattern "evaluation_*.yml" \
    --models-pattern "champion_*"

# Run campaign with suite filtering
python batch_evaluation.py campaign \
    --config-dir configs/ \
    --models-pattern "v2_*" \
    --suites-pattern "base|hard"
```

### Scheduled Evaluation

Setup automated periodic evaluation:

```bash
# Schedule evaluation every 24 hours at 2 AM
python batch_evaluation.py schedule \
    --interval 24h \
    --time 02:00 \
    --models-pattern "latest_*" \
    --config configs/nightly_evaluation.yml
```

### Model Generation Comparison

Compare different generations of models:

```bash
# Compare v1 vs v2 models
python batch_evaluation.py compare-generations \
    --base-pattern "v1_*" \
    --new-pattern "v2_*" \
    --output generation_comparison.json
```

---

## Evaluation Monitor

### Terminal Monitoring

#### Rich Terminal Interface (Recommended)
```bash
# Install rich for enhanced terminal UI
pip install rich

# Start rich terminal monitoring
python evaluation_monitor.py terminal --follow
```

#### Basic Terminal Monitoring
```bash
# Basic progress monitoring
python evaluation_monitor.py terminal --follow
```

#### One-time Status
```bash
# Get current status snapshot
python evaluation_monitor.py terminal
```

### Web Dashboard

```bash
# Install Flask for web dashboard
pip install flask

# Start web dashboard
python evaluation_monitor.py web --port 8080 --host localhost
```

Access dashboard at: http://localhost:8080

### Progress Reports

```bash
# Generate HTML progress report
python evaluation_monitor.py report \
    --output progress_report.html \
    --format html

# Generate JSON progress report
python evaluation_monitor.py report \
    --output progress_data.json \
    --format json
```

### Monitoring with Alerts

```bash
# Enable alerts with email notifications
python evaluation_monitor.py terminal \
    --follow \
    --alerts \
    --email admin@example.com researcher@example.com
```

---

## Evaluation Analysis

### Advanced Result Querying

#### Filter by Model and Metrics
```bash
# Query models with high success rate
python evaluation_analysis.py query \
    --model "champion_*" \
    --metric success_rate \
    --min-value 0.8 \
    --format table

# Query specific suite results
python evaluation_analysis.py query \
    --suite base \
    --policy-mode deterministic \
    --date-from 2024-08-01 \
    --output base_results.csv \
    --format csv
```

### Performance Trends Analysis

```bash
# Analyze trends for multiple models
python evaluation_analysis.py trends \
    --models champion_v1 champion_v2 experimental_v1 \
    --time-range 30d \
    --output trends_analysis.json

# Short-term trend analysis
python evaluation_analysis.py trends \
    --models latest_model \
    --time-range 7d
```

### Model Performance Comparison

```bash
# Compare models on specific metrics
python evaluation_analysis.py compare \
    --models champion_v1 baseline_v2 experimental_v1 \
    --metrics success_rate mean_reward mean_lateral_deviation \
    --output detailed_comparison.json

# Quick comparison
python evaluation_analysis.py compare \
    --models model_a model_b
```

### Data Export

```bash
# Export all results to CSV
python evaluation_analysis.py export \
    --format csv \
    --output all_results.csv \
    --include-metadata

# Export filtered results
python evaluation_analysis.py export \
    --format json \
    --output champion_results.json \
    --model "champion_*" \
    --suite base
```

### Insights Generation

```bash
# Generate comprehensive insights report
python evaluation_analysis.py insights \
    --output insights_report.html \
    --include-plots

# Quick insights without plots
python evaluation_analysis.py insights \
    --output quick_insights.html
```

### Database Operations

```bash
# Sync results to database for faster querying
python evaluation_analysis.py sync --db-path evaluation_results.db

# Use database for queries
python evaluation_analysis.py query \
    --db-path evaluation_results.db \
    --model "champion_*" \
    --format json
```

---

## Configuration Management

### Evaluation Configuration

Create and manage evaluation configurations:

```bash
# Create basic configuration
python config/evaluation_config_cli.py create \
    --template basic \
    --output basic_eval.yml

# Create comprehensive configuration
python config/evaluation_config_cli.py create \
    --template comprehensive \
    --output comprehensive_eval.yml

# Validate configuration
python config/evaluation_config_cli.py validate \
    --config my_config.yml

# Show configuration information
python config/evaluation_config_cli.py info \
    --config my_config.yml
```

### Configuration Updates

```bash
# Update configuration parameters
python config/evaluation_config_cli.py update \
    --config my_config.yml \
    --parallel-workers 8 \
    --bootstrap-resamples 15000 \
    --output updated_config.yml

# Compare configurations
python config/evaluation_config_cli.py compare \
    --config1 basic.yml \
    --config2 comprehensive.yml
```

---

## Common Workflows

### 1. Complete Evaluation Workflow

```bash
# 1. Register models
python evaluation_cli.py register --model-path models/new_model.pth --model-id new_model_v1

# 2. Start evaluation with monitoring
python evaluation_cli.py evaluate --models new_model_v1 --suites base hard --monitor

# 3. Query results
python evaluation_cli.py results --model new_model_v1 --format json --output results.json

# 4. Generate insights
python evaluation_analysis.py insights --output insights.html
```

### 2. Batch Evaluation Campaign

```bash
# 1. Register all models in directory
python batch_evaluation.py register-batch --models-dir models/ --patterns "*.pth"

# 2. Run comprehensive campaign
python batch_evaluation.py campaign --config-dir configs/ --models-pattern "*"

# 3. Monitor progress
python evaluation_monitor.py web --port 8080

# 4. Analyze results
python evaluation_analysis.py compare --models $(python evaluation_cli.py list-models | grep "ID:" | cut -d' ' -f2)
```

### 3. Continuous Monitoring Setup

```bash
# 1. Setup scheduled evaluation
python batch_evaluation.py schedule --interval 24h --time 02:00 &

# 2. Start web monitoring dashboard
python evaluation_monitor.py web --port 8080 &

# 3. Setup database for efficient querying
python evaluation_analysis.py sync --db-path evaluation_results.db
```

### 4. Research Analysis Workflow

```bash
# 1. Export all data
python evaluation_analysis.py export --format csv --output research_data.csv --include-metadata

# 2. Analyze trends
python evaluation_analysis.py trends --models $(cat model_list.txt) --time-range 90d --output trends.json

# 3. Generate comprehensive insights
python evaluation_analysis.py insights --output research_insights.html --include-plots

# 4. Compare model generations
python batch_evaluation.py compare-generations --base-pattern "v1_*" --new-pattern "v2_*" --output generation_analysis.json
```

---

## Troubleshooting

### Common Issues

#### 1. Model Registration Fails
```bash
# Check if model file exists
ls -la models/your_model.pth

# Verify model type
python evaluation_cli.py register --model-path models/your_model.pth --model-type pytorch
```

#### 2. Evaluation Hangs
```bash
# Check system resources
python evaluation_monitor.py terminal

# Stop evaluation if needed
python evaluation_cli.py stop
```

#### 3. No Results Found
```bash
# List all registered models
python evaluation_cli.py list-models

# Check evaluation status
python evaluation_cli.py monitor

# Sync database if using analysis tools
python evaluation_analysis.py sync
```

#### 4. Configuration Errors
```bash
# Validate configuration
python config/evaluation_config_cli.py validate --config your_config.yml

# Show configuration details
python config/evaluation_config_cli.py info --config your_config.yml
```

### Performance Optimization

#### 1. Increase Parallel Workers
```bash
# Update configuration for more parallel workers
python config/evaluation_config_cli.py update \
    --config config.yml \
    --parallel-workers 8
```

#### 2. Use Database for Large Datasets
```bash
# Sync to database for faster queries
python evaluation_analysis.py sync --db-path fast_queries.db

# Use database for analysis
python evaluation_analysis.py query --db-path fast_queries.db --model "*"
```

#### 3. Reduce Bootstrap Resamples for Faster Analysis
```bash
# Update for faster analysis (less statistical rigor)
python config/evaluation_config_cli.py update \
    --config config.yml \
    --bootstrap-resamples 1000
```

---

## API Integration

### Python API Usage

```python
from evaluation_cli import EvaluationCLI
from evaluation_analysis import EvaluationAnalyzer, QueryFilter

# Initialize CLI
cli = EvaluationCLI()

# Register model programmatically
orchestrator = cli._get_orchestrator()
model_id = orchestrator.register_model(
    model_path="models/my_model.pth",
    model_type="checkpoint",
    metadata={"version": "1.0", "algorithm": "PPO"}
)

# Schedule evaluation
task_ids = orchestrator.schedule_evaluation(
    model_ids=[model_id],
    suite_names=["base", "hard"]
)

# Start evaluation
orchestrator.start_evaluation()

# Query results with analyzer
analyzer = EvaluationAnalyzer()
filter_criteria = QueryFilter(model_pattern=model_id)
results = analyzer.query_results(filter_criteria)
```

### REST API (Future Enhancement)

The CLI tools can be extended with a REST API for web integration:

```python
# Future: REST API endpoints
GET /api/models                    # List models
POST /api/models                   # Register model
GET /api/evaluations               # List evaluations
POST /api/evaluations              # Start evaluation
GET /api/results                   # Query results
GET /api/progress                  # Get progress
```

---

## Best Practices

### 1. Model Management
- Use descriptive model IDs with version numbers
- Include comprehensive metadata during registration
- Tag models for easy filtering and organization

### 2. Evaluation Planning
- Start with basic suites before running comprehensive evaluations
- Use appropriate number of seeds for statistical significance
- Monitor system resources during large evaluations

### 3. Result Analysis
- Sync results to database for large-scale analysis
- Use filters to focus on relevant subsets of data
- Generate regular insights reports for trend monitoring

### 4. Performance Monitoring
- Use web dashboard for long-running evaluations
- Set up alerts for failure rate monitoring
- Monitor system resources to prevent bottlenecks

### 5. Data Management
- Export results regularly for backup
- Use version control for configuration files
- Maintain evaluation history for reproducibility

---

## Advanced Features

### 1. Custom Evaluation Suites
Define custom evaluation suites in configuration:

```yaml
suites:
  custom_suite:
    name: custom_suite
    enabled: true
    seeds_per_map: 30
    maps: ['custom_map_1', 'custom_map_2']
    environmental_noise: 0.5
    traffic_density: 0.3
```

### 2. Evaluation Hooks
Set up pre/post evaluation hooks:

```python
def pre_evaluation_hook(model_id, suite_name):
    print(f"Starting evaluation: {model_id} on {suite_name}")

def post_evaluation_hook(model_id, suite_name, results):
    print(f"Completed evaluation: {model_id} - Success Rate: {results['success_rate']}")

# Register hooks with orchestrator
orchestrator.add_pre_evaluation_hook(pre_evaluation_hook)
orchestrator.add_post_evaluation_hook(post_evaluation_hook)
```

### 3. Custom Metrics
Add custom metrics to evaluation results:

```python
def custom_metric_calculator(episode_data):
    # Calculate custom metric from episode data
    return custom_value

# Register custom metric
orchestrator.register_custom_metric("custom_metric", custom_metric_calculator)
```

---

## Support and Contributing

### Getting Help
- Check the troubleshooting section above
- Review log files in `logs/` directory
- Use `--verbose` flag for detailed debugging

### Contributing
- Follow the existing code style and patterns
- Add comprehensive tests for new features
- Update documentation for any CLI changes
- Submit pull requests with clear descriptions

### Reporting Issues
Include the following information when reporting issues:
- CLI command that failed
- Full error message and stack trace
- System information (OS, Python version)
- Configuration files used
- Log files from `logs/` directory

---

This documentation provides comprehensive coverage of all evaluation CLI tools. For specific implementation details, refer to the source code and inline documentation in each module.