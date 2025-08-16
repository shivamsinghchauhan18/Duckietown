# Evaluation Orchestrator API Documentation

## Overview

The Evaluation Orchestrator provides a comprehensive framework for systematically evaluating reinforcement learning models across diverse scenarios with statistical rigor. This document provides detailed API documentation for all evaluation components.

## Core Components

### EvaluationOrchestrator

The main orchestrator class that coordinates evaluation execution across multiple models and test suites.

#### Class Definition

```python
class EvaluationOrchestrator:
    """
    Main orchestrator for systematic model evaluation across standardized test suites.
    
    Coordinates evaluation workflow, manages reproducible seed sets, and ensures
    statistical rigor across all evaluation components.
    """
    
    def __init__(self, config: EvaluationConfig, logger: Optional[Logger] = None):
        """
        Initialize the evaluation orchestrator.
        
        Args:
            config: Evaluation configuration with suite settings and parameters
            logger: Optional logger instance for evaluation tracking
        """
```

#### Key Methods

##### evaluate_models()

```python
def evaluate_models(
    self, 
    model_paths: List[str], 
    model_ids: Optional[List[str]] = None
) -> Dict[str, ModelEvaluationResults]:
    """
    Evaluate multiple models across all configured test suites.
    
    Args:
        model_paths: List of paths to trained model files
        model_ids: Optional custom identifiers for models
        
    Returns:
        Dictionary mapping model IDs to comprehensive evaluation results
        
    Raises:
        EvaluationError: If model loading or evaluation fails
        ConfigurationError: If evaluation configuration is invalid
    """
```

##### compare_with_champion()

```python
def compare_with_champion(
    self, 
    candidate_results: ModelEvaluationResults,
    champion_results: Optional[ModelEvaluationResults] = None
) -> ComparisonReport:
    """
    Compare candidate model with current champion using statistical tests.
    
    Args:
        candidate_results: Evaluation results for candidate model
        champion_results: Optional champion results (loads from registry if None)
        
    Returns:
        Detailed comparison report with statistical significance tests
    """
```

##### run_single_evaluation()

```python
def run_single_evaluation(
    self, 
    model_path: str, 
    model_id: str,
    suites: Optional[List[str]] = None
) -> ModelEvaluationResults:
    """
    Run evaluation for a single model across specified suites.
    
    Args:
        model_path: Path to trained model file
        model_id: Unique identifier for the model
        suites: Optional list of suite names (uses config default if None)
        
    Returns:
        Comprehensive evaluation results for the model
    """
```

### SuiteManager

Manages different evaluation test suites with standardized protocols.

#### Class Definition

```python
class SuiteManager:
    """
    Manages evaluation test suites with standardized protocols and configurations.
    
    Each suite represents a different testing scenario (base, hard randomization,
    out-of-distribution, etc.) with specific environmental parameters.
    """
    
    def __init__(self, config: EvaluationConfig, seed_manager: SeedManager):
        """
        Initialize suite manager with configuration and seed management.
        
        Args:
            config: Evaluation configuration
            seed_manager: Seed manager for reproducible evaluations
        """
```

#### Key Methods

##### run_suite()

```python
def run_suite(
    self, 
    suite_name: str, 
    model: Any, 
    seeds: List[int],
    policy_modes: List[str] = ['deterministic']
) -> SuiteResults:
    """
    Execute a complete evaluation suite for a model.
    
    Args:
        suite_name: Name of the suite to run ('base', 'hard', 'law', 'ood', 'stress')
        model: Loaded RL model for evaluation
        seeds: List of random seeds for reproducible evaluation
        policy_modes: List of policy modes ('deterministic', 'stochastic')
        
    Returns:
        Complete suite results with episode-level data and aggregated metrics
    """
```

##### get_suite_config()

```python
def get_suite_config(self, suite_name: str) -> SuiteConfig:
    """
    Get configuration parameters for a specific test suite.
    
    Args:
        suite_name: Name of the suite
        
    Returns:
        Suite configuration with environmental parameters and test conditions
    """
```

##### list_available_suites()

```python
def list_available_suites(self) -> List[str]:
    """
    Get list of all available evaluation suites.
    
    Returns:
        List of suite names that can be used for evaluation
    """
```

### MetricsCalculator

Computes comprehensive performance metrics with statistical rigor.

#### Class Definition

```python
class MetricsCalculator:
    """
    Calculates comprehensive performance metrics for RL model evaluation.
    
    Computes primary metrics (success rate, reward, etc.) and composite scores
    with proper normalization and statistical confidence intervals.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize metrics calculator with evaluation configuration.
        
        Args:
            config: Evaluation configuration with metric parameters
        """
```

#### Key Methods

##### calculate_episode_metrics()

```python
def calculate_episode_metrics(self, episode_data: EpisodeData) -> EpisodeMetrics:
    """
    Calculate metrics for a single episode.
    
    Args:
        episode_data: Raw episode data from simulation
        
    Returns:
        Calculated metrics for the episode
    """
```

##### aggregate_suite_metrics()

```python
def aggregate_suite_metrics(
    self, 
    episode_results: List[EpisodeResult]
) -> SuiteMetrics:
    """
    Aggregate episode-level metrics into suite-level statistics.
    
    Args:
        episode_results: List of episode results from suite execution
        
    Returns:
        Aggregated suite metrics with confidence intervals
    """
```

##### calculate_composite_score()

```python
def calculate_composite_score(
    self, 
    metrics: Dict[str, float],
    normalization_data: Optional[Dict[str, Tuple[float, float]]] = None
) -> float:
    """
    Calculate weighted composite score from individual metrics.
    
    Args:
        metrics: Dictionary of metric name to value
        normalization_data: Optional (min, max) values for normalization
        
    Returns:
        Composite score between 0 and 1
    """
```

### StatisticalAnalyzer

Provides rigorous statistical analysis with confidence intervals and significance testing.

#### Class Definition

```python
class StatisticalAnalyzer:
    """
    Provides statistical analysis for model evaluation results.
    
    Implements confidence intervals, significance testing, and multiple
    comparison corrections for rigorous statistical evaluation.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize statistical analyzer with configuration.
        
        Args:
            config: Evaluation configuration with statistical parameters
        """
```

#### Key Methods

##### compute_confidence_intervals()

```python
def compute_confidence_intervals(
    self, 
    data: np.ndarray, 
    confidence_level: float = 0.95,
    method: str = 'bootstrap'
) -> ConfidenceInterval:
    """
    Compute confidence intervals for metric data.
    
    Args:
        data: Array of metric values
        confidence_level: Confidence level (default 0.95)
        method: Method to use ('bootstrap', 'wilson', 'normal')
        
    Returns:
        Confidence interval with lower and upper bounds
    """
```

##### compare_models()

```python
def compare_models(
    self, 
    model_a_results: List[float], 
    model_b_results: List[float],
    test_type: str = 'paired_t'
) -> ComparisonResult:
    """
    Compare two models using statistical significance tests.
    
    Args:
        model_a_results: Results for first model
        model_b_results: Results for second model
        test_type: Statistical test ('paired_t', 'wilcoxon', 'mann_whitney')
        
    Returns:
        Comparison result with p-value and effect size
    """
```

##### correct_multiple_comparisons()

```python
def correct_multiple_comparisons(
    self, 
    p_values: List[float],
    method: str = 'benjamini_hochberg'
) -> List[float]:
    """
    Apply multiple comparison correction to p-values.
    
    Args:
        p_values: List of uncorrected p-values
        method: Correction method ('benjamini_hochberg', 'bonferroni')
        
    Returns:
        List of corrected p-values
    """
```

### FailureAnalyzer

Comprehensive failure classification and diagnostic analysis.

#### Class Definition

```python
class FailureAnalyzer:
    """
    Analyzes failure modes and provides diagnostic information for model debugging.
    
    Classifies failures, captures state traces, and generates visualizations
    for understanding model weaknesses.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize failure analyzer with configuration.
        
        Args:
            config: Evaluation configuration with failure analysis parameters
        """
```

#### Key Methods

##### analyze_failures()

```python
def analyze_failures(
    self, 
    episode_results: List[EpisodeResult]
) -> FailureAnalysis:
    """
    Analyze failure patterns across episode results.
    
    Args:
        episode_results: List of episode results including failures
        
    Returns:
        Comprehensive failure analysis with classifications and patterns
    """
```

##### classify_failure()

```python
def classify_failure(self, episode_data: EpisodeData) -> FailureClassification:
    """
    Classify the type of failure for a single episode.
    
    Args:
        episode_data: Episode data including termination reason
        
    Returns:
        Failure classification with primary and secondary causes
    """
```

##### generate_failure_heatmaps()

```python
def generate_failure_heatmaps(
    self, 
    failure_data: List[FailureEvent],
    map_name: str,
    output_dir: str
) -> List[str]:
    """
    Generate spatial heatmaps of failure locations.
    
    Args:
        failure_data: List of failure events with spatial coordinates
        map_name: Name of the map for layout reference
        output_dir: Directory to save heatmap images
        
    Returns:
        List of generated heatmap file paths
    """
```

### RobustnessAnalyzer

Evaluates model sensitivity across environmental parameter sweeps.

#### Class Definition

```python
class RobustnessAnalyzer:
    """
    Analyzes model robustness across environmental parameter variations.
    
    Performs parameter sweeps and generates robustness curves to understand
    model sensitivity and operating boundaries.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize robustness analyzer with configuration.
        
        Args:
            config: Evaluation configuration with robustness parameters
        """
```

#### Key Methods

##### analyze_robustness()

```python
def analyze_robustness(
    self, 
    model: Any, 
    parameter_ranges: Dict[str, List[float]],
    base_config: Dict[str, Any]
) -> RobustnessAnalysis:
    """
    Perform comprehensive robustness analysis across parameter sweeps.
    
    Args:
        model: Trained RL model for evaluation
        parameter_ranges: Dictionary of parameter names to value ranges
        base_config: Base configuration for parameter sweeps
        
    Returns:
        Robustness analysis with curves and sensitivity metrics
    """
```

##### calculate_auc_robustness()

```python
def calculate_auc_robustness(
    self, 
    parameter_values: List[float], 
    success_rates: List[float]
) -> float:
    """
    Calculate Area Under Curve robustness metric.
    
    Args:
        parameter_values: Parameter values tested
        success_rates: Corresponding success rates
        
    Returns:
        AUC robustness score between 0 and 1
    """
```

### ChampionSelector

Automated model ranking and champion selection with statistical validation.

#### Class Definition

```python
class ChampionSelector:
    """
    Selects champion models based on comprehensive ranking criteria.
    
    Implements multi-criteria ranking with statistical validation and
    Pareto front analysis for trade-off evaluation.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize champion selector with configuration.
        
        Args:
            config: Evaluation configuration with selection criteria
        """
```

#### Key Methods

##### select_champion()

```python
def select_champion(
    self, 
    candidate_results: List[ModelEvaluationResults],
    current_champion: Optional[ModelEvaluationResults] = None
) -> ChampionSelection:
    """
    Select champion from candidate models using ranking criteria.
    
    Args:
        candidate_results: List of evaluation results for candidate models
        current_champion: Optional current champion for comparison
        
    Returns:
        Champion selection with ranking justification
    """
```

##### rank_models()

```python
def rank_models(
    self, 
    model_results: List[ModelEvaluationResults]
) -> List[ModelRanking]:
    """
    Rank models according to selection criteria.
    
    Args:
        model_results: List of model evaluation results
        
    Returns:
        List of model rankings in descending order of performance
    """
```

##### analyze_pareto_front()

```python
def analyze_pareto_front(
    self, 
    model_results: List[ModelEvaluationResults],
    objectives: List[str]
) -> ParetoAnalysis:
    """
    Analyze Pareto front for multi-objective trade-offs.
    
    Args:
        model_results: List of model evaluation results
        objectives: List of objective names for Pareto analysis
        
    Returns:
        Pareto front analysis with dominated/non-dominated models
    """
```

### ReportGenerator

Generates comprehensive evaluation reports with visualizations.

#### Class Definition

```python
class ReportGenerator:
    """
    Generates comprehensive evaluation reports with statistical analysis and visualizations.
    
    Creates leaderboards, performance tables, plots, and executive summaries
    for evaluation results communication.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize report generator with configuration.
        
        Args:
            config: Evaluation configuration with reporting parameters
        """
```

#### Key Methods

##### generate_comprehensive_report()

```python
def generate_comprehensive_report(
    self, 
    evaluation_results: Dict[str, ModelEvaluationResults],
    output_dir: str,
    report_name: str = "evaluation_report"
) -> ReportArtifacts:
    """
    Generate comprehensive evaluation report with all components.
    
    Args:
        evaluation_results: Dictionary of model results
        output_dir: Directory to save report artifacts
        report_name: Base name for report files
        
    Returns:
        Report artifacts with file paths and metadata
    """
```

##### create_leaderboard()

```python
def create_leaderboard(
    self, 
    model_results: List[ModelEvaluationResults],
    output_path: str,
    format: str = 'csv'
) -> str:
    """
    Create model leaderboard with rankings and confidence intervals.
    
    Args:
        model_results: List of model evaluation results
        output_path: Path to save leaderboard file
        format: Output format ('csv', 'json', 'html')
        
    Returns:
        Path to generated leaderboard file
    """
```

##### generate_performance_plots()

```python
def generate_performance_plots(
    self, 
    model_results: List[ModelEvaluationResults],
    output_dir: str,
    plot_types: List[str] = ['pareto', 'robustness', 'comparison']
) -> List[str]:
    """
    Generate performance visualization plots.
    
    Args:
        model_results: List of model evaluation results
        output_dir: Directory to save plot files
        plot_types: Types of plots to generate
        
    Returns:
        List of generated plot file paths
    """
```

## Data Models

### Core Data Structures

#### EvaluationConfig

```python
@dataclass
class EvaluationConfig:
    """Configuration for evaluation orchestrator."""
    
    # Suite Configuration
    suites: List[str] = field(default_factory=lambda: ['base', 'hard', 'law', 'ood'])
    seeds_per_map: int = 50
    policy_modes: List[str] = field(default_factory=lambda: ['deterministic', 'stochastic'])
    
    # Metrics Configuration
    compute_ci: bool = True
    bootstrap_resamples: int = 10000
    significance_correction: str = 'benjamini_hochberg'
    
    # Scoring Configuration
    use_composite: bool = True
    normalization_scope: str = 'per_map_suite'
    pareto_axes: List[List[str]] = field(default_factory=lambda: [['SR', '-D', '-J']])
    
    # Artifact Configuration
    keep_top_k: int = 5
    export_csv_json: bool = True
    export_plots: bool = True
    record_videos: bool = True
    save_worst_k: int = 5
    
    # Reproducibility Configuration
    fix_seed_list: bool = True
    cudnn_deterministic: bool = True
    log_git_sha: bool = True
```

#### EpisodeResult

```python
@dataclass
class EpisodeResult:
    """Results from a single episode evaluation."""
    
    run_id: str
    model_id: str
    mode: str  # 'deterministic' | 'stochastic'
    suite: str
    map_name: str
    seed: int
    success: bool
    collision: bool
    off_lane: bool
    violations: Dict[str, int]
    reward_mean: float
    lap_time_s: float
    deviation_m: float
    heading_deg: float
    jerk_mean: float
    stability_mu_over_sigma: float
    episode_len_steps: int
    video_path: Optional[str]
    trace_path: Optional[str]
    config_hash: str
    env_build: str
    timestamp: str
```

#### ModelEvaluationResults

```python
@dataclass
class ModelEvaluationResults:
    """Comprehensive evaluation results for a single model."""
    
    model_id: str
    global_score: float
    global_score_ci: Tuple[float, float]
    suite_results: Dict[str, SuiteResults]
    pareto_rank: int
    champion_comparison: Optional[ComparisonResult]
    failure_analysis: FailureAnalysis
    robustness_analysis: RobustnessAnalysis
    metadata: Dict[str, Any]
```

## Usage Examples

### Basic Evaluation

```python
from duckietown_utils.evaluation_orchestrator import EvaluationOrchestrator
from config.evaluation_config import EvaluationConfig

# Create configuration
config = EvaluationConfig(
    suites=['base', 'hard'],
    seeds_per_map=25,
    compute_ci=True
)

# Initialize orchestrator
orchestrator = EvaluationOrchestrator(config)

# Evaluate models
model_paths = ['model1.pkl', 'model2.pkl']
results = orchestrator.evaluate_models(model_paths)

# Print results
for model_id, result in results.items():
    print(f"{model_id}: Score = {result.global_score:.3f}")
```

### Champion Selection

```python
from duckietown_utils.champion_selector import ChampionSelector

# Initialize selector
selector = ChampionSelector(config)

# Select champion from results
champion_selection = selector.select_champion(
    list(results.values())
)

print(f"Champion: {champion_selection.champion_id}")
print(f"Ranking criteria: {champion_selection.ranking_justification}")
```

### Report Generation

```python
from duckietown_utils.report_generator import ReportGenerator

# Initialize report generator
reporter = ReportGenerator(config)

# Generate comprehensive report
artifacts = reporter.generate_comprehensive_report(
    results, 
    output_dir='evaluation_reports',
    report_name='model_comparison'
)

print(f"Report saved to: {artifacts.html_report_path}")
```

## Error Handling

### Common Exceptions

#### EvaluationError

```python
class EvaluationError(Exception):
    """Raised when evaluation execution fails."""
    pass
```

#### ConfigurationError

```python
class ConfigurationError(Exception):
    """Raised when evaluation configuration is invalid."""
    pass
```

#### ModelLoadError

```python
class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass
```

### Error Handling Best Practices

1. **Graceful Degradation**: Continue evaluation with remaining models if one fails
2. **Detailed Logging**: Log all errors with context and stack traces
3. **Validation**: Validate configuration and inputs before execution
4. **Recovery**: Implement retry mechanisms for transient failures
5. **User Feedback**: Provide clear error messages with suggested solutions

## Performance Considerations

### Optimization Guidelines

1. **Parallel Execution**: Use multiprocessing for independent evaluations
2. **Memory Management**: Clear episode data after processing to prevent memory leaks
3. **Caching**: Cache expensive computations like confidence intervals
4. **Batch Processing**: Process multiple episodes in batches for efficiency
5. **Resource Monitoring**: Monitor CPU, memory, and GPU usage during evaluation

### Scalability

- **Model Count**: Tested with up to 50 models simultaneously
- **Episode Count**: Supports 1000+ episodes per suite
- **Suite Count**: All 5 suites can run in parallel
- **Memory Usage**: ~2GB per model during evaluation
- **Execution Time**: ~30 minutes for complete evaluation of 5 models

## Integration

### With Training Pipeline

```python
# Integrate with training callback
class EvaluationCallback:
    def on_training_complete(self, model_path: str):
        results = orchestrator.run_single_evaluation(model_path, 'latest')
        if results.global_score > current_champion_score:
            update_champion(model_path, results)
```

### With CI/CD Pipeline

```python
# Automated evaluation in CI/CD
def evaluate_pr_model():
    config = load_config('ci_evaluation.yml')
    orchestrator = EvaluationOrchestrator(config)
    
    results = orchestrator.run_single_evaluation('pr_model.pkl', 'pr_candidate')
    
    if results.global_score < minimum_threshold:
        raise Exception("Model performance below threshold")
    
    return results
```

This API documentation provides comprehensive coverage of all evaluation orchestrator components with detailed method signatures, usage examples, and integration guidelines.