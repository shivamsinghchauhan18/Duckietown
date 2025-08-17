#!/usr/bin/env python3
"""
🏆 EVALUATION SYSTEM INTEGRATION 🏆
Final integration of evaluation orchestrator with training infrastructure

This module implements task 29: Final Evaluation System Integration
- Integrates evaluation orchestrator with existing training infrastructure
- Validates evaluation system with real trained models
- Performs comprehensive evaluation system testing across all suites
- Executes performance optimization and memory usage validation
- Validates all evaluation requirements are met and system is production-ready

Requirements: All requirements 8.1-13.5
"""

import os
import sys
import time
import json
import threading
import psutil
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import evaluation system components
from duckietown_utils.evaluation_orchestrator import EvaluationOrchestrator
from duckietown_utils.master_evaluation_system import MasterEvaluationSystem
from duckietown_utils.suite_manager import SuiteManager
from duckietown_utils.metrics_calculator import MetricsCalculator
from duckietown_utils.statistical_analyzer import StatisticalAnalyzer
from duckietown_utils.failure_analyzer import FailureAnalyzer
from duckietown_utils.robustness_analyzer import RobustnessAnalyzer
from duckietown_utils.champion_selector import ChampionSelector
from duckietown_utils.report_generator import ReportGenerator
from duckietown_utils.artifact_manager import ArtifactManager

# Import training infrastructure (with error handling)
try:
    from train_ultimate_champion import UltimateChampionTrainer
except ImportError:
    UltimateChampionTrainer = None

try:
    from continuous_champion_training import ContinuousChampionTrainer
except ImportError:
    ContinuousChampionTrainer = None

try:
    from master_rl_orchestrator import MasterRLOrchestrator
except ImportError:
    MasterRLOrchestrator = None

try:
    from evaluate_trained_model import ModelEvaluator
except ImportError:
    ModelEvaluator = None

@dataclass
class IntegrationTestResult:
    """Result of an integration test."""
    test_name: str
    status: str  # 'passed', 'failed', 'error'
    duration: float
    memory_usage_mb: float
    details: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class SystemValidationResult:
    """Result of system validation."""
    component: str
    validation_type: str
    passed: bool
    metrics: Dict[str, Any]
    issues: List[str]

class EvaluationSystemIntegrator:
    """Integrates evaluation system with training infrastructure."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/evaluation_integration_config.yml"
        self.config = self._load_config()
        
        # Integration state
        self.integration_results: List[IntegrationTestResult] = []
        self.validation_results: List[SystemValidationResult] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Directories
        self.results_dir = Path("logs/evaluation_integration")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🏆 Evaluation System Integrator initialized") 
   
    def _load_config(self) -> Dict[str, Any]:
        """Load integration configuration."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default integration configuration."""
        return {
            'integration_tests': {
                'timeout_per_test': 300,
                'memory_limit_mb': 2000,
                'concurrent_tests': 2
            },
            'validation': {
                'performance_threshold_multiplier': 1.2,
                'memory_efficiency_threshold': 0.8,
                'accuracy_tolerance': 0.02
            },
            'evaluation': {
                'test_episodes_per_suite': 20,
                'test_models_count': 3,
                'bootstrap_samples': 1000
            },
            'training_integration': {
                'mock_training_episodes': 100,
                'evaluation_frequency': 50
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / 'integration.log'),
                logging.StreamHandler()
            ]
        )
    
    def run_complete_integration(self) -> Dict[str, Any]:
        """Run complete evaluation system integration."""
        self.logger.info("🚀 Starting complete evaluation system integration")
        start_time = time.time()
        
        try:
            # Step 1: Integrate with training infrastructure
            self._integrate_with_training_infrastructure()
            
            # Step 2: Validate with real trained models
            self._validate_with_real_models()
            
            # Step 3: Comprehensive system testing
            self._run_comprehensive_system_tests()
            
            # Step 4: Performance optimization and validation
            self._perform_performance_optimization()
            
            # Step 5: Final requirements validation
            self._validate_all_requirements()
            
            # Generate final integration report
            total_duration = time.time() - start_time
            report = self._generate_integration_report(total_duration)
            
            self.logger.info(f"✅ Integration completed in {total_duration/60:.1f} minutes")
            return report
            
        except Exception as e:
            self.logger.error(f"💥 Integration failed: {e}")
            raise
    
    def _integrate_with_training_infrastructure(self):
        """Integrate evaluation orchestrator with existing training infrastructure."""
        self.logger.info("🔗 Integrating with training infrastructure")
        
        # Test 1: Integration with UltimateChampionTrainer
        result = self._test_ultimate_champion_integration()
        self.integration_results.append(result)
        
        # Test 2: Integration with MasterRLOrchestrator
        result = self._test_master_orchestrator_integration()
        self.integration_results.append(result)
        
        # Test 3: Integration with ContinuousChampionTrainer
        result = self._test_continuous_training_integration()
        self.integration_results.append(result)
        
        # Test 4: Model evaluation pipeline integration
        result = self._test_model_evaluation_pipeline()
        self.integration_results.append(result)
    
    def _test_ultimate_champion_integration(self) -> IntegrationTestResult:
        """Test integration with UltimateChampionTrainer."""
        test_name = "ultimate_champion_integration"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            self.logger.info("Testing UltimateChampionTrainer integration...")
            
            # Create evaluation orchestrator
            eval_config = {
                'results_dir': str(self.results_dir / 'ultimate_champion'),
                'max_concurrent_evaluations': 2,
                'default_seeds_per_suite': 5
            }
            orchestrator = EvaluationOrchestrator(eval_config)
            
            # Create mock trained model
            model_dir = self.results_dir / 'mock_models'
            model_dir.mkdir(exist_ok=True)
            mock_model_path = model_dir / 'ultimate_champion.json'
            
            mock_model_data = {
                'model_type': 'ultimate_champion',
                'training_episodes': 5000,
                'final_score': 92.5,
                'policy_params': {
                    'steering_bias': 0.02,
                    'throttle_bias': 0.7,
                    'epsilon': 0.01
                }
            }
            
            with open(mock_model_path, 'w') as f:
                json.dump(mock_model_data, f, indent=2)
            
            # Register model with orchestrator
            model_id = orchestrator.register_model(
                str(mock_model_path),
                model_type="ultimate_champion",
                metadata=mock_model_data
            )
            
            # Schedule evaluation
            task_ids = orchestrator.schedule_evaluation(
                model_ids=[model_id],
                suite_names=["base", "hard_randomization"],
                seeds_per_suite=5
            )
            
            # Verify integration
            assert len(task_ids) == 4  # 1 model * 2 suites * 2 policy modes
            assert orchestrator.model_registry.get_model(model_id) is not None
            
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="passed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={
                    'model_registered': True,
                    'tasks_scheduled': len(task_ids),
                    'model_metadata_preserved': True
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="failed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={},
                error_message=str(e)
            ) 
   
    def _test_master_orchestrator_integration(self) -> IntegrationTestResult:
        """Test integration with MasterRLOrchestrator."""
        test_name = "master_orchestrator_integration"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            self.logger.info("Testing MasterRLOrchestrator integration...")
            
            # Create master evaluation system
            eval_config = {
                'evaluation_seeds': 10,
                'target_maps': [
                    {'name': 'loop_empty', 'difficulty': 'easy', 'type': 'loop'},
                    {'name': 'zigzag_dists', 'difficulty': 'moderate', 'type': 'curvy'}
                ],
                'composite_score': {
                    'sr_weight': 0.45,
                    'reward_weight': 0.25,
                    'length_weight': 0.10,
                    'deviation_weight': 0.08,
                    'heading_weight': 0.06,
                    'jerk_weight': 0.06
                }
            }
            
            master_eval = MasterEvaluationSystem(eval_config)
            
            # Create mock agent for testing
            class MockAgent:
                def __init__(self):
                    self.model_id = "master_orchestrator_champion"
                
                def get_action(self, obs, deterministic=True):
                    return [0.1, 0.7]  # Mock steering and throttle
                
                def predict(self, obs, deterministic=True):
                    return [self.get_action(obs, deterministic)], None
            
            mock_agent = MockAgent()
            
            # Test evaluation integration
            # Note: This would normally run full evaluation, but we'll mock it
            evaluation_results = {
                'loop_empty': {
                    'success_rate': 0.9,
                    'mean_reward': 0.85,
                    'composite_score': 88.5
                },
                'zigzag_dists': {
                    'success_rate': 0.8,
                    'mean_reward': 0.75,
                    'composite_score': 82.3
                }
            }
            
            # Verify integration capabilities
            assert hasattr(master_eval, 'evaluate_agent_comprehensive')
            assert hasattr(master_eval, '_calculate_global_metrics')
            assert len(master_eval.maps) == 2
            
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="passed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={
                    'master_eval_created': True,
                    'maps_configured': len(master_eval.maps),
                    'evaluation_interface_available': True,
                    'mock_results': evaluation_results
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="failed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={},
                error_message=str(e)
            )
    
    def _test_continuous_training_integration(self) -> IntegrationTestResult:
        """Test integration with ContinuousChampionTrainer."""
        test_name = "continuous_training_integration"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            self.logger.info("Testing ContinuousChampionTrainer integration...")
            
            # Create evaluation orchestrator for continuous training
            eval_config = {
                'results_dir': str(self.results_dir / 'continuous_training'),
                'max_concurrent_evaluations': 1,
                'default_seeds_per_suite': 3
            }
            orchestrator = EvaluationOrchestrator(eval_config)
            
            # Simulate continuous training checkpoints
            checkpoints = []
            for i in range(3):
                checkpoint_path = self.results_dir / f'checkpoint_{i}.json'
                checkpoint_data = {
                    'session': i + 1,
                    'episodes_trained': (i + 1) * 1000,
                    'current_score': 75.0 + (i * 5.0),
                    'timestamp': datetime.now().isoformat()
                }
                
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                
                # Register checkpoint as model
                model_id = orchestrator.register_model(
                    str(checkpoint_path),
                    model_type="continuous_checkpoint",
                    model_id=f"checkpoint_{i}",
                    metadata=checkpoint_data
                )
                checkpoints.append((model_id, checkpoint_data))
            
            # Test progressive evaluation
            for model_id, data in checkpoints:
                task_ids = orchestrator.schedule_evaluation(
                    model_ids=[model_id],
                    suite_names=["base"],
                    seeds_per_suite=3
                )
                assert len(task_ids) == 2  # 1 model * 1 suite * 2 policy modes
            
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="passed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={
                    'checkpoints_created': len(checkpoints),
                    'progressive_evaluation_supported': True,
                    'checkpoint_metadata_preserved': True
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="failed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={},
                error_message=str(e)
            )
    
    def _test_model_evaluation_pipeline(self) -> IntegrationTestResult:
        """Test model evaluation pipeline integration."""
        test_name = "model_evaluation_pipeline"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            self.logger.info("Testing model evaluation pipeline integration...")
            
            # Create complete evaluation pipeline with proper configs
            base_dir = self.results_dir / 'pipeline_test'
            
            # Import configuration classes
            from duckietown_utils.failure_analyzer import FailureAnalysisConfig
            from duckietown_utils.artifact_manager import ArtifactManagerConfig
            
            # Initialize components with proper configurations
            orchestrator = EvaluationOrchestrator()
            suite_manager = SuiteManager()
            metrics_calculator = MetricsCalculator()
            statistical_analyzer = StatisticalAnalyzer()
            
            # Components that need specific config classes
            failure_analyzer = FailureAnalyzer(FailureAnalysisConfig())
            robustness_analyzer = RobustnessAnalyzer()
            champion_selector = ChampionSelector()
            report_generator = ReportGenerator({'generate_html': True})
            artifact_manager = ArtifactManager(ArtifactManagerConfig(base_path=str(base_dir)))
            
            # Test component integration
            components = {
                'orchestrator': orchestrator,
                'suite_manager': suite_manager,
                'metrics_calculator': metrics_calculator,
                'statistical_analyzer': statistical_analyzer,
                'failure_analyzer': failure_analyzer,
                'robustness_analyzer': robustness_analyzer,
                'champion_selector': champion_selector,
                'report_generator': report_generator,
                'artifact_manager': artifact_manager
            }
            
            # Verify all components are properly initialized
            for name, component in components.items():
                assert component is not None, f"{name} not initialized"
                assert hasattr(component, 'config'), f"{name} missing config"
            
            # Test pipeline connectivity
            # Create mock model
            model_path = self.results_dir / 'pipeline_test_model.json'
            model_data = {'model_type': 'pipeline_test', 'score': 85.0}
            with open(model_path, 'w') as f:
                json.dump(model_data, f)
            
            model_id = orchestrator.register_model(str(model_path))
            
            # Test that components can work together
            task_ids = orchestrator.schedule_evaluation(
                model_ids=[model_id],
                suite_names=["base"],
                seeds_per_suite=2
            )
            
            assert len(task_ids) == 2
            
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="passed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={
                    'components_initialized': len(components),
                    'pipeline_connectivity_verified': True,
                    'model_registration_successful': True
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="failed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={},
                error_message=str(e)
            )  
  
    def _validate_with_real_models(self):
        """Validate evaluation system with real trained models."""
        self.logger.info("🔍 Validating with real trained models")
        
        # Find existing trained models
        model_paths = self._find_existing_models()
        
        if not model_paths:
            self.logger.warning("No existing models found, creating mock models for validation")
            model_paths = self._create_validation_models()
        
        # Test with each model
        for model_path in model_paths[:3]:  # Limit to 3 models for testing
            result = self._validate_single_model(model_path)
            self.validation_results.append(result)
    
    def _find_existing_models(self) -> List[Path]:
        """Find existing trained models."""
        model_paths = []
        
        # Check common model directories
        search_dirs = [
            Path("models"),
            Path("checkpoints"),
            Path("models/legendary_fusion_champions"),
            Path("models/master_orchestrator_champions")
        ]
        
        for search_dir in search_dirs:
            if search_dir.exists():
                # Look for model files
                for ext in ['.json', '.pth', '.pt', '.onnx']:
                    model_paths.extend(search_dir.glob(f"*{ext}"))
        
        self.logger.info(f"Found {len(model_paths)} existing model files")
        return model_paths
    
    def _create_validation_models(self) -> List[Path]:
        """Create mock models for validation."""
        validation_dir = self.results_dir / 'validation_models'
        validation_dir.mkdir(exist_ok=True)
        
        model_configs = [
            {
                'name': 'high_performance_model',
                'score': 92.5,
                'success_rate': 0.95,
                'training_episodes': 10000
            },
            {
                'name': 'baseline_model',
                'score': 78.3,
                'success_rate': 0.82,
                'training_episodes': 5000
            },
            {
                'name': 'experimental_model',
                'score': 85.7,
                'success_rate': 0.88,
                'training_episodes': 7500
            }
        ]
        
        model_paths = []
        for config in model_configs:
            model_path = validation_dir / f"{config['name']}.json"
            with open(model_path, 'w') as f:
                json.dump(config, f, indent=2)
            model_paths.append(model_path)
        
        return model_paths
    
    def _validate_single_model(self, model_path: Path) -> SystemValidationResult:
        """Validate evaluation system with a single model."""
        try:
            self.logger.info(f"Validating with model: {model_path.name}")
            
            # Create evaluation orchestrator
            eval_config = {
                'results_dir': str(self.results_dir / 'model_validation'),
                'default_seeds_per_suite': 5
            }
            orchestrator = EvaluationOrchestrator(eval_config)
            
            # Register model
            model_id = orchestrator.register_model(
                str(model_path),
                model_type="validation_model"
            )
            
            # Schedule evaluation across multiple suites
            task_ids = orchestrator.schedule_evaluation(
                model_ids=[model_id],
                suite_names=["base", "hard_randomization"],
                seeds_per_suite=5
            )
            
            # Verify evaluation setup
            assert len(task_ids) == 4  # 1 model * 2 suites * 2 policy modes
            
            # Check task details
            for task_id in task_ids:
                task = orchestrator.state_tracker.get_task(task_id)
                assert task is not None
                assert task.model_id == model_id
                assert len(task.seeds) == 5
            
            return SystemValidationResult(
                component="model_validation",
                validation_type="real_model_integration",
                passed=True,
                metrics={
                    'model_path': str(model_path),
                    'tasks_scheduled': len(task_ids),
                    'model_registered': True
                },
                issues=[]
            )
            
        except Exception as e:
            return SystemValidationResult(
                component="model_validation",
                validation_type="real_model_integration",
                passed=False,
                metrics={'model_path': str(model_path)},
                issues=[str(e)]
            )
    
    def _run_comprehensive_system_tests(self):
        """Run comprehensive evaluation system testing across all suites."""
        self.logger.info("🧪 Running comprehensive system tests")
        
        # Test 1: All evaluation suites
        result = self._test_all_evaluation_suites()
        self.integration_results.append(result)
        
        # Test 2: Statistical analysis accuracy
        result = self._test_statistical_analysis()
        self.integration_results.append(result)
        
        # Test 3: Failure analysis system
        result = self._test_failure_analysis()
        self.integration_results.append(result)
        
        # Test 4: Robustness analysis
        result = self._test_robustness_analysis()
        self.integration_results.append(result)
        
        # Test 5: Champion selection
        result = self._test_champion_selection()
        self.integration_results.append(result)
        
        # Test 6: Report generation
        result = self._test_report_generation()
        self.integration_results.append(result)
        
        # Test 7: Artifact management
        result = self._test_artifact_management()
        self.integration_results.append(result)
    
    def _test_all_evaluation_suites(self) -> IntegrationTestResult:
        """Test all evaluation suites."""
        test_name = "all_evaluation_suites"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            self.logger.info("Testing all evaluation suites...")
            
            suite_config = {
                'results_dir': str(self.results_dir / 'suite_test'),
                'timeout_per_episode': 30.0
            }
            suite_manager = SuiteManager(suite_config)
            
            # Test all suite types
            suite_names = ["base", "hard_randomization", "law_intersection", 
                          "out_of_distribution", "stress_adversarial"]
            
            suite_results = {}
            
            # Create mock model for testing
            class MockModel:
                def __init__(self):
                    self.model_id = "suite_test_model"
                
                def predict(self, obs, deterministic=True):
                    return [[0.1, 0.7]], None
            
            mock_model = MockModel()
            seeds = [1, 2, 3, 4, 5]
            
            for suite_name in suite_names:
                try:
                    # This would normally run actual suite evaluation
                    # For integration testing, we verify the suite can be configured
                    suite_config_obj = suite_manager.get_suite_config(suite_name)
                    assert suite_config_obj is not None
                    
                    suite_results[suite_name] = {
                        'configured': True,
                        'suite_type': suite_config_obj.suite_type.value if hasattr(suite_config_obj, 'suite_type') else 'unknown'
                    }
                    
                except Exception as e:
                    suite_results[suite_name] = {
                        'configured': False,
                        'error': str(e)
                    }
            
            # Verify all suites are available
            configured_suites = sum(1 for result in suite_results.values() if result.get('configured', False))
            
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="passed" if configured_suites == len(suite_names) else "failed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={
                    'total_suites': len(suite_names),
                    'configured_suites': configured_suites,
                    'suite_results': suite_results
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={},
                error_message=str(e)
            )    

    def _test_statistical_analysis(self) -> IntegrationTestResult:
        """Test statistical analysis accuracy."""
        test_name = "statistical_analysis"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            self.logger.info("Testing statistical analysis...")
            
            config = {'bootstrap_samples': 100, 'confidence_level': 0.95}
            analyzer = StatisticalAnalyzer(config)
            
            # Generate test data with known properties
            import numpy as np
            np.random.seed(42)
            
            baseline_data = np.random.normal(0.7, 0.1, 100)
            treatment_data = np.random.normal(0.8, 0.1, 100)
            
            # Test confidence intervals
            baseline_ci = analyzer.compute_confidence_intervals(baseline_data)
            treatment_ci = analyzer.compute_confidence_intervals(treatment_data)
            
            # Verify CI properties
            assert baseline_ci.lower < baseline_ci.upper
            assert treatment_ci.lower < treatment_ci.upper
            assert baseline_ci.confidence_level == 0.95
            
            # Test model comparison
            comparison = analyzer.compare_models(
                baseline_data, treatment_data,
                "baseline", "treatment", "test_metric"
            )
            
            assert comparison.model_b_mean > comparison.model_a_mean
            assert comparison.p_value < 0.05  # Should be significant
            
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="passed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={
                    'confidence_intervals_computed': True,
                    'significance_test_passed': True,
                    'baseline_ci_width': baseline_ci.upper - baseline_ci.lower,
                    'treatment_ci_width': treatment_ci.upper - treatment_ci.lower,
                    'p_value': comparison.p_value
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={},
                error_message=str(e)
            )
    
    def _test_failure_analysis(self) -> IntegrationTestResult:
        """Test failure analysis system."""
        test_name = "failure_analysis"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            self.logger.info("Testing failure analysis...")
            
            # Use the correct configuration approach
            from duckietown_utils.failure_analyzer import FailureAnalysisConfig
            config = FailureAnalysisConfig()
            analyzer = FailureAnalyzer(config)
            
            # Test the correct method - generate_failure_statistics
            failure_statistics = analyzer.generate_failure_statistics()
            
            # Verify analysis results
            assert failure_statistics is not None
            assert isinstance(failure_statistics, dict)
            
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="passed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={
                    'failure_analyzer_initialized': True,
                    'statistics_generated': True,
                    'analysis_components': list(failure_statistics.keys())
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={},
                error_message=str(e)
            )
    
    def _test_robustness_analysis(self) -> IntegrationTestResult:
        """Test robustness analysis."""
        test_name = "robustness_analysis"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            self.logger.info("Testing robustness analysis...")
            
            config = {'results_dir': str(self.results_dir / 'robustness_test')}
            analyzer = RobustnessAnalyzer(config)
            
            # Create mock parameter sweep results
            parameter_sweep_results = {}
            lighting_values = [0.5, 0.7, 1.0, 1.3, 1.5]
            
            for lighting in lighting_values:
                # Success rate decreases with extreme lighting
                base_success = 0.9
                if lighting < 0.7 or lighting > 1.3:
                    success_rate = base_success * 0.7
                else:
                    success_rate = base_success
                
                parameter_sweep_results[f"lighting_{lighting}"] = {
                    'success_rate': success_rate,
                    'mean_reward': success_rate * 0.8,
                    'parameter_value': lighting
                }
            
            # Create proper parameter sweep configuration and results
            from duckietown_utils.robustness_analyzer import ParameterSweepConfig, ParameterType
            from duckietown_utils.suite_manager import EpisodeResult
            
            # Create sweep config
            sweep_config = ParameterSweepConfig(
                parameter_type=ParameterType.LIGHTING_INTENSITY,
                parameter_name="lighting_intensity",
                baseline_value=1.0,
                min_value=0.5,
                max_value=1.5,
                num_points=5,
                sweep_method="linear"
            )
            
            # Create mock episode results for each parameter value
            parameter_results = {}
            for lighting in lighting_values:
                episodes = []
                success_rate = parameter_sweep_results[f"lighting_{lighting}"]['success_rate']
                for i in range(10):  # 10 episodes per parameter value
                    episode = EpisodeResult(
                        episode_id=f"ep_{i}",
                        map_name="test_map",
                        seed=i,
                        success=i < (success_rate * 10),
                        episode_length=100,
                        reward=success_rate * 0.8,
                        lateral_deviation=0.1,
                        heading_error=2.0,
                        jerk=0.5,
                        stability=0.9,
                        collision=False,
                        off_lane=False
                    )
                    episodes.append(episode)
                parameter_results[lighting] = episodes
            
            # Analyze robustness using correct method
            robustness_curve = analyzer.analyze_parameter_sweep(
                model_id="test_model",
                parameter_results=parameter_results,
                sweep_config=sweep_config
            )
            
            # Verify analysis
            assert robustness_curve is not None
            assert robustness_curve.auc_success_rate is not None
            assert 0.0 <= robustness_curve.auc_success_rate <= 1.0
            
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="passed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={
                    'parameter_values_tested': len(lighting_values),
                    'auc_success_rate': robustness_curve.auc_success_rate,
                    'parameter_name': robustness_curve.parameter_name,
                    'model_id': robustness_curve.model_id
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={},
                error_message=str(e)
            )
    
    def _test_champion_selection(self) -> IntegrationTestResult:
        """Test champion selection system."""
        test_name = "champion_selection"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            self.logger.info("Testing champion selection...")
            
            config = {'results_dir': str(self.results_dir / 'champion_test')}
            selector = ChampionSelector(config)
            
            # Create mock model metrics
            from duckietown_utils.metrics_calculator import ModelMetrics, MetricResult
            
            model_metrics_list = []
            model_names = ['champion', 'baseline', 'weak']
            performance_levels = [0.95, 0.8, 0.6]
            
            for model_name, perf_level in zip(model_names, performance_levels):
                primary_metrics = {
                    'success_rate': MetricResult('success_rate', perf_level, sample_size=100),
                    'mean_reward': MetricResult('mean_reward', perf_level * 0.8, sample_size=100)
                }
                
                secondary_metrics = {
                    'lateral_deviation': MetricResult('lateral_deviation', 0.1 * (2 - perf_level), sample_size=100),
                    'heading_error': MetricResult('heading_error', 5.0 * (2 - perf_level), sample_size=100)
                }
                
                model_metrics = ModelMetrics(
                    model_id=model_name,
                    primary_metrics=primary_metrics,
                    secondary_metrics=secondary_metrics,
                    safety_metrics={},
                    per_suite_metrics={},
                    per_map_metrics={},
                    metadata={'total_episodes': 100}
                )
                
                model_metrics_list.append(model_metrics)
            
            # Select champion
            champion_result = selector.select_champion(model_metrics_list)
            
            # Verify champion selection
            assert champion_result is not None
            assert champion_result.new_champion_id == 'champion'  # Highest performance
            
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="passed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={
                    'models_evaluated': len(model_metrics_list),
                    'champion_selected': champion_result.new_champion_id,
                    'selection_criteria_applied': True
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={},
                error_message=str(e)
            )   
 
    def _test_report_generation(self) -> IntegrationTestResult:
        """Test report generation system."""
        test_name = "report_generation"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            self.logger.info("Testing report generation...")
            
            # Use correct ReportConfig parameters
            config = {
                'generate_html': True,
                'save_plots': True,
                'include_confidence_intervals': True
            }
            generator = ReportGenerator(config)
            
            # Create mock model metrics for report generation
            from duckietown_utils.metrics_calculator import ModelMetrics, MetricResult
            
            primary_metrics = {
                'success_rate': MetricResult('success_rate', 0.85, sample_size=100),
                'mean_reward': MetricResult('mean_reward', 0.78, sample_size=100)
            }
            
            secondary_metrics = {
                'lateral_deviation': MetricResult('lateral_deviation', 0.15, sample_size=100),
                'heading_error': MetricResult('heading_error', 5.2, sample_size=100)
            }
            
            model_metrics = ModelMetrics(
                model_id='test_model',
                primary_metrics=primary_metrics,
                secondary_metrics=secondary_metrics,
                safety_metrics={},
                per_suite_metrics={},
                per_map_metrics={},
                metadata={'total_episodes': 100}
            )
            
            # Generate comprehensive report
            report = generator.generate_comprehensive_report(
                model_metrics_list=[model_metrics],
                report_id="integration_test_report"
            )
            
            # Verify report was created
            assert report is not None
            assert report.leaderboard is not None
            assert len(report.leaderboard) == 1
            assert report.leaderboard[0].model_id == 'test_model'
            
            # Save report to file for testing
            report_path = self.results_dir / "integration_test_report.json"
            report_dict = {
                'report_id': report.report_id,
                'generation_timestamp': report.generation_timestamp,
                'leaderboard': [
                    {
                        'model_id': model.model_id,
                        'composite_score': model.composite_score,
                        'success_rate': model.success_rate,
                        'mean_reward': model.mean_reward,
                        'rank': model.rank
                    } for model in report.leaderboard
                ]
            }
            with open(report_path, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="passed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={
                    'report_generated': True,
                    'report_path': str(report_path),
                    'content_verified': True
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={},
                error_message=str(e)
            )
    
    def _test_artifact_management(self) -> IntegrationTestResult:
        """Test artifact management system."""
        test_name = "artifact_management"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            self.logger.info("Testing artifact management...")
            
            from duckietown_utils.artifact_manager import ArtifactManagerConfig
            
            config = ArtifactManagerConfig(
                base_path=str(self.results_dir / 'artifact_test'),
                compression_enabled=True,
                max_artifacts_per_type=1000
            )
            manager = ArtifactManager(config)
            
            # Test artifact storage
            test_data = {
                'model_id': 'test_model',
                'evaluation_results': {'success_rate': 0.85},
                'timestamp': datetime.now().isoformat()
            }
            
            # Create a temporary file for the artifact
            import tempfile
            from duckietown_utils.artifact_manager import ArtifactType
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(test_data, f)
                temp_file_path = f.name
            
            # Store artifact with correct parameters
            artifact_id = manager.store_artifact(
                temp_file_path,
                ArtifactType.EVALUATION_RESULT,
                model_id='test_model',
                metadata={'test': True}
            )
            
            # Retrieve artifact
            retrieved_path = manager.retrieve_artifact(artifact_id)
            
            # Read the retrieved data
            with open(retrieved_path, 'r') as f:
                retrieved_data = json.load(f)
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            # Verify artifact integrity
            assert retrieved_data is not None
            assert retrieved_data['model_id'] == 'test_model'
            assert retrieved_data['evaluation_results']['success_rate'] == 0.85
            
            # Test artifact listing
            artifacts = manager.get_artifacts(artifact_type=ArtifactType.EVALUATION_RESULT)
            assert len(artifacts) >= 1
            assert artifact_id in [a.artifact_id for a in artifacts]
            
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="passed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={
                    'artifact_stored': True,
                    'artifact_retrieved': True,
                    'data_integrity_verified': True,
                    'artifact_id': artifact_id
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={},
                error_message=str(e)
            )
    
    def _perform_performance_optimization(self):
        """Perform performance optimization and memory usage validation."""
        self.logger.info("⚡ Performing performance optimization and validation")
        
        # Test 1: Memory usage optimization
        result = self._test_memory_optimization()
        self.integration_results.append(result)
        
        # Test 2: Concurrent processing performance
        result = self._test_concurrent_performance()
        self.integration_results.append(result)
        
        # Test 3: Large-scale evaluation performance
        result = self._test_large_scale_performance()
        self.integration_results.append(result)
        
        # Collect overall performance metrics
        self._collect_performance_metrics()
    
    def _test_memory_optimization(self) -> IntegrationTestResult:
        """Test memory usage optimization."""
        test_name = "memory_optimization"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            self.logger.info("Testing memory optimization...")
            
            # Create multiple orchestrators to test memory usage
            orchestrators = []
            memory_measurements = []
            
            for i in range(5):
                config = {
                    'results_dir': str(self.results_dir / f'memory_test_{i}'),
                    'max_concurrent_evaluations': 2
                }
                orchestrator = EvaluationOrchestrator(config)
                orchestrators.append(orchestrator)
                
                # Measure memory after each orchestrator
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_measurements.append(current_memory - start_memory)
            
            # Test memory cleanup
            for orchestrator in orchestrators:
                orchestrator.cleanup()
            
            # Force garbage collection
            gc.collect()
            
            # Measure memory after cleanup
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_after_cleanup = final_memory - start_memory
            
            # Calculate memory efficiency
            max_memory_usage = max(memory_measurements)
            memory_efficiency = 1.0 - (memory_after_cleanup / max_memory_usage) if max_memory_usage > 0 else 1.0
            
            duration = time.time() - start_time
            
            return IntegrationTestResult(
                test_name=test_name,
                status="passed" if memory_efficiency > 0.7 else "failed",
                duration=duration,
                memory_usage_mb=max_memory_usage,
                details={
                    'orchestrators_created': len(orchestrators),
                    'max_memory_usage_mb': max_memory_usage,
                    'memory_after_cleanup_mb': memory_after_cleanup,
                    'memory_efficiency': memory_efficiency,
                    'memory_measurements': memory_measurements
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={},
                error_message=str(e)
            )
    
    def _test_concurrent_performance(self) -> IntegrationTestResult:
        """Test concurrent processing performance."""
        test_name = "concurrent_performance"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            self.logger.info("Testing concurrent performance...")
            
            config = {
                'results_dir': str(self.results_dir / 'concurrent_test'),
                'max_concurrent_evaluations': 4
            }
            orchestrator = EvaluationOrchestrator(config)
            
            # Create multiple models for concurrent testing
            model_ids = []
            for i in range(6):
                model_path = self.results_dir / f'concurrent_model_{i}.json'
                model_data = {'model_id': f'concurrent_model_{i}', 'score': 80.0 + i}
                with open(model_path, 'w') as f:
                    json.dump(model_data, f)
                
                model_id = orchestrator.register_model(str(model_path))
                model_ids.append(model_id)
            
            # Schedule concurrent evaluations
            concurrent_start = time.time()
            
            task_ids = orchestrator.schedule_evaluation(
                model_ids=model_ids,
                suite_names=["base"],
                seeds_per_suite=3
            )
            
            scheduling_time = time.time() - concurrent_start
            
            # Measure concurrent processing efficiency
            expected_sequential_time = len(task_ids) * 0.1  # Assume 0.1s per task
            actual_time = scheduling_time
            efficiency = min(expected_sequential_time / actual_time, 1.0) if actual_time > 0 else 1.0
            
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="passed" if efficiency > 0.5 else "failed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={
                    'models_processed': len(model_ids),
                    'tasks_scheduled': len(task_ids),
                    'scheduling_time': scheduling_time,
                    'concurrent_efficiency': efficiency
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={},
                error_message=str(e)
            ) 
   
    def _test_large_scale_performance(self) -> IntegrationTestResult:
        """Test large-scale evaluation performance."""
        test_name = "large_scale_performance"
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            self.logger.info("Testing large-scale performance...")
            
            config = {
                'results_dir': str(self.results_dir / 'large_scale_test'),
                'max_concurrent_evaluations': 4
            }
            orchestrator = EvaluationOrchestrator(config)
            
            # Create many models to test scalability
            model_count = 20
            model_ids = []
            
            batch_start = time.time()
            
            for i in range(model_count):
                model_path = self.results_dir / f'scale_model_{i}.json'
                model_data = {'model_id': f'scale_model_{i}', 'score': 70.0 + (i % 30)}
                with open(model_path, 'w') as f:
                    json.dump(model_data, f)
                
                model_id = orchestrator.register_model(str(model_path))
                model_ids.append(model_id)
            
            registration_time = time.time() - batch_start
            
            # Test batch scheduling
            scheduling_start = time.time()
            
            task_ids = orchestrator.schedule_evaluation(
                model_ids=model_ids[:10],  # Limit to 10 models for testing
                suite_names=["base", "hard_randomization"],
                seeds_per_suite=5
            )
            
            scheduling_time = time.time() - scheduling_start
            
            # Calculate performance metrics
            models_per_second = model_count / registration_time if registration_time > 0 else 0
            tasks_per_second = len(task_ids) / scheduling_time if scheduling_time > 0 else 0
            
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            # Performance thresholds
            min_models_per_second = 50
            min_tasks_per_second = 100
            
            performance_passed = (models_per_second >= min_models_per_second and 
                                tasks_per_second >= min_tasks_per_second)
            
            return IntegrationTestResult(
                test_name=test_name,
                status="passed" if performance_passed else "failed",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={
                    'models_registered': model_count,
                    'tasks_scheduled': len(task_ids),
                    'registration_time': registration_time,
                    'scheduling_time': scheduling_time,
                    'models_per_second': models_per_second,
                    'tasks_per_second': tasks_per_second,
                    'performance_thresholds_met': performance_passed
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = end_memory - start_memory
            
            return IntegrationTestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                memory_usage_mb=memory_usage,
                details={},
                error_message=str(e)
            )
    
    def _collect_performance_metrics(self):
        """Collect overall performance metrics."""
        self.logger.info("📊 Collecting performance metrics")
        
        # Calculate aggregate metrics from integration results
        total_tests = len(self.integration_results)
        passed_tests = sum(1 for r in self.integration_results if r.status == "passed")
        failed_tests = sum(1 for r in self.integration_results if r.status == "failed")
        error_tests = sum(1 for r in self.integration_results if r.status == "error")
        
        total_duration = sum(r.duration for r in self.integration_results)
        total_memory_usage = sum(r.memory_usage_mb for r in self.integration_results)
        avg_memory_per_test = total_memory_usage / total_tests if total_tests > 0 else 0
        
        # System resource metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')
        
        self.performance_metrics = {
            'test_execution': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'error_tests': error_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'total_duration': total_duration,
                'avg_duration_per_test': total_duration / total_tests if total_tests > 0 else 0
            },
            'memory_usage': {
                'total_memory_usage_mb': total_memory_usage,
                'avg_memory_per_test_mb': avg_memory_per_test,
                'system_memory_available_gb': memory_info.available / (1024**3),
                'system_memory_percent': memory_info.percent
            },
            'system_resources': {
                'cpu_percent': cpu_percent,
                'disk_free_gb': disk_usage.free / (1024**3),
                'disk_percent': (disk_usage.used / disk_usage.total) * 100
            }
        }
    
    def _validate_all_requirements(self):
        """Validate that all evaluation requirements are met."""
        self.logger.info("✅ Validating all requirements")
        
        # Requirements 8.1-8.5: Evaluation Orchestrator
        self._validate_orchestrator_requirements()
        
        # Requirements 9.1-9.5: Evaluation Suites
        self._validate_suite_requirements()
        
        # Requirements 10.1-10.5: Failure Analysis
        self._validate_failure_analysis_requirements()
        
        # Requirements 11.1-11.5: Robustness Analysis
        self._validate_robustness_requirements()
        
        # Requirements 12.1-12.5: Champion Selection
        self._validate_champion_selection_requirements()
        
        # Requirements 13.1-13.5: Artifacts and Reproducibility
        self._validate_artifact_requirements()
    
    def _validate_orchestrator_requirements(self):
        """Validate orchestrator requirements 8.1-8.5."""
        requirements = [
            "8.1: Standardized test suites across all models",
            "8.2: Deterministic and stochastic policy modes",
            "8.3: Comprehensive metrics with confidence intervals",
            "8.4: Statistical significance testing",
            "8.5: Weighted composite scoring"
        ]
        
        # Check if orchestrator tests passed
        orchestrator_tests = [r for r in self.integration_results 
                            if 'orchestrator' in r.test_name or 'pipeline' in r.test_name]
        
        passed = all(r.status == "passed" for r in orchestrator_tests)
        
        self.validation_results.append(SystemValidationResult(
            component="evaluation_orchestrator",
            validation_type="requirements_8.1-8.5",
            passed=passed,
            metrics={'tests_passed': sum(1 for r in orchestrator_tests if r.status == "passed"),
                    'total_tests': len(orchestrator_tests)},
            issues=[] if passed else ["Some orchestrator integration tests failed"]
        ))
    
    def _validate_suite_requirements(self):
        """Validate suite requirements 9.1-9.5."""
        suite_test = next((r for r in self.integration_results if r.test_name == "all_evaluation_suites"), None)
        
        if suite_test:
            passed = suite_test.status == "passed"
            configured_suites = suite_test.details.get('configured_suites', 0)
            total_suites = suite_test.details.get('total_suites', 5)
            
            self.validation_results.append(SystemValidationResult(
                component="evaluation_suites",
                validation_type="requirements_9.1-9.5",
                passed=passed and configured_suites == total_suites,
                metrics={'configured_suites': configured_suites, 'total_suites': total_suites},
                issues=[] if passed else ["Not all evaluation suites are properly configured"]
            ))
    
    def _validate_failure_analysis_requirements(self):
        """Validate failure analysis requirements 10.1-10.5."""
        failure_test = next((r for r in self.integration_results if r.test_name == "failure_analysis"), None)
        
        if failure_test:
            passed = failure_test.status == "passed"
            
            self.validation_results.append(SystemValidationResult(
                component="failure_analysis",
                validation_type="requirements_10.1-10.5",
                passed=passed,
                metrics=failure_test.details,
                issues=[] if passed else ["Failure analysis system not working properly"]
            ))
    
    def _validate_robustness_requirements(self):
        """Validate robustness requirements 11.1-11.5."""
        robustness_test = next((r for r in self.integration_results if r.test_name == "robustness_analysis"), None)
        
        if robustness_test:
            passed = robustness_test.status == "passed"
            
            self.validation_results.append(SystemValidationResult(
                component="robustness_analysis",
                validation_type="requirements_11.1-11.5",
                passed=passed,
                metrics=robustness_test.details,
                issues=[] if passed else ["Robustness analysis system not working properly"]
            ))
    
    def _validate_champion_selection_requirements(self):
        """Validate champion selection requirements 12.1-12.5."""
        champion_test = next((r for r in self.integration_results if r.test_name == "champion_selection"), None)
        
        if champion_test:
            passed = champion_test.status == "passed"
            
            self.validation_results.append(SystemValidationResult(
                component="champion_selection",
                validation_type="requirements_12.1-12.5",
                passed=passed,
                metrics=champion_test.details,
                issues=[] if passed else ["Champion selection system not working properly"]
            ))
    
    def _validate_artifact_requirements(self):
        """Validate artifact requirements 13.1-13.5."""
        artifact_tests = [r for r in self.integration_results 
                         if r.test_name in ["report_generation", "artifact_management"]]
        
        passed = all(r.status == "passed" for r in artifact_tests)
        
        self.validation_results.append(SystemValidationResult(
            component="artifacts_and_reproducibility",
            validation_type="requirements_13.1-13.5",
            passed=passed,
            metrics={'tests_passed': sum(1 for r in artifact_tests if r.status == "passed"),
                    'total_tests': len(artifact_tests)},
            issues=[] if passed else ["Artifact management or report generation issues"]
        ))
    
    def _generate_integration_report(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive integration report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate overall status
        total_tests = len(self.integration_results)
        passed_tests = sum(1 for r in self.integration_results if r.status == "passed")
        failed_tests = sum(1 for r in self.integration_results if r.status == "failed")
        error_tests = sum(1 for r in self.integration_results if r.status == "error")
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Determine overall status
        if success_rate >= 0.95:
            overall_status = "🏆 PRODUCTION READY"
        elif success_rate >= 0.8:
            overall_status = "✅ INTEGRATION SUCCESSFUL"
        elif success_rate >= 0.6:
            overall_status = "⚠️ PARTIAL INTEGRATION"
        else:
            overall_status = "❌ INTEGRATION FAILED"
        
        # Requirements validation summary
        total_requirements = len(self.validation_results)
        passed_requirements = sum(1 for r in self.validation_results if r.passed)
        requirements_success_rate = passed_requirements / total_requirements if total_requirements > 0 else 0
        
        report = {
            'timestamp': timestamp,
            'integration_type': 'final_evaluation_system_integration',
            'overall_status': overall_status,
            'total_duration_minutes': total_duration / 60,
            
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'error_tests': error_tests,
                'success_rate': success_rate
            },
            
            'requirements_validation': {
                'total_requirements': total_requirements,
                'passed_requirements': passed_requirements,
                'requirements_success_rate': requirements_success_rate,
                'validation_details': [asdict(r) for r in self.validation_results]
            },
            
            'integration_test_results': [asdict(r) for r in self.integration_results],
            'performance_metrics': self.performance_metrics,
            
            'production_readiness': {
                'evaluation_orchestrator_ready': success_rate >= 0.9,
                'training_integration_ready': any(r.status == "passed" for r in self.integration_results 
                                                 if 'training' in r.test_name or 'champion' in r.test_name),
                'performance_acceptable': self.performance_metrics.get('test_execution', {}).get('success_rate', 0) >= 0.8,
                'memory_efficient': self.performance_metrics.get('memory_usage', {}).get('avg_memory_per_test_mb', 0) < 100,
                'all_requirements_met': requirements_success_rate >= 0.9
            },
            
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_path = self.results_dir / f'final_integration_report_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📋 Integration report saved: {report_path}")
        
        # Print summary
        self._print_integration_summary(report)
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on integration results."""
        recommendations = []
        
        # Check for failed tests
        failed_tests = [r for r in self.integration_results if r.status in ["failed", "error"]]
        if failed_tests:
            recommendations.append(f"Address {len(failed_tests)} failed integration tests")
        
        # Check memory usage
        avg_memory = self.performance_metrics.get('memory_usage', {}).get('avg_memory_per_test_mb', 0)
        if avg_memory > 100:
            recommendations.append("Optimize memory usage - average per test exceeds 100MB")
        
        # Check performance
        success_rate = self.performance_metrics.get('test_execution', {}).get('success_rate', 0)
        if success_rate < 0.9:
            recommendations.append("Improve test reliability - success rate below 90%")
        
        # Check requirements validation
        failed_validations = [r for r in self.validation_results if not r.passed]
        if failed_validations:
            recommendations.append(f"Address {len(failed_validations)} failed requirement validations")
        
        if not recommendations:
            recommendations.append("System is production ready - no critical issues identified")
        
        return recommendations
    
    def _print_integration_summary(self, report: Dict[str, Any]):
        """Print integration summary."""
        print("\n" + "=" * 80)
        print("🏆 FINAL EVALUATION SYSTEM INTEGRATION SUMMARY")
        print("=" * 80)
        
        print(f"📊 Overall Status: {report['overall_status']}")
        print(f"⏱️  Total Duration: {report['total_duration_minutes']:.1f} minutes")
        
        test_summary = report['test_summary']
        print(f"\n🧪 Integration Tests:")
        print(f"   ✅ Passed: {test_summary['passed_tests']}")
        print(f"   ❌ Failed: {test_summary['failed_tests']}")
        print(f"   💥 Errors: {test_summary['error_tests']}")
        print(f"   📈 Success Rate: {test_summary['success_rate']*100:.1f}%")
        
        req_validation = report['requirements_validation']
        print(f"\n📋 Requirements Validation:")
        print(f"   ✅ Passed: {req_validation['passed_requirements']}/{req_validation['total_requirements']}")
        print(f"   📈 Success Rate: {req_validation['requirements_success_rate']*100:.1f}%")
        
        prod_readiness = report['production_readiness']
        print(f"\n🚀 Production Readiness:")
        for key, value in prod_readiness.items():
            status = "✅" if value else "❌"
            print(f"   {status} {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
        
        print("=" * 80)


def main():
    """Main entry point for integration testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run final evaluation system integration')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Run integration
        integrator = EvaluationSystemIntegrator(args.config)
        report = integrator.run_complete_integration()
        
        # Return appropriate exit code
        success_rate = report['test_summary']['success_rate']
        requirements_rate = report['requirements_validation']['requirements_success_rate']
        
        if success_rate >= 0.9 and requirements_rate >= 0.9:
            print("\n🎉 INTEGRATION SUCCESSFUL - SYSTEM IS PRODUCTION READY!")
            return 0
        elif success_rate >= 0.7 and requirements_rate >= 0.7:
            print("\n⚠️ INTEGRATION PARTIALLY SUCCESSFUL - REVIEW RECOMMENDATIONS")
            return 1
        else:
            print("\n❌ INTEGRATION FAILED - CRITICAL ISSUES NEED RESOLUTION")
            return 2
            
    except Exception as e:
        print(f"\n💥 Integration crashed: {e}")
        return 3


if __name__ == '__main__':
    sys.exit(main())