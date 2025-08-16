#!/usr/bin/env python3
"""
Unit tests for the ReportGenerator class.

This module tests:
- Report generation functionality
- Leaderboard creation with confidence intervals
- Performance table generation
- Statistical comparison matrices
- Visualization generation
- Executive summary creation
"""

import unittest
import tempfile
import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.report_generator import (
    ReportGenerator, ReportConfig, LeaderboardEntry, PerformanceTable,
    ExecutiveSummary, EvaluationReport
)
from duckietown_utils.metrics_calculator import ModelMetrics, MetricResult, ConfidenceInterval
from duckietown_utils.champion_selector import ChampionSelectionResult, RankingResult, ParetoFront, ParetoPoint
from duckietown_utils.statistical_analyzer import ComparisonResult, MultipleComparisonResult
from duckietown_utils.robustness_analyzer import RobustnessAnalysisResult, RobustnessCurve, ParameterPoint
# Note: failure_analyzer returns Dict[str, Any] rather than a specific result class

class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'save_plots': False,  # Disable plotting for tests
            'generate_html': False
        }
        self.generator = ReportGenerator(self.config)
        
        # Override output directory to use temp directory
        self.generator.output_dir = Path(self.temp_dir)
        
        # Create sample model metrics
        self.sample_metrics = self._create_sample_metrics()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_sample_metrics(self):
        """Create sample model metrics for testing."""
        metrics_list = []
        
        for i, (model_id, base_score) in enumerate([
            ("model_a", 0.9),
            ("model_b", 0.8),
            ("model_c", 0.7)
        ]):
            # Primary metrics
            primary_metrics = {
                'success_rate': MetricResult(
                    name='success_rate',
                    value=base_score,
                    confidence_interval=ConfidenceInterval(
                        lower=base_score - 0.05,
                        upper=base_score + 0.05,
                        confidence_level=0.95,
                        method="wilson"
                    ),
                    sample_size=100
                ),
                'mean_reward': MetricResult(name='mean_reward', value=base_score * 0.8),
                'lateral_deviation': MetricResult(name='lateral_deviation', value=0.1 * (1 - base_score)),
                'smoothness': MetricResult(name='smoothness', value=0.05 * (1 - base_score)),
                'episode_length': MetricResult(name='episode_length', value=500 + i * 50)
            }
            
            # Secondary metrics
            secondary_metrics = {
                'stability': MetricResult(name='stability', value=base_score * 1.1)
            }
            
            # Safety metrics
            safety_metrics = {
                'collision_rate': MetricResult(name='collision_rate', value=0.1 * (1 - base_score)),
                'off_lane_rate': MetricResult(name='off_lane_rate', value=0.05 * (1 - base_score))
            }
            
            # Composite score
            composite_score = MetricResult(
                name='composite_score',
                value=base_score,
                confidence_interval=ConfidenceInterval(
                    lower=base_score - 0.02,
                    upper=base_score + 0.02,
                    confidence_level=0.95,
                    method="bootstrap"
                ),
                sample_size=100
            )
            
            # Per-map metrics
            per_map_metrics = {
                'map_1': {
                    'success_rate': MetricResult(name='success_rate', value=base_score + 0.05),
                    'lateral_deviation': MetricResult(name='lateral_deviation', value=0.08)
                },
                'map_2': {
                    'success_rate': MetricResult(name='success_rate', value=base_score - 0.05),
                    'lateral_deviation': MetricResult(name='lateral_deviation', value=0.12)
                }
            }
            
            # Per-suite metrics
            per_suite_metrics = {
                'base': {
                    'success_rate': MetricResult(name='success_rate', value=base_score),
                    'mean_reward': MetricResult(name='mean_reward', value=base_score * 0.8)
                },
                'hard': {
                    'success_rate': MetricResult(name='success_rate', value=base_score * 0.9),
                    'mean_reward': MetricResult(name='mean_reward', value=base_score * 0.7)
                }
            }
            
            metrics = ModelMetrics(
                model_id=model_id,
                primary_metrics=primary_metrics,
                secondary_metrics=secondary_metrics,
                safety_metrics=safety_metrics,
                composite_score=composite_score,
                per_map_metrics=per_map_metrics,
                per_suite_metrics=per_suite_metrics,
                metadata={'total_episodes': 100}
            )
            
            metrics_list.append(metrics)
        
        return metrics_list
    
    def test_initialization(self):
        """Test ReportGenerator initialization."""
        # Test default initialization
        generator = ReportGenerator()
        self.assertIsInstance(generator.config, ReportConfig)
        self.assertIsNotNone(generator.statistical_analyzer)
        
        # Test custom configuration
        custom_config = {
            'include_confidence_intervals': False,
            'plot_dpi': 150
        }
        generator = ReportGenerator(custom_config)
        self.assertFalse(generator.config.include_confidence_intervals)
        self.assertEqual(generator.config.plot_dpi, 150)
    
    def test_generate_leaderboard(self):
        """Test leaderboard generation."""
        leaderboard = self.generator._generate_leaderboard(self.sample_metrics)
        
        # Check basic structure
        self.assertEqual(len(leaderboard), 3)
        self.assertIsInstance(leaderboard[0], LeaderboardEntry)
        
        # Check sorting (highest composite score first)
        self.assertEqual(leaderboard[0].model_id, "model_a")
        self.assertEqual(leaderboard[1].model_id, "model_b")
        self.assertEqual(leaderboard[2].model_id, "model_c")
        
        # Check ranks
        self.assertEqual(leaderboard[0].rank, 1)
        self.assertEqual(leaderboard[1].rank, 2)
        self.assertEqual(leaderboard[2].rank, 3)
        
        # Check confidence intervals
        self.assertIsNotNone(leaderboard[0].composite_score_ci)
        self.assertIsNotNone(leaderboard[0].success_rate_ci)
    
    def test_generate_leaderboard_with_champion_selection(self):
        """Test leaderboard generation with champion selection results."""
        # Create mock champion selection result
        mock_ranking = Mock()
        mock_ranking.model_id = "model_a"
        mock_ranking.pareto_rank = 1
        mock_ranking.validation = Mock()
        mock_ranking.validation.status.value = "valid"
        mock_ranking.regression_analysis = None
        
        mock_champion_result = Mock()
        mock_champion_result.new_champion_id = "model_a"
        mock_champion_result.rankings = [mock_ranking]
        
        leaderboard = self.generator._generate_leaderboard(
            self.sample_metrics, mock_champion_result
        )
        
        # Check champion status
        champion_entry = next(e for e in leaderboard if e.model_id == "model_a")
        self.assertEqual(champion_entry.champion_status, "champion")
        self.assertEqual(champion_entry.pareto_rank, 1)
        self.assertEqual(champion_entry.validation_status, "valid")
    
    def test_generate_performance_tables(self):
        """Test performance table generation."""
        tables = self.generator._generate_performance_tables(self.sample_metrics)
        
        # Check that tables were generated
        self.assertIn('per_map_success_rate', tables)
        self.assertIn('per_suite_success_rate', tables)
        self.assertIn('overall_metrics', tables)
        
        # Check per-map table structure
        map_table = tables['per_map_success_rate']
        self.assertIsInstance(map_table, PerformanceTable)
        self.assertIsInstance(map_table.table_data, pd.DataFrame)
        self.assertEqual(map_table.table_data.shape[0], 2)  # 2 maps
        self.assertEqual(map_table.table_data.shape[1], 3)  # 3 models
        
        # Check best performers
        self.assertIsInstance(map_table.best_performers, dict)
        self.assertTrue(len(map_table.best_performers) > 0)
        
        # Check overall metrics table
        overall_table = tables['overall_metrics']
        self.assertTrue(overall_table.table_data.shape[0] >= 3)  # At least 3 metrics
        self.assertEqual(overall_table.table_data.shape[1], 3)  # 3 models
    
    def test_generate_statistical_comparisons(self):
        """Test statistical comparison generation."""
        comparisons = self.generator._generate_statistical_comparisons(self.sample_metrics)
        
        # Check structure
        self.assertIsInstance(comparisons, MultipleComparisonResult)
        self.assertEqual(len(comparisons.comparisons), 3)  # 3 choose 2 = 3 comparisons
        
        # Check comparison structure
        comparison = comparisons.comparisons[0]
        self.assertIsInstance(comparison, ComparisonResult)
        self.assertIn(comparison.model_a_id, ["model_a", "model_b", "model_c"])
        self.assertIn(comparison.model_b_id, ["model_a", "model_b", "model_c"])
        self.assertEqual(comparison.metric_name, 'composite_score')
    
    def test_generate_statistical_comparisons_single_model(self):
        """Test statistical comparison with single model."""
        single_model = [self.sample_metrics[0]]
        comparisons = self.generator._generate_statistical_comparisons(single_model)
        
        # Should return empty comparisons
        self.assertEqual(len(comparisons.comparisons), 0)
        self.assertEqual(comparisons.num_significant_before, 0)
        self.assertEqual(comparisons.num_significant_after, 0)
    
    def test_generate_executive_summary(self):
        """Test executive summary generation."""
        # Create mock inputs
        leaderboard = self.generator._generate_leaderboard(self.sample_metrics)
        performance_tables = self.generator._generate_performance_tables(self.sample_metrics)
        statistical_comparisons = self.generator._generate_statistical_comparisons(self.sample_metrics)
        
        summary = self.generator._generate_executive_summary(
            leaderboard, performance_tables, statistical_comparisons
        )
        
        # Check structure
        self.assertIsInstance(summary, ExecutiveSummary)
        self.assertEqual(summary.champion_model, "model_a")
        self.assertEqual(summary.total_models_evaluated, 3)
        
        # Check key findings
        self.assertTrue(len(summary.key_findings) > 0)
        self.assertIsInstance(summary.key_findings[0], str)
        
        # Check recommendations
        self.assertTrue(len(summary.recommendations) > 0)
        self.assertIsInstance(summary.recommendations[0], str)
        
        # Check risk assessment
        self.assertIsInstance(summary.risk_assessment, dict)
        self.assertIn('performance', summary.risk_assessment)
        
        # Check deployment readiness
        self.assertIn(summary.deployment_readiness, ['ready', 'conditional', 'not_ready'])
    
    def test_generate_executive_summary_empty_leaderboard(self):
        """Test executive summary with empty leaderboard."""
        summary = self.generator._generate_executive_summary(
            [], {}, MultipleComparisonResult([], "none", 0.05, 0, 0)
        )
        
        self.assertEqual(summary.champion_model, "none")
        self.assertEqual(summary.total_models_evaluated, 0)
        self.assertEqual(summary.deployment_readiness, "not_ready")
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_leaderboard_plot(self, mock_close, mock_savefig):
        """Test leaderboard plot generation."""
        leaderboard = self.generator._generate_leaderboard(self.sample_metrics)
        
        # Enable plotting for this test
        self.generator.config.save_plots = True
        
        plots = self.generator._generate_leaderboard_plot(leaderboard, Path(self.temp_dir))
        
        # Check that plot was generated
        self.assertIn('leaderboard', plots)
        self.assertTrue(plots['leaderboard'].endswith('.png'))
        
        # Check that matplotlib functions were called
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_generate_performance_heatmaps(self, mock_close, mock_savefig):
        """Test performance heatmap generation."""
        performance_tables = self.generator._generate_performance_tables(self.sample_metrics)
        
        # Enable plotting for this test
        self.generator.config.save_plots = True
        
        plots = self.generator._generate_performance_heatmaps(
            performance_tables, Path(self.temp_dir)
        )
        
        # Check that plots were generated
        self.assertTrue(len(plots) > 0)
        for plot_name, plot_path in plots.items():
            self.assertTrue(plot_name.startswith('heatmap_'))
            self.assertTrue(plot_path.endswith('.png'))
        
        # Check that matplotlib functions were called
        self.assertTrue(mock_savefig.call_count > 0)
        self.assertTrue(mock_close.call_count > 0)
    
    def test_compile_pareto_analysis(self):
        """Test Pareto analysis compilation."""
        # Create mock Pareto front
        mock_point = Mock()
        mock_point.model_id = "model_a"
        mock_point.coordinates = {'success_rate': 0.9, 'lateral_deviation': 0.1}
        mock_point.is_dominated = False
        
        mock_front = Mock()
        mock_front.axes = ['success_rate', 'lateral_deviation']
        mock_front.points = [mock_point]
        mock_front.non_dominated_models = ['model_a']
        mock_front.dominated_models = ['model_b']
        mock_front.trade_off_analysis = {'test': 'data'}
        
        mock_champion_result = Mock()
        mock_champion_result.pareto_fronts = [mock_front]
        
        analysis = self.generator._compile_pareto_analysis(mock_champion_result)
        
        # Check structure
        self.assertEqual(analysis['total_fronts_analyzed'], 1)
        self.assertEqual(len(analysis['fronts']), 1)
        
        front_data = analysis['fronts'][0]
        self.assertEqual(front_data['axes'], ['success_rate', 'lateral_deviation'])
        self.assertEqual(front_data['non_dominated_models'], ['model_a'])
        self.assertEqual(front_data['dominated_models'], ['model_b'])
    
    def test_compile_robustness_analysis(self):
        """Test robustness analysis compilation."""
        # Create mock robustness results
        mock_curve = Mock()
        mock_curve.auc_success_rate = 0.8
        
        mock_result = Mock()
        mock_result.overall_robustness_score = 0.85
        mock_result.parameter_curves = {'lighting': mock_curve}
        
        robustness_results = {'model_a': mock_result}
        
        analysis = self.generator._compile_robustness_analysis(robustness_results)
        
        # Check structure
        self.assertEqual(analysis['total_models_analyzed'], 1)
        self.assertEqual(len(analysis['overall_rankings']), 1)
        self.assertEqual(analysis['overall_rankings'][0], ('model_a', 0.85))
        
        # Check parameter analysis
        self.assertIn('lighting', analysis['parameter_analysis'])
        self.assertEqual(analysis['parameter_analysis']['lighting']['best_model'], 'model_a')
    
    def test_compile_failure_analysis(self):
        """Test failure analysis compilation."""
        # Create mock failure results matching the expected format
        mock_result = {
            'failure_statistics': {
                'failure_type_counts': {'collision': 10, 'off_lane': 5}
            }
        }
        
        failure_results = {'model_a': mock_result}
        
        analysis = self.generator._compile_failure_analysis(failure_results)
        
        # Check structure
        self.assertEqual(analysis['total_models_analyzed'], 1)
        self.assertIn('collision', analysis['failure_patterns'])
        self.assertIn('off_lane', analysis['failure_patterns'])
        
        # Check percentages
        total_failures = 15
        expected_collision_pct = (10 / total_failures) * 100
        self.assertAlmostEqual(analysis['failure_patterns']['collision'], expected_collision_pct)
    
    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        report = self.generator.generate_comprehensive_report(
            model_metrics_list=self.sample_metrics,
            report_id="test_report"
        )
        
        # Check basic structure
        self.assertIsInstance(report, EvaluationReport)
        self.assertEqual(report.report_id, "test_report")
        self.assertIsNotNone(report.generation_timestamp)
        
        # Check components
        self.assertEqual(len(report.leaderboard), 3)
        self.assertTrue(len(report.performance_tables) > 0)
        self.assertIsInstance(report.statistical_comparisons, MultipleComparisonResult)
        self.assertIsInstance(report.executive_summary, ExecutiveSummary)
        
        # Check metadata
        self.assertEqual(report.metadata['total_models'], 3)
        self.assertIn('generation_time', report.metadata)
    
    def test_generate_comprehensive_report_with_all_components(self):
        """Test comprehensive report generation with all optional components."""
        # Create mock champion selection result
        mock_champion_result = Mock()
        mock_champion_result.new_champion_id = "model_a"
        mock_champion_result.rankings = []
        mock_champion_result.pareto_fronts = []
        
        # Create mock robustness results
        mock_robustness = Mock()
        mock_robustness.overall_robustness_score = 0.8
        mock_robustness.parameter_curves = {}
        robustness_results = {'model_a': mock_robustness}
        
        # Create mock failure results
        mock_failure = {
            'failure_statistics': {
                'failure_type_counts': {'collision': 5}
            }
        }
        failure_results = {'model_a': mock_failure}
        
        report = self.generator.generate_comprehensive_report(
            model_metrics_list=self.sample_metrics,
            champion_selection_result=mock_champion_result,
            robustness_results=robustness_results,
            failure_results=failure_results,
            report_id="comprehensive_test"
        )
        
        # Check that all components are included
        self.assertIsNotNone(report.pareto_analysis)
        self.assertIsNotNone(report.robustness_analysis)
        self.assertIsNotNone(report.failure_analysis)
    
    def test_generate_comprehensive_report_empty_input(self):
        """Test comprehensive report generation with empty input."""
        with self.assertRaises(ValueError):
            self.generator.generate_comprehensive_report(
                model_metrics_list=[],
                report_id="empty_test"
            )
    
    def test_save_report(self):
        """Test report saving functionality."""
        report = self.generator.generate_comprehensive_report(
            model_metrics_list=self.sample_metrics,
            report_id="save_test"
        )
        
        report_dir = Path(self.temp_dir) / "save_test"
        report_dir.mkdir(exist_ok=True)
        
        self.generator._save_report(report, report_dir)
        
        # Check that files were created
        self.assertTrue((report_dir / "evaluation_report.json").exists())
        self.assertTrue((report_dir / "leaderboard.csv").exists())
        
        # Check JSON content
        with open(report_dir / "evaluation_report.json", 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data['report_id'], "save_test")
        self.assertEqual(len(saved_data['leaderboard']), 3)
        
        # Check CSV content
        leaderboard_df = pd.read_csv(report_dir / "leaderboard.csv")
        self.assertEqual(len(leaderboard_df), 3)
        self.assertIn('model_id', leaderboard_df.columns)
        self.assertIn('composite_score', leaderboard_df.columns)
    
    def test_report_config_validation(self):
        """Test report configuration validation."""
        # Test valid configuration
        valid_config = {
            'include_confidence_intervals': True,
            'plot_dpi': 300,
            'plot_format': 'pdf'
        }
        
        config = ReportConfig(**valid_config)
        self.assertTrue(config.include_confidence_intervals)
        self.assertEqual(config.plot_dpi, 300)
        self.assertEqual(config.plot_format, 'pdf')
        
        # Test default values
        default_config = ReportConfig()
        self.assertTrue(default_config.include_confidence_intervals)
        self.assertEqual(default_config.plot_dpi, 300)
        self.assertEqual(default_config.plot_format, 'png')
    
    def test_leaderboard_entry_creation(self):
        """Test LeaderboardEntry creation and validation."""
        entry = LeaderboardEntry(
            rank=1,
            model_id="test_model",
            composite_score=0.85,
            success_rate=0.90,
            mean_reward=0.75,
            lateral_deviation=0.08,
            smoothness=0.05,
            stability=0.92
        )
        
        self.assertEqual(entry.rank, 1)
        self.assertEqual(entry.model_id, "test_model")
        self.assertEqual(entry.composite_score, 0.85)
        self.assertEqual(entry.champion_status, "candidate")  # default
        self.assertEqual(entry.validation_status, "valid")  # default
    
    def test_performance_table_creation(self):
        """Test PerformanceTable creation and validation."""
        # Create sample data
        data = pd.DataFrame({
            'model_a': [0.9, 0.8],
            'model_b': [0.85, 0.75]
        }, index=['map_1', 'map_2'])
        
        table = PerformanceTable(
            metric_name='success_rate',
            table_data=data,
            best_performers={'map_1': 'model_a', 'map_2': 'model_a'}
        )
        
        self.assertEqual(table.metric_name, 'success_rate')
        self.assertEqual(table.table_data.shape, (2, 2))
        self.assertEqual(table.best_performers['map_1'], 'model_a')
    
    def test_executive_summary_creation(self):
        """Test ExecutiveSummary creation and validation."""
        summary = ExecutiveSummary(
            champion_model="test_champion",
            total_models_evaluated=5,
            evaluation_timestamp="2024-01-01T00:00:00",
            key_findings=["Finding 1", "Finding 2"],
            performance_highlights={'score': 0.9},
            recommendations=["Recommendation 1"],
            risk_assessment={'performance': 'Low'},
            deployment_readiness="ready"
        )
        
        self.assertEqual(summary.champion_model, "test_champion")
        self.assertEqual(summary.total_models_evaluated, 5)
        self.assertEqual(len(summary.key_findings), 2)
        self.assertEqual(len(summary.recommendations), 1)
        self.assertEqual(summary.deployment_readiness, "ready")

class TestReportGeneratorIntegration(unittest.TestCase):
    """Integration tests for ReportGenerator."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ReportGenerator({'save_plots': False, 'generate_html': False})
        self.generator.output_dir = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_report_generation(self):
        """Test complete end-to-end report generation workflow."""
        # Create comprehensive test data
        model_metrics = []
        
        for i, model_id in enumerate(["champion", "runner_up", "baseline"]):
            base_score = 0.9 - i * 0.1
            
            metrics = ModelMetrics(
                model_id=model_id,
                primary_metrics={
                    'success_rate': MetricResult(
                        name='success_rate',
                        value=base_score,
                        confidence_interval=ConfidenceInterval(
                            lower=base_score - 0.05,
                            upper=base_score + 0.05,
                            confidence_level=0.95,
                            method="wilson"
                        ),
                        sample_size=100
                    ),
                    'mean_reward': MetricResult(name='mean_reward', value=base_score * 0.8),
                    'lateral_deviation': MetricResult(name='lateral_deviation', value=0.1 * (1 - base_score)),
                    'smoothness': MetricResult(name='smoothness', value=0.05 * (1 - base_score)),
                    'episode_length': MetricResult(name='episode_length', value=500 + i * 50)
                },
                secondary_metrics={
                    'stability': MetricResult(name='stability', value=base_score * 1.1)
                },
                safety_metrics={
                    'collision_rate': MetricResult(name='collision_rate', value=0.1 * (1 - base_score))
                },
                composite_score=MetricResult(
                    name='composite_score',
                    value=base_score,
                    confidence_interval=ConfidenceInterval(
                        lower=base_score - 0.02,
                        upper=base_score + 0.02,
                        confidence_level=0.95,
                        method="bootstrap"
                    ),
                    sample_size=100
                ),
                per_map_metrics={
                    f'map_{j}': {
                        'success_rate': MetricResult(name='success_rate', value=base_score + 0.05 * j),
                        'lateral_deviation': MetricResult(name='lateral_deviation', value=0.08 + 0.02 * j)
                    }
                    for j in range(3)
                },
                per_suite_metrics={
                    suite: {
                        'success_rate': MetricResult(name='success_rate', value=base_score * (1 - 0.1 * k)),
                        'mean_reward': MetricResult(name='mean_reward', value=base_score * 0.8 * (1 - 0.1 * k))
                    }
                    for k, suite in enumerate(['base', 'hard', 'ood'])
                },
                metadata={'total_episodes': 100 + i * 50}
            )
            
            model_metrics.append(metrics)
        
        # Generate comprehensive report
        report = self.generator.generate_comprehensive_report(
            model_metrics_list=model_metrics,
            report_id="integration_test"
        )
        
        # Validate complete report structure
        self.assertEqual(report.report_id, "integration_test")
        self.assertEqual(len(report.leaderboard), 3)
        self.assertEqual(report.leaderboard[0].model_id, "champion")
        self.assertEqual(report.executive_summary.champion_model, "champion")
        self.assertIn("ready", report.executive_summary.deployment_readiness)
        
        # Validate performance tables
        self.assertTrue(len(report.performance_tables) >= 3)
        self.assertIn('overall_metrics', report.performance_tables)
        
        # Validate statistical comparisons
        self.assertEqual(len(report.statistical_comparisons.comparisons), 3)  # 3 choose 2
        
        # Validate metadata
        self.assertEqual(report.metadata['total_models'], 3)
        self.assertIsInstance(report.metadata['config_used'], dict)

if __name__ == '__main__':
    unittest.main()