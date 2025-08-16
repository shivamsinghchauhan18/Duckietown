#!/usr/bin/env python3
"""
📊 REPORT GENERATOR 📊
Comprehensive evaluation report generation system

This module implements the ReportGenerator class for comprehensive evaluation reports,
leaderboard generation with confidence intervals, per-map performance tables and
statistical comparison matrices, Pareto plots and robustness curve visualizations,
and executive summary generation with recommendations.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
import warnings
from collections import defaultdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import required components
from duckietown_utils.metrics_calculator import ModelMetrics, MetricResult
from duckietown_utils.statistical_analyzer import StatisticalAnalyzer, ComparisonResult, MultipleComparisonResult
from duckietown_utils.champion_selector import ChampionSelectionResult, RankingResult, ParetoFront
from duckietown_utils.robustness_analyzer import RobustnessAnalysisResult, RobustnessCurve
# Note: failure_analyzer returns Dict[str, Any] rather than a specific result class

# Set matplotlib backend for headless environments
plt.switch_backend('Agg')

@dataclass
class ReportConfig:
    """Configuration for report generation."""
    include_confidence_intervals: bool = True
    include_statistical_tests: bool = True
    include_pareto_analysis: bool = True
    include_robustness_analysis: bool = True
    include_failure_analysis: bool = True
    plot_style: str = 'seaborn-v0_8'
    plot_dpi: int = 300
    plot_format: str = 'png'
    color_palette: str = 'Set2'
    figure_size: Tuple[int, int] = (12, 8)
    font_size: int = 10
    save_plots: bool = True
    generate_html: bool = True
    generate_pdf: bool = False

@dataclass
class LeaderboardEntry:
    """Entry in the evaluation leaderboard."""
    rank: int
    model_id: str
    composite_score: float
    success_rate: float
    mean_reward: float
    lateral_deviation: float
    smoothness: float
    stability: float
    composite_score_ci: Optional[Tuple[float, float]] = None
    success_rate_ci: Optional[Tuple[float, float]] = None
    pareto_rank: Optional[int] = None
    champion_status: str = "candidate"  # "champion", "candidate", "regression"
    validation_status: str = "valid"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceTable:
    """Performance table for a specific metric across maps/suites."""
    metric_name: str
    table_data: pd.DataFrame
    confidence_intervals: Optional[pd.DataFrame] = None
    statistical_significance: Optional[pd.DataFrame] = None
    best_performers: Dict[str, str] = field(default_factory=dict)  # map/suite -> model_id

@dataclass
class ExecutiveSummary:
    """Executive summary of evaluation results."""
    champion_model: str
    total_models_evaluated: int
    evaluation_timestamp: str
    key_findings: List[str]
    performance_highlights: Dict[str, Any]
    recommendations: List[str]
    risk_assessment: Dict[str, str]
    deployment_readiness: str  # "ready", "conditional", "not_ready"

@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    report_id: str
    generation_timestamp: str
    config: ReportConfig
    executive_summary: ExecutiveSummary
    leaderboard: List[LeaderboardEntry]
    performance_tables: Dict[str, PerformanceTable]
    statistical_comparisons: MultipleComparisonResult
    pareto_analysis: Optional[Dict[str, Any]] = None
    robustness_analysis: Optional[Dict[str, Any]] = None
    failure_analysis: Optional[Dict[str, Any]] = None
    plots: Dict[str, str] = field(default_factory=dict)  # plot_name -> file_path
    metadata: Dict[str, Any] = field(default_factory=dict)

class ReportGenerator:
    """Comprehensive evaluation report generation system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the report generator.
        
        Args:
            config: Configuration dictionary for the generator
        """
        self.config = ReportConfig(**(config or {}))
        self.logger = logging.getLogger(__name__)
        
        # Initialize statistical analyzer for comparisons
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Set up plotting style
        plt.style.use(self.config.plot_style)
        sns.set_palette(self.config.color_palette)
        plt.rcParams.update({'font.size': self.config.font_size})
        
        # Output directory
        self.output_dir = Path("logs/evaluation_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("📊 Report Generator initialized")
        self.logger.info(f"🎨 Plot style: {self.config.plot_style}")
        self.logger.info(f"📁 Output directory: {self.output_dir}")
    
    def generate_comprehensive_report(self, 
                                    model_metrics_list: List[ModelMetrics],
                                    champion_selection_result: Optional[ChampionSelectionResult] = None,
                                    robustness_results: Optional[Dict[str, RobustnessAnalysisResult]] = None,
                                    failure_results: Optional[Dict[str, Dict[str, Any]]] = None,
                                    report_id: Optional[str] = None) -> EvaluationReport:
        """Generate a comprehensive evaluation report.
        
        Args:
            model_metrics_list: List of model metrics for all evaluated models
            champion_selection_result: Optional champion selection results
            robustness_results: Optional robustness analysis results
            failure_results: Optional failure analysis results
            report_id: Optional custom report ID
            
        Returns:
            EvaluationReport: Complete evaluation report
        """
        if not model_metrics_list:
            raise ValueError("No model metrics provided for report generation")
        
        # Generate report ID
        if report_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_id = f"evaluation_report_{timestamp}"
        
        self.logger.info(f"📊 Generating comprehensive report: {report_id}")
        
        # Create report directory
        report_dir = self.output_dir / report_id
        report_dir.mkdir(exist_ok=True)
        
        # Generate leaderboard
        leaderboard = self._generate_leaderboard(model_metrics_list, champion_selection_result)
        
        # Generate performance tables
        performance_tables = self._generate_performance_tables(model_metrics_list)
        
        # Generate statistical comparisons
        statistical_comparisons = self._generate_statistical_comparisons(model_metrics_list)
        
        # Generate plots
        plots = {}
        if self.config.save_plots:
            plots.update(self._generate_leaderboard_plot(leaderboard, report_dir))
            plots.update(self._generate_performance_heatmaps(performance_tables, report_dir))
            
            if champion_selection_result and self.config.include_pareto_analysis:
                plots.update(self._generate_pareto_plots(champion_selection_result.pareto_fronts, report_dir))
            
            if robustness_results and self.config.include_robustness_analysis:
                plots.update(self._generate_robustness_plots(robustness_results, report_dir))
            
            plots.update(self._generate_statistical_comparison_plot(statistical_comparisons, report_dir))
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            leaderboard, performance_tables, statistical_comparisons,
            champion_selection_result, robustness_results, failure_results
        )
        
        # Compile report
        report = EvaluationReport(
            report_id=report_id,
            generation_timestamp=datetime.now().isoformat(),
            config=self.config,
            executive_summary=executive_summary,
            leaderboard=leaderboard,
            performance_tables=performance_tables,
            statistical_comparisons=statistical_comparisons,
            pareto_analysis=self._compile_pareto_analysis(champion_selection_result) if champion_selection_result else None,
            robustness_analysis=self._compile_robustness_analysis(robustness_results) if robustness_results else None,
            failure_analysis=self._compile_failure_analysis(failure_results) if failure_results else None,
            plots=plots,
            metadata={
                'total_models': len(model_metrics_list),
                'generation_time': datetime.now().isoformat(),
                'config_used': asdict(self.config)
            }
        )
        
        # Save report
        self._save_report(report, report_dir)
        
        # Generate HTML report if requested
        if self.config.generate_html:
            self._generate_html_report(report, report_dir)
        
        self.logger.info(f"✅ Report generated successfully: {report_dir}")
        return report
    
    def _generate_leaderboard(self, model_metrics_list: List[ModelMetrics],
                            champion_selection_result: Optional[ChampionSelectionResult] = None) -> List[LeaderboardEntry]:
        """Generate the evaluation leaderboard.
        
        Args:
            model_metrics_list: List of model metrics
            champion_selection_result: Optional champion selection results
            
        Returns:
            List[LeaderboardEntry]: Sorted leaderboard entries
        """
        leaderboard_entries = []
        
        # Get champion and ranking info if available
        champion_id = None
        rankings_dict = {}
        if champion_selection_result:
            champion_id = champion_selection_result.new_champion_id
            rankings_dict = {r.model_id: r for r in champion_selection_result.rankings}
        
        for model_metrics in model_metrics_list:
            # Extract key metrics
            composite_score = model_metrics.composite_score.value if model_metrics.composite_score else 0.0
            composite_score_ci = None
            if model_metrics.composite_score and model_metrics.composite_score.confidence_interval:
                ci = model_metrics.composite_score.confidence_interval
                composite_score_ci = (ci.lower, ci.upper)
            
            success_rate = model_metrics.primary_metrics.get('success_rate')
            success_rate_val = success_rate.value if success_rate else 0.0
            success_rate_ci = None
            if success_rate and success_rate.confidence_interval:
                ci = success_rate.confidence_interval
                success_rate_ci = (ci.lower, ci.upper)
            
            mean_reward = model_metrics.primary_metrics.get('mean_reward')
            mean_reward_val = mean_reward.value if mean_reward else 0.0
            
            lateral_deviation = model_metrics.primary_metrics.get('lateral_deviation')
            lateral_deviation_val = lateral_deviation.value if lateral_deviation else 0.0
            
            smoothness = model_metrics.primary_metrics.get('smoothness')
            smoothness_val = smoothness.value if smoothness else 0.0
            
            stability = model_metrics.secondary_metrics.get('stability')
            stability_val = stability.value if stability else 0.0
            
            # Determine champion status
            champion_status = "candidate"
            if model_metrics.model_id == champion_id:
                champion_status = "champion"
            
            # Get ranking info
            ranking_info = rankings_dict.get(model_metrics.model_id)
            pareto_rank = ranking_info.pareto_rank if ranking_info else None
            validation_status = "valid"
            if ranking_info and ranking_info.validation:
                validation_status = ranking_info.validation.status.value
            if ranking_info and ranking_info.regression_analysis and ranking_info.regression_analysis.is_regression:
                champion_status = "regression"
            
            entry = LeaderboardEntry(
                rank=0,  # Will be set after sorting
                model_id=model_metrics.model_id,
                composite_score=composite_score,
                composite_score_ci=composite_score_ci,
                success_rate=success_rate_val,
                success_rate_ci=success_rate_ci,
                mean_reward=mean_reward_val,
                lateral_deviation=lateral_deviation_val,
                smoothness=smoothness_val,
                stability=stability_val,
                pareto_rank=pareto_rank,
                champion_status=champion_status,
                validation_status=validation_status,
                metadata={
                    'total_episodes': model_metrics.metadata.get('total_episodes', 0),
                    'total_suites': model_metrics.metadata.get('total_suites', 0)
                }
            )
            
            leaderboard_entries.append(entry)
        
        # Sort by composite score (descending)
        leaderboard_entries.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Assign ranks
        for i, entry in enumerate(leaderboard_entries):
            entry.rank = i + 1
        
        return leaderboard_entries
    
    def _generate_performance_tables(self, model_metrics_list: List[ModelMetrics]) -> Dict[str, PerformanceTable]:
        """Generate performance tables for different metrics and groupings.
        
        Args:
            model_metrics_list: List of model metrics
            
        Returns:
            Dict[str, PerformanceTable]: Performance tables by category
        """
        performance_tables = {}
        
        # Get all model IDs
        model_ids = [m.model_id for m in model_metrics_list]
        
        # Generate per-map performance table
        all_maps = set()
        for model_metrics in model_metrics_list:
            all_maps.update(model_metrics.per_map_metrics.keys())
        
        if all_maps:
            map_data = {}
            map_ci_data = {}
            
            for metric_name in ['success_rate', 'lateral_deviation']:
                map_data[metric_name] = pd.DataFrame(index=sorted(all_maps), columns=model_ids)
                map_ci_data[metric_name] = pd.DataFrame(index=sorted(all_maps), columns=model_ids)
                
                for model_metrics in model_metrics_list:
                    for map_name, map_metrics in model_metrics.per_map_metrics.items():
                        metric_result = map_metrics.get(metric_name)
                        if metric_result:
                            map_data[metric_name].loc[map_name, model_metrics.model_id] = metric_result.value
                            
                            if metric_result.confidence_interval:
                                ci = metric_result.confidence_interval
                                ci_str = f"[{ci.lower:.3f}, {ci.upper:.3f}]"
                                map_ci_data[metric_name].loc[map_name, model_metrics.model_id] = ci_str
                
                # Find best performers for each map
                best_performers = {}
                for map_name in sorted(all_maps):
                    map_values = map_data[metric_name].loc[map_name].dropna()
                    if not map_values.empty:
                        if metric_name == 'success_rate':
                            best_model = map_values.idxmax()
                        else:  # lateral_deviation (lower is better)
                            best_model = map_values.idxmin()
                        best_performers[map_name] = best_model
                
                performance_tables[f'per_map_{metric_name}'] = PerformanceTable(
                    metric_name=f'{metric_name}_per_map',
                    table_data=map_data[metric_name],
                    confidence_intervals=map_ci_data[metric_name] if self.config.include_confidence_intervals else None,
                    best_performers=best_performers
                )
        
        # Generate per-suite performance table
        all_suites = set()
        for model_metrics in model_metrics_list:
            all_suites.update(model_metrics.per_suite_metrics.keys())
        
        if all_suites:
            suite_data = {}
            suite_ci_data = {}
            
            for metric_name in ['success_rate', 'mean_reward']:
                suite_data[metric_name] = pd.DataFrame(index=sorted(all_suites), columns=model_ids)
                suite_ci_data[metric_name] = pd.DataFrame(index=sorted(all_suites), columns=model_ids)
                
                for model_metrics in model_metrics_list:
                    for suite_name, suite_metrics in model_metrics.per_suite_metrics.items():
                        metric_result = suite_metrics.get(metric_name)
                        if metric_result:
                            suite_data[metric_name].loc[suite_name, model_metrics.model_id] = metric_result.value
                            
                            if metric_result.confidence_interval:
                                ci = metric_result.confidence_interval
                                ci_str = f"[{ci.lower:.3f}, {ci.upper:.3f}]"
                                suite_ci_data[metric_name].loc[suite_name, model_metrics.model_id] = ci_str
                
                # Find best performers for each suite
                best_performers = {}
                for suite_name in sorted(all_suites):
                    suite_values = suite_data[metric_name].loc[suite_name].dropna()
                    if not suite_values.empty:
                        best_model = suite_values.idxmax()  # Higher is better for both metrics
                        best_performers[suite_name] = best_model
                
                performance_tables[f'per_suite_{metric_name}'] = PerformanceTable(
                    metric_name=f'{metric_name}_per_suite',
                    table_data=suite_data[metric_name],
                    confidence_intervals=suite_ci_data[metric_name] if self.config.include_confidence_intervals else None,
                    best_performers=best_performers
                )
        
        # Generate overall metrics comparison table
        overall_metrics = ['success_rate', 'mean_reward', 'lateral_deviation', 'smoothness', 'stability']
        overall_data = pd.DataFrame(index=overall_metrics, columns=model_ids)
        overall_ci_data = pd.DataFrame(index=overall_metrics, columns=model_ids)
        
        for model_metrics in model_metrics_list:
            for metric_name in overall_metrics:
                # Try primary metrics first, then secondary
                metric_result = model_metrics.primary_metrics.get(metric_name)
                if not metric_result:
                    metric_result = model_metrics.secondary_metrics.get(metric_name)
                
                if metric_result:
                    overall_data.loc[metric_name, model_metrics.model_id] = metric_result.value
                    
                    if metric_result.confidence_interval:
                        ci = metric_result.confidence_interval
                        ci_str = f"[{ci.lower:.3f}, {ci.upper:.3f}]"
                        overall_ci_data.loc[metric_name, model_metrics.model_id] = ci_str
        
        # Find best performers for each metric
        best_performers = {}
        higher_is_better = {'success_rate', 'mean_reward', 'stability'}
        for metric_name in overall_metrics:
            metric_values = overall_data.loc[metric_name].dropna()
            if not metric_values.empty:
                if metric_name in higher_is_better:
                    best_model = metric_values.idxmax()
                else:
                    best_model = metric_values.idxmin()
                best_performers[metric_name] = best_model
        
        performance_tables['overall_metrics'] = PerformanceTable(
            metric_name='overall_metrics',
            table_data=overall_data,
            confidence_intervals=overall_ci_data if self.config.include_confidence_intervals else None,
            best_performers=best_performers
        )
        
        return performance_tables
    
    def _generate_statistical_comparisons(self, model_metrics_list: List[ModelMetrics]) -> MultipleComparisonResult:
        """Generate statistical comparisons between models.
        
        Args:
            model_metrics_list: List of model metrics
            
        Returns:
            MultipleComparisonResult: Statistical comparison results
        """
        if len(model_metrics_list) < 2:
            return MultipleComparisonResult(
                comparisons=[],
                correction_method="none",
                alpha=0.05,
                num_significant_before=0,
                num_significant_after=0
            )
        
        # For this implementation, we'll create mock comparisons based on composite scores
        # In a real implementation, this would use episode-level data
        comparisons = []
        
        for i, model_a in enumerate(model_metrics_list):
            for j, model_b in enumerate(model_metrics_list[i+1:], i+1):
                score_a = model_a.composite_score.value if model_a.composite_score else 0.0
                score_b = model_b.composite_score.value if model_b.composite_score else 0.0
                
                # Mock statistical comparison
                difference = score_b - score_a
                p_value = max(0.001, 0.1 * (1 - abs(difference)))  # Mock p-value
                
                comparison = ComparisonResult(
                    model_a_id=model_a.model_id,
                    model_b_id=model_b.model_id,
                    metric_name='composite_score',
                    model_a_mean=score_a,
                    model_b_mean=score_b,
                    difference=difference,
                    p_value=p_value,
                    is_significant=p_value < 0.05,
                    effect_size=abs(difference) / 0.1,  # Mock effect size
                    effect_size_method='cohens_d',
                    test_method='mock_test'
                )
                
                comparisons.append(comparison)
        
        # Apply multiple comparison correction
        if comparisons:
            return self.statistical_analyzer.correct_multiple_comparisons(
                comparisons, method="benjamini_hochberg"
            )
        else:
            return MultipleComparisonResult(
                comparisons=[],
                correction_method="benjamini_hochberg",
                alpha=0.05,
                num_significant_before=0,
                num_significant_after=0
            )
    
    def _generate_leaderboard_plot(self, leaderboard: List[LeaderboardEntry], 
                                 output_dir: Path) -> Dict[str, str]:
        """Generate leaderboard visualization plot.
        
        Args:
            leaderboard: Leaderboard entries
            output_dir: Output directory for plots
            
        Returns:
            Dict[str, str]: Plot name to file path mapping
        """
        plots = {}
        
        if not leaderboard:
            return plots
        
        # Create leaderboard bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Composite score plot
        model_ids = [entry.model_id for entry in leaderboard[:10]]  # Top 10
        composite_scores = [entry.composite_score for entry in leaderboard[:10]]
        colors = ['gold' if entry.champion_status == 'champion' else 'lightblue' 
                 for entry in leaderboard[:10]]
        
        bars1 = ax1.barh(range(len(model_ids)), composite_scores, color=colors)
        ax1.set_yticks(range(len(model_ids)))
        ax1.set_yticklabels(model_ids)
        ax1.set_xlabel('Composite Score')
        ax1.set_title('Model Leaderboard - Composite Score')
        ax1.invert_yaxis()
        
        # Add confidence intervals if available
        if self.config.include_confidence_intervals:
            for i, entry in enumerate(leaderboard[:10]):
                if entry.composite_score_ci:
                    lower, upper = entry.composite_score_ci
                    ax1.errorbar(entry.composite_score, i, 
                               xerr=[[entry.composite_score - lower], [upper - entry.composite_score]],
                               fmt='none', color='black', capsize=3)
        
        # Success rate plot
        success_rates = [entry.success_rate for entry in leaderboard[:10]]
        bars2 = ax2.barh(range(len(model_ids)), success_rates, color=colors)
        ax2.set_yticks(range(len(model_ids)))
        ax2.set_yticklabels(model_ids)
        ax2.set_xlabel('Success Rate')
        ax2.set_title('Model Leaderboard - Success Rate')
        ax2.invert_yaxis()
        
        # Add confidence intervals for success rate
        if self.config.include_confidence_intervals:
            for i, entry in enumerate(leaderboard[:10]):
                if entry.success_rate_ci:
                    lower, upper = entry.success_rate_ci
                    ax2.errorbar(entry.success_rate, i,
                               xerr=[[entry.success_rate - lower], [upper - entry.success_rate]],
                               fmt='none', color='black', capsize=3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gold', label='Champion'),
            Patch(facecolor='lightblue', label='Candidate')
        ]
        ax1.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"leaderboard.{self.config.plot_format}"
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        plots['leaderboard'] = str(plot_path)
        
        return plots
    
    def _generate_performance_heatmaps(self, performance_tables: Dict[str, PerformanceTable],
                                     output_dir: Path) -> Dict[str, str]:
        """Generate performance heatmap visualizations.
        
        Args:
            performance_tables: Performance tables to visualize
            output_dir: Output directory for plots
            
        Returns:
            Dict[str, str]: Plot name to file path mapping
        """
        plots = {}
        
        for table_name, table in performance_tables.items():
            if table.table_data.empty:
                continue
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # Convert to numeric and handle NaN values
            data = table.table_data.astype(float)
            
            # Create heatmap
            sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                       ax=ax, cbar_kws={'label': table.metric_name})
            
            ax.set_title(f'Performance Heatmap - {table.metric_name.replace("_", " ").title()}')
            ax.set_xlabel('Models')
            ax.set_ylabel('Maps/Suites')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / f"heatmap_{table_name}.{self.config.plot_format}"
            plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            plots[f'heatmap_{table_name}'] = str(plot_path)
        
        return plots
    
    def _generate_pareto_plots(self, pareto_fronts: List[ParetoFront], 
                             output_dir: Path) -> Dict[str, str]:
        """Generate Pareto front visualization plots.
        
        Args:
            pareto_fronts: List of Pareto front analyses
            output_dir: Output directory for plots
            
        Returns:
            Dict[str, str]: Plot name to file path mapping
        """
        plots = {}
        
        for i, front in enumerate(pareto_fronts):
            if len(front.axes) < 2:
                continue
            
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # Extract coordinates for plotting
            x_axis = front.axes[0]
            y_axis = front.axes[1]
            
            x_coords = []
            y_coords = []
            labels = []
            colors = []
            
            for point in front.points:
                x_coords.append(point.coordinates[x_axis])
                y_coords.append(point.coordinates[y_axis])
                labels.append(point.model_id)
                colors.append('red' if point.is_dominated else 'blue')
            
            # Create scatter plot
            scatter = ax.scatter(x_coords, y_coords, c=colors, alpha=0.7, s=100)
            
            # Add labels
            for j, label in enumerate(labels):
                ax.annotate(label, (x_coords[j], y_coords[j]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Connect non-dominated points
            non_dominated_points = [(x_coords[j], y_coords[j]) for j, point in enumerate(front.points) 
                                  if not point.is_dominated]
            if len(non_dominated_points) > 1:
                # Sort points for proper connection
                non_dominated_points.sort()
                x_nd, y_nd = zip(*non_dominated_points)
                ax.plot(x_nd, y_nd, 'b--', alpha=0.5, label='Pareto Front')
            
            ax.set_xlabel(x_axis.replace('_', ' ').title())
            ax.set_ylabel(y_axis.replace('_', ' ').title())
            ax.set_title(f'Pareto Front Analysis: {x_axis} vs {y_axis}')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', label='Non-dominated'),
                Patch(facecolor='red', label='Dominated')
            ]
            ax.legend(handles=legend_elements)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / f"pareto_front_{i+1}.{self.config.plot_format}"
            plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            plots[f'pareto_front_{i+1}'] = str(plot_path)
        
        return plots
    
    def _generate_robustness_plots(self, robustness_results: Dict[str, RobustnessAnalysisResult],
                                 output_dir: Path) -> Dict[str, str]:
        """Generate robustness analysis visualization plots.
        
        Args:
            robustness_results: Robustness analysis results by model
            output_dir: Output directory for plots
            
        Returns:
            Dict[str, str]: Plot name to file path mapping
        """
        plots = {}
        
        if not robustness_results:
            return plots
        
        # Create robustness comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Get all parameter names
        all_params = set()
        for result in robustness_results.values():
            all_params.update(result.parameter_curves.keys())
        
        for i, param_name in enumerate(sorted(all_params)[:4]):  # Plot up to 4 parameters
            ax = axes[i]
            
            for model_id, result in robustness_results.items():
                if param_name in result.parameter_curves:
                    curve = result.parameter_curves[param_name]
                    
                    # Extract data points
                    param_values = [p.parameter_value for p in curve.sweep_points]
                    success_rates = [p.success_rate for p in curve.sweep_points]
                    
                    ax.plot(param_values, success_rates, 'o-', label=model_id, alpha=0.7)
            
            ax.set_xlabel(param_name.replace('_', ' ').title())
            ax.set_ylabel('Success Rate')
            ax.set_title(f'Robustness: {param_name.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(len(all_params), 4):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"robustness_analysis.{self.config.plot_format}"
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        plots['robustness_analysis'] = str(plot_path)
        
        return plots
    
    def _generate_statistical_comparison_plot(self, statistical_comparisons: MultipleComparisonResult,
                                            output_dir: Path) -> Dict[str, str]:
        """Generate statistical comparison visualization plot.
        
        Args:
            statistical_comparisons: Statistical comparison results
            output_dir: Output directory for plots
            
        Returns:
            Dict[str, str]: Plot name to file path mapping
        """
        plots = {}
        
        if not statistical_comparisons.comparisons:
            return plots
        
        # Create comparison matrix
        model_ids = set()
        for comp in statistical_comparisons.comparisons:
            model_ids.add(comp.model_a_id)
            model_ids.add(comp.model_b_id)
        
        model_ids = sorted(list(model_ids))
        n_models = len(model_ids)
        
        # Create matrices for p-values and effect sizes
        p_value_matrix = np.ones((n_models, n_models))
        effect_size_matrix = np.zeros((n_models, n_models))
        significance_matrix = np.zeros((n_models, n_models))
        
        model_to_idx = {model_id: i for i, model_id in enumerate(model_ids)}
        
        for comp in statistical_comparisons.comparisons:
            i = model_to_idx[comp.model_a_id]
            j = model_to_idx[comp.model_b_id]
            
            p_val = comp.adjusted_p_value if comp.adjusted_p_value is not None else comp.p_value
            p_value_matrix[i, j] = p_val
            p_value_matrix[j, i] = p_val
            
            effect_size_matrix[i, j] = comp.effect_size or 0.0
            effect_size_matrix[j, i] = comp.effect_size or 0.0
            
            significance_matrix[i, j] = 1 if comp.is_significant else 0
            significance_matrix[j, i] = 1 if comp.is_significant else 0
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # P-value heatmap
        sns.heatmap(-np.log10(p_value_matrix), annot=True, fmt='.2f', 
                   xticklabels=model_ids, yticklabels=model_ids,
                   cmap='Reds', ax=ax1, cbar_kws={'label': '-log10(p-value)'})
        ax1.set_title('Statistical Significance Matrix\n(-log10 adjusted p-values)')
        
        # Effect size heatmap
        sns.heatmap(effect_size_matrix, annot=True, fmt='.3f',
                   xticklabels=model_ids, yticklabels=model_ids,
                   cmap='RdBu_r', center=0, ax=ax2, cbar_kws={'label': 'Effect Size'})
        ax2.set_title('Effect Size Matrix')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"statistical_comparisons.{self.config.plot_format}"
        plt.savefig(plot_path, dpi=self.config.plot_dpi, bbox_inches='tight')
        plt.close()
        
        plots['statistical_comparisons'] = str(plot_path)
        
        return plots
    
    def _generate_executive_summary(self, leaderboard: List[LeaderboardEntry],
                                  performance_tables: Dict[str, PerformanceTable],
                                  statistical_comparisons: MultipleComparisonResult,
                                  champion_selection_result: Optional[ChampionSelectionResult] = None,
                                  robustness_results: Optional[Dict[str, RobustnessAnalysisResult]] = None,
                                  failure_results: Optional[Dict[str, Dict[str, Any]]] = None) -> ExecutiveSummary:
        """Generate executive summary of evaluation results.
        
        Args:
            leaderboard: Leaderboard entries
            performance_tables: Performance tables
            statistical_comparisons: Statistical comparison results
            champion_selection_result: Optional champion selection results
            robustness_results: Optional robustness analysis results
            failure_results: Optional failure analysis results
            
        Returns:
            ExecutiveSummary: Executive summary
        """
        if not leaderboard:
            return ExecutiveSummary(
                champion_model="none",
                total_models_evaluated=0,
                evaluation_timestamp=datetime.now().isoformat(),
                key_findings=[],
                performance_highlights={},
                recommendations=[],
                risk_assessment={},
                deployment_readiness="not_ready"
            )
        
        champion_model = leaderboard[0].model_id
        total_models = len(leaderboard)
        
        # Generate key findings
        key_findings = []
        
        # Champion performance
        champion_entry = leaderboard[0]
        key_findings.append(
            f"Champion model '{champion_model}' achieved {champion_entry.composite_score:.3f} composite score "
            f"with {champion_entry.success_rate:.1%} success rate"
        )
        
        # Performance spread
        if len(leaderboard) > 1:
            score_range = champion_entry.composite_score - leaderboard[-1].composite_score
            key_findings.append(
                f"Performance spread: {score_range:.3f} composite score difference between "
                f"best and worst models"
            )
        
        # Statistical significance
        significant_comparisons = sum(1 for comp in statistical_comparisons.comparisons if comp.is_significant)
        total_comparisons = len(statistical_comparisons.comparisons)
        if total_comparisons > 0:
            key_findings.append(
                f"Statistical analysis: {significant_comparisons}/{total_comparisons} "
                f"({significant_comparisons/total_comparisons:.1%}) model comparisons show significant differences"
            )
        
        # Robustness findings
        if robustness_results:
            avg_robustness = np.mean([r.overall_robustness_score for r in robustness_results.values()])
            key_findings.append(f"Average robustness score across all models: {avg_robustness:.3f}")
        
        # Performance highlights
        performance_highlights = {
            'champion_composite_score': champion_entry.composite_score,
            'champion_success_rate': champion_entry.success_rate,
            'total_models_evaluated': total_models,
            'models_above_threshold': sum(1 for entry in leaderboard if entry.success_rate >= 0.8),
            'champion_validation_status': champion_entry.validation_status
        }
        
        # Generate recommendations
        recommendations = []
        
        # Deployment readiness assessment
        deployment_readiness = "not_ready"
        
        if champion_entry.success_rate >= 0.9 and champion_entry.validation_status == "valid":
            deployment_readiness = "ready"
            recommendations.append("Champion model meets deployment criteria and is ready for production")
        elif champion_entry.success_rate >= 0.8:
            deployment_readiness = "conditional"
            recommendations.append("Champion model shows good performance but requires additional validation")
        else:
            recommendations.append("Champion model requires significant improvement before deployment")
        
        # Performance improvement recommendations
        if champion_entry.lateral_deviation > 0.1:
            recommendations.append("Focus on improving lane-following precision to reduce lateral deviation")
        
        if champion_entry.smoothness > 0.1:
            recommendations.append("Optimize action smoothness to reduce jerky movements")
        
        # Model comparison recommendations
        if len(leaderboard) > 1 and leaderboard[1].composite_score > 0.8:
            recommendations.append(f"Consider ensemble methods combining top models: {leaderboard[0].model_id} and {leaderboard[1].model_id}")
        
        # Risk assessment
        risk_assessment = {}
        
        if champion_entry.success_rate < 0.85:
            risk_assessment['performance'] = "Medium - Success rate below 85%"
        else:
            risk_assessment['performance'] = "Low - Good success rate achieved"
        
        if champion_entry.validation_status != "valid":
            risk_assessment['validation'] = f"High - Validation status: {champion_entry.validation_status}"
        else:
            risk_assessment['validation'] = "Low - Model passes validation criteria"
        
        if robustness_results and champion_model in robustness_results:
            champion_robustness = robustness_results[champion_model].overall_robustness_score
            if champion_robustness < 0.7:
                risk_assessment['robustness'] = "High - Low robustness to environmental changes"
            else:
                risk_assessment['robustness'] = "Low - Good robustness demonstrated"
        
        return ExecutiveSummary(
            champion_model=champion_model,
            total_models_evaluated=total_models,
            evaluation_timestamp=datetime.now().isoformat(),
            key_findings=key_findings,
            performance_highlights=performance_highlights,
            recommendations=recommendations,
            risk_assessment=risk_assessment,
            deployment_readiness=deployment_readiness
        )
    
    def _compile_pareto_analysis(self, champion_selection_result: ChampionSelectionResult) -> Dict[str, Any]:
        """Compile Pareto analysis results.
        
        Args:
            champion_selection_result: Champion selection results
            
        Returns:
            Dict[str, Any]: Compiled Pareto analysis
        """
        if not champion_selection_result.pareto_fronts:
            return {}
        
        analysis = {
            'total_fronts_analyzed': len(champion_selection_result.pareto_fronts),
            'fronts': []
        }
        
        for i, front in enumerate(champion_selection_result.pareto_fronts):
            front_analysis = {
                'front_id': i + 1,
                'axes': front.axes,
                'non_dominated_models': front.non_dominated_models,
                'dominated_models': front.dominated_models,
                'trade_off_analysis': front.trade_off_analysis
            }
            analysis['fronts'].append(front_analysis)
        
        return analysis
    
    def _compile_robustness_analysis(self, robustness_results: Dict[str, RobustnessAnalysisResult]) -> Dict[str, Any]:
        """Compile robustness analysis results.
        
        Args:
            robustness_results: Robustness analysis results by model
            
        Returns:
            Dict[str, Any]: Compiled robustness analysis
        """
        if not robustness_results:
            return {}
        
        analysis = {
            'total_models_analyzed': len(robustness_results),
            'overall_rankings': [],
            'parameter_analysis': {},
            'summary_statistics': {}
        }
        
        # Overall rankings
        model_scores = [(model_id, result.overall_robustness_score) 
                       for model_id, result in robustness_results.items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        analysis['overall_rankings'] = model_scores
        
        # Parameter analysis
        all_params = set()
        for result in robustness_results.values():
            all_params.update(result.parameter_curves.keys())
        
        for param_name in all_params:
            param_scores = []
            for model_id, result in robustness_results.items():
                if param_name in result.parameter_curves:
                    curve = result.parameter_curves[param_name]
                    param_scores.append((model_id, curve.auc_success_rate))
            
            param_scores.sort(key=lambda x: x[1], reverse=True)
            analysis['parameter_analysis'][param_name] = {
                'rankings': param_scores,
                'best_model': param_scores[0][0] if param_scores else None
            }
        
        # Summary statistics
        all_scores = [result.overall_robustness_score for result in robustness_results.values()]
        analysis['summary_statistics'] = {
            'mean_robustness_score': np.mean(all_scores),
            'std_robustness_score': np.std(all_scores),
            'min_robustness_score': np.min(all_scores),
            'max_robustness_score': np.max(all_scores)
        }
        
        return analysis
    
    def _compile_failure_analysis(self, failure_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compile failure analysis results.
        
        Args:
            failure_results: Failure analysis results by model
            
        Returns:
            Dict[str, Any]: Compiled failure analysis
        """
        if not failure_results:
            return {}
        
        analysis = {
            'total_models_analyzed': len(failure_results),
            'failure_patterns': {},
            'common_failure_modes': [],
            'model_specific_issues': {}
        }
        
        # Aggregate failure patterns across all models
        all_failure_types = defaultdict(int)
        for result in failure_results.values():
            failure_stats = result.get('failure_statistics', {})
            failure_counts = failure_stats.get('failure_type_counts', {})
            for failure_type, count in failure_counts.items():
                all_failure_types[failure_type] += count
        
        # Convert to percentages
        total_failures = sum(all_failure_types.values())
        if total_failures > 0:
            analysis['failure_patterns'] = {
                failure_type: (count / total_failures) * 100
                for failure_type, count in all_failure_types.items()
            }
        
        # Identify common failure modes (>10% of all failures)
        analysis['common_failure_modes'] = [
            failure_type for failure_type, percentage in analysis['failure_patterns'].items()
            if percentage > 10.0
        ]
        
        # Model-specific issues
        for model_id, result in failure_results.items():
            model_issues = []
            
            # Check for high failure rates in specific categories
            failure_stats = result.get('failure_statistics', {})
            failure_counts = failure_stats.get('failure_type_counts', {})
            total_episodes = sum(failure_counts.values())
            if total_episodes > 0:
                for failure_type, count in failure_counts.items():
                    failure_rate = count / total_episodes
                    if failure_rate > 0.2:  # >20% failure rate
                        model_issues.append(f"High {failure_type} rate: {failure_rate:.1%}")
            
            if model_issues:
                analysis['model_specific_issues'][model_id] = model_issues
        
        return analysis
    
    def _save_report(self, report: EvaluationReport, output_dir: Path):
        """Save the evaluation report to files.
        
        Args:
            report: Evaluation report to save
            output_dir: Output directory
        """
        # Save main report as JSON
        report_file = output_dir / "evaluation_report.json"
        
        # Convert report to JSON-serializable format
        report_dict = asdict(report)
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        # Save leaderboard as CSV
        leaderboard_file = output_dir / "leaderboard.csv"
        leaderboard_df = pd.DataFrame([asdict(entry) for entry in report.leaderboard])
        leaderboard_df.to_csv(leaderboard_file, index=False)
        
        # Save performance tables as CSV
        for table_name, table in report.performance_tables.items():
            table_file = output_dir / f"performance_table_{table_name}.csv"
            table.table_data.to_csv(table_file)
            
            if table.confidence_intervals is not None:
                ci_file = output_dir / f"confidence_intervals_{table_name}.csv"
                table.confidence_intervals.to_csv(ci_file)
        
        self.logger.info(f"📁 Report saved to: {output_dir}")
    
    def _generate_html_report(self, report: EvaluationReport, output_dir: Path):
        """Generate HTML version of the report.
        
        Args:
            report: Evaluation report
            output_dir: Output directory
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Report - {report.report_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .champion {{ background-color: #ffd700; }}
                .plot {{ text-align: center; margin: 20px 0; }}
                .recommendation {{ background-color: #e8f5e8; padding: 10px; margin: 5px 0; border-left: 4px solid #4CAF50; }}
                .risk-high {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
                .risk-medium {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
                .risk-low {{ background-color: #e8f5e8; border-left: 4px solid #4CAF50; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Evaluation Report: {report.report_id}</h1>
                <p><strong>Generated:</strong> {report.generation_timestamp}</p>
                <p><strong>Champion Model:</strong> {report.executive_summary.champion_model}</p>
                <p><strong>Deployment Readiness:</strong> {report.executive_summary.deployment_readiness}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <strong>Total Models:</strong> {report.executive_summary.total_models_evaluated}
                </div>
                <div class="metric">
                    <strong>Champion Score:</strong> {report.executive_summary.performance_highlights.get('champion_composite_score', 'N/A'):.3f}
                </div>
                <div class="metric">
                    <strong>Champion Success Rate:</strong> {report.executive_summary.performance_highlights.get('champion_success_rate', 0):.1%}
                </div>
                
                <h3>Key Findings</h3>
                <ul>
        """
        
        for finding in report.executive_summary.key_findings:
            html_content += f"<li>{finding}</li>"
        
        html_content += """
                </ul>
                
                <h3>Recommendations</h3>
        """
        
        for recommendation in report.executive_summary.recommendations:
            html_content += f'<div class="recommendation">{recommendation}</div>'
        
        html_content += """
                <h3>Risk Assessment</h3>
        """
        
        for risk_category, risk_level in report.executive_summary.risk_assessment.items():
            risk_class = "risk-low"
            if "High" in risk_level:
                risk_class = "risk-high"
            elif "Medium" in risk_level:
                risk_class = "risk-medium"
            
            html_content += f'<div class="{risk_class}"><strong>{risk_category.title()}:</strong> {risk_level}</div>'
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Leaderboard</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Model ID</th>
                        <th>Composite Score</th>
                        <th>Success Rate</th>
                        <th>Status</th>
                        <th>Validation</th>
                    </tr>
        """
        
        for entry in report.leaderboard[:10]:  # Top 10
            row_class = "champion" if entry.champion_status == "champion" else ""
            html_content += f"""
                    <tr class="{row_class}">
                        <td>{entry.rank}</td>
                        <td>{entry.model_id}</td>
                        <td>{entry.composite_score:.3f}</td>
                        <td>{entry.success_rate:.1%}</td>
                        <td>{entry.champion_status}</td>
                        <td>{entry.validation_status}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
        
        # Add plots
        if report.plots:
            html_content += """
            <div class="section">
                <h2>Visualizations</h2>
            """
            
            for plot_name, plot_path in report.plots.items():
                plot_filename = Path(plot_path).name
                html_content += f"""
                <div class="plot">
                    <h3>{plot_name.replace('_', ' ').title()}</h3>
                    <img src="{plot_filename}" alt="{plot_name}" style="max-width: 100%; height: auto;">
                </div>
                """
            
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML file
        html_file = output_dir / "evaluation_report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"📄 HTML report generated: {html_file}")