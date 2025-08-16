#!/usr/bin/env python3
"""
📈 EVALUATION ANALYSIS 📈
Advanced querying and analysis utilities for evaluation results

This module provides:
- Advanced result querying with filters and aggregations
- Statistical analysis and trend detection
- Performance comparison and ranking
- Data export and visualization
- Automated insights and recommendations

Usage Examples:
    # Query results with filters
    python evaluation_analysis.py query --model "champion_*" --suite base --metric success_rate --min-value 0.8
    
    # Generate performance trends
    python evaluation_analysis.py trends --models champion_v1 champion_v2 --time-range 30d
    
    # Compare model performance
    python evaluation_analysis.py compare --models champion_v1 baseline_v2 --metrics success_rate mean_reward
    
    # Export results to CSV
    python evaluation_analysis.py export --format csv --output results.csv --include-metadata
    
    # Generate insights report
    python evaluation_analysis.py insights --output insights_report.html
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import re
import sqlite3
from collections import defaultdict
import statistics

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from evaluation_cli import EvaluationCLI
from duckietown_utils.evaluation_orchestrator import EvaluationOrchestrator
from duckietown_utils.statistical_analyzer import StatisticalAnalyzer

# Optional dependencies for advanced analysis
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QueryFilter:
    """Filter criteria for result queries."""
    model_pattern: Optional[str] = None
    suite_pattern: Optional[str] = None
    policy_mode: Optional[str] = None
    metric_name: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass
class AnalysisResult:
    """Result of analysis operation."""
    analysis_type: str
    timestamp: str
    data: Dict[str, Any]
    summary: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)


class ResultsDatabase:
    """SQLite database for efficient result querying."""
    
    def __init__(self, db_path: str = "evaluation_results.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables."""
        cursor = self.conn.cursor()
        
        # Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                model_path TEXT,
                model_type TEXT,
                registration_time TEXT,
                metadata TEXT
            )
        """)
        
        # Evaluations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE,
                model_id TEXT,
                suite_name TEXT,
                policy_mode TEXT,
                timestamp TEXT,
                success_rate REAL,
                mean_reward REAL,
                mean_episode_length REAL,
                mean_lateral_deviation REAL,
                mean_heading_error REAL,
                mean_jerk REAL,
                stability REAL,
                evaluation_time REAL,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_id ON evaluations (model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_suite_name ON evaluations (suite_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON evaluations (timestamp)")
        
        self.conn.commit()
    
    def insert_model(self, model_info: Dict[str, Any]):
        """Insert model information."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO models 
            (model_id, model_path, model_type, registration_time, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            model_info['model_id'],
            model_info['model_path'],
            model_info['model_type'],
            model_info['registration_time'],
            json.dumps(model_info.get('metadata', {}))
        ))
        self.conn.commit()
    
    def insert_evaluation(self, evaluation_result: Dict[str, Any]):
        """Insert evaluation result."""
        cursor = self.conn.cursor()
        
        results = evaluation_result.get('results', {})
        
        cursor.execute("""
            INSERT OR REPLACE INTO evaluations 
            (task_id, model_id, suite_name, policy_mode, timestamp,
             success_rate, mean_reward, mean_episode_length, mean_lateral_deviation,
             mean_heading_error, mean_jerk, stability, evaluation_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            evaluation_result['task_id'],
            evaluation_result['model_id'],
            evaluation_result['suite_name'],
            evaluation_result['policy_mode'],
            results.get('timestamp', datetime.now().isoformat()),
            results.get('success_rate'),
            results.get('mean_reward'),
            results.get('mean_episode_length'),
            results.get('mean_lateral_deviation'),
            results.get('mean_heading_error'),
            results.get('mean_jerk'),
            results.get('stability'),
            results.get('evaluation_time')
        ))
        self.conn.commit()
    
    def query_results(self, filter_criteria: QueryFilter) -> List[Dict[str, Any]]:
        """Query results with filters."""
        cursor = self.conn.cursor()
        
        # Build query
        query = """
            SELECT e.*, m.model_path, m.model_type, m.metadata
            FROM evaluations e
            JOIN models m ON e.model_id = m.model_id
            WHERE 1=1
        """
        params = []
        
        if filter_criteria.model_pattern:
            query += " AND e.model_id GLOB ?"
            params.append(filter_criteria.model_pattern)
        
        if filter_criteria.suite_pattern:
            query += " AND e.suite_name GLOB ?"
            params.append(filter_criteria.suite_pattern)
        
        if filter_criteria.policy_mode:
            query += " AND e.policy_mode = ?"
            params.append(filter_criteria.policy_mode)
        
        if filter_criteria.date_from:
            query += " AND e.timestamp >= ?"
            params.append(filter_criteria.date_from)
        
        if filter_criteria.date_to:
            query += " AND e.timestamp <= ?"
            params.append(filter_criteria.date_to)
        
        # Add metric filters
        if filter_criteria.metric_name and filter_criteria.min_value is not None:
            query += f" AND e.{filter_criteria.metric_name} >= ?"
            params.append(filter_criteria.min_value)
        
        if filter_criteria.metric_name and filter_criteria.max_value is not None:
            query += f" AND e.{filter_criteria.metric_name} <= ?"
            params.append(filter_criteria.max_value)
        
        query += " ORDER BY e.timestamp DESC"
        
        cursor.execute(query, params)
        columns = [description[0] for description in cursor.description]
        
        results = []
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            # Parse metadata JSON
            if result['metadata']:
                result['metadata'] = json.loads(result['metadata'])
            results.append(result)
        
        return results
    
    def get_model_performance_trends(self, model_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get performance trends for a model over time."""
        cursor = self.conn.cursor()
        
        date_from = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT timestamp, suite_name, success_rate, mean_reward, mean_lateral_deviation
            FROM evaluations
            WHERE model_id = ? AND timestamp >= ?
            ORDER BY timestamp
        """, (model_id, date_from))
        
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_suite_statistics(self, suite_name: str) -> Dict[str, Any]:
        """Get statistics for a specific suite across all models."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as evaluation_count,
                AVG(success_rate) as avg_success_rate,
                MIN(success_rate) as min_success_rate,
                MAX(success_rate) as max_success_rate,
                AVG(mean_reward) as avg_reward,
                AVG(mean_lateral_deviation) as avg_lateral_deviation
            FROM evaluations
            WHERE suite_name = ?
        """, (suite_name,))
        
        result = cursor.fetchone()
        columns = [description[0] for description in cursor.description]
        return dict(zip(columns, result)) if result else {}
    
    def close(self):
        """Close database connection."""
        self.conn.close()


class EvaluationAnalyzer:
    """Advanced analysis utilities for evaluation results."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.cli = EvaluationCLI()
        self.db = ResultsDatabase(db_path) if db_path else None
        self.statistical_analyzer = StatisticalAnalyzer()
    
    def sync_database(self):
        """Sync orchestrator results to database."""
        if not self.db:
            logger.warning("No database configured")
            return
        
        orchestrator = self.cli._get_orchestrator()
        
        # Sync models
        models = orchestrator.model_registry.list_models()
        for model in models:
            model_info = {
                'model_id': model.model_id,
                'model_path': model.model_path,
                'model_type': model.model_type,
                'registration_time': model.registration_time,
                'metadata': model.metadata
            }
            self.db.insert_model(model_info)
        
        # Sync evaluation results
        results = orchestrator.get_results()
        for result in results:
            self.db.insert_evaluation(result)
        
        logger.info(f"Synced {len(models)} models and {len(results)} evaluation results")
    
    def query_results(self, filter_criteria: QueryFilter) -> List[Dict[str, Any]]:
        """Query evaluation results with advanced filters."""
        if self.db:
            return self.db.query_results(filter_criteria)
        else:
            # Fallback to orchestrator query
            orchestrator = self.cli._get_orchestrator()
            results = orchestrator.get_results(
                model_id=filter_criteria.model_pattern,
                suite_name=filter_criteria.suite_pattern
            )
            
            # Apply additional filters
            filtered_results = []
            for result in results:
                if self._matches_filter(result, filter_criteria):
                    filtered_results.append(result)
            
            return filtered_results
    
    def _matches_filter(self, result: Dict[str, Any], filter_criteria: QueryFilter) -> bool:
        """Check if result matches filter criteria."""
        if filter_criteria.policy_mode and result.get('policy_mode') != filter_criteria.policy_mode:
            return False
        
        if filter_criteria.metric_name and 'results' in result:
            metric_value = result['results'].get(filter_criteria.metric_name)
            if metric_value is None:
                return False
            
            if filter_criteria.min_value is not None and metric_value < filter_criteria.min_value:
                return False
            
            if filter_criteria.max_value is not None and metric_value > filter_criteria.max_value:
                return False
        
        return True
    
    def analyze_performance_trends(self, model_ids: List[str], 
                                 time_range_days: int = 30) -> AnalysisResult:
        """Analyze performance trends for models over time."""
        trends_data = {}
        
        for model_id in model_ids:
            if self.db:
                trend_data = self.db.get_model_performance_trends(model_id, time_range_days)
            else:
                # Fallback implementation
                trend_data = self._get_trend_data_fallback(model_id, time_range_days)
            
            trends_data[model_id] = trend_data
        
        # Analyze trends
        summary = self._analyze_trend_summary(trends_data)
        recommendations = self._generate_trend_recommendations(trends_data, summary)
        
        return AnalysisResult(
            analysis_type="performance_trends",
            timestamp=datetime.now().isoformat(),
            data=trends_data,
            summary=summary,
            recommendations=recommendations
        )
    
    def _get_trend_data_fallback(self, model_id: str, days: int) -> List[Dict[str, Any]]:
        """Fallback method to get trend data without database."""
        orchestrator = self.cli._get_orchestrator()
        results = orchestrator.get_results(model_id=model_id)
        
        # Filter by date range
        cutoff_date = datetime.now() - timedelta(days=days)
        
        trend_data = []
        for result in results:
            if 'results' in result and result['results']:
                result_time = result['results'].get('timestamp')
                if result_time and datetime.fromisoformat(result_time) >= cutoff_date:
                    trend_data.append({
                        'timestamp': result_time,
                        'suite_name': result['suite_name'],
                        'success_rate': result['results'].get('success_rate'),
                        'mean_reward': result['results'].get('mean_reward'),
                        'mean_lateral_deviation': result['results'].get('mean_lateral_deviation')
                    })
        
        return sorted(trend_data, key=lambda x: x['timestamp'])
    
    def _analyze_trend_summary(self, trends_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze trend summary statistics."""
        summary = {}
        
        for model_id, data in trends_data.items():
            if not data:
                continue
            
            # Extract metrics
            success_rates = [d['success_rate'] for d in data if d['success_rate'] is not None]
            rewards = [d['mean_reward'] for d in data if d['mean_reward'] is not None]
            
            if success_rates:
                # Calculate trend direction
                if len(success_rates) > 1:
                    trend_slope = self._calculate_trend_slope(success_rates)
                    trend_direction = "improving" if trend_slope > 0.01 else "declining" if trend_slope < -0.01 else "stable"
                else:
                    trend_direction = "insufficient_data"
                
                summary[model_id] = {
                    'data_points': len(success_rates),
                    'latest_success_rate': success_rates[-1],
                    'avg_success_rate': statistics.mean(success_rates),
                    'success_rate_std': statistics.stdev(success_rates) if len(success_rates) > 1 else 0,
                    'trend_direction': trend_direction,
                    'trend_slope': trend_slope if len(success_rates) > 1 else 0
                }
                
                if rewards:
                    summary[model_id].update({
                        'latest_reward': rewards[-1],
                        'avg_reward': statistics.mean(rewards),
                        'reward_std': statistics.stdev(rewards) if len(rewards) > 1 else 0
                    })
        
        return summary
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression."""
        if len(values) < 2:
            return 0
        
        x = list(range(len(values)))
        
        if SCIPY_AVAILABLE:
            slope, _, _, _, _ = stats.linregress(x, values)
            return slope
        else:
            # Simple slope calculation
            n = len(values)
            sum_x = sum(x)
            sum_y = sum(values)
            sum_xy = sum(x[i] * values[i] for i in range(n))
            sum_x2 = sum(xi * xi for xi in x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
    
    def _generate_trend_recommendations(self, trends_data: Dict, summary: Dict) -> List[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        for model_id, model_summary in summary.items():
            if model_summary['trend_direction'] == 'declining':
                recommendations.append(
                    f"Model {model_id} shows declining performance. Consider retraining or hyperparameter tuning."
                )
            elif model_summary['trend_direction'] == 'improving':
                recommendations.append(
                    f"Model {model_id} shows improving performance. Continue current training approach."
                )
            
            if model_summary['success_rate_std'] > 0.1:
                recommendations.append(
                    f"Model {model_id} shows high performance variability. Consider more stable training."
                )
        
        return recommendations
    
    def compare_models(self, model_ids: List[str], 
                      metrics: List[str] = None) -> AnalysisResult:
        """Compare performance between multiple models."""
        if not metrics:
            metrics = ['success_rate', 'mean_reward', 'mean_lateral_deviation']
        
        # Get results for all models
        comparison_data = {}
        for model_id in model_ids:
            filter_criteria = QueryFilter(model_pattern=model_id)
            results = self.query_results(filter_criteria)
            comparison_data[model_id] = results
        
        # Perform statistical comparison
        summary = self._perform_model_comparison(comparison_data, metrics)
        recommendations = self._generate_comparison_recommendations(summary)
        
        return AnalysisResult(
            analysis_type="model_comparison",
            timestamp=datetime.now().isoformat(),
            data=comparison_data,
            summary=summary,
            recommendations=recommendations
        )
    
    def _perform_model_comparison(self, comparison_data: Dict, metrics: List[str]) -> Dict[str, Any]:
        """Perform statistical comparison between models."""
        summary = {
            'metrics_compared': metrics,
            'model_statistics': {},
            'pairwise_comparisons': {},
            'rankings': {}
        }
        
        # Calculate statistics for each model
        for model_id, results in comparison_data.items():
            model_stats = {}
            
            for metric in metrics:
                values = []
                for result in results:
                    if 'results' in result and result['results']:
                        value = result['results'].get(metric)
                        if value is not None:
                            values.append(value)
                
                if values:
                    model_stats[metric] = {
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'std': statistics.stdev(values) if len(values) > 1 else 0,
                        'min': min(values),
                        'max': max(values),
                        'median': statistics.median(values)
                    }
            
            summary['model_statistics'][model_id] = model_stats
        
        # Perform pairwise comparisons
        for i, model_a in enumerate(model_ids):
            for model_b in model_ids[i+1:]:
                comparison_key = f"{model_a}_vs_{model_b}"
                summary['pairwise_comparisons'][comparison_key] = self._compare_model_pair(
                    comparison_data[model_a], comparison_data[model_b], metrics
                )
        
        # Generate rankings
        for metric in metrics:
            ranking = []
            for model_id in model_ids:
                if model_id in summary['model_statistics'] and metric in summary['model_statistics'][model_id]:
                    mean_value = summary['model_statistics'][model_id][metric]['mean']
                    ranking.append((model_id, mean_value))
            
            # Sort by metric (higher is better for most metrics except lateral_deviation)
            reverse_sort = metric not in ['mean_lateral_deviation', 'mean_heading_error']
            ranking.sort(key=lambda x: x[1], reverse=reverse_sort)
            summary['rankings'][metric] = ranking
        
        return summary
    
    def _compare_model_pair(self, results_a: List[Dict], results_b: List[Dict], 
                           metrics: List[str]) -> Dict[str, Any]:
        """Compare two models statistically."""
        comparison = {}
        
        for metric in metrics:
            values_a = []
            values_b = []
            
            for result in results_a:
                if 'results' in result and result['results']:
                    value = result['results'].get(metric)
                    if value is not None:
                        values_a.append(value)
            
            for result in results_b:
                if 'results' in result and result['results']:
                    value = result['results'].get(metric)
                    if value is not None:
                        values_b.append(value)
            
            if values_a and values_b and len(values_a) > 1 and len(values_b) > 1:
                # Perform statistical test
                if SCIPY_AVAILABLE:
                    statistic, p_value = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')
                    effect_size = self._calculate_effect_size(values_a, values_b)
                else:
                    # Simple comparison
                    mean_a = statistics.mean(values_a)
                    mean_b = statistics.mean(values_b)
                    p_value = 0.05  # Placeholder
                    effect_size = abs(mean_a - mean_b) / max(statistics.stdev(values_a), statistics.stdev(values_b))
                
                comparison[metric] = {
                    'mean_a': statistics.mean(values_a),
                    'mean_b': statistics.mean(values_b),
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': p_value < 0.05,
                    'better_model': 'a' if statistics.mean(values_a) > statistics.mean(values_b) else 'b'
                }
        
        return comparison
    
    def _calculate_effect_size(self, values_a: List[float], values_b: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        mean_a = statistics.mean(values_a)
        mean_b = statistics.mean(values_b)
        std_a = statistics.stdev(values_a) if len(values_a) > 1 else 0
        std_b = statistics.stdev(values_b) if len(values_b) > 1 else 0
        
        pooled_std = ((len(values_a) - 1) * std_a**2 + (len(values_b) - 1) * std_b**2) / (len(values_a) + len(values_b) - 2)
        pooled_std = pooled_std**0.5
        
        if pooled_std == 0:
            return 0
        
        return abs(mean_a - mean_b) / pooled_std
    
    def _generate_comparison_recommendations(self, summary: Dict) -> List[str]:
        """Generate recommendations based on model comparison."""
        recommendations = []
        
        # Check rankings
        for metric, ranking in summary['rankings'].items():
            if ranking:
                best_model = ranking[0][0]
                worst_model = ranking[-1][0]
                
                recommendations.append(
                    f"For {metric}, {best_model} performs best with score {ranking[0][1]:.3f}"
                )
                
                if len(ranking) > 1:
                    performance_gap = abs(ranking[0][1] - ranking[-1][1])
                    if performance_gap > 0.1:  # Significant gap
                        recommendations.append(
                            f"Large performance gap in {metric} between {best_model} and {worst_model} ({performance_gap:.3f})"
                        )
        
        # Check for significant differences
        for comparison_key, comparison_data in summary['pairwise_comparisons'].items():
            models = comparison_key.split('_vs_')
            for metric, metric_data in comparison_data.items():
                if metric_data['significant'] and metric_data['effect_size'] > 0.5:
                    better_model = models[0] if metric_data['better_model'] == 'a' else models[1]
                    recommendations.append(
                        f"Significant difference in {metric}: {better_model} outperforms with effect size {metric_data['effect_size']:.2f}"
                    )
        
        return recommendations
    
    def generate_insights_report(self, output_path: str, 
                               include_plots: bool = True) -> AnalysisResult:
        """Generate comprehensive insights report."""
        # Collect all available data
        all_results = self.query_results(QueryFilter())
        
        if not all_results:
            logger.warning("No evaluation results found")
            return AnalysisResult(
                analysis_type="insights_report",
                timestamp=datetime.now().isoformat(),
                data={},
                summary={'status': 'no_data'},
                recommendations=["No evaluation data available for analysis"]
            )
        
        # Generate insights
        insights = {
            'total_evaluations': len(all_results),
            'unique_models': len(set(r['model_id'] for r in all_results)),
            'unique_suites': len(set(r['suite_name'] for r in all_results)),
            'date_range': self._get_date_range(all_results),
            'performance_summary': self._generate_performance_summary(all_results),
            'suite_analysis': self._analyze_suites(all_results),
            'model_rankings': self._generate_model_rankings(all_results)
        }
        
        # Generate recommendations
        recommendations = self._generate_insights_recommendations(insights)
        
        # Create HTML report
        self._create_insights_html_report(insights, recommendations, output_path, include_plots)
        
        return AnalysisResult(
            analysis_type="insights_report",
            timestamp=datetime.now().isoformat(),
            data=insights,
            summary={'report_path': output_path, 'total_evaluations': len(all_results)},
            recommendations=recommendations
        )
    
    def _get_date_range(self, results: List[Dict]) -> Dict[str, str]:
        """Get date range of evaluation results."""
        timestamps = []
        for result in results:
            if 'results' in result and result['results']:
                timestamp = result['results'].get('timestamp')
                if timestamp:
                    timestamps.append(timestamp)
        
        if timestamps:
            timestamps.sort()
            return {
                'earliest': timestamps[0],
                'latest': timestamps[-1],
                'span_days': (datetime.fromisoformat(timestamps[-1]) - datetime.fromisoformat(timestamps[0])).days
            }
        
        return {'earliest': None, 'latest': None, 'span_days': 0}
    
    def _generate_performance_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate overall performance summary."""
        success_rates = []
        rewards = []
        lateral_deviations = []
        
        for result in results:
            if 'results' in result and result['results']:
                r = result['results']
                if r.get('success_rate') is not None:
                    success_rates.append(r['success_rate'])
                if r.get('mean_reward') is not None:
                    rewards.append(r['mean_reward'])
                if r.get('mean_lateral_deviation') is not None:
                    lateral_deviations.append(r['mean_lateral_deviation'])
        
        summary = {}
        
        if success_rates:
            summary['success_rate'] = {
                'mean': statistics.mean(success_rates),
                'std': statistics.stdev(success_rates) if len(success_rates) > 1 else 0,
                'min': min(success_rates),
                'max': max(success_rates),
                'count': len(success_rates)
            }
        
        if rewards:
            summary['reward'] = {
                'mean': statistics.mean(rewards),
                'std': statistics.stdev(rewards) if len(rewards) > 1 else 0,
                'min': min(rewards),
                'max': max(rewards),
                'count': len(rewards)
            }
        
        if lateral_deviations:
            summary['lateral_deviation'] = {
                'mean': statistics.mean(lateral_deviations),
                'std': statistics.stdev(lateral_deviations) if len(lateral_deviations) > 1 else 0,
                'min': min(lateral_deviations),
                'max': max(lateral_deviations),
                'count': len(lateral_deviations)
            }
        
        return summary
    
    def _analyze_suites(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance by suite."""
        suite_data = defaultdict(list)
        
        for result in results:
            suite_name = result['suite_name']
            if 'results' in result and result['results']:
                suite_data[suite_name].append(result['results'])
        
        suite_analysis = {}
        for suite_name, suite_results in suite_data.items():
            success_rates = [r.get('success_rate') for r in suite_results if r.get('success_rate') is not None]
            
            if success_rates:
                suite_analysis[suite_name] = {
                    'evaluation_count': len(suite_results),
                    'mean_success_rate': statistics.mean(success_rates),
                    'std_success_rate': statistics.stdev(success_rates) if len(success_rates) > 1 else 0,
                    'difficulty_rank': None  # Will be set after all suites are processed
                }
        
        # Rank suites by difficulty (lower success rate = higher difficulty)
        sorted_suites = sorted(suite_analysis.items(), key=lambda x: x[1]['mean_success_rate'])
        for i, (suite_name, _) in enumerate(sorted_suites):
            suite_analysis[suite_name]['difficulty_rank'] = i + 1
        
        return suite_analysis
    
    def _generate_model_rankings(self, results: List[Dict]) -> Dict[str, List[Tuple[str, float]]]:
        """Generate model rankings by metric."""
        model_data = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            model_id = result['model_id']
            if 'results' in result and result['results']:
                r = result['results']
                for metric in ['success_rate', 'mean_reward', 'mean_lateral_deviation']:
                    if r.get(metric) is not None:
                        model_data[model_id][metric].append(r[metric])
        
        rankings = {}
        for metric in ['success_rate', 'mean_reward', 'mean_lateral_deviation']:
            model_scores = []
            for model_id, metrics in model_data.items():
                if metric in metrics and metrics[metric]:
                    mean_score = statistics.mean(metrics[metric])
                    model_scores.append((model_id, mean_score))
            
            # Sort (higher is better except for lateral deviation)
            reverse_sort = metric != 'mean_lateral_deviation'
            model_scores.sort(key=lambda x: x[1], reverse=reverse_sort)
            rankings[metric] = model_scores
        
        return rankings
    
    def _generate_insights_recommendations(self, insights: Dict) -> List[str]:
        """Generate recommendations based on insights."""
        recommendations = []
        
        # Check data coverage
        if insights['total_evaluations'] < 100:
            recommendations.append("Consider running more evaluations for more reliable insights")
        
        # Check performance variability
        perf_summary = insights.get('performance_summary', {})
        if 'success_rate' in perf_summary:
            sr_std = perf_summary['success_rate']['std']
            if sr_std > 0.2:
                recommendations.append("High variability in success rates suggests inconsistent model performance")
        
        # Check suite difficulty
        suite_analysis = insights.get('suite_analysis', {})
        if suite_analysis:
            easiest_suite = min(suite_analysis.items(), key=lambda x: x[1]['difficulty_rank'])
            hardest_suite = max(suite_analysis.items(), key=lambda x: x[1]['difficulty_rank'])
            
            recommendations.append(f"Easiest suite: {easiest_suite[0]} (SR: {easiest_suite[1]['mean_success_rate']:.3f})")
            recommendations.append(f"Hardest suite: {hardest_suite[0]} (SR: {hardest_suite[1]['mean_success_rate']:.3f})")
        
        # Check model performance
        model_rankings = insights.get('model_rankings', {})
        if 'success_rate' in model_rankings and model_rankings['success_rate']:
            best_model = model_rankings['success_rate'][0]
            recommendations.append(f"Best performing model: {best_model[0]} (SR: {best_model[1]:.3f})")
        
        return recommendations
    
    def _create_insights_html_report(self, insights: Dict, recommendations: List[str], 
                                   output_path: str, include_plots: bool):
        """Create HTML insights report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation Insights Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                .metric {{ font-size: 18px; font-weight: bold; color: #2196F3; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .recommendation {{ background: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📈 Evaluation Insights Report</h1>
                <p>Generated: {datetime.now().isoformat()}</p>
                
                <div class="section">
                    <h2>Overview</h2>
                    <p>Total Evaluations: <span class="metric">{insights['total_evaluations']}</span></p>
                    <p>Unique Models: <span class="metric">{insights['unique_models']}</span></p>
                    <p>Unique Suites: <span class="metric">{insights['unique_suites']}</span></p>
                    <p>Date Range: {insights['date_range']['earliest']} to {insights['date_range']['latest']}</p>
                </div>
                
                <div class="section">
                    <h2>Performance Summary</h2>
                    {self._format_performance_summary_html(insights.get('performance_summary', {}))}
                </div>
                
                <div class="section">
                    <h2>Suite Analysis</h2>
                    {self._format_suite_analysis_html(insights.get('suite_analysis', {}))}
                </div>
                
                <div class="section">
                    <h2>Model Rankings</h2>
                    {self._format_model_rankings_html(insights.get('model_rankings', {}))}
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    {''.join([f'<div class="recommendation">• {rec}</div>' for rec in recommendations])}
                </div>
                
                <div class="section">
                    <h2>Raw Data</h2>
                    <pre>{json.dumps(insights, indent=2)}</pre>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Insights report saved to: {output_path}")
    
    def _format_performance_summary_html(self, perf_summary: Dict) -> str:
        """Format performance summary as HTML."""
        if not perf_summary:
            return "<p>No performance data available</p>"
        
        html = "<table><tr><th>Metric</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Count</th></tr>"
        
        for metric, stats in perf_summary.items():
            html += f"""
            <tr>
                <td>{metric.replace('_', ' ').title()}</td>
                <td>{stats['mean']:.3f}</td>
                <td>{stats['std']:.3f}</td>
                <td>{stats['min']:.3f}</td>
                <td>{stats['max']:.3f}</td>
                <td>{stats['count']}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _format_suite_analysis_html(self, suite_analysis: Dict) -> str:
        """Format suite analysis as HTML."""
        if not suite_analysis:
            return "<p>No suite data available</p>"
        
        html = "<table><tr><th>Suite</th><th>Evaluations</th><th>Mean Success Rate</th><th>Std</th><th>Difficulty Rank</th></tr>"
        
        # Sort by difficulty rank
        sorted_suites = sorted(suite_analysis.items(), key=lambda x: x[1]['difficulty_rank'])
        
        for suite_name, stats in sorted_suites:
            html += f"""
            <tr>
                <td>{suite_name}</td>
                <td>{stats['evaluation_count']}</td>
                <td>{stats['mean_success_rate']:.3f}</td>
                <td>{stats['std_success_rate']:.3f}</td>
                <td>{stats['difficulty_rank']}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _format_model_rankings_html(self, model_rankings: Dict) -> str:
        """Format model rankings as HTML."""
        if not model_rankings:
            return "<p>No ranking data available</p>"
        
        html = ""
        for metric, rankings in model_rankings.items():
            if rankings:
                html += f"<h3>{metric.replace('_', ' ').title()}</h3>"
                html += "<table><tr><th>Rank</th><th>Model</th><th>Score</th></tr>"
                
                for i, (model_id, score) in enumerate(rankings[:10]):  # Top 10
                    html += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{model_id}</td>
                        <td>{score:.3f}</td>
                    </tr>
                    """
                
                html += "</table><br>"
        
        return html
    
    def export_results(self, output_path: str, format_type: str = "csv", 
                      filter_criteria: Optional[QueryFilter] = None,
                      include_metadata: bool = False) -> int:
        """Export evaluation results to file."""
        # Get results
        if filter_criteria is None:
            filter_criteria = QueryFilter()
        
        results = self.query_results(filter_criteria)
        
        if not results:
            logger.warning("No results found to export")
            return 0
        
        # Export based on format
        if format_type.lower() == 'csv':
            self._export_to_csv(results, output_path, include_metadata)
        elif format_type.lower() == 'json':
            self._export_to_json(results, output_path, include_metadata)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        logger.info(f"Exported {len(results)} results to {output_path}")
        return len(results)
    
    def _export_to_csv(self, results: List[Dict], output_path: str, include_metadata: bool):
        """Export results to CSV format."""
        if not results:
            return
        
        # Flatten results for CSV
        flattened_results = []
        for result in results:
            flat_result = {
                'model_id': result.get('model_id'),
                'suite_name': result.get('suite_name'),
                'policy_mode': result.get('policy_mode'),
                'timestamp': result.get('timestamp')
            }
            
            # Add evaluation metrics
            if 'results' in result and result['results']:
                for key, value in result['results'].items():
                    flat_result[f"result_{key}"] = value
            
            # Add metadata if requested
            if include_metadata and 'metadata' in result and result['metadata']:
                for key, value in result['metadata'].items():
                    flat_result[f"metadata_{key}"] = value
            
            flattened_results.append(flat_result)
        
        # Write to CSV
        df = pd.DataFrame(flattened_results)
        df.to_csv(output_path, index=False)
    
    def _export_to_json(self, results: List[Dict], output_path: str, include_metadata: bool):
        """Export results to JSON format."""
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_results': len(results),
            'include_metadata': include_metadata,
            'results': results if include_metadata else [
                {k: v for k, v in result.items() if k != 'metadata'}
                for result in results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)


def create_analysis_parser() -> argparse.ArgumentParser:
    """Create argument parser for evaluation analysis."""
    parser = argparse.ArgumentParser(
        description="Evaluation Results Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query evaluation results')
    query_parser.add_argument('--model', help='Model ID pattern')
    query_parser.add_argument('--suite', help='Suite name pattern')
    query_parser.add_argument('--policy-mode', choices=['deterministic', 'stochastic'], help='Policy mode')
    query_parser.add_argument('--metric', help='Metric name to filter by')
    query_parser.add_argument('--min-value', type=float, help='Minimum metric value')
    query_parser.add_argument('--max-value', type=float, help='Maximum metric value')
    query_parser.add_argument('--date-from', help='Start date (ISO format)')
    query_parser.add_argument('--date-to', help='End date (ISO format)')
    query_parser.add_argument('--output', help='Output file path')
    query_parser.add_argument('--format', choices=['table', 'json', 'csv'], default='table', help='Output format')
    
    # Trends command
    trends_parser = subparsers.add_parser('trends', help='Analyze performance trends')
    trends_parser.add_argument('--models', nargs='+', required=True, help='Model IDs to analyze')
    trends_parser.add_argument('--time-range', default='30d', help='Time range (e.g., 30d, 7d)')
    trends_parser.add_argument('--output', help='Output file path')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare model performance')
    compare_parser.add_argument('--models', nargs='+', required=True, help='Model IDs to compare')
    compare_parser.add_argument('--metrics', nargs='*', help='Metrics to compare')
    compare_parser.add_argument('--output', help='Output file path')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export evaluation results')
    export_parser.add_argument('--format', choices=['csv', 'json'], default='csv', help='Export format')
    export_parser.add_argument('--output', required=True, help='Output file path')
    export_parser.add_argument('--include-metadata', action='store_true', help='Include model metadata')
    export_parser.add_argument('--model', help='Model ID pattern filter')
    export_parser.add_argument('--suite', help='Suite name pattern filter')
    
    # Insights command
    insights_parser = subparsers.add_parser('insights', help='Generate insights report')
    insights_parser.add_argument('--output', required=True, help='Output HTML file path')
    insights_parser.add_argument('--include-plots', action='store_true', help='Include plots in report')
    
    # Sync command
    sync_parser = subparsers.add_parser('sync', help='Sync results to database')
    sync_parser.add_argument('--db-path', help='Database file path')
    
    return parser


def main():
    """Main entry point for evaluation analysis."""
    parser = create_analysis_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create analyzer
    db_path = getattr(args, 'db_path', None)
    analyzer = EvaluationAnalyzer(db_path)
    
    try:
        if args.command == 'query':
            filter_criteria = QueryFilter(
                model_pattern=args.model,
                suite_pattern=args.suite,
                policy_mode=args.policy_mode,
                metric_name=args.metric,
                min_value=args.min_value,
                max_value=args.max_value,
                date_from=args.date_from,
                date_to=args.date_to
            )
            
            results = analyzer.query_results(filter_criteria)
            
            if args.format == 'json':
                output = json.dumps(results, indent=2)
            elif args.format == 'csv':
                # Convert to CSV format
                import io
                import csv
                output_io = io.StringIO()
                if results:
                    writer = csv.DictWriter(output_io, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
                output = output_io.getvalue()
            else:  # table
                output = f"Found {len(results)} results\n"
                for result in results[:10]:  # Show first 10
                    output += f"Model: {result.get('model_id')}, Suite: {result.get('suite_name')}\n"
            
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                print(f"Results saved to {args.output}")
            else:
                print(output)
        
        elif args.command == 'trends':
            # Parse time range
            time_match = re.match(r'(\d+)d', args.time_range)
            days = int(time_match.group(1)) if time_match else 30
            
            analysis_result = analyzer.analyze_performance_trends(args.models, days)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(analysis_result.__dict__, f, indent=2)
                print(f"Trends analysis saved to {args.output}")
            else:
                print(f"Trends Analysis for {args.models}")
                print(f"Summary: {analysis_result.summary}")
                print(f"Recommendations: {analysis_result.recommendations}")
        
        elif args.command == 'compare':
            analysis_result = analyzer.compare_models(args.models, args.metrics)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(analysis_result.__dict__, f, indent=2)
                print(f"Comparison analysis saved to {args.output}")
            else:
                print(f"Model Comparison: {args.models}")
                print(f"Summary: {analysis_result.summary}")
                print(f"Recommendations: {analysis_result.recommendations}")
        
        elif args.command == 'export':
            filter_criteria = QueryFilter(
                model_pattern=args.model,
                suite_pattern=args.suite
            )
            
            count = analyzer.export_results(
                output_path=args.output,
                format_type=args.format,
                filter_criteria=filter_criteria,
                include_metadata=args.include_metadata
            )
            print(f"Exported {count} results to {args.output}")
        
        elif args.command == 'insights':
            analysis_result = analyzer.generate_insights_report(
                output_path=args.output,
                include_plots=args.include_plots
            )
            print(f"Insights report generated: {args.output}")
            print(f"Total evaluations analyzed: {analysis_result.summary.get('total_evaluations', 0)}")
        
        elif args.command == 'sync':
            analyzer.sync_database()
            print("Database sync completed")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    finally:
        if analyzer.db:
            analyzer.db.close()


if __name__ == '__main__':
    sys.exit(main())