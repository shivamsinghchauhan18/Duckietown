#!/usr/bin/env python3
"""
🚀 BATCH EVALUATION SCRIPTS 🚀
Automated batch evaluation utilities for multiple model comparison

This module provides utilities for:
- Batch model registration from directories
- Automated evaluation campaigns
- Scheduled evaluation runs
- Multi-configuration evaluation
- Comparative analysis workflows

Usage Examples:
    # Register all models in a directory
    python batch_evaluation.py register-batch --models-dir models/ --pattern "*.pth"
    
    # Run comprehensive evaluation campaign
    python batch_evaluation.py campaign --config-dir configs/ --models-pattern "champion_*"
    
    # Schedule periodic evaluation
    python batch_evaluation.py schedule --interval 24h --models-pattern "latest_*"
    
    # Compare model generations
    python batch_evaluation.py compare-generations --base-pattern "v1_*" --new-pattern "v2_*"
"""

import os
import sys
import json
import time
import glob
import argparse
import logging
import schedule
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Pattern
from dataclasses import dataclass, field
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from evaluation_cli import EvaluationCLI
from config.evaluation_config import EvaluationConfig
from duckietown_utils.evaluation_orchestrator import EvaluationOrchestrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BatchEvaluationConfig:
    """Configuration for batch evaluation operations."""
    models_directory: str = "models"
    results_directory: str = "batch_evaluation_results"
    config_directory: str = "config"
    
    # Model discovery
    model_patterns: List[str] = field(default_factory=lambda: ["*.pth", "*.onnx", "*.pt"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["*temp*", "*backup*"])
    
    # Evaluation settings
    max_concurrent_evaluations: int = 2
    evaluation_timeout_hours: float = 12.0
    retry_failed_evaluations: bool = True
    max_retries: int = 3
    
    # Reporting
    generate_reports: bool = True
    report_formats: List[str] = field(default_factory=lambda: ["html", "json"])
    
    # Scheduling
    schedule_enabled: bool = False
    schedule_interval_hours: int = 24
    schedule_time: str = "02:00"  # 2 AM


class BatchEvaluationManager:
    """Manager for batch evaluation operations."""
    
    def __init__(self, config: BatchEvaluationConfig):
        self.config = config
        self.cli = EvaluationCLI()
        
        # Create directories
        Path(self.config.results_directory).mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.evaluation_history: List[Dict[str, Any]] = []
        self.failed_evaluations: List[Dict[str, Any]] = []
        
        # Scheduling
        self._scheduler_running = False
        self._scheduler_thread: Optional[threading.Thread] = None
    
    def discover_models(self, directory: str, patterns: List[str], 
                       exclude_patterns: Optional[List[str]] = None) -> List[Path]:
        """Discover model files in directory matching patterns."""
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Models directory not found: {directory}")
            return []
        
        exclude_patterns = exclude_patterns or []
        discovered_models = []
        
        for pattern in patterns:
            # Find files matching pattern
            matches = list(directory.rglob(pattern))
            
            for match in matches:
                # Check exclude patterns
                excluded = False
                for exclude_pattern in exclude_patterns:
                    if match.match(exclude_pattern):
                        excluded = True
                        break
                
                if not excluded and match.is_file():
                    discovered_models.append(match)
        
        # Remove duplicates and sort
        discovered_models = sorted(list(set(discovered_models)))
        
        logger.info(f"Discovered {len(discovered_models)} models in {directory}")
        return discovered_models
    
    def register_models_batch(self, models_directory: str, patterns: List[str],
                             exclude_patterns: Optional[List[str]] = None,
                             model_type: str = "checkpoint") -> List[str]:
        """Register multiple models from directory."""
        models = self.discover_models(models_directory, patterns, exclude_patterns)
        
        if not models:
            logger.warning("No models found to register")
            return []
        
        registered_ids = []
        
        for model_path in models:
            try:
                # Generate model ID from filename
                model_id = model_path.stem
                
                # Add timestamp if ID might conflict
                if any(existing_id == model_id for existing_id in registered_ids):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_id = f"{model_id}_{timestamp}"
                
                # Prepare metadata
                metadata = {
                    'batch_registered': True,
                    'source_directory': str(models_directory),
                    'file_size_bytes': model_path.stat().st_size,
                    'file_modified': datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
                }
                
                # Register model using CLI
                orchestrator = self.cli._get_orchestrator()
                registered_id = orchestrator.register_model(
                    model_path=str(model_path),
                    model_type=model_type,
                    model_id=model_id,
                    metadata=metadata
                )
                
                registered_ids.append(registered_id)
                logger.info(f"✅ Registered: {registered_id}")
                
            except Exception as e:
                logger.error(f"Failed to register {model_path}: {e}")
                continue
        
        logger.info(f"Successfully registered {len(registered_ids)} models")
        return registered_ids
    
    def run_evaluation_campaign(self, config_files: List[str], 
                               model_pattern: Optional[str] = None,
                               suite_pattern: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation campaign across multiple configurations."""
        campaign_id = f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        campaign_results = {
            'campaign_id': campaign_id,
            'start_time': datetime.now().isoformat(),
            'config_files': config_files,
            'model_pattern': model_pattern,
            'suite_pattern': suite_pattern,
            'evaluations': [],
            'summary': {}
        }
        
        logger.info(f"🚀 Starting evaluation campaign: {campaign_id}")
        
        # Get models to evaluate
        orchestrator = self.cli._get_orchestrator()
        all_models = orchestrator.model_registry.list_models()
        
        if model_pattern:
            models_to_evaluate = [
                model for model in all_models 
                if re.match(model_pattern, model.model_id)
            ]
        else:
            models_to_evaluate = all_models
        
        if not models_to_evaluate:
            logger.error("No models found matching pattern")
            return campaign_results
        
        model_ids = [model.model_id for model in models_to_evaluate]
        logger.info(f"Evaluating {len(model_ids)} models: {model_ids}")
        
        # Run evaluation for each configuration
        for config_file in config_files:
            try:
                config_path = Path(config_file)
                if not config_path.exists():
                    logger.error(f"Configuration file not found: {config_path}")
                    continue
                
                logger.info(f"📋 Running evaluation with config: {config_path}")
                
                # Load configuration
                config = EvaluationConfig.from_yaml(config_path)
                
                # Filter suites if pattern provided
                suites_to_run = config.get_enabled_suites()
                if suite_pattern:
                    suites_to_run = [
                        suite for suite in suites_to_run
                        if re.match(suite_pattern, suite)
                    ]
                
                if not suites_to_run:
                    logger.warning(f"No suites found matching pattern in {config_path}")
                    continue
                
                # Create new CLI instance with this configuration
                campaign_cli = EvaluationCLI(config_path=str(config_path))
                campaign_orchestrator = campaign_cli._get_orchestrator()
                
                # Register models in this orchestrator
                for model in models_to_evaluate:
                    campaign_orchestrator.register_model(
                        model_path=model.model_path,
                        model_type=model.model_type,
                        model_id=model.model_id,
                        metadata=model.metadata
                    )
                
                # Schedule and run evaluation
                task_ids = campaign_orchestrator.schedule_evaluation(
                    model_ids=model_ids,
                    suite_names=suites_to_run
                )
                
                if campaign_orchestrator.start_evaluation():
                    # Wait for completion
                    self._wait_for_evaluation_completion(campaign_orchestrator)
                    
                    # Collect results
                    results = campaign_orchestrator.get_results()
                    
                    evaluation_record = {
                        'config_file': str(config_path),
                        'suites_run': suites_to_run,
                        'models_evaluated': model_ids,
                        'task_count': len(task_ids),
                        'results_count': len(results),
                        'completion_time': datetime.now().isoformat()
                    }
                    
                    campaign_results['evaluations'].append(evaluation_record)
                    logger.info(f"✅ Completed evaluation with {config_path}")
                else:
                    logger.error(f"Failed to start evaluation with {config_path}")
                
            except Exception as e:
                logger.error(f"Evaluation failed for {config_file}: {e}")
                continue
        
        # Generate campaign summary
        campaign_results['end_time'] = datetime.now().isoformat()
        campaign_results['summary'] = self._generate_campaign_summary(campaign_results)
        
        # Save campaign results
        campaign_file = Path(self.config.results_directory) / f"{campaign_id}_results.json"
        with open(campaign_file, 'w') as f:
            json.dump(campaign_results, f, indent=2)
        
        logger.info(f"🎉 Campaign completed: {campaign_id}")
        logger.info(f"Results saved to: {campaign_file}")
        
        return campaign_results
    
    def _wait_for_evaluation_completion(self, orchestrator: EvaluationOrchestrator, 
                                      timeout_hours: Optional[float] = None):
        """Wait for evaluation to complete with timeout."""
        timeout_hours = timeout_hours or self.config.evaluation_timeout_hours
        start_time = time.time()
        timeout_seconds = timeout_hours * 3600
        
        while True:
            progress = orchestrator.get_progress()
            
            # Check if completed
            if progress.running_tasks == 0 and progress.pending_tasks == 0:
                break
            
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                logger.warning(f"Evaluation timeout after {timeout_hours} hours")
                orchestrator.stop_evaluation()
                break
            
            # Wait before next check
            time.sleep(10)
    
    def _generate_campaign_summary(self, campaign_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for campaign."""
        evaluations = campaign_results['evaluations']
        
        if not evaluations:
            return {'status': 'no_evaluations_completed'}
        
        total_tasks = sum(eval_record['task_count'] for eval_record in evaluations)
        total_results = sum(eval_record['results_count'] for eval_record in evaluations)
        
        # Calculate duration
        start_time = datetime.fromisoformat(campaign_results['start_time'])
        end_time = datetime.fromisoformat(campaign_results['end_time'])
        duration_hours = (end_time - start_time).total_seconds() / 3600
        
        return {
            'status': 'completed',
            'total_evaluations': len(evaluations),
            'total_tasks': total_tasks,
            'total_results': total_results,
            'success_rate': total_results / max(total_tasks, 1),
            'duration_hours': duration_hours,
            'configs_processed': len(campaign_results['config_files']),
            'models_evaluated': len(set(
                model_id 
                for eval_record in evaluations 
                for model_id in eval_record['models_evaluated']
            ))
        }
    
    def compare_model_generations(self, base_pattern: str, new_pattern: str,
                                 output_file: Optional[str] = None) -> Dict[str, Any]:
        """Compare two generations of models."""
        orchestrator = self.cli._get_orchestrator()
        all_models = orchestrator.model_registry.list_models()
        
        # Find models matching patterns
        base_models = [
            model for model in all_models 
            if re.match(base_pattern, model.model_id)
        ]
        new_models = [
            model for model in all_models 
            if re.match(new_pattern, model.model_id)
        ]
        
        if not base_models or not new_models:
            logger.error("Insufficient models found for comparison")
            return {}
        
        logger.info(f"Comparing {len(base_models)} base models vs {len(new_models)} new models")
        
        # Get results for both generations
        base_results = {}
        for model in base_models:
            results = orchestrator.get_results(model_id=model.model_id)
            if results:
                base_results[model.model_id] = results
        
        new_results = {}
        for model in new_models:
            results = orchestrator.get_results(model_id=model.model_id)
            if results:
                new_results[model.model_id] = results
        
        # Generate comparison
        comparison = {
            'comparison_id': f"generation_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'base_pattern': base_pattern,
            'new_pattern': new_pattern,
            'base_models': list(base_results.keys()),
            'new_models': list(new_results.keys()),
            'comparison_time': datetime.now().isoformat(),
            'summary': self._compare_generation_results(base_results, new_results)
        }
        
        # Save comparison if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(comparison, f, indent=2)
            logger.info(f"Comparison saved to: {output_file}")
        
        return comparison
    
    def _compare_generation_results(self, base_results: Dict, new_results: Dict) -> Dict[str, Any]:
        """Compare results between two model generations."""
        # Aggregate metrics for each generation
        def aggregate_metrics(results_dict):
            all_metrics = []
            for model_results in results_dict.values():
                for result in model_results:
                    if 'results' in result and result['results']:
                        all_metrics.append(result['results'])
            
            if not all_metrics:
                return {}
            
            # Calculate averages
            metric_keys = all_metrics[0].keys()
            aggregated = {}
            for key in metric_keys:
                values = [m.get(key, 0) for m in all_metrics if isinstance(m.get(key), (int, float))]
                if values:
                    aggregated[f"mean_{key}"] = sum(values) / len(values)
                    aggregated[f"max_{key}"] = max(values)
                    aggregated[f"min_{key}"] = min(values)
            
            return aggregated
        
        base_aggregated = aggregate_metrics(base_results)
        new_aggregated = aggregate_metrics(new_results)
        
        # Calculate improvements
        improvements = {}
        for key in base_aggregated.keys():
            if key in new_aggregated:
                base_val = base_aggregated[key]
                new_val = new_aggregated[key]
                if base_val != 0:
                    improvement = (new_val - base_val) / base_val * 100
                    improvements[key] = improvement
        
        return {
            'base_generation': base_aggregated,
            'new_generation': new_aggregated,
            'improvements_percent': improvements,
            'base_model_count': len(base_results),
            'new_model_count': len(new_results)
        }
    
    def setup_scheduled_evaluation(self, interval_hours: int = 24, 
                                  time_of_day: str = "02:00",
                                  model_pattern: Optional[str] = None,
                                  config_file: Optional[str] = None):
        """Setup scheduled periodic evaluation."""
        def scheduled_job():
            logger.info("🕐 Running scheduled evaluation")
            try:
                # Discover and register new models
                new_models = self.register_models_batch(
                    self.config.models_directory,
                    self.config.model_patterns,
                    self.config.exclude_patterns
                )
                
                if new_models:
                    logger.info(f"Found {len(new_models)} new models")
                    
                    # Run evaluation
                    if config_file:
                        config_files = [config_file]
                    else:
                        # Use default configuration
                        config_files = [str(Path(self.config.config_directory) / "evaluation_config.yml")]
                    
                    campaign_results = self.run_evaluation_campaign(
                        config_files=config_files,
                        model_pattern=model_pattern
                    )
                    
                    logger.info(f"Scheduled evaluation completed: {campaign_results['campaign_id']}")
                else:
                    logger.info("No new models found for scheduled evaluation")
                    
            except Exception as e:
                logger.error(f"Scheduled evaluation failed: {e}")
        
        # Schedule the job
        schedule.every(interval_hours).hours.do(scheduled_job)
        
        # Also schedule at specific time if provided
        if time_of_day:
            schedule.every().day.at(time_of_day).do(scheduled_job)
        
        logger.info(f"📅 Scheduled evaluation every {interval_hours} hours at {time_of_day}")
        
        # Start scheduler thread
        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(target=self._run_scheduler)
        self._scheduler_thread.daemon = True
        self._scheduler_thread.start()
    
    def _run_scheduler(self):
        """Run the scheduler in background thread."""
        while self._scheduler_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def stop_scheduler(self):
        """Stop the scheduled evaluation."""
        self._scheduler_running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        schedule.clear()
        logger.info("📅 Scheduled evaluation stopped")


def create_batch_parser() -> argparse.ArgumentParser:
    """Create argument parser for batch evaluation."""
    parser = argparse.ArgumentParser(
        description="Batch Evaluation Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Register batch command
    register_parser = subparsers.add_parser('register-batch', help='Register multiple models')
    register_parser.add_argument('--models-dir', required=True, help='Directory containing models')
    register_parser.add_argument('--patterns', nargs='*', default=['*.pth', '*.onnx'], 
                                help='File patterns to match')
    register_parser.add_argument('--exclude', nargs='*', default=['*temp*', '*backup*'],
                                help='Patterns to exclude')
    register_parser.add_argument('--model-type', default='checkpoint', help='Model type')
    
    # Campaign command
    campaign_parser = subparsers.add_parser('campaign', help='Run evaluation campaign')
    campaign_parser.add_argument('--config-dir', required=True, help='Directory with config files')
    campaign_parser.add_argument('--config-pattern', default='*.yml', help='Config file pattern')
    campaign_parser.add_argument('--models-pattern', help='Model ID pattern to match')
    campaign_parser.add_argument('--suites-pattern', help='Suite name pattern to match')
    
    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Setup scheduled evaluation')
    schedule_parser.add_argument('--interval', default='24h', help='Evaluation interval (e.g., 24h, 12h)')
    schedule_parser.add_argument('--time', default='02:00', help='Time of day to run (HH:MM)')
    schedule_parser.add_argument('--models-pattern', help='Model pattern for scheduled runs')
    schedule_parser.add_argument('--config', help='Configuration file for scheduled runs')
    
    # Compare generations command
    compare_parser = subparsers.add_parser('compare-generations', help='Compare model generations')
    compare_parser.add_argument('--base-pattern', required=True, help='Pattern for base models')
    compare_parser.add_argument('--new-pattern', required=True, help='Pattern for new models')
    compare_parser.add_argument('--output', help='Output file for comparison results')
    
    return parser


def main():
    """Main entry point for batch evaluation."""
    parser = create_batch_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create batch manager
    config = BatchEvaluationConfig()
    manager = BatchEvaluationManager(config)
    
    try:
        if args.command == 'register-batch':
            registered_ids = manager.register_models_batch(
                models_directory=args.models_dir,
                patterns=args.patterns,
                exclude_patterns=args.exclude,
                model_type=args.model_type
            )
            print(f"Registered {len(registered_ids)} models")
            
        elif args.command == 'campaign':
            # Find config files
            config_dir = Path(args.config_dir)
            config_files = list(config_dir.glob(args.config_pattern))
            
            if not config_files:
                logger.error(f"No config files found in {config_dir}")
                return 1
            
            results = manager.run_evaluation_campaign(
                config_files=[str(f) for f in config_files],
                model_pattern=args.models_pattern,
                suite_pattern=args.suites_pattern
            )
            print(f"Campaign completed: {results['campaign_id']}")
            
        elif args.command == 'schedule':
            # Parse interval
            interval_match = re.match(r'(\d+)h', args.interval)
            if not interval_match:
                logger.error("Invalid interval format. Use format like '24h'")
                return 1
            
            interval_hours = int(interval_match.group(1))
            
            manager.setup_scheduled_evaluation(
                interval_hours=interval_hours,
                time_of_day=args.time,
                model_pattern=args.models_pattern,
                config_file=args.config
            )
            
            print(f"Scheduled evaluation setup. Running every {interval_hours} hours at {args.time}")
            print("Press Ctrl+C to stop...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                manager.stop_scheduler()
                print("Scheduler stopped")
            
        elif args.command == 'compare-generations':
            comparison = manager.compare_model_generations(
                base_pattern=args.base_pattern,
                new_pattern=args.new_pattern,
                output_file=args.output
            )
            
            if comparison:
                print(f"Generation comparison completed: {comparison['comparison_id']}")
                summary = comparison['summary']
                print(f"Base models: {summary['base_model_count']}")
                print(f"New models: {summary['new_model_count']}")
            
        return 0
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())