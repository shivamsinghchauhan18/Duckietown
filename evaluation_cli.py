#!/usr/bin/env python3
"""
🏆 EVALUATION CLI 🏆
Command-line interface for the Enhanced Duckietown RL Evaluation Orchestrator

This CLI provides comprehensive tools for:
- Model registration and management
- Batch evaluation execution
- Progress monitoring and reporting
- Result querying and analysis
- Evaluation orchestration

Usage Examples:
    # Register models for evaluation
    python evaluation_cli.py register --model-path models/champion.pth --model-id champion_v1
    
    # Run comprehensive evaluation
    python evaluation_cli.py evaluate --models champion_v1 baseline_v2 --suites base hard law
    
    # Monitor evaluation progress
    python evaluation_cli.py monitor --follow
    
    # Query evaluation results
    python evaluation_cli.py results --model champion_v1 --format json
    
    # Generate comparison report
    python evaluation_cli.py compare --models champion_v1 baseline_v2 --output comparison_report.html
"""

import os
import sys
import json
import time
import argparse
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import asdict

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from duckietown_utils.evaluation_orchestrator import (
    EvaluationOrchestrator, 
    EvaluationProgress,
    PolicyMode,
    EvaluationStatus
)
from config.evaluation_config import EvaluationConfig
from duckietown_utils.report_generator import ReportGenerator
from duckietown_utils.champion_selector import ChampionSelector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationCLI:
    """Main CLI class for evaluation orchestrator."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the CLI with optional configuration."""
        self.config_path = config_path
        self.orchestrator: Optional[EvaluationOrchestrator] = None
        self.config: Optional[EvaluationConfig] = None
        
        # Default paths
        self.default_config_path = Path("config/evaluation_config.yml")
        self.results_dir = Path("evaluation_results")
        self.models_registry_path = Path("evaluation_models_registry.json")
        
        # Progress monitoring
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def _load_config(self) -> EvaluationConfig:
        """Load evaluation configuration."""
        if self.config is not None:
            return self.config
        
        config_path = Path(self.config_path) if self.config_path else self.default_config_path
        
        if config_path.exists():
            logger.info(f"Loading configuration from {config_path}")
            self.config = EvaluationConfig.from_yaml(config_path)
        else:
            logger.info("Using default configuration")
            from config.evaluation_config import create_comprehensive_evaluation_config
            self.config = create_comprehensive_evaluation_config()
        
        return self.config
    
    def _get_orchestrator(self) -> EvaluationOrchestrator:
        """Get or create evaluation orchestrator."""
        if self.orchestrator is None:
            config = self._load_config()
            orchestrator_config = {
                'max_concurrent_evaluations': config.max_parallel_workers,
                'evaluation_timeout': config.evaluation_timeout_hours * 3600,
                'results_dir': str(self.results_dir),
                'base_seed': config.reproducibility.seed_base
            }
            self.orchestrator = EvaluationOrchestrator(orchestrator_config)
        
        return self.orchestrator
    
    def _save_models_registry(self, models: Dict[str, Any]):
        """Save models registry to file."""
        with open(self.models_registry_path, 'w') as f:
            json.dump(models, f, indent=2)
    
    def _load_models_registry(self) -> Dict[str, Any]:
        """Load models registry from file."""
        if self.models_registry_path.exists():
            with open(self.models_registry_path, 'r') as f:
                return json.load(f)
        return {}
    
    def register_model(self, args) -> int:
        """Register a model for evaluation."""
        try:
            orchestrator = self._get_orchestrator()
            
            # Validate model path
            model_path = Path(args.model_path)
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return 1
            
            # Prepare metadata
            metadata = {
                'description': args.description or f"Model registered on {datetime.now().isoformat()}",
                'tags': args.tags or [],
                'registration_time': datetime.now().isoformat()
            }
            
            if args.metadata:
                try:
                    additional_metadata = json.loads(args.metadata)
                    metadata.update(additional_metadata)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid metadata JSON: {e}")
                    return 1
            
            # Register model
            model_id = orchestrator.register_model(
                model_path=str(model_path),
                model_type=args.model_type,
                model_id=args.model_id,
                metadata=metadata
            )
            
            logger.info(f"✅ Successfully registered model: {model_id}")
            logger.info(f"   Path: {model_path}")
            logger.info(f"   Type: {args.model_type}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return 1
    
    def list_models(self, args) -> int:
        """List registered models."""
        try:
            orchestrator = self._get_orchestrator()
            models = orchestrator.model_registry.list_models()
            
            if not models:
                print("No models registered")
                return 0
            
            print(f"Registered Models ({len(models)} total):")
            print("=" * 80)
            
            for model in models:
                print(f"ID: {model.model_id}")
                print(f"  Path: {model.model_path}")
                print(f"  Type: {model.model_type}")
                print(f"  Registered: {model.registration_time}")
                
                if model.metadata:
                    if 'description' in model.metadata:
                        print(f"  Description: {model.metadata['description']}")
                    if 'tags' in model.metadata and model.metadata['tags']:
                        print(f"  Tags: {', '.join(model.metadata['tags'])}")
                
                print()
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return 1
    
    def evaluate_models(self, args) -> int:
        """Run evaluation on specified models and suites."""
        try:
            orchestrator = self._get_orchestrator()
            config = self._load_config()
            
            # Parse model IDs
            model_ids = args.models
            if args.all_models:
                registered_models = orchestrator.model_registry.list_models()
                model_ids = [model.model_id for model in registered_models]
            
            if not model_ids:
                logger.error("No models specified for evaluation")
                return 1
            
            # Parse suite names
            suite_names = args.suites or config.get_enabled_suites()
            if not suite_names:
                logger.error("No evaluation suites specified or enabled")
                return 1
            
            # Parse policy modes
            policy_modes = []
            if args.deterministic:
                policy_modes.append(PolicyMode.DETERMINISTIC)
            if args.stochastic:
                policy_modes.append(PolicyMode.STOCHASTIC)
            if not policy_modes:
                policy_modes = [PolicyMode.DETERMINISTIC, PolicyMode.STOCHASTIC]
            
            # Validate models exist
            for model_id in model_ids:
                if not orchestrator.model_registry.get_model(model_id):
                    logger.error(f"Model not found: {model_id}")
                    return 1
            
            logger.info(f"🚀 Starting evaluation:")
            logger.info(f"   Models: {model_ids}")
            logger.info(f"   Suites: {suite_names}")
            logger.info(f"   Policy modes: {[mode.value for mode in policy_modes]}")
            
            # Schedule evaluation tasks
            task_ids = orchestrator.schedule_evaluation(
                model_ids=model_ids,
                suite_names=suite_names,
                policy_modes=policy_modes,
                seeds_per_suite=args.seeds_per_suite
            )
            
            logger.info(f"📋 Scheduled {len(task_ids)} evaluation tasks")
            
            # Start evaluation
            if orchestrator.start_evaluation():
                logger.info("✅ Evaluation started successfully")
                
                # Monitor progress if requested
                if args.monitor:
                    self._start_progress_monitoring()
                    
                    # Wait for completion
                    try:
                        while True:
                            progress = orchestrator.get_progress()
                            if progress.running_tasks == 0 and progress.pending_tasks == 0:
                                break
                            time.sleep(5)
                    except KeyboardInterrupt:
                        logger.info("Stopping evaluation...")
                        orchestrator.stop_evaluation()
                        return 1
                    finally:
                        self._stop_progress_monitoring()
                
                logger.info("🎉 Evaluation completed!")
                return 0
            else:
                logger.error("Failed to start evaluation")
                return 1
                
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 1
    
    def monitor_progress(self, args) -> int:
        """Monitor evaluation progress."""
        try:
            orchestrator = self._get_orchestrator()
            
            if args.follow:
                logger.info("📊 Monitoring evaluation progress (Ctrl+C to stop)...")
                self._start_progress_monitoring()
                
                try:
                    while True:
                        time.sleep(2)
                except KeyboardInterrupt:
                    pass
                finally:
                    self._stop_progress_monitoring()
            else:
                # Show current progress once
                progress = orchestrator.get_progress()
                self._display_progress(progress)
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to monitor progress: {e}")
            return 1
    
    def _start_progress_monitoring(self):
        """Start progress monitoring in background thread."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._progress_monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def _stop_progress_monitoring(self):
        """Stop progress monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
    
    def _progress_monitor_loop(self):
        """Progress monitoring loop."""
        orchestrator = self._get_orchestrator()
        last_update = time.time()
        
        while self._monitoring:
            try:
                progress = orchestrator.get_progress()
                
                # Update display every 5 seconds or when significant change
                current_time = time.time()
                if current_time - last_update >= 5:
                    self._display_progress(progress)
                    last_update = current_time
                
                time.sleep(1)
            except Exception as e:
                logger.error(f"Progress monitoring error: {e}")
                break
    
    def _display_progress(self, progress: EvaluationProgress):
        """Display evaluation progress."""
        print(f"\r📊 Progress: {progress.overall_progress:.1f}% | "
              f"✅ {progress.completed_tasks} | "
              f"🏃 {progress.running_tasks} | "
              f"⏳ {progress.pending_tasks} | "
              f"❌ {progress.failed_tasks}", end="", flush=True)
        
        if progress.estimated_time_remaining:
            eta_minutes = progress.estimated_time_remaining / 60
            print(f" | ETA: {eta_minutes:.1f}m", end="", flush=True)
        
        if progress.overall_progress >= 100 or (progress.running_tasks == 0 and progress.pending_tasks == 0):
            print()  # New line when complete
    
    def query_results(self, args) -> int:
        """Query and display evaluation results."""
        try:
            orchestrator = self._get_orchestrator()
            
            # Get results with optional filtering
            results = orchestrator.get_results(
                model_id=args.model,
                suite_name=args.suite
            )
            
            if not results:
                print("No evaluation results found")
                return 0
            
            # Format output
            if args.format == 'json':
                output = json.dumps(results, indent=2)
            elif args.format == 'csv':
                output = self._format_results_as_csv(results)
            else:  # table format
                output = self._format_results_as_table(results)
            
            # Output to file or stdout
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                logger.info(f"Results saved to {args.output}")
            else:
                print(output)
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to query results: {e}")
            return 1
    
    def _format_results_as_table(self, results: List[Dict[str, Any]]) -> str:
        """Format results as a table."""
        if not results:
            return "No results to display"
        
        # Extract key metrics for table display
        table_data = []
        for result in results:
            if 'results' in result and result['results']:
                r = result['results']
                table_data.append({
                    'Model': result['model_id'],
                    'Suite': result['suite_name'],
                    'Mode': result['policy_mode'],
                    'Success Rate': f"{r.get('success_rate', 0):.3f}",
                    'Mean Reward': f"{r.get('mean_reward', 0):.3f}",
                    'Episode Length': f"{r.get('mean_episode_length', 0):.1f}",
                    'Lateral Dev': f"{r.get('mean_lateral_deviation', 0):.3f}",
                    'Heading Error': f"{r.get('mean_heading_error', 0):.1f}°"
                })
        
        if not table_data:
            return "No valid results to display"
        
        # Create table
        headers = list(table_data[0].keys())
        col_widths = {h: max(len(h), max(len(str(row[h])) for row in table_data)) for h in headers}
        
        # Header
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        separator = "-" * len(header_line)
        
        # Rows
        rows = []
        for row in table_data:
            row_line = " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers)
            rows.append(row_line)
        
        return f"{header_line}\n{separator}\n" + "\n".join(rows)
    
    def _format_results_as_csv(self, results: List[Dict[str, Any]]) -> str:
        """Format results as CSV."""
        import csv
        import io
        
        if not results:
            return ""
        
        output = io.StringIO()
        
        # Extract all unique keys from results
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
            if 'results' in result and result['results']:
                all_keys.update(f"result_{k}" for k in result['results'].keys())
        
        writer = csv.writer(output)
        writer.writerow(sorted(all_keys))
        
        for result in results:
            row = {}
            row.update(result)
            if 'results' in result and result['results']:
                for k, v in result['results'].items():
                    row[f"result_{k}"] = v
            
            writer.writerow([row.get(k, '') for k in sorted(all_keys)])
        
        return output.getvalue()
    
    def compare_models(self, args) -> int:
        """Compare evaluation results between models."""
        try:
            orchestrator = self._get_orchestrator()
            
            if len(args.models) < 2:
                logger.error("At least 2 models required for comparison")
                return 1
            
            # Get results for all models
            all_results = {}
            for model_id in args.models:
                results = orchestrator.get_results(model_id=model_id)
                if not results:
                    logger.warning(f"No results found for model: {model_id}")
                    continue
                all_results[model_id] = results
            
            if len(all_results) < 2:
                logger.error("Insufficient results for comparison")
                return 1
            
            # Generate comparison report
            if args.output:
                output_path = Path(args.output)
                if output_path.suffix.lower() == '.html':
                    self._generate_html_comparison(all_results, output_path)
                else:
                    self._generate_text_comparison(all_results, output_path)
                logger.info(f"Comparison report saved to {output_path}")
            else:
                # Print to stdout
                comparison_text = self._generate_text_comparison(all_results)
                print(comparison_text)
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            return 1
    
    def _generate_text_comparison(self, results: Dict[str, List[Dict]], output_path: Optional[Path] = None) -> str:
        """Generate text-based comparison report."""
        lines = []
        lines.append("Model Comparison Report")
        lines.append("=" * 50)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Models: {list(results.keys())}")
        lines.append("")
        
        # Aggregate results by suite
        suite_comparisons = {}
        for model_id, model_results in results.items():
            for result in model_results:
                suite_name = result['suite_name']
                if suite_name not in suite_comparisons:
                    suite_comparisons[suite_name] = {}
                
                if 'results' in result and result['results']:
                    suite_comparisons[suite_name][model_id] = result['results']
        
        # Generate comparison for each suite
        for suite_name, suite_results in suite_comparisons.items():
            lines.append(f"Suite: {suite_name}")
            lines.append("-" * 30)
            
            # Key metrics comparison
            metrics = ['success_rate', 'mean_reward', 'mean_episode_length', 
                      'mean_lateral_deviation', 'mean_heading_error']
            
            for metric in metrics:
                lines.append(f"  {metric}:")
                for model_id, model_results in suite_results.items():
                    value = model_results.get(metric, 0)
                    lines.append(f"    {model_id}: {value:.4f}")
                lines.append("")
            
            lines.append("")
        
        comparison_text = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(comparison_text)
        
        return comparison_text
    
    def _generate_html_comparison(self, results: Dict[str, List[Dict]], output_path: Path):
        """Generate HTML comparison report using ReportGenerator."""
        try:
            # This would use the ReportGenerator class for comprehensive HTML reports
            # For now, create a simple HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Comparison Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .metric {{ font-weight: bold; }}
                </style>
            </head>
            <body>
                <h1>Model Comparison Report</h1>
                <p>Generated: {datetime.now().isoformat()}</p>
                <p>Models: {', '.join(results.keys())}</p>
                
                <h2>Results Summary</h2>
                <p>Detailed comparison would be generated here using ReportGenerator</p>
                
                <h2>Raw Data</h2>
                <pre>{json.dumps(results, indent=2)}</pre>
            </body>
            </html>
            """
            
            with open(output_path, 'w') as f:
                f.write(html_content)
                
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            raise
    
    def stop_evaluation(self, args) -> int:
        """Stop running evaluation."""
        try:
            orchestrator = self._get_orchestrator()
            orchestrator.stop_evaluation()
            logger.info("🛑 Evaluation stopped")
            return 0
        except Exception as e:
            logger.error(f"Failed to stop evaluation: {e}")
            return 1
    
    def cleanup(self, args) -> int:
        """Cleanup evaluation resources."""
        try:
            if self.orchestrator:
                self.orchestrator.cleanup()
            logger.info("🧹 Cleanup completed")
            return 0
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="Enhanced Duckietown RL Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register a model
  python evaluation_cli.py register --model-path models/champion.pth --model-id champion_v1
  
  # List registered models
  python evaluation_cli.py list-models
  
  # Run evaluation on specific models and suites
  python evaluation_cli.py evaluate --models champion_v1 baseline_v2 --suites base hard
  
  # Monitor evaluation progress
  python evaluation_cli.py monitor --follow
  
  # Query results
  python evaluation_cli.py results --model champion_v1 --format json
  
  # Compare models
  python evaluation_cli.py compare --models champion_v1 baseline_v2 --output comparison.html
        """
    )
    
    parser.add_argument('--config', '-c', help='Path to evaluation configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a model for evaluation')
    register_parser.add_argument('--model-path', required=True, help='Path to model file')
    register_parser.add_argument('--model-id', help='Custom model ID (auto-generated if not provided)')
    register_parser.add_argument('--model-type', default='checkpoint', 
                                choices=['checkpoint', 'onnx', 'pytorch', 'tensorflow'],
                                help='Type of model file')
    register_parser.add_argument('--description', help='Model description')
    register_parser.add_argument('--tags', nargs='*', help='Model tags')
    register_parser.add_argument('--metadata', help='Additional metadata as JSON string')
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List registered models')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run model evaluation')
    eval_parser.add_argument('--models', nargs='*', help='Model IDs to evaluate')
    eval_parser.add_argument('--all-models', action='store_true', help='Evaluate all registered models')
    eval_parser.add_argument('--suites', nargs='*', help='Evaluation suites to run')
    eval_parser.add_argument('--deterministic', action='store_true', help='Run deterministic policy mode')
    eval_parser.add_argument('--stochastic', action='store_true', help='Run stochastic policy mode')
    eval_parser.add_argument('--seeds-per-suite', type=int, help='Number of seeds per suite')
    eval_parser.add_argument('--monitor', action='store_true', help='Monitor progress during evaluation')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor evaluation progress')
    monitor_parser.add_argument('--follow', action='store_true', help='Continuously monitor progress')
    
    # Results command
    results_parser = subparsers.add_parser('results', help='Query evaluation results')
    results_parser.add_argument('--model', help='Filter by model ID')
    results_parser.add_argument('--suite', help='Filter by suite name')
    results_parser.add_argument('--format', choices=['table', 'json', 'csv'], default='table',
                               help='Output format')
    results_parser.add_argument('--output', help='Output file path')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare model evaluation results')
    compare_parser.add_argument('--models', nargs='+', required=True, help='Model IDs to compare')
    compare_parser.add_argument('--output', help='Output file path for comparison report')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop running evaluation')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup evaluation resources')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create CLI instance
    cli = EvaluationCLI(config_path=args.config)
    
    # Execute command
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Map commands to methods
        command_map = {
            'register': cli.register_model,
            'list-models': cli.list_models,
            'evaluate': cli.evaluate_models,
            'monitor': cli.monitor_progress,
            'results': cli.query_results,
            'compare': cli.compare_models,
            'stop': cli.stop_evaluation,
            'cleanup': cli.cleanup
        }
        
        if args.command in command_map:
            return command_map[args.command](args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())