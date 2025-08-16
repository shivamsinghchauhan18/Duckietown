#!/usr/bin/env python3
"""
🎯 EVALUATION CLI EXAMPLES 🎯
Comprehensive examples and tutorials for using the evaluation CLI tools

This module provides practical examples for:
- Basic evaluation workflows
- Advanced analysis scenarios
- Batch processing examples
- Monitoring and reporting
- Integration patterns

Run examples:
    python examples/evaluation_cli_examples.py basic_workflow
    python examples/evaluation_cli_examples.py batch_campaign
    python examples/evaluation_cli_examples.py analysis_workflow
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluation_cli import EvaluationCLI
from evaluation_analysis import EvaluationAnalyzer, QueryFilter
from batch_evaluation import BatchEvaluationManager, BatchEvaluationConfig
from evaluation_monitor import EvaluationMonitor, MonitoringConfig


class EvaluationCLIExamples:
    """Collection of evaluation CLI usage examples."""
    
    def __init__(self):
        self.cli = EvaluationCLI()
        self.examples_dir = Path("examples/evaluation_outputs")
        self.examples_dir.mkdir(parents=True, exist_ok=True)
    
    def basic_workflow_example(self):
        """Example 1: Basic evaluation workflow."""
        print("🎯 Example 1: Basic Evaluation Workflow")
        print("=" * 50)
        
        # Step 1: Register a model
        print("\n📝 Step 1: Registering a model...")
        
        # Create a mock model file for demonstration
        mock_model_path = self.examples_dir / "demo_model.pth"
        with open(mock_model_path, 'w') as f:
            f.write("# Mock model file for demonstration")
        
        try:
            orchestrator = self.cli._get_orchestrator()
            model_id = orchestrator.register_model(
                model_path=str(mock_model_path),
                model_type="checkpoint",
                model_id="demo_model_v1",
                metadata={
                    "description": "Demo model for CLI examples",
                    "algorithm": "PPO",
                    "training_steps": 500000,
                    "created_by": "evaluation_cli_examples"
                }
            )
            print(f"✅ Registered model: {model_id}")
            
            # Step 2: List registered models
            print("\n📋 Step 2: Listing registered models...")
            models = orchestrator.model_registry.list_models()
            for model in models:
                print(f"  - {model.model_id}: {model.model_path}")
            
            # Step 3: Schedule evaluation
            print("\n🚀 Step 3: Scheduling evaluation...")
            task_ids = orchestrator.schedule_evaluation(
                model_ids=[model_id],
                suite_names=["base"],
                seeds_per_suite=5  # Small number for demo
            )
            print(f"✅ Scheduled {len(task_ids)} evaluation tasks")
            
            # Step 4: Start evaluation (simulated)
            print("\n⚡ Step 4: Starting evaluation...")
            if orchestrator.start_evaluation():
                print("✅ Evaluation started successfully")
                
                # Wait a bit for some progress
                time.sleep(10)
                
                # Check progress
                progress = orchestrator.get_progress()
                print(f"📊 Progress: {progress.overall_progress:.1f}% complete")
                print(f"   Completed: {progress.completed_tasks}")
                print(f"   Running: {progress.running_tasks}")
                print(f"   Failed: {progress.failed_tasks}")
                
                # Wait for completion (with timeout)
                timeout = 60  # 1 minute timeout for demo
                start_time = time.time()
                
                while time.time() - start_time < timeout:
                    progress = orchestrator.get_progress()
                    if progress.running_tasks == 0 and progress.pending_tasks == 0:
                        break
                    time.sleep(2)
                
                # Step 5: Query results
                print("\n📈 Step 5: Querying results...")
                results = orchestrator.get_results(model_id=model_id)
                
                if results:
                    print(f"✅ Found {len(results)} evaluation results")
                    for result in results:
                        if 'results' in result and result['results']:
                            r = result['results']
                            print(f"  Suite: {result['suite_name']}")
                            print(f"    Success Rate: {r.get('success_rate', 0):.3f}")
                            print(f"    Mean Reward: {r.get('mean_reward', 0):.3f}")
                else:
                    print("⚠️  No results found (evaluation may still be running)")
            
            print("\n🎉 Basic workflow completed!")
            
        except Exception as e:
            print(f"❌ Error in basic workflow: {e}")
    
    def batch_campaign_example(self):
        """Example 2: Batch evaluation campaign."""
        print("🚀 Example 2: Batch Evaluation Campaign")
        print("=" * 50)
        
        # Create mock models directory
        models_dir = self.examples_dir / "mock_models"
        models_dir.mkdir(exist_ok=True)
        
        # Create several mock model files
        model_files = []
        for i in range(3):
            model_path = models_dir / f"model_v{i+1}.pth"
            with open(model_path, 'w') as f:
                f.write(f"# Mock model v{i+1}")
            model_files.append(model_path)
        
        print(f"📁 Created {len(model_files)} mock models in {models_dir}")
        
        try:
            # Initialize batch manager
            config = BatchEvaluationConfig(
                models_directory=str(models_dir),
                max_concurrent_evaluations=2,
                evaluation_timeout_hours=1.0  # Short timeout for demo
            )
            
            batch_manager = BatchEvaluationManager(config)
            
            # Step 1: Batch register models
            print("\n📝 Step 1: Batch registering models...")
            registered_ids = batch_manager.register_models_batch(
                models_directory=str(models_dir),
                patterns=["*.pth"],
                model_type="checkpoint"
            )
            
            print(f"✅ Registered {len(registered_ids)} models: {registered_ids}")
            
            # Step 2: Create mock configuration files
            print("\n⚙️  Step 2: Creating evaluation configurations...")
            configs_dir = self.examples_dir / "mock_configs"
            configs_dir.mkdir(exist_ok=True)
            
            # Create basic config
            basic_config = {
                'suites': {
                    'demo_base': {
                        'name': 'demo_base',
                        'enabled': True,
                        'seeds_per_map': 3,
                        'maps': ['loop_empty'],
                        'policy_modes': ['deterministic']
                    }
                },
                'metrics': {
                    'compute_confidence_intervals': False,
                    'bootstrap_resamples': 100
                },
                'max_parallel_workers': 1
            }
            
            config_file = configs_dir / "demo_config.yml"
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(basic_config, f)
            
            print(f"✅ Created configuration: {config_file}")
            
            # Step 3: Run evaluation campaign (simulated)
            print("\n🏃 Step 3: Running evaluation campaign...")
            print("⚠️  Note: This is a demonstration - actual evaluation would take longer")
            
            # Simulate campaign results
            campaign_results = {
                'campaign_id': f"demo_campaign_{int(time.time())}",
                'start_time': datetime.now().isoformat(),
                'config_files': [str(config_file)],
                'evaluations': [
                    {
                        'config_file': str(config_file),
                        'suites_run': ['demo_base'],
                        'models_evaluated': registered_ids,
                        'task_count': len(registered_ids) * 1,  # 1 suite
                        'results_count': len(registered_ids) * 1,
                        'completion_time': datetime.now().isoformat()
                    }
                ],
                'summary': {
                    'status': 'simulated',
                    'total_evaluations': 1,
                    'models_evaluated': len(registered_ids)
                }
            }
            
            # Save campaign results
            campaign_file = self.examples_dir / f"{campaign_results['campaign_id']}_results.json"
            with open(campaign_file, 'w') as f:
                json.dump(campaign_results, f, indent=2)
            
            print(f"✅ Campaign completed: {campaign_results['campaign_id']}")
            print(f"📄 Results saved to: {campaign_file}")
            
            # Step 4: Model generation comparison
            print("\n🔍 Step 4: Model generation comparison...")
            
            # Simulate comparison between model versions
            comparison_result = {
                'comparison_id': f"generation_comparison_{int(time.time())}",
                'base_pattern': 'model_v1*',
                'new_pattern': 'model_v[23]*',
                'base_models': [registered_ids[0]] if registered_ids else [],
                'new_models': registered_ids[1:] if len(registered_ids) > 1 else [],
                'summary': {
                    'base_model_count': 1,
                    'new_model_count': len(registered_ids) - 1,
                    'improvements_percent': {
                        'mean_success_rate': 5.2,  # 5.2% improvement
                        'mean_reward': 3.8
                    }
                }
            }
            
            comparison_file = self.examples_dir / f"{comparison_result['comparison_id']}.json"
            with open(comparison_file, 'w') as f:
                json.dump(comparison_result, f, indent=2)
            
            print(f"✅ Generation comparison completed")
            print(f"📄 Comparison saved to: {comparison_file}")
            
            print("\n🎉 Batch campaign example completed!")
            
        except Exception as e:
            print(f"❌ Error in batch campaign: {e}")
    
    def monitoring_example(self):
        """Example 3: Evaluation monitoring."""
        print("📊 Example 3: Evaluation Monitoring")
        print("=" * 50)
        
        try:
            # Initialize monitoring
            config = MonitoringConfig(
                update_interval_seconds=1.0,
                enable_alerts=True,
                alert_thresholds={
                    'failure_rate': 0.2,  # 20% failure rate threshold
                    'memory_usage_gb': 4.0  # 4GB memory threshold
                }
            )
            
            monitor = EvaluationMonitor(config)
            
            print("🚀 Starting evaluation monitoring...")
            monitor.start_monitoring()
            
            # Simulate monitoring for a short period
            print("📊 Monitoring evaluation progress...")
            for i in range(10):
                status = monitor.get_current_status()
                
                if status['evaluation_metrics']:
                    metrics = status['evaluation_metrics']
                    print(f"  Progress: {metrics['overall_progress']:.1f}% | "
                          f"Completed: {metrics['completed_tasks']} | "
                          f"Running: {metrics['running_tasks']} | "
                          f"Failed: {metrics['failed_tasks']}")
                else:
                    print(f"  Monitoring active: {status['monitoring_active']}")
                
                time.sleep(2)
            
            # Generate monitoring report
            print("\n📄 Generating monitoring report...")
            report_file = self.examples_dir / "monitoring_report.json"
            
            final_status = monitor.get_current_status()
            with open(report_file, 'w') as f:
                json.dump(final_status, f, indent=2)
            
            print(f"✅ Monitoring report saved to: {report_file}")
            
            # Stop monitoring
            monitor.stop_monitoring()
            print("🛑 Monitoring stopped")
            
            print("\n🎉 Monitoring example completed!")
            
        except Exception as e:
            print(f"❌ Error in monitoring example: {e}")
    
    def analysis_workflow_example(self):
        """Example 4: Advanced analysis workflow."""
        print("📈 Example 4: Advanced Analysis Workflow")
        print("=" * 50)
        
        try:
            # Initialize analyzer
            analyzer = EvaluationAnalyzer()
            
            # Step 1: Create mock evaluation data
            print("📝 Step 1: Creating mock evaluation data...")
            
            # Register some mock models and results
            orchestrator = self.cli._get_orchestrator()
            
            mock_models = []
            for i in range(3):
                model_path = self.examples_dir / f"analysis_model_{i+1}.pth"
                with open(model_path, 'w') as f:
                    f.write(f"# Analysis model {i+1}")
                
                model_id = orchestrator.register_model(
                    model_path=str(model_path),
                    model_type="checkpoint",
                    model_id=f"analysis_model_{i+1}",
                    metadata={
                        "version": f"1.{i}",
                        "generation": 1 if i < 2 else 2
                    }
                )
                mock_models.append(model_id)
            
            print(f"✅ Created {len(mock_models)} mock models for analysis")
            
            # Step 2: Query results with filters
            print("\n🔍 Step 2: Querying results with filters...")
            
            # Create various query filters
            filters = [
                QueryFilter(model_pattern="analysis_model_*"),
                QueryFilter(suite_pattern="base"),
                QueryFilter(metric_name="success_rate", min_value=0.7)
            ]
            
            for i, filter_criteria in enumerate(filters):
                results = analyzer.query_results(filter_criteria)
                print(f"  Filter {i+1}: Found {len(results)} results")
            
            # Step 3: Performance trends analysis
            print("\n📊 Step 3: Analyzing performance trends...")
            
            trends_analysis = analyzer.analyze_performance_trends(
                model_ids=mock_models[:2],  # Analyze first 2 models
                time_range_days=7
            )
            
            trends_file = self.examples_dir / "trends_analysis.json"
            with open(trends_file, 'w') as f:
                json.dump(trends_analysis.__dict__, f, indent=2)
            
            print(f"✅ Trends analysis completed")
            print(f"📄 Results saved to: {trends_file}")
            print(f"🔍 Recommendations: {len(trends_analysis.recommendations)}")
            for rec in trends_analysis.recommendations[:3]:  # Show first 3
                print(f"  - {rec}")
            
            # Step 4: Model comparison
            print("\n⚖️  Step 4: Comparing model performance...")
            
            comparison_analysis = analyzer.compare_models(
                model_ids=mock_models,
                metrics=['success_rate', 'mean_reward', 'mean_lateral_deviation']
            )
            
            comparison_file = self.examples_dir / "model_comparison.json"
            with open(comparison_file, 'w') as f:
                json.dump(comparison_analysis.__dict__, f, indent=2)
            
            print(f"✅ Model comparison completed")
            print(f"📄 Results saved to: {comparison_file}")
            
            # Show rankings
            if 'rankings' in comparison_analysis.summary:
                rankings = comparison_analysis.summary['rankings']
                for metric, ranking in rankings.items():
                    if ranking:
                        print(f"  {metric}: {ranking[0][0]} leads with {ranking[0][1]:.3f}")
            
            # Step 5: Export results
            print("\n💾 Step 5: Exporting results...")
            
            # Export to CSV
            csv_file = self.examples_dir / "analysis_results.csv"
            count = analyzer.export_results(
                output_path=str(csv_file),
                format_type="csv",
                filter_criteria=QueryFilter(model_pattern="analysis_model_*"),
                include_metadata=True
            )
            
            print(f"✅ Exported {count} results to CSV: {csv_file}")
            
            # Step 6: Generate insights report
            print("\n🧠 Step 6: Generating insights report...")
            
            insights_file = self.examples_dir / "insights_report.html"
            insights_analysis = analyzer.generate_insights_report(
                output_path=str(insights_file),
                include_plots=False  # Skip plots for demo
            )
            
            print(f"✅ Insights report generated: {insights_file}")
            print(f"📊 Total evaluations analyzed: {insights_analysis.summary.get('total_evaluations', 0)}")
            print(f"💡 Recommendations: {len(insights_analysis.recommendations)}")
            
            print("\n🎉 Analysis workflow completed!")
            
        except Exception as e:
            print(f"❌ Error in analysis workflow: {e}")
    
    def integration_example(self):
        """Example 5: Integration with external systems."""
        print("🔗 Example 5: Integration Patterns")
        print("=" * 50)
        
        try:
            # Example 1: Programmatic API usage
            print("🐍 Programmatic API Usage:")
            
            cli = EvaluationCLI()
            orchestrator = cli._get_orchestrator()
            
            # Register model programmatically
            model_path = self.examples_dir / "integration_model.pth"
            with open(model_path, 'w') as f:
                f.write("# Integration example model")
            
            model_id = orchestrator.register_model(
                model_path=str(model_path),
                model_type="checkpoint",
                metadata={"source": "integration_example"}
            )
            
            print(f"  ✅ Registered model: {model_id}")
            
            # Schedule evaluation with callback
            def progress_callback(progress):
                if progress.overall_progress % 25 == 0:  # Every 25%
                    print(f"    Progress update: {progress.overall_progress:.1f}%")
            
            orchestrator.add_progress_callback(progress_callback)
            
            # Example 2: Command-line integration
            print("\n💻 Command-line Integration:")
            
            # Demonstrate CLI command construction
            cli_commands = [
                "python evaluation_cli.py list-models",
                f"python evaluation_cli.py results --model {model_id} --format json",
                "python evaluation_analysis.py query --model '*' --format csv --output results.csv"
            ]
            
            for cmd in cli_commands:
                print(f"  Command: {cmd}")
            
            # Example 3: Configuration-driven workflow
            print("\n⚙️  Configuration-driven Workflow:")
            
            workflow_config = {
                "models": {
                    "register_pattern": "models/*.pth",
                    "exclude_pattern": "*temp*"
                },
                "evaluation": {
                    "suites": ["base", "hard"],
                    "policy_modes": ["deterministic"],
                    "seeds_per_suite": 50
                },
                "analysis": {
                    "generate_trends": True,
                    "compare_models": True,
                    "export_format": "csv",
                    "insights_report": True
                },
                "monitoring": {
                    "web_dashboard": True,
                    "port": 8080,
                    "alerts_enabled": True
                }
            }
            
            config_file = self.examples_dir / "workflow_config.json"
            with open(config_file, 'w') as f:
                json.dump(workflow_config, f, indent=2)
            
            print(f"  ✅ Created workflow configuration: {config_file}")
            
            # Example 4: Results processing pipeline
            print("\n🔄 Results Processing Pipeline:")
            
            pipeline_steps = [
                "1. Register models from directory",
                "2. Run evaluation campaign",
                "3. Monitor progress with web dashboard",
                "4. Export results to database",
                "5. Generate analysis reports",
                "6. Send notifications on completion"
            ]
            
            for step in pipeline_steps:
                print(f"  {step}")
            
            # Example 5: Custom metrics integration
            print("\n📊 Custom Metrics Integration:")
            
            def custom_safety_metric(episode_data):
                """Example custom safety metric."""
                # This would calculate a custom safety score
                return 0.95  # Mock value
            
            def custom_efficiency_metric(episode_data):
                """Example custom efficiency metric."""
                # This would calculate efficiency based on episode data
                return 0.87  # Mock value
            
            custom_metrics = {
                "safety_score": custom_safety_metric,
                "efficiency_score": custom_efficiency_metric
            }
            
            print(f"  ✅ Defined {len(custom_metrics)} custom metrics")
            
            # Example 6: Automated reporting
            print("\n📧 Automated Reporting:")
            
            report_config = {
                "schedule": "daily",
                "time": "09:00",
                "recipients": ["team@example.com"],
                "include_plots": True,
                "metrics_threshold": {
                    "success_rate": 0.8,
                    "failure_rate": 0.1
                },
                "alert_conditions": [
                    "success_rate < 0.7",
                    "failure_rate > 0.2",
                    "no_evaluations_24h"
                ]
            }
            
            report_config_file = self.examples_dir / "reporting_config.json"
            with open(report_config_file, 'w') as f:
                json.dump(report_config, f, indent=2)
            
            print(f"  ✅ Created reporting configuration: {report_config_file}")
            
            print("\n🎉 Integration examples completed!")
            
        except Exception as e:
            print(f"❌ Error in integration example: {e}")
    
    def performance_optimization_example(self):
        """Example 6: Performance optimization techniques."""
        print("⚡ Example 6: Performance Optimization")
        print("=" * 50)
        
        try:
            # Optimization 1: Database usage for large datasets
            print("🗄️  Database Optimization:")
            
            db_path = self.examples_dir / "optimized_results.db"
            analyzer = EvaluationAnalyzer(str(db_path))
            
            # Sync results to database
            analyzer.sync_database()
            print(f"  ✅ Results synced to database: {db_path}")
            
            # Optimization 2: Parallel evaluation configuration
            print("\n🚀 Parallel Evaluation Configuration:")
            
            optimal_config = {
                "max_parallel_workers": 8,  # Adjust based on system
                "evaluation_timeout_hours": 6.0,
                "batch_size": 4,
                "memory_limit_gb": 16
            }
            
            print("  Recommended settings:")
            for key, value in optimal_config.items():
                print(f"    {key}: {value}")
            
            # Optimization 3: Efficient querying patterns
            print("\n🔍 Efficient Querying Patterns:")
            
            # Use specific filters to reduce data processing
            efficient_filters = [
                QueryFilter(model_pattern="champion_*", suite_pattern="base"),
                QueryFilter(metric_name="success_rate", min_value=0.8),
                QueryFilter(date_from="2024-08-01", date_to="2024-08-31")
            ]
            
            print("  Efficient filter examples:")
            for i, filter_obj in enumerate(efficient_filters):
                print(f"    Filter {i+1}: {filter_obj.__dict__}")
            
            # Optimization 4: Memory management
            print("\n💾 Memory Management:")
            
            memory_tips = [
                "Use database for large result sets",
                "Process results in batches",
                "Clear completed tasks regularly",
                "Limit bootstrap resamples for faster analysis",
                "Use streaming for large exports"
            ]
            
            print("  Memory optimization tips:")
            for tip in memory_tips:
                print(f"    • {tip}")
            
            # Optimization 5: Monitoring efficiency
            print("\n📊 Monitoring Efficiency:")
            
            monitoring_config = MonitoringConfig(
                update_interval_seconds=5.0,  # Less frequent updates
                history_length=50,  # Smaller history buffer
                alert_thresholds={
                    'failure_rate': 0.15,
                    'memory_usage_gb': 12.0,
                    'cpu_usage_percent': 85.0
                }
            )
            
            print("  Optimized monitoring configuration:")
            print(f"    Update interval: {monitoring_config.update_interval_seconds}s")
            print(f"    History length: {monitoring_config.history_length}")
            print(f"    Alert thresholds: {monitoring_config.alert_thresholds}")
            
            print("\n🎉 Performance optimization examples completed!")
            
        except Exception as e:
            print(f"❌ Error in performance optimization: {e}")
    
    def run_example(self, example_name: str):
        """Run a specific example by name."""
        examples = {
            'basic_workflow': self.basic_workflow_example,
            'batch_campaign': self.batch_campaign_example,
            'monitoring': self.monitoring_example,
            'analysis_workflow': self.analysis_workflow_example,
            'integration': self.integration_example,
            'performance': self.performance_optimization_example
        }
        
        if example_name in examples:
            print(f"🚀 Running example: {example_name}")
            print(f"📁 Output directory: {self.examples_dir}")
            print()
            
            examples[example_name]()
            
            print(f"\n📁 Check {self.examples_dir} for generated files")
        else:
            print(f"❌ Unknown example: {example_name}")
            print(f"Available examples: {list(examples.keys())}")
    
    def run_all_examples(self):
        """Run all examples in sequence."""
        examples = [
            'basic_workflow',
            'batch_campaign', 
            'monitoring',
            'analysis_workflow',
            'integration',
            'performance'
        ]
        
        print("🎯 Running all evaluation CLI examples")
        print("=" * 60)
        
        for example in examples:
            try:
                self.run_example(example)
                print("\n" + "─" * 60 + "\n")
                time.sleep(2)  # Brief pause between examples
            except Exception as e:
                print(f"❌ Example {example} failed: {e}")
                continue
        
        print("🎉 All examples completed!")
        print(f"📁 Check {self.examples_dir} for all generated files")


def main():
    """Main entry point for examples."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluation CLI Examples and Tutorials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Examples:
  basic_workflow    - Basic evaluation workflow
  batch_campaign    - Batch evaluation campaign
  monitoring        - Evaluation monitoring
  analysis_workflow - Advanced analysis workflow
  integration       - Integration patterns
  performance       - Performance optimization
  all              - Run all examples

Examples:
  python examples/evaluation_cli_examples.py basic_workflow
  python examples/evaluation_cli_examples.py all
        """
    )
    
    parser.add_argument('example', nargs='?', default='basic_workflow',
                       help='Example to run (default: basic_workflow)')
    
    args = parser.parse_args()
    
    # Create examples instance
    examples = EvaluationCLIExamples()
    
    # Run requested example
    if args.example == 'all':
        examples.run_all_examples()
    else:
        examples.run_example(args.example)


if __name__ == '__main__':
    main()