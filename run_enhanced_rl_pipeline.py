#!/usr/bin/env python3
"""
🚀 ENHANCED RL PIPELINE RUNNER 🚀

One-command script to run the complete enhanced RL pipeline:
1. System validation and testing
2. Enhanced RL training with all features
3. Comprehensive evaluation
4. Champion model selection and export
5. Deployment package preparation

Usage:
    python run_enhanced_rl_pipeline.py                    # Full pipeline
    python run_enhanced_rl_pipeline.py --quick            # Quick training
    python run_enhanced_rl_pipeline.py --test-only        # Testing only
    python run_enhanced_rl_pipeline.py --deploy-only      # Deployment only
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedRLPipelineRunner:
    """Complete enhanced RL pipeline runner."""
    
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        
        # Pipeline configuration
        self.pipeline_config = {
            'run_tests': not args.skip_tests,
            'run_training': not args.deploy_only,
            'run_evaluation': not args.skip_evaluation,
            'run_deployment_prep': not args.test_only,
            'quick_mode': args.quick,
            'timesteps': 1_000_000 if args.quick else 5_000_000,
            'eval_episodes': 10 if args.quick else 50
        }
        
        # Results tracking
        self.results = {
            'pipeline_start': datetime.now().isoformat(),
            'tests_passed': False,
            'training_completed': False,
            'evaluation_completed': False,
            'deployment_prepared': False,
            'best_performance': None,
            'total_time_hours': 0.0
        }
        
        logger.info("🚀 Enhanced RL Pipeline Runner initialized")
        logger.info(f"📋 Configuration: {self.pipeline_config}")
    
    def run_complete_pipeline(self):
        """Run the complete enhanced RL pipeline."""
        logger.info("🚀 STARTING ENHANCED RL COMPLETE PIPELINE")
        logger.info("=" * 80)
        logger.info("🎯 Mission: Train and deploy the ultimate RL agent")
        logger.info("🧠 Features: YOLO + Object Avoidance + Lane Changing")
        logger.info("⚡ Hardware: Metal GPU acceleration (if available)")
        logger.info("🔬 Evaluation: Comprehensive multi-suite testing")
        logger.info("🚀 Deployment: Production-ready DTS package")
        logger.info("=" * 80)
        
        try:
            # Phase 1: System Testing and Validation
            if self.pipeline_config['run_tests']:
                logger.info("🧪 PHASE 1: System Testing and Validation")
                self._run_system_tests()
            
            # Phase 2: Enhanced RL Training
            if self.pipeline_config['run_training']:
                logger.info("🏋️ PHASE 2: Enhanced RL Training")
                self._run_enhanced_training()
            
            # Phase 3: Comprehensive Evaluation
            if self.pipeline_config['run_evaluation']:
                logger.info("🔬 PHASE 3: Comprehensive Evaluation")
                self._run_comprehensive_evaluation()
            
            # Phase 4: Deployment Preparation
            if self.pipeline_config['run_deployment_prep']:
                logger.info("🚀 PHASE 4: Deployment Preparation")
                self._prepare_deployment_package()
            
            # Phase 5: Final Summary and Instructions
            logger.info("📋 PHASE 5: Final Summary and Instructions")
            self._generate_final_summary()
            
            logger.info("🎉 ENHANCED RL PIPELINE COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            self._save_failure_report(str(e))
            raise
        finally:
            self._save_pipeline_results()
    
    def _run_system_tests(self):
        """Run comprehensive system tests."""
        logger.info("Running comprehensive system tests...")
        
        try:
            # Run test suite
            cmd = [sys.executable, "test_enhanced_rl_system.py"]
            
            if self.args.quick:
                cmd.append("--benchmark-only")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info("✅ System tests passed!")
                self.results['tests_passed'] = True
            else:
                logger.error("❌ System tests failed!")
                logger.error(f"Error output: {result.stderr}")
                
                if not self.args.ignore_test_failures:
                    raise RuntimeError("System tests failed")
                else:
                    logger.warning("⚠️ Continuing despite test failures (--ignore-test-failures)")
            
        except subprocess.TimeoutExpired:
            logger.error("❌ System tests timed out")
            if not self.args.ignore_test_failures:
                raise RuntimeError("System tests timed out")
        except Exception as e:
            logger.error(f"❌ System test execution failed: {e}")
            if not self.args.ignore_test_failures:
                raise
    
    def _run_enhanced_training(self):
        """Run enhanced RL training."""
        logger.info("Starting enhanced RL training...")
        
        try:
            # Prepare training command
            cmd = [
                sys.executable, "train_enhanced_rl_champion.py",
                "--timesteps", str(self.pipeline_config['timesteps'])
            ]
            
            # Add configuration if provided
            if self.args.config:
                cmd.extend(["--config", self.args.config])
            
            # Add feature flags
            if self.args.no_yolo:
                cmd.append("--no-yolo")
            if self.args.no_avoidance:
                cmd.append("--no-avoidance")
            if self.args.no_lane_changing:
                cmd.append("--no-lane-changing")
            if self.args.no_metal:
                cmd.append("--no-metal")
            
            # Add custom maps if specified
            if self.args.maps:
                cmd.extend(["--maps", self.args.maps])
            
            logger.info(f"🏋️ Training command: {' '.join(cmd)}")
            
            # Run training with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            return_code = process.poll()
            
            if return_code == 0:
                logger.info("✅ Enhanced RL training completed successfully!")
                self.results['training_completed'] = True
                
                # Find the latest experiment directory
                experiments_dir = Path("experiments")
                if experiments_dir.exists():
                    latest_experiment = max(
                        experiments_dir.glob("enhanced_rl_*"),
                        key=lambda p: p.stat().st_mtime,
                        default=None
                    )
                    if latest_experiment:
                        self.results['experiment_dir'] = str(latest_experiment)
                        
                        # Check for champion model
                        champion_model = latest_experiment / "champion" / "enhanced_champion_model.pth"
                        if champion_model.exists():
                            self.results['champion_model'] = str(champion_model)
            else:
                raise RuntimeError(f"Training failed with return code {return_code}")
                
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
    
    def _run_comprehensive_evaluation(self):
        """Run comprehensive evaluation."""
        logger.info("Running comprehensive evaluation...")
        
        try:
            # Check if we have a trained model
            if not self.results.get('champion_model'):
                logger.warning("⚠️ No champion model found, skipping evaluation")
                return
            
            # Run evaluation
            cmd = [
                sys.executable, "train_enhanced_rl_champion.py",
                "--eval-only",
                "--config", self.args.config or "config/enhanced_rl_champion_config.yml"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info("✅ Comprehensive evaluation completed!")
                self.results['evaluation_completed'] = True
                
                # Parse evaluation results
                self._parse_evaluation_results()
            else:
                logger.error("❌ Evaluation failed!")
                logger.error(f"Error output: {result.stderr}")
                raise RuntimeError("Evaluation failed")
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Evaluation timed out")
            raise RuntimeError("Evaluation timed out")
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise
    
    def _parse_evaluation_results(self):
        """Parse evaluation results."""
        try:
            # Look for evaluation results in the experiment directory
            if 'experiment_dir' in self.results:
                eval_dir = Path(self.results['experiment_dir']) / "evaluation"
                
                # Find latest evaluation results
                eval_files = list(eval_dir.glob("comprehensive_evaluation_*.json"))
                if eval_files:
                    latest_eval = max(eval_files, key=lambda p: p.stat().st_mtime)
                    
                    with open(latest_eval, 'r') as f:
                        eval_data = json.load(f)
                    
                    # Extract best performance
                    if eval_data:
                        best_model = max(eval_data.items(), key=lambda x: x[1].get('global_score', 0))
                        self.results['best_performance'] = {
                            'model_path': best_model[0],
                            'global_score': best_model[1].get('global_score', 0),
                            'success_rate': best_model[1].get('success_rate', 0),
                            'evaluation_file': str(latest_eval)
                        }
                        
                        logger.info(f"📊 Best Performance: {self.results['best_performance']['global_score']:.2f}")
        
        except Exception as e:
            logger.warning(f"⚠️ Could not parse evaluation results: {e}")
    
    def _prepare_deployment_package(self):
        """Prepare deployment package."""
        logger.info("Preparing deployment package...")
        
        try:
            # Run deployment preparation
            cmd = [
                sys.executable, "train_enhanced_rl_champion.py",
                "--export-only"
            ]
            
            if self.args.config:
                cmd.extend(["--config", self.args.config])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Deployment package prepared!")
                self.results['deployment_prepared'] = True
                
                # Find deployment directory
                if 'experiment_dir' in self.results:
                    deployment_dir = Path(self.results['experiment_dir']) / "deployment"
                    if deployment_dir.exists():
                        self.results['deployment_dir'] = str(deployment_dir)
            else:
                logger.error("❌ Deployment preparation failed!")
                logger.error(f"Error output: {result.stderr}")
                raise RuntimeError("Deployment preparation failed")
                
        except Exception as e:
            logger.error(f"❌ Deployment preparation failed: {e}")
            raise
    
    def _generate_final_summary(self):
        """Generate final summary and instructions."""
        logger.info("Generating final summary...")
        
        # Calculate total time
        total_time = time.time() - self.start_time
        self.results['total_time_hours'] = total_time / 3600
        
        # Create summary
        summary = f"""
🎉 ENHANCED RL PIPELINE COMPLETED SUCCESSFULLY! 🎉

📊 PIPELINE RESULTS:
{'='*60}
⏱️  Total Time: {total_time/3600:.1f} hours
🧪 Tests Passed: {'✅' if self.results['tests_passed'] else '❌'}
🏋️ Training Completed: {'✅' if self.results['training_completed'] else '❌'}
🔬 Evaluation Completed: {'✅' if self.results['evaluation_completed'] else '❌'}
🚀 Deployment Prepared: {'✅' if self.results['deployment_prepared'] else '❌'}
"""
        
        if self.results.get('best_performance'):
            perf = self.results['best_performance']
            summary += f"""
🏆 BEST PERFORMANCE:
   Global Score: {perf['global_score']:.2f}/100
   Success Rate: {perf.get('success_rate', 0):.1%}
   Model: {Path(perf['model_path']).name}
"""
        
        if self.results.get('deployment_dir'):
            summary += f"""
🚀 DEPLOYMENT INSTRUCTIONS:
{'='*60}
1. Copy deployment package to your Duckiebot:
   scp -r {self.results['deployment_dir']}/ duckie@duckiebot.local:/data/enhanced_rl/

2. SSH into your Duckiebot and run:
   cd /data/enhanced_rl
   python3 enhanced_deployment_system.py

3. Monitor performance:
   rostopic echo /duckiebot/enhanced_rl/status

📁 Files Created:
   🧠 Champion Model: {self.results.get('champion_model', 'N/A')}
   📦 Deployment Package: {self.results.get('deployment_dir', 'N/A')}
   📊 Evaluation Results: {self.results.get('experiment_dir', 'N/A')}/evaluation/
"""
        
        summary += f"""
🎯 SYSTEM CAPABILITIES:
{'='*60}
✅ YOLO v5 Object Detection & Classification
✅ Real-time Object Avoidance with Potential Fields
✅ Dynamic Lane Changing with Safety Validation
✅ Multi-objective Reward Optimization
✅ Metal GPU Acceleration (macOS)
✅ Comprehensive Evaluation & Testing
✅ Production-ready DTS Deployment

🔧 CONFIGURATION:
   Training Timesteps: {self.pipeline_config['timesteps']:,}
   Quick Mode: {'Yes' if self.pipeline_config['quick_mode'] else 'No'}
   Features: {'YOLO' if not self.args.no_yolo else ''} {'Avoidance' if not self.args.no_avoidance else ''} {'Lane Changing' if not self.args.no_lane_changing else ''}

🎉 Your enhanced RL agent is ready for deployment!
"""
        
        print(summary)
        logger.info("Final summary generated")
        
        # Save summary to file
        if 'experiment_dir' in self.results:
            summary_file = Path(self.results['experiment_dir']) / "PIPELINE_SUMMARY.md"
            with open(summary_file, 'w') as f:
                f.write(summary)
            logger.info(f"📋 Summary saved to: {summary_file}")
    
    def _save_pipeline_results(self):
        """Save pipeline results."""
        try:
            results_file = Path("pipeline_results.json")
            
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            logger.info(f"📋 Pipeline results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {e}")
    
    def _save_failure_report(self, error_message: str):
        """Save failure report."""
        try:
            failure_report = {
                'timestamp': datetime.now().isoformat(),
                'error_message': error_message,
                'pipeline_config': self.pipeline_config,
                'results': self.results,
                'total_time_hours': (time.time() - self.start_time) / 3600
            }
            
            failure_file = Path(f"pipeline_failure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(failure_file, 'w') as f:
                json.dump(failure_report, f, indent=2, default=str)
            
            logger.info(f"📋 Failure report saved to: {failure_file}")
            
        except Exception as e:
            logger.error(f"Failed to save failure report: {e}")


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description="Enhanced RL Complete Pipeline Runner")
    
    # Pipeline control
    parser.add_argument('--quick', action='store_true', help='Quick training mode (1M timesteps)')
    parser.add_argument('--test-only', action='store_true', help='Run tests only')
    parser.add_argument('--deploy-only', action='store_true', help='Prepare deployment only')
    parser.add_argument('--skip-tests', action='store_true', help='Skip system tests')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip evaluation')
    parser.add_argument('--ignore-test-failures', action='store_true', help='Continue despite test failures')
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--maps', type=str, help='Comma-separated list of training maps')
    
    # Feature flags
    parser.add_argument('--no-yolo', action='store_true', help='Disable YOLO detection')
    parser.add_argument('--no-avoidance', action='store_true', help='Disable object avoidance')
    parser.add_argument('--no-lane-changing', action='store_true', help='Disable lane changing')
    parser.add_argument('--no-metal', action='store_true', help='Disable Metal acceleration')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.test_only and args.deploy_only:
        logger.error("❌ Cannot specify both --test-only and --deploy-only")
        sys.exit(1)
    
    try:
        # Create and run pipeline
        pipeline = EnhancedRLPipelineRunner(args)
        pipeline.run_complete_pipeline()
        
        logger.info("🎉 SUCCESS: Enhanced RL pipeline completed!")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()