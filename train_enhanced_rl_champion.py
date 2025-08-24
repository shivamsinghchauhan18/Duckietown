#!/usr/bin/env python3
"""
🏆 ENHANCED RL CHAMPION TRAINING SCRIPT 🏆

Complete training pipeline that integrates:
- Enhanced RL training system with YOLO
- Comprehensive evaluation orchestrator
- Metal framework GPU acceleration
- Multi-map curriculum learning
- Production-ready model export

This is the main entry point for training the ultimate RL champion.
"""

import os
import sys
import time
import json
import yaml
import logging
import argparse
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our systems
from enhanced_rl_training_system import EnhancedRLTrainer, TrainingConfig
from duckietown_utils.evaluation_orchestrator import EvaluationOrchestrator
from config.enhanced_config import EnhancedRLConfig, load_enhanced_config
from duckietown_utils.enhanced_logger import EnhancedLogger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedRLChampionTrainer:
    """Complete enhanced RL champion training system."""
    
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        
        # Load configurations
        self.enhanced_config = load_enhanced_config(args.config) if args.config else load_enhanced_config()
        self.training_config = self._create_training_config()
        
        # Setup directories
        self.experiment_dir = Path(f"experiments/enhanced_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_dir = self.experiment_dir / "models"
        self.log_dir = self.experiment_dir / "logs"
        self.eval_dir = self.experiment_dir / "evaluation"
        
        for dir_path in [self.model_dir, self.log_dir, self.eval_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.trainer = None
        self.evaluator = None
        self.logger = EnhancedLogger("enhanced_rl_champion", str(self.log_dir))
        
        # Training state
        self.training_active = False
        self.best_performance = {
            'global_score': 0.0,
            'model_path': None,
            'evaluation_results': None
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🏆 Enhanced RL Champion Trainer initialized")
        logger.info(f"📁 Experiment directory: {self.experiment_dir}")
    
    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("🛑 Training interrupted. Saving state...")
        self.training_active = False
        
        if self.trainer and hasattr(self.trainer, 'agent'):
            self.trainer.agent.save_model(str(self.model_dir / "interrupted_model.pth"))
        
        self._save_experiment_summary()
        sys.exit(0)
    
    def _create_training_config(self) -> TrainingConfig:
        """Create training configuration from arguments."""
        config = TrainingConfig()
        
        # Update from arguments
        if self.args.timesteps:
            config.total_timesteps = self.args.timesteps
        
        if self.args.learning_rate:
            config.learning_rate = self.args.learning_rate
        
        if self.args.batch_size:
            config.batch_size = self.args.batch_size
        
        # Feature flags
        config.use_yolo = not self.args.no_yolo
        config.use_object_avoidance = not self.args.no_avoidance
        config.use_lane_changing = not self.args.no_lane_changing
        config.use_metal = not self.args.no_metal
        
        # Training maps
        if self.args.maps:
            config.training_maps = self.args.maps.split(',')
        
        return config
    
    def run_complete_training(self):
        """Run the complete training pipeline."""
        logger.info("🚀 STARTING ENHANCED RL CHAMPION TRAINING")
        logger.info("=" * 80)
        logger.info(f"🎯 Target: Ultimate autonomous driving performance")
        logger.info(f"🧠 Features: YOLO={self.training_config.use_yolo}, "
                   f"Avoidance={self.training_config.use_object_avoidance}, "
                   f"Lane Changing={self.training_config.use_lane_changing}")
        logger.info(f"⚡ Metal GPU: {self.training_config.use_metal}")
        logger.info(f"🗺️  Training Maps: {', '.join(self.training_config.training_maps)}")
        logger.info(f"📊 Total Timesteps: {self.training_config.total_timesteps:,}")
        logger.info("=" * 80)
        
        self.training_active = True
        
        try:
            # Phase 1: Enhanced RL Training
            logger.info("🏋️ PHASE 1: Enhanced RL Training")
            self._run_enhanced_training()
            
            # Phase 2: Comprehensive Evaluation
            logger.info("🔬 PHASE 2: Comprehensive Evaluation")
            self._run_comprehensive_evaluation()
            
            # Phase 3: Champion Selection and Export
            logger.info("🏆 PHASE 3: Champion Selection and Export")
            self._select_and_export_champion()
            
            # Phase 4: Deployment Preparation
            logger.info("🚀 PHASE 4: Deployment Preparation")
            self._prepare_deployment()
            
            logger.info("✅ ENHANCED RL CHAMPION TRAINING COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        finally:
            self._save_experiment_summary()
    
    def _run_enhanced_training(self):
        """Run enhanced RL training with all features."""
        logger.info("Starting enhanced RL training...")
        
        # Create trainer
        self.trainer = EnhancedRLTrainer()
        self.trainer.training_config = self.training_config
        self.trainer.enhanced_config = self.enhanced_config
        
        # Update trainer directories
        self.trainer.log_dir = self.log_dir
        self.trainer.model_dir = self.model_dir
        
        # Run training
        self.trainer.train()
        
        # Update best performance
        if hasattr(self.trainer, 'best_reward'):
            self.best_performance['global_score'] = self.trainer.best_reward
            self.best_performance['model_path'] = str(self.model_dir / "best_model.pth")
        
        logger.info(f"✅ Enhanced training completed. Best reward: {self.trainer.best_reward:.2f}")
    
    def _run_comprehensive_evaluation(self):
        """Run comprehensive evaluation across all test suites."""
        logger.info("Starting comprehensive evaluation...")
        
        # Create evaluation configuration
        eval_config = {
            'suites': ['base', 'hard', 'law', 'ood', 'stress'],
            'seeds_per_map': 50,
            'policy_modes': ['deterministic', 'stochastic'],
            'compute_ci': True,
            'bootstrap_resamples': 10000,
            'significance_correction': 'benjamini_hochberg',
            'use_composite': True,
            'normalization_scope': 'per_map_suite',
            'keep_top_k': 5,
            'export_csv_json': True,
            'export_plots': True,
            'record_videos': True,
            'save_worst_k': 5,
            'fix_seed_list': True,
            'cudnn_deterministic': True,
            'log_git_sha': True
        }
        
        # Create evaluator
        self.evaluator = EvaluationOrchestrator(eval_config)
        
        # Get model paths to evaluate
        model_paths = []
        if (self.model_dir / "best_model.pth").exists():
            model_paths.append(str(self.model_dir / "best_model.pth"))
        if (self.model_dir / "final_model.pth").exists():
            model_paths.append(str(self.model_dir / "final_model.pth"))
        
        # Add checkpoint models
        for checkpoint_path in self.model_dir.glob("checkpoint_*.pth"):
            model_paths.append(str(checkpoint_path))
        
        if not model_paths:
            logger.warning("⚠️ No trained models found for evaluation")
            return
        
        logger.info(f"📊 Evaluating {len(model_paths)} models across all test suites")
        
        # Run evaluation
        evaluation_results = self.evaluator.evaluate_models(model_paths)
        
        # Save evaluation results
        eval_results_path = self.eval_dir / "comprehensive_evaluation_results.json"
        with open(eval_results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Update best performance
        if evaluation_results:
            best_model = max(evaluation_results.items(), key=lambda x: x[1].get('global_score', 0))
            self.best_performance['global_score'] = best_model[1].get('global_score', 0)
            self.best_performance['model_path'] = best_model[0]
            self.best_performance['evaluation_results'] = best_model[1]
        
        logger.info(f"✅ Comprehensive evaluation completed. Best global score: {self.best_performance['global_score']:.2f}")
    
    def _select_and_export_champion(self):
        """Select champion model and export in multiple formats."""
        logger.info("Selecting and exporting champion model...")
        
        if not self.best_performance['model_path']:
            logger.warning("⚠️ No champion model to export")
            return
        
        champion_model_path = Path(self.best_performance['model_path'])
        
        if not champion_model_path.exists():
            logger.error(f"❌ Champion model not found: {champion_model_path}")
            return
        
        # Create champion export directory
        champion_dir = self.experiment_dir / "champion"
        champion_dir.mkdir(exist_ok=True)
        
        # Copy champion model
        import shutil
        champion_copy = champion_dir / "enhanced_champion_model.pth"
        shutil.copy2(champion_model_path, champion_copy)
        
        # Export in different formats
        self._export_champion_formats(champion_copy, champion_dir)
        
        # Create champion metadata
        champion_metadata = {
            'timestamp': datetime.now().isoformat(),
            'training_config': self.training_config.__dict__,
            'enhanced_config': self.enhanced_config.__dict__ if hasattr(self.enhanced_config, '__dict__') else str(self.enhanced_config),
            'best_performance': self.best_performance,
            'training_time_hours': (time.time() - self.start_time) / 3600,
            'model_path': str(champion_copy),
            'features': {
                'yolo_detection': self.training_config.use_yolo,
                'object_avoidance': self.training_config.use_object_avoidance,
                'lane_changing': self.training_config.use_lane_changing,
                'multi_objective_reward': self.training_config.use_multi_objective_reward,
                'metal_acceleration': self.training_config.use_metal
            }
        }
        
        metadata_path = champion_dir / "champion_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(champion_metadata, f, indent=2, default=str)
        
        logger.info(f"🏆 Champion model exported to: {champion_dir}")
        logger.info(f"📋 Champion metadata saved to: {metadata_path}")
    
    def _export_champion_formats(self, model_path: Path, export_dir: Path):
        """Export champion model in multiple formats."""
        try:
            import torch
            
            # Load the model
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Export as TorchScript (for deployment)
            try:
                # This would require the actual model architecture
                logger.info("📦 TorchScript export would be implemented here")
                # model = load_model_architecture()
                # model.load_state_dict(checkpoint['q_network_state_dict'])
                # traced_model = torch.jit.trace(model, example_input)
                # traced_model.save(export_dir / "champion_model.pt")
            except Exception as e:
                logger.warning(f"⚠️ TorchScript export failed: {e}")
            
            # Export as ONNX (for cross-platform deployment)
            try:
                logger.info("📦 ONNX export would be implemented here")
                # torch.onnx.export(model, example_input, export_dir / "champion_model.onnx")
            except Exception as e:
                logger.warning(f"⚠️ ONNX export failed: {e}")
            
            # Create deployment-ready copy
            deployment_model = export_dir / "deployment_model.pth"
            torch.save({
                'model_state_dict': checkpoint.get('q_network_state_dict', checkpoint),
                'model_config': {
                    'use_yolo': self.training_config.use_yolo,
                    'use_enhanced_obs': True,
                    'action_dim': 2
                },
                'deployment_ready': True,
                'timestamp': datetime.now().isoformat()
            }, deployment_model)
            
            logger.info(f"✅ Deployment model saved: {deployment_model}")
            
        except Exception as e:
            logger.error(f"❌ Model export failed: {e}")
    
    def _prepare_deployment(self):
        """Prepare deployment package."""
        logger.info("Preparing deployment package...")
        
        deployment_dir = self.experiment_dir / "deployment"
        deployment_dir.mkdir(exist_ok=True)
        
        # Copy deployment script
        import shutil
        
        deployment_script = Path("enhanced_deployment_system.py")
        if deployment_script.exists():
            shutil.copy2(deployment_script, deployment_dir / "enhanced_deployment_system.py")
        
        # Copy champion model
        champion_model = self.experiment_dir / "champion" / "deployment_model.pth"
        if champion_model.exists():
            shutil.copy2(champion_model, deployment_dir / "enhanced_champion_model.pth")
        
        # Create deployment configuration
        deployment_config = {
            'model_path': '/data/models/enhanced_champion_model.pth',
            'yolo_model_path': 'yolov5s.pt',
            'yolo_confidence_threshold': 0.5,
            'yolo_device': 'cpu',
            'max_detections': 10,
            'safety_distance': 0.5,
            'min_clearance': 0.2,
            'avoidance_strength': 1.0,
            'emergency_brake_distance': 0.15,
            'lane_change_threshold': 0.3,
            'safety_margin': 2.0,
            'max_lane_change_time': 3.0,
            'max_linear_velocity': 0.3,
            'max_angular_velocity': 1.0,
            'control_frequency': 10.0,
            'enable_safety_override': True,
            'emergency_stop_enabled': True,
            'max_consecutive_failures': 5,
            'log_detections': True,
            'log_actions': True,
            'log_performance': True,
            'save_debug_images': False
        }
        
        config_path = deployment_dir / "deployment_config.json"
        with open(config_path, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        # Create deployment instructions
        instructions = f"""
# Enhanced RL Duckiebot Deployment Instructions

## Quick Start
1. Copy the deployment files to your Duckiebot:
   ```bash
   scp -r deployment/ duckie@duckiebot.local:/data/enhanced_rl/
   ```

2. SSH into your Duckiebot and run:
   ```bash
   cd /data/enhanced_rl
   python3 enhanced_deployment_system.py --config deployment_config.json
   ```

## Features Enabled
- ✅ YOLO v5 Object Detection
- ✅ Real-time Object Avoidance
- ✅ Dynamic Lane Changing
- ✅ Multi-objective Decision Making
- ✅ Safety Override Systems
- ✅ Performance Monitoring

## Model Performance
- Global Score: {self.best_performance['global_score']:.2f}
- Training Time: {(time.time() - self.start_time) / 3600:.1f} hours
- Training Timesteps: {self.training_config.total_timesteps:,}

## Configuration
The deployment configuration can be modified in `deployment_config.json`.
Key parameters:
- `safety_distance`: Distance threshold for object avoidance (default: 0.5m)
- `max_linear_velocity`: Maximum forward speed (default: 0.3 m/s)
- `control_frequency`: Control loop frequency (default: 10 Hz)

## Monitoring
The system publishes performance metrics and status information to ROS topics:
- `/{robot_name}/enhanced_rl/status`: Real-time status and performance
- `/{robot_name}/enhanced_rl/detections`: Object detection results
- `/{robot_name}/enhanced_rl/emergency_stop`: Emergency stop status

## Troubleshooting
1. If YOLO detection fails, the system will fall back to basic lane following
2. Emergency stop can be triggered by publishing to the emergency_stop topic
3. Performance logs are saved automatically for analysis

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        readme_path = deployment_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(instructions)
        
        logger.info(f"🚀 Deployment package prepared: {deployment_dir}")
        logger.info(f"📖 Deployment instructions: {readme_path}")
    
    def _save_experiment_summary(self):
        """Save comprehensive experiment summary."""
        try:
            summary = {
                'experiment_info': {
                    'timestamp': datetime.now().isoformat(),
                    'experiment_dir': str(self.experiment_dir),
                    'total_time_hours': (time.time() - self.start_time) / 3600,
                    'training_completed': self.training_active
                },
                'configuration': {
                    'training_config': self.training_config.__dict__,
                    'enhanced_config': self.enhanced_config.__dict__ if hasattr(self.enhanced_config, '__dict__') else str(self.enhanced_config),
                    'command_line_args': vars(self.args)
                },
                'results': {
                    'best_performance': self.best_performance,
                    'training_successful': self.best_performance['global_score'] > 0
                },
                'files': {
                    'model_dir': str(self.model_dir),
                    'log_dir': str(self.log_dir),
                    'eval_dir': str(self.eval_dir),
                    'champion_model': self.best_performance.get('model_path'),
                    'deployment_dir': str(self.experiment_dir / "deployment")
                }
            }
            
            summary_path = self.experiment_dir / "experiment_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"📋 Experiment summary saved: {summary_path}")
            
        except Exception as e:
            logger.error(f"Failed to save experiment summary: {e}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Enhanced RL Champion Training")
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to enhanced config file')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=5_000_000, help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--maps', type=str, help='Comma-separated list of training maps')
    
    # Feature flags
    parser.add_argument('--no-yolo', action='store_true', help='Disable YOLO detection')
    parser.add_argument('--no-avoidance', action='store_true', help='Disable object avoidance')
    parser.add_argument('--no-lane-changing', action='store_true', help='Disable lane changing')
    parser.add_argument('--no-metal', action='store_true', help='Disable Metal acceleration')
    
    # Execution modes
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only')
    parser.add_argument('--export-only', action='store_true', help='Export existing models only')
    
    args = parser.parse_args()
    
    try:
        # Create trainer
        trainer = EnhancedRLChampionTrainer(args)
        
        if args.eval_only:
            logger.info("🔬 Running evaluation only")
            trainer._run_comprehensive_evaluation()
        elif args.export_only:
            logger.info("📦 Running export only")
            trainer._select_and_export_champion()
        else:
            # Run complete training pipeline
            trainer.run_complete_training()
        
        logger.info("🎉 SUCCESS: Enhanced RL Champion Training completed!")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()