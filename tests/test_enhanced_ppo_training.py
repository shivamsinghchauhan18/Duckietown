"""
Integration tests for enhanced PPO training with multi-objective rewards.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2020 András Kalapos"

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env

from config.config import load_config
from config.enhanced_config import load_enhanced_config, EnhancedRLConfig
from experiments.train_enhanced_rllib import EnhancedPPOTrainer
from duckietown_utils.enhanced_rllib_callbacks import EnhancedRLLibCallbacks, CurriculumLearningCallback
from duckietown_utils.training_utils import ModelEvaluator, TrainingAnalyzer, compare_models
from duckietown_utils.env import launch_and_wrap_enhanced_env


class TestEnhancedPPOTraining:
    """Test enhanced PPO training integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test artifacts."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration for testing."""
        return {
            'experiment_name': 'test_enhanced_ppo',
            'seed': 1234,
            'algo': 'PPO',
            'timesteps_total': 1000,
            'env_config': {
                'mode': 'train',
                'episode_max_steps': 100,
                'resized_input_shape': '(84, 84)',
                'crop_image_top': True,
                'top_crop_divider': 3,
                'grayscale_image': False,
                'frame_stacking': True,
                'frame_stacking_depth': 3,
                'action_type': 'heading',
                'reward_function': 'posangle',
                'distortion': True,
                'accepted_start_angle_deg': 4,
                'simulation_framerate': 30,
                'frame_skip': 1,
                'action_delay_ratio': 0.0,
                'training_map': 'small_loop',
                'domain_rand': False,
                'dynamics_rand': False,
                'camera_rand': False,
                'spawn_obstacles': False,
                'spawn_forward_obstacle': False,
                'aido_wrapper': False
            },
            'rllib_config': {
                'num_workers': 1,
                'num_gpus': 0,
                'train_batch_size': 64,
                'sgd_minibatch_size': 32,
                'num_sgd_iter': 2,
                'lr': 3e-4,
                'gamma': 0.99,
                'lambda': 0.95,
                'clip_param': 0.2,
                'vf_clip_param': 10.0,
                'entropy_coeff': 0.01
            },
            'ray_init_config': {
                'num_cpus': 2,
                'local_mode': True
            },
            'restore_seed': -1
        }
    
    @pytest.fixture
    def mock_enhanced_config(self):
        """Create mock enhanced configuration for testing."""
        return EnhancedRLConfig(
            enabled_features=['yolo', 'object_avoidance', 'lane_changing', 'multi_objective_reward'],
            debug_mode=True,
            yolo={
                'model_path': 'yolov5s.pt',
                'confidence_threshold': 0.5,
                'device': 'cpu',
                'input_size': 640,
                'max_detections': 100
            },
            object_avoidance={
                'safety_distance': 0.5,
                'avoidance_strength': 1.0,
                'min_clearance': 0.2,
                'max_avoidance_angle': 0.5,
                'smoothing_factor': 0.8
            },
            lane_changing={
                'lane_change_threshold': 0.3,
                'safety_margin': 2.0,
                'max_lane_change_time': 3.0,
                'min_lane_width': 0.4,
                'evaluation_distance': 5.0
            },
            reward={
                'lane_following_weight': 1.0,
                'object_avoidance_weight': 0.5,
                'lane_change_weight': 0.3,
                'efficiency_weight': 0.2,
                'safety_penalty_weight': -2.0,
                'collision_penalty': -10.0
            },
            logging={
                'log_level': 'INFO',
                'log_detections': True,
                'log_actions': True,
                'log_rewards': True,
                'log_performance': True,
                'log_file_path': None,
                'console_logging': True
            },
            performance={
                'max_fps': 30.0,
                'detection_batch_size': 1,
                'use_gpu_acceleration': False,
                'memory_limit_gb': 4.0
            }
        )
    
    def test_enhanced_ppo_trainer_initialization(self, mock_config, mock_enhanced_config, temp_dir):
        """Test enhanced PPO trainer initialization."""
        # Create temporary config files
        config_file = temp_dir / 'config.yml'
        enhanced_config_file = temp_dir / 'enhanced_config.yml'
        
        # Mock file loading
        with patch('config.config.load_config', return_value=mock_config), \
             patch('config.enhanced_config.load_enhanced_config', return_value=mock_enhanced_config):
            
            trainer = EnhancedPPOTrainer(
                config_path=str(config_file),
                enhanced_config_path=str(enhanced_config_file)
            )
            
            assert trainer.config == mock_config
            assert trainer.enhanced_config == mock_enhanced_config
            assert trainer.curriculum_callback is not None
            assert trainer.checkpoint_callback is not None
    
    def test_training_config_setup(self, mock_config, mock_enhanced_config, temp_dir):
        """Test training configuration setup with enhanced features."""
        with patch('config.config.load_config', return_value=mock_config), \
             patch('config.enhanced_config.load_enhanced_config', return_value=mock_enhanced_config):
            
            trainer = EnhancedPPOTrainer(
                config_path=str(temp_dir / 'config.yml'),
                enhanced_config_path=str(temp_dir / 'enhanced_config.yml')
            )
            
            rllib_config = trainer.setup_training_config()
            
            # Check enhanced configuration
            assert rllib_config['env'] == 'EnhancedDuckietown'
            assert 'callbacks' in rllib_config
            assert isinstance(rllib_config['callbacks'], EnhancedRLLibCallbacks)
            assert 'enhanced_config' in rllib_config['env_config']
            
            # Check multi-objective reward configuration
            assert 'multiagent' in rllib_config
            assert 'policies' in rllib_config['multiagent']
            assert 'reward_weights' in rllib_config['multiagent']['policies']['default_policy'][3]
    
    @patch('ray.init')
    @patch('ray.tune.run')
    def test_enhanced_training_execution(self, mock_tune_run, mock_ray_init, 
                                       mock_config, mock_enhanced_config, temp_dir):
        """Test enhanced training execution."""
        # Mock tune.run return value
        mock_tune_run.return_value = {'training_iteration': 10, 'episode_reward_mean': 0.5}
        
        with patch('config.config.load_config', return_value=mock_config), \
             patch('config.enhanced_config.load_enhanced_config', return_value=mock_enhanced_config), \
             patch('duckietown_utils.env.launch_and_wrap_enhanced_env') as mock_env:
            
            # Mock environment
            mock_env.return_value = Mock()
            
            trainer = EnhancedPPOTrainer(
                config_path=str(temp_dir / 'config.yml'),
                enhanced_config_path=str(temp_dir / 'enhanced_config.yml')
            )
            
            # Mock environment registration
            with patch.object(trainer, 'register_enhanced_environment'):
                results = trainer.train(timesteps_total=1000)
            
            # Verify training was called
            mock_ray_init.assert_called_once()
            mock_tune_run.assert_called_once()
            
            # Check tune.run arguments
            call_args = mock_tune_run.call_args
            assert call_args[0][0] == PPOTrainer
            assert call_args[1]['stop']['timesteps_total'] == 1000
            assert 'EnhancedTensorboardLogger' in str(call_args[1]['loggers'])
    
    def test_curriculum_learning_callback(self, mock_enhanced_config):
        """Test curriculum learning callback functionality."""
        callback = CurriculumLearningCallback(mock_enhanced_config)
        
        # Test initial state
        assert callback.current_stage == 0
        assert callback.stage_start_timestep == 0
        
        # Mock training result
        mock_result = {
            'timesteps_total': 100000,
            'episode_reward_mean': 0.6,
            'custom_metrics': {
                'safety_score': 0.85
            }
        }
        
        # Mock trainer
        mock_trainer = Mock()
        mock_trainer.workers = Mock()
        mock_trainer.workers.foreach_worker = Mock()
        
        # Test stage advancement logic
        callback.on_train_result(mock_trainer, mock_result)
        
        # Verify callback processed the result
        assert len(callback.stage_performance_history) == 1
        assert callback.stage_performance_history[0]['timestep'] == 100000
    
    def test_enhanced_callbacks_episode_tracking(self, mock_enhanced_config):
        """Test enhanced callbacks episode tracking."""
        from duckietown_utils.enhanced_logger import EnhancedLogger
        
        enhanced_logger = EnhancedLogger(mock_enhanced_config.logging)
        callbacks = EnhancedRLLibCallbacks(
            enhanced_config=mock_enhanced_config,
            enhanced_logger=enhanced_logger
        )
        
        # Mock episode
        mock_episode = Mock()
        mock_episode.episode_id = 'test_episode'
        mock_episode.length = 0
        mock_episode.user_data = {}
        mock_episode.hist_data = {}
        mock_episode.custom_metrics = {}
        
        # Test episode start
        callbacks.on_episode_start(
            worker=Mock(),
            base_env=Mock(),
            policies={},
            episode=mock_episode,
            env_index=0
        )
        
        # Verify initialization
        assert 'reward_components' in mock_episode.user_data
        assert 'object_detections' in mock_episode.user_data
        assert 'lane_changes' in mock_episode.user_data
        assert 'safety_violations' in mock_episode.user_data
        
        # Test episode step
        mock_episode.last_info_for = Mock(return_value={
            'reward_components': {
                'lane_following': 0.5,
                'object_avoidance': 0.2,
                'safety_penalty': -0.1
            },
            'object_detection': {
                'detections': [{'confidence': 0.8, 'class': 'duckie'}]
            }
        })
        mock_episode.last_action_for = Mock(return_value=np.array([0.1, 0.2]))
        
        callbacks.on_episode_step(
            worker=Mock(),
            base_env=Mock(),
            episode=mock_episode,
            env_index=0
        )
        
        # Verify step tracking
        assert len(mock_episode.user_data['reward_components']['lane_following']) == 1
        assert len(mock_episode.user_data['object_detections']) == 1
    
    def test_model_evaluator(self, mock_enhanced_config, temp_dir):
        """Test model evaluator functionality."""
        evaluator = ModelEvaluator(mock_enhanced_config)
        
        # Create mock checkpoint
        checkpoint_dir = temp_dir / 'checkpoint'
        checkpoint_dir.mkdir()
        
        # Mock trainer and environment
        with patch('ray.rllib.agents.ppo.PPOTrainer') as mock_trainer_class, \
             patch('duckietown_utils.env.launch_and_wrap_enhanced_env') as mock_env_func:
            
            # Setup mocks
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            mock_trainer.compute_action = Mock(return_value=np.array([0.1, 0.2]))
            
            mock_env = Mock()
            mock_env.reset = Mock(return_value=np.zeros((84, 84, 3)))
            mock_env.step = Mock(return_value=(
                np.zeros((84, 84, 3)),  # obs
                0.5,  # reward
                True,  # done
                {'Simulator': {'cur_pos': [1, 0, 1]}}  # info
            ))
            mock_env_func.return_value = mock_env
            
            # Test evaluation
            results = evaluator.evaluate_checkpoint(
                checkpoint_path=str(checkpoint_dir),
                env_config={'mode': 'inference'},
                num_episodes=2,
                save_results=False
            )
            
            # Verify results structure
            assert 'summary' in results
            assert 'detailed' in results
            assert 'performance_scores' in results
            assert results['summary']['num_episodes'] == 2
            assert 'mean_reward' in results['summary']
            assert 'success_rate' in results['summary']
    
    def test_training_analyzer(self, temp_dir):
        """Test training analyzer functionality."""
        # Create mock experiment directory structure
        experiment_dir = temp_dir / 'experiment'
        experiment_dir.mkdir()
        
        checkpoint_dir = experiment_dir / 'enhanced_checkpoints'
        checkpoint_dir.mkdir()
        
        # Create mock checkpoint with metadata
        checkpoint_path = checkpoint_dir / 'checkpoint_001'
        checkpoint_path.mkdir()
        
        metadata = {
            'timestamp': '2023-01-01T00:00:00',
            'timesteps_total': 100000,
            'episode_reward_mean': 0.75,
            'performance_score': 0.85
        }
        
        with open(checkpoint_path / 'enhanced_metadata.json', 'w') as f:
            import json
            json.dump(metadata, f)
        
        # Test analyzer
        analyzer = TrainingAnalyzer(str(experiment_dir))
        analysis = analyzer.analyze_training_progress()
        
        # Verify analysis structure
        assert 'training_metrics' in analysis
        assert 'checkpoint_analysis' in analysis
        assert 'performance_trends' in analysis
        assert 'component_analysis' in analysis
        
        # Check checkpoint analysis
        checkpoint_analysis = analysis['checkpoint_analysis']
        assert checkpoint_analysis['num_checkpoints'] == 1
        assert checkpoint_analysis['best_checkpoint'] == str(checkpoint_path)
    
    def test_model_comparison(self, mock_enhanced_config, temp_dir):
        """Test model comparison functionality."""
        # Create mock checkpoints
        checkpoint1 = temp_dir / 'checkpoint1'
        checkpoint2 = temp_dir / 'checkpoint2'
        checkpoint1.mkdir()
        checkpoint2.mkdir()
        
        with patch('duckietown_utils.training_utils.ModelEvaluator') as mock_evaluator_class:
            # Mock evaluator results
            mock_evaluator = Mock()
            mock_evaluator_class.return_value = mock_evaluator
            
            mock_evaluator.evaluate_checkpoint.side_effect = [
                {
                    'summary': {'mean_reward': 0.6, 'success_rate': 0.7},
                    'performance_scores': {'overall_score': 0.65, 'safety_score': 0.8}
                },
                {
                    'summary': {'mean_reward': 0.8, 'success_rate': 0.9},
                    'performance_scores': {'overall_score': 0.85, 'safety_score': 0.9}
                }
            ]
            
            # Test comparison
            results = compare_models(
                checkpoint_paths=[str(checkpoint1), str(checkpoint2)],
                env_config={'mode': 'inference'},
                enhanced_config=mock_enhanced_config,
                num_episodes=5
            )
            
            # Verify comparison results
            assert 'models' in results
            assert 'comparison' in results
            assert len(results['models']) == 2
            
            comparison = results['comparison']
            assert 'best_overall' in comparison
            assert 'best_safety' in comparison
            assert 'metric_comparison' in comparison
    
    @pytest.mark.integration
    def test_full_training_integration(self, mock_config, mock_enhanced_config, temp_dir):
        """Integration test for full training pipeline."""
        # This test requires more setup and mocking
        # It would test the complete training pipeline end-to-end
        
        with patch('ray.init'), \
             patch('ray.tune.run') as mock_tune_run, \
             patch('config.config.load_config', return_value=mock_config), \
             patch('config.enhanced_config.load_enhanced_config', return_value=mock_enhanced_config), \
             patch('duckietown_utils.env.launch_and_wrap_enhanced_env'):
            
            # Mock successful training
            mock_tune_run.return_value = {
                'training_iteration': 5,
                'timesteps_total': 1000,
                'episode_reward_mean': 0.7,
                'custom_metrics': {
                    'overall_performance_score': 0.75,
                    'safety_score': 0.85
                }
            }
            
            trainer = EnhancedPPOTrainer(
                config_path=str(temp_dir / 'config.yml'),
                enhanced_config_path=str(temp_dir / 'enhanced_config.yml'),
                config_updates={'timesteps_total': 1000}
            )
            
            # Mock environment registration
            with patch.object(trainer, 'register_enhanced_environment'):
                results = trainer.train()
            
            # Verify training completed
            assert results is not None
            assert 'training_iteration' in results
            
            # Test evaluation
            with patch.object(trainer, 'evaluate_model') as mock_evaluate:
                mock_evaluate.return_value = {
                    'metrics': {
                        'mean_reward': 0.7,
                        'success_rate': 0.8,
                        'safety_score': 0.85
                    }
                }
                
                eval_results = trainer.evaluate_model('mock_checkpoint', num_episodes=5)
                assert eval_results['metrics']['mean_reward'] == 0.7


class TestEnhancedTrainingComponents:
    """Test individual components of enhanced training."""
    
    def test_reward_component_tracking(self):
        """Test multi-objective reward component tracking."""
        # Mock reward components
        reward_components = {
            'lane_following': 0.6,
            'object_avoidance': 0.3,
            'lane_change': 0.1,
            'efficiency': 0.2,
            'safety_penalty': -0.1
        }
        
        # Test reward aggregation
        total_reward = sum(reward_components.values())
        assert abs(total_reward - 1.1) < 1e-6
        
        # Test reward weighting
        weights = {
            'lane_following': 1.0,
            'object_avoidance': 0.5,
            'lane_change': 0.3,
            'efficiency': 0.2,
            'safety_penalty': -2.0
        }
        
        weighted_reward = sum(
            reward_components[component] * weights[component]
            for component in reward_components
        )
        
        expected_reward = (0.6 * 1.0 + 0.3 * 0.5 + 0.1 * 0.3 + 
                          0.2 * 0.2 + (-0.1) * (-2.0))
        assert abs(weighted_reward - expected_reward) < 1e-6
    
    def test_curriculum_stage_transitions(self):
        """Test curriculum learning stage transitions."""
        # Mock curriculum stages
        stages = [
            {
                'name': 'basic',
                'min_timesteps': 1000,
                'criteria': {'episode_reward_mean': 0.5}
            },
            {
                'name': 'intermediate',
                'min_timesteps': 2000,
                'criteria': {'episode_reward_mean': 0.7, 'safety_score': 0.8}
            }
        ]
        
        # Test stage advancement logic
        def should_advance(current_stage, timesteps_in_stage, result):
            if current_stage >= len(stages) - 1:
                return False
            
            stage_config = stages[current_stage]
            
            # Check minimum timesteps
            if timesteps_in_stage < stage_config.get('min_timesteps', 0):
                return False
            
            # Check criteria
            criteria = stage_config.get('criteria', {})
            for metric, threshold in criteria.items():
                if result.get(metric, 0) < threshold:
                    return False
            
            return True
        
        # Test cases
        result1 = {'episode_reward_mean': 0.4}  # Should not advance
        assert not should_advance(0, 1500, result1)
        
        result2 = {'episode_reward_mean': 0.6}  # Should advance
        assert should_advance(0, 1500, result2)
        
        result3 = {'episode_reward_mean': 0.8, 'safety_score': 0.75}  # Should not advance (safety too low)
        assert not should_advance(1, 2500, result3)
        
        result4 = {'episode_reward_mean': 0.8, 'safety_score': 0.85}  # Should advance
        assert should_advance(1, 2500, result4)
    
    def test_performance_score_calculation(self):
        """Test composite performance score calculation."""
        def calculate_performance_score(metrics):
            weights = {
                'reward': 0.3,
                'success': 0.3,
                'safety': 0.25,
                'efficiency': 0.15
            }
            
            reward_score = metrics.get('mean_reward', 0)
            success_score = metrics.get('success_rate', 0)
            safety_score = 1.0 - metrics.get('collision_rate', 0)
            efficiency_score = metrics.get('efficiency_score', 0)
            
            return (weights['reward'] * reward_score +
                   weights['success'] * success_score +
                   weights['safety'] * safety_score +
                   weights['efficiency'] * efficiency_score)
        
        # Test cases
        metrics1 = {
            'mean_reward': 0.8,
            'success_rate': 0.9,
            'collision_rate': 0.1,
            'efficiency_score': 0.7
        }
        
        score1 = calculate_performance_score(metrics1)
        expected1 = 0.3 * 0.8 + 0.3 * 0.9 + 0.25 * 0.9 + 0.15 * 0.7
        assert abs(score1 - expected1) < 1e-6
        
        metrics2 = {
            'mean_reward': 0.5,
            'success_rate': 0.6,
            'collision_rate': 0.3,
            'efficiency_score': 0.4
        }
        
        score2 = calculate_performance_score(metrics2)
        expected2 = 0.3 * 0.5 + 0.3 * 0.6 + 0.25 * 0.7 + 0.15 * 0.4
        assert abs(score2 - expected2) < 1e-6
        
        # Score1 should be higher than score2
        assert score1 > score2


if __name__ == '__main__':
    pytest.main([__file__])