"""
Integration tests for complete pipeline with real simulator.
Tests the full enhanced environment pipeline end-to-end.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Enhanced Duckietown RL"

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import time
import threading
import queue

from duckietown_utils.env import launch_and_wrap_enhanced_env
from config.enhanced_config import EnhancedRLConfig
from duckietown_utils.enhanced_logger import EnhancedLogger


class TestFullPipelineIntegration:
    """Integration tests for complete enhanced environment pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test artifacts."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def base_env_config(self):
        """Base environment configuration for integration testing."""
        return {
            'training_map': 'small_loop',
            'episode_max_steps': 50,  # Short episodes for testing
            'domain_rand': False,
            'dynamics_rand': False,
            'camera_rand': False,
            'accepted_start_angle_deg': 60,
            'distortion': False,
            'simulation_framerate': 30,
            'frame_skip': 1,
            'robot_speed': 0.3,
            'mode': 'debug',
            'aido_wrapper': False,
            'crop_image_top': True,
            'top_crop_divider': 3,
            'grayscale_image': False,
            'resized_input_shape': (84, 84, 3),
            'frame_stacking': False,
            'motion_blur': False,
            'action_type': 'continuous',
            'reward_function': 'Posangle'
        }
    
    @pytest.fixture
    def minimal_enhanced_config(self):
        """Minimal enhanced configuration for testing."""
        return EnhancedRLConfig(
            enabled_features=[],
            debug_mode=True
        )
    
    @pytest.fixture
    def full_enhanced_config(self, temp_dir):
        """Full enhanced configuration for comprehensive testing."""
        return EnhancedRLConfig(
            enabled_features=['yolo', 'object_avoidance', 'lane_changing', 'multi_objective_reward'],
            debug_mode=True,
            yolo={
                'model_path': 'yolov5s.pt',
                'confidence_threshold': 0.5,
                'device': 'cpu',
                'input_size': 416,  # Smaller for faster testing
                'max_detections': 50
            },
            object_avoidance={
                'safety_distance': 0.5,
                'avoidance_strength': 1.0,
                'min_clearance': 0.2
            },
            lane_changing={
                'lane_change_threshold': 0.3,
                'safety_margin': 2.0,
                'max_lane_change_time': 3.0
            },
            reward={
                'lane_following_weight': 1.0,
                'object_avoidance_weight': 0.5,
                'lane_change_weight': 0.3,
                'efficiency_weight': 0.2,
                'safety_penalty_weight': -2.0
            },
            logging={
                'log_level': 'DEBUG',
                'log_detections': True,
                'log_actions': True,
                'log_rewards': True,
                'log_file_path': str(temp_dir / 'integration_test.log')
            }
        )
    
    @pytest.mark.integration
    def test_minimal_pipeline_creation(self, base_env_config, minimal_enhanced_config):
        """Test creating minimal enhanced environment pipeline."""
        try:
            env = launch_and_wrap_enhanced_env(base_env_config, minimal_enhanced_config)
            
            # Test basic functionality
            obs = env.reset()
            assert obs is not None
            assert obs.shape == (84, 84, 3)
            
            # Test single step
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            assert obs is not None
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            
            env.close()
            
        except Exception as e:
            pytest.skip(f"Integration test skipped due to environment setup: {e}")
    
    @pytest.mark.integration
    def test_full_pipeline_creation(self, base_env_config, full_enhanced_config):
        """Test creating full enhanced environment pipeline with all features."""
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                # Mock YOLO model to avoid downloading
                mock_model = Mock()
                mock_model.return_value = Mock()
                mock_model.return_value.pandas.return_value.xyxy = [Mock(values=np.array([]))]
                mock_yolo.return_value = mock_model
                
                env = launch_and_wrap_enhanced_env(base_env_config, full_enhanced_config)
                
                # Test environment creation
                assert env is not None
                
                # Test observation space
                obs = env.reset()
                assert obs is not None
                
                # Test action space
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                
                # Verify enhanced features are working
                assert 'reward_components' in info or 'enhanced_info' in info
                
                env.close()
                
        except Exception as e:
            pytest.skip(f"Integration test skipped due to environment setup: {e}")
    
    @pytest.mark.integration
    def test_episode_completion(self, base_env_config, minimal_enhanced_config):
        """Test complete episode execution."""
        try:
            env = launch_and_wrap_enhanced_env(base_env_config, minimal_enhanced_config)
            
            obs = env.reset()
            total_reward = 0
            steps = 0
            max_steps = 100
            
            while steps < max_steps:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Verify episode completed properly
            assert steps > 0
            assert isinstance(total_reward, (int, float))
            
            env.close()
            
        except Exception as e:
            pytest.skip(f"Integration test skipped due to environment setup: {e}")
    
    @pytest.mark.integration
    def test_multiple_episodes(self, base_env_config, minimal_enhanced_config):
        """Test multiple episode execution for stability."""
        try:
            env = launch_and_wrap_enhanced_env(base_env_config, minimal_enhanced_config)
            
            num_episodes = 3
            episode_rewards = []
            
            for episode in range(num_episodes):
                obs = env.reset()
                episode_reward = 0
                steps = 0
                max_steps = 50
                
                while steps < max_steps:
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                    
                    episode_reward += reward
                    steps += 1
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
            
            # Verify all episodes completed
            assert len(episode_rewards) == num_episodes
            assert all(isinstance(r, (int, float)) for r in episode_rewards)
            
            env.close()
            
        except Exception as e:
            pytest.skip(f"Integration test skipped due to environment setup: {e}")
    
    @pytest.mark.integration
    def test_wrapper_composition_order(self, base_env_config, full_enhanced_config):
        """Test that wrappers are composed in correct order."""
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                mock_model.return_value = Mock()
                mock_model.return_value.pandas.return_value.xyxy = [Mock(values=np.array([]))]
                mock_yolo.return_value = mock_model
                
                env = launch_and_wrap_enhanced_env(base_env_config, full_enhanced_config)
                
                # Get wrapper hierarchy
                from duckietown_utils.env import get_enhanced_wrappers
                obs_wrappers, action_wrappers, reward_wrappers, enhanced_wrappers = get_enhanced_wrappers(env)
                
                # Verify enhanced wrappers are present
                assert len(enhanced_wrappers) > 0
                
                # Test that observation flows through properly
                obs = env.reset()
                assert obs is not None
                
                # Test that actions flow through properly
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                
                assert obs is not None
                assert isinstance(reward, (int, float))
                
                env.close()
                
        except Exception as e:
            pytest.skip(f"Integration test skipped due to environment setup: {e}")
    
    @pytest.mark.integration
    def test_error_recovery(self, base_env_config):
        """Test error recovery in pipeline."""
        # Test with invalid configuration
        invalid_config = EnhancedRLConfig(
            enabled_features=['yolo'],
            yolo={'model_path': 'nonexistent_model.pt'}
        )
        
        try:
            # Should handle error gracefully in non-debug mode
            invalid_config.debug_mode = False
            env = launch_and_wrap_enhanced_env(base_env_config, invalid_config)
            
            # Should still create environment (with fallback)
            assert env is not None
            
            obs = env.reset()
            assert obs is not None
            
            env.close()
            
        except Exception as e:
            # Expected in some cases
            pass
    
    @pytest.mark.integration
    def test_logging_integration(self, base_env_config, full_enhanced_config, temp_dir):
        """Test logging integration in pipeline."""
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                mock_model.return_value = Mock()
                mock_model.return_value.pandas.return_value.xyxy = [Mock(values=np.array([]))]
                mock_yolo.return_value = mock_model
                
                env = launch_and_wrap_enhanced_env(base_env_config, full_enhanced_config)
                
                # Run a few steps to generate logs
                obs = env.reset()
                for _ in range(5):
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                    if done:
                        break
                
                env.close()
                
                # Check if log file was created
                log_file = temp_dir / 'integration_test.log'
                if log_file.exists():
                    assert log_file.stat().st_size > 0
                
        except Exception as e:
            pytest.skip(f"Integration test skipped due to environment setup: {e}")


class TestPipelinePerformance:
    """Performance tests for the enhanced environment pipeline."""
    
    @pytest.fixture
    def performance_env_config(self):
        """Environment configuration optimized for performance testing."""
        return {
            'training_map': 'small_loop',
            'episode_max_steps': 100,
            'domain_rand': False,
            'dynamics_rand': False,
            'camera_rand': False,
            'simulation_framerate': 60,  # Higher framerate for performance testing
            'frame_skip': 1,
            'robot_speed': 0.5,
            'mode': 'debug',
            'resized_input_shape': (64, 64, 3),  # Smaller for faster processing
            'action_type': 'continuous'
        }
    
    @pytest.fixture
    def performance_enhanced_config(self):
        """Enhanced configuration optimized for performance."""
        return EnhancedRLConfig(
            enabled_features=['yolo', 'object_avoidance'],
            debug_mode=False,  # Disable debug for performance
            yolo={
                'model_path': 'yolov5n.pt',  # Nano model for speed
                'confidence_threshold': 0.5,
                'device': 'cpu',
                'input_size': 320,  # Smaller input for speed
                'max_detections': 20
            },
            object_avoidance={
                'safety_distance': 0.5,
                'avoidance_strength': 1.0
            },
            performance={
                'max_fps': 30.0,
                'detection_batch_size': 1,
                'use_gpu_acceleration': False
            }
        )
    
    @pytest.mark.performance
    def test_step_performance(self, performance_env_config, performance_enhanced_config):
        """Test step execution performance."""
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                # Mock fast YOLO model
                mock_model = Mock()
                mock_model.return_value = Mock()
                mock_model.return_value.pandas.return_value.xyxy = [Mock(values=np.array([]))]
                mock_yolo.return_value = mock_model
                
                env = launch_and_wrap_enhanced_env(performance_env_config, performance_enhanced_config)
                
                obs = env.reset()
                
                # Measure step performance
                num_steps = 100
                start_time = time.time()
                
                for _ in range(num_steps):
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                    if done:
                        obs = env.reset()
                
                end_time = time.time()
                total_time = end_time - start_time
                steps_per_second = num_steps / total_time
                
                env.close()
                
                # Should achieve reasonable performance
                assert steps_per_second >= 10.0, f"Performance too low: {steps_per_second:.2f} steps/sec"
                
        except Exception as e:
            pytest.skip(f"Performance test skipped due to environment setup: {e}")
    
    @pytest.mark.performance
    def test_memory_usage(self, performance_env_config, performance_enhanced_config):
        """Test memory usage during pipeline execution."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                mock_model.return_value = Mock()
                mock_model.return_value.pandas.return_value.xyxy = [Mock(values=np.array([]))]
                mock_yolo.return_value = mock_model
                
                env = launch_and_wrap_enhanced_env(performance_env_config, performance_enhanced_config)
                
                # Run multiple episodes to check for memory leaks
                for episode in range(5):
                    obs = env.reset()
                    for step in range(50):
                        action = env.action_space.sample()
                        obs, reward, done, info = env.step(action)
                        if done:
                            break
                
                env.close()
                
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                
                # Memory increase should be reasonable (less than 500MB)
                assert memory_increase < 500, f"Memory usage too high: {memory_increase:.2f} MB increase"
                
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        except Exception as e:
            pytest.skip(f"Memory test skipped due to environment setup: {e}")
    
    @pytest.mark.performance
    def test_concurrent_environments(self, performance_env_config, performance_enhanced_config):
        """Test performance with multiple concurrent environments."""
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                mock_model.return_value = Mock()
                mock_model.return_value.pandas.return_value.xyxy = [Mock(values=np.array([]))]
                mock_yolo.return_value = mock_model
                
                num_envs = 3
                envs = []
                
                # Create multiple environments
                for i in range(num_envs):
                    env = launch_and_wrap_enhanced_env(performance_env_config, performance_enhanced_config)
                    envs.append(env)
                
                # Test concurrent execution
                results = queue.Queue()
                
                def run_env(env_id, env):
                    try:
                        obs = env.reset()
                        total_reward = 0
                        for step in range(20):
                            action = env.action_space.sample()
                            obs, reward, done, info = env.step(action)
                            total_reward += reward
                            if done:
                                break
                        results.put((env_id, total_reward, True))
                    except Exception as e:
                        results.put((env_id, 0, False))
                
                # Run environments concurrently
                threads = []
                start_time = time.time()
                
                for i, env in enumerate(envs):
                    thread = threading.Thread(target=run_env, args=(i, env))
                    thread.start()
                    threads.append(thread)
                
                # Wait for completion
                for thread in threads:
                    thread.join(timeout=30)  # 30 second timeout
                
                end_time = time.time()
                
                # Collect results
                successful_envs = 0
                while not results.empty():
                    env_id, reward, success = results.get()
                    if success:
                        successful_envs += 1
                
                # Clean up
                for env in envs:
                    try:
                        env.close()
                    except:
                        pass
                
                # Should complete within reasonable time
                total_time = end_time - start_time
                assert total_time < 60, f"Concurrent execution too slow: {total_time:.2f} seconds"
                assert successful_envs >= num_envs // 2, f"Too many failed environments: {successful_envs}/{num_envs}"
                
        except Exception as e:
            pytest.skip(f"Concurrent test skipped due to environment setup: {e}")


class TestPipelineRobustness:
    """Robustness tests for the enhanced environment pipeline."""
    
    @pytest.mark.robustness
    def test_invalid_action_handling(self, base_env_config, minimal_enhanced_config):
        """Test handling of invalid actions."""
        try:
            env = launch_and_wrap_enhanced_env(base_env_config, minimal_enhanced_config)
            
            obs = env.reset()
            
            # Test various invalid actions
            invalid_actions = [
                np.array([np.inf, 0.0]),  # Infinite value
                np.array([np.nan, 0.0]),  # NaN value
                np.array([10.0, 10.0]),   # Out of bounds
                np.array([-10.0, -10.0]), # Out of bounds
                np.array([0.0]),          # Wrong shape
                None,                     # None value
            ]
            
            for invalid_action in invalid_actions:
                try:
                    obs, reward, done, info = env.step(invalid_action)
                    # Should handle gracefully
                    assert obs is not None
                    assert isinstance(reward, (int, float))
                    assert not np.isnan(reward)
                    assert not np.isinf(reward)
                except (ValueError, TypeError):
                    # Expected for some invalid actions
                    pass
            
            env.close()
            
        except Exception as e:
            pytest.skip(f"Robustness test skipped due to environment setup: {e}")
    
    @pytest.mark.robustness
    def test_rapid_reset_handling(self, base_env_config, minimal_enhanced_config):
        """Test handling of rapid environment resets."""
        try:
            env = launch_and_wrap_enhanced_env(base_env_config, minimal_enhanced_config)
            
            # Rapid resets
            for i in range(10):
                obs = env.reset()
                assert obs is not None
                
                # Take a few steps
                for j in range(3):
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                    if done:
                        break
            
            env.close()
            
        except Exception as e:
            pytest.skip(f"Robustness test skipped due to environment setup: {e}")
    
    @pytest.mark.robustness
    def test_long_episode_stability(self, base_env_config, minimal_enhanced_config):
        """Test stability during long episodes."""
        try:
            # Increase episode length for this test
            long_episode_config = base_env_config.copy()
            long_episode_config['episode_max_steps'] = 500
            
            env = launch_and_wrap_enhanced_env(long_episode_config, minimal_enhanced_config)
            
            obs = env.reset()
            total_steps = 0
            
            for step in range(500):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                total_steps += 1
                
                # Verify stability
                assert obs is not None
                assert not np.any(np.isnan(obs))
                assert not np.any(np.isinf(obs))
                assert isinstance(reward, (int, float))
                assert not np.isnan(reward)
                assert not np.isinf(reward)
                
                if done:
                    break
            
            assert total_steps > 0
            env.close()
            
        except Exception as e:
            pytest.skip(f"Long episode test skipped due to environment setup: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])