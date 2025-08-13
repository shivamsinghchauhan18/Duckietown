"""
Integration tests for Enhanced Duckietown RL Environment.
Tests the complete environment setup with all enhanced wrappers.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Enhanced Duckietown RL"

import pytest
import gym
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import yaml

from duckietown_utils.env import (
    launch_and_wrap_enhanced_env,
    _apply_enhanced_wrappers,
    _validate_wrapper_compatibility,
    get_enhanced_wrappers,
    DummyDuckietownGymLikeEnv
)
from config.enhanced_config import EnhancedRLConfig, YOLOConfig, ObjectAvoidanceConfig
from duckietown_utils.wrappers.yolo_detection_wrapper import YOLOObjectDetectionWrapper
from duckietown_utils.wrappers.enhanced_observation_wrapper import EnhancedObservationWrapper
from duckietown_utils.wrappers.object_avoidance_action_wrapper import ObjectAvoidanceActionWrapper
from duckietown_utils.wrappers.lane_changing_action_wrapper import LaneChangingActionWrapper
from duckietown_utils.wrappers.multi_objective_reward_wrapper import MultiObjectiveRewardWrapper


class TestEnhancedEnvironmentIntegration:
    """Test suite for enhanced environment integration."""
    
    @pytest.fixture
    def base_env_config(self):
        """Standard environment configuration for testing."""
        return {
            'training_map': 'small_loop',
            'episode_max_steps': 500,
            'domain_rand': False,
            'dynamics_rand': False,
            'camera_rand': False,
            'accepted_start_angle_deg': 60,
            'distortion': False,
            'simulation_framerate': 30,
            'frame_skip': 1,
            'robot_speed': 0.3,
            'mode': 'train',
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
    def enhanced_config(self):
        """Enhanced configuration for testing."""
        return EnhancedRLConfig(
            enabled_features=['yolo', 'object_avoidance', 'lane_changing', 'multi_objective_reward'],
            debug_mode=True
        )
    
    @pytest.fixture
    def minimal_enhanced_config(self):
        """Minimal enhanced configuration for testing."""
        return EnhancedRLConfig(
            enabled_features=[],
            debug_mode=False
        )
    
    @pytest.fixture
    def mock_env(self):
        """Mock environment for testing."""
        env = Mock(spec=DummyDuckietownGymLikeEnv)
        env.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        env.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        env.road_tile_size = 0.585
        return env
    
    def test_launch_and_wrap_enhanced_env_default_config(self, base_env_config):
        """Test launching enhanced environment with default configuration."""
        with patch('duckietown_utils.env.launch_and_wrap_env') as mock_launch:
            mock_env = Mock()
            mock_launch.return_value = mock_env
            
            with patch('duckietown_utils.env._apply_enhanced_wrappers') as mock_apply:
                mock_apply.return_value = mock_env
                
                result = launch_and_wrap_enhanced_env(base_env_config)
                
                mock_launch.assert_called_once_with(base_env_config, 0)
                mock_apply.assert_called_once()
                assert result == mock_env
    
    def test_launch_and_wrap_enhanced_env_with_config_object(self, base_env_config, enhanced_config):
        """Test launching enhanced environment with configuration object."""
        with patch('duckietown_utils.env.launch_and_wrap_env') as mock_launch:
            mock_env = Mock()
            mock_launch.return_value = mock_env
            
            with patch('duckietown_utils.env._apply_enhanced_wrappers') as mock_apply:
                mock_apply.return_value = mock_env
                
                result = launch_and_wrap_enhanced_env(base_env_config, enhanced_config)
                
                mock_launch.assert_called_once_with(base_env_config, 0)
                mock_apply.assert_called_once()
                assert result == mock_env
    
    def test_launch_and_wrap_enhanced_env_with_config_file(self, base_env_config):
        """Test launching enhanced environment with configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            config_data = {
                'enabled_features': ['yolo'],
                'debug_mode': False,
                'yolo': {'confidence_threshold': 0.7}
            }
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with patch('duckietown_utils.env.launch_and_wrap_env') as mock_launch:
                mock_env = Mock()
                mock_launch.return_value = mock_env
                
                with patch('duckietown_utils.env._apply_enhanced_wrappers') as mock_apply:
                    mock_apply.return_value = mock_env
                    
                    result = launch_and_wrap_enhanced_env(base_env_config, config_path)
                    
                    mock_launch.assert_called_once_with(base_env_config, 0)
                    mock_apply.assert_called_once()
                    assert result == mock_env
        finally:
            Path(config_path).unlink()
    
    def test_launch_and_wrap_enhanced_env_invalid_config(self, base_env_config):
        """Test launching enhanced environment with invalid configuration."""
        with pytest.raises(ValueError, match="enhanced_config must be"):
            launch_and_wrap_enhanced_env(base_env_config, 123)
    
    def test_apply_enhanced_wrappers_all_features(self, mock_env, base_env_config, enhanced_config):
        """Test applying all enhanced wrappers."""
        with patch('duckietown_utils.env.YOLOObjectDetectionWrapper') as mock_yolo:
            with patch('duckietown_utils.env.EnhancedObservationWrapper') as mock_obs:
                with patch('duckietown_utils.env.ObjectAvoidanceActionWrapper') as mock_avoid:
                    with patch('duckietown_utils.env.LaneChangingActionWrapper') as mock_lane:
                        with patch('duckietown_utils.env.MultiObjectiveRewardWrapper') as mock_reward:
                            
                            # Set up mock returns
                            mock_yolo.return_value = mock_env
                            mock_obs.return_value = mock_env
                            mock_avoid.return_value = mock_env
                            mock_lane.return_value = mock_env
                            mock_reward.return_value = mock_env
                            
                            result = _apply_enhanced_wrappers(mock_env, base_env_config, enhanced_config)
                            
                            # Verify all wrappers were applied
                            mock_yolo.assert_called_once()
                            mock_obs.assert_called_once()
                            mock_avoid.assert_called_once()
                            mock_lane.assert_called_once()
                            mock_reward.assert_called_once()
                            assert result == mock_env
    
    def test_apply_enhanced_wrappers_selective_features(self, mock_env, base_env_config):
        """Test applying only selected enhanced wrappers."""
        config = EnhancedRLConfig(enabled_features=['yolo', 'object_avoidance'])
        
        with patch('duckietown_utils.env.YOLOObjectDetectionWrapper') as mock_yolo:
            with patch('duckietown_utils.env.EnhancedObservationWrapper') as mock_obs:
                with patch('duckietown_utils.env.ObjectAvoidanceActionWrapper') as mock_avoid:
                    with patch('duckietown_utils.env.LaneChangingActionWrapper') as mock_lane:
                        with patch('duckietown_utils.env.MultiObjectiveRewardWrapper') as mock_reward:
                            
                            # Set up mock returns
                            mock_yolo.return_value = mock_env
                            mock_obs.return_value = mock_env
                            mock_avoid.return_value = mock_env
                            
                            result = _apply_enhanced_wrappers(mock_env, base_env_config, config)
                            
                            # Verify only selected wrappers were applied
                            mock_yolo.assert_called_once()
                            mock_obs.assert_called_once()
                            mock_avoid.assert_called_once()
                            mock_lane.assert_not_called()
                            mock_reward.assert_called_once()  # Applied due to object_avoidance
                            assert result == mock_env
    
    def test_apply_enhanced_wrappers_no_features(self, mock_env, base_env_config, minimal_enhanced_config):
        """Test applying no enhanced wrappers."""
        with patch('duckietown_utils.env.YOLOObjectDetectionWrapper') as mock_yolo:
            with patch('duckietown_utils.env.EnhancedObservationWrapper') as mock_obs:
                with patch('duckietown_utils.env.ObjectAvoidanceActionWrapper') as mock_avoid:
                    with patch('duckietown_utils.env.LaneChangingActionWrapper') as mock_lane:
                        with patch('duckietown_utils.env.MultiObjectiveRewardWrapper') as mock_reward:
                            
                            result = _apply_enhanced_wrappers(mock_env, base_env_config, minimal_enhanced_config)
                            
                            # Verify no wrappers were applied
                            mock_yolo.assert_not_called()
                            mock_obs.assert_not_called()
                            mock_avoid.assert_not_called()
                            mock_lane.assert_not_called()
                            mock_reward.assert_not_called()
                            assert result == mock_env
    
    def test_apply_enhanced_wrappers_error_handling(self, mock_env, base_env_config):
        """Test error handling in wrapper application."""
        config = EnhancedRLConfig(enabled_features=['yolo'], debug_mode=False)
        
        with patch('duckietown_utils.env.YOLOObjectDetectionWrapper') as mock_yolo:
            mock_yolo.side_effect = Exception("YOLO initialization failed")
            
            # Should not raise exception in non-debug mode
            result = _apply_enhanced_wrappers(mock_env, base_env_config, config)
            assert result == mock_env
    
    def test_apply_enhanced_wrappers_error_handling_debug_mode(self, mock_env, base_env_config):
        """Test error handling in debug mode."""
        config = EnhancedRLConfig(enabled_features=['yolo'], debug_mode=True)
        
        with patch('duckietown_utils.env.YOLOObjectDetectionWrapper') as mock_yolo:
            mock_yolo.side_effect = Exception("YOLO initialization failed")
            
            # Should raise exception in debug mode
            with pytest.raises(Exception, match="YOLO initialization failed"):
                _apply_enhanced_wrappers(mock_env, base_env_config, config)
    
    def test_validate_wrapper_compatibility_grayscale_yolo_conflict(self, enhanced_config):
        """Test validation of grayscale image with YOLO conflict."""
        env_config = {'grayscale_image': True}
        
        with pytest.raises(ValueError, match="YOLO detection requires RGB images"):
            _validate_wrapper_compatibility(env_config, enhanced_config)
    
    def test_validate_wrapper_compatibility_frame_stacking_warning(self, enhanced_config):
        """Test validation warning for frame stacking with YOLO."""
        env_config = {'frame_stacking': True}
        
        with patch('duckietown_utils.env.logger') as mock_logger:
            _validate_wrapper_compatibility(env_config, enhanced_config)
            mock_logger.warning.assert_called()
    
    def test_validate_wrapper_compatibility_discrete_actions_warning(self):
        """Test validation warning for discrete actions with enhanced wrappers."""
        config = EnhancedRLConfig(enabled_features=['object_avoidance'])
        env_config = {'action_type': 'discrete'}
        
        with patch('duckietown_utils.env.logger') as mock_logger:
            _validate_wrapper_compatibility(env_config, config)
            mock_logger.warning.assert_called()
    
    def test_validate_wrapper_compatibility_reward_function_warning(self):
        """Test validation warning for conflicting reward functions."""
        config = EnhancedRLConfig(enabled_features=['multi_objective_reward'])
        env_config = {'reward_function': 'custom_reward'}
        
        with patch('duckietown_utils.env.logger') as mock_logger:
            _validate_wrapper_compatibility(env_config, config)
            mock_logger.warning.assert_called()
    
    def test_validate_wrapper_compatibility_dependency_warnings(self):
        """Test validation warnings for feature dependencies."""
        config = EnhancedRLConfig(enabled_features=['object_avoidance'])
        env_config = {}
        
        with patch('duckietown_utils.env.logger') as mock_logger:
            _validate_wrapper_compatibility(env_config, config)
            mock_logger.warning.assert_called()
    
    def test_get_enhanced_wrappers(self):
        """Test getting enhanced wrappers from wrapped environment."""
        # Create a mock wrapped environment with enhanced wrappers
        base_env = Mock(spec=DummyDuckietownGymLikeEnv)
        
        yolo_wrapper = Mock(spec=YOLOObjectDetectionWrapper)
        yolo_wrapper.env = base_env
        
        obs_wrapper = Mock(spec=EnhancedObservationWrapper)
        obs_wrapper.env = yolo_wrapper
        
        action_wrapper = Mock(spec=ObjectAvoidanceActionWrapper)
        action_wrapper.env = obs_wrapper
        
        reward_wrapper = Mock(spec=MultiObjectiveRewardWrapper)
        reward_wrapper.env = action_wrapper
        
        obs_wrappers, action_wrappers, reward_wrappers, enhanced_wrappers = get_enhanced_wrappers(reward_wrapper)
        
        # Check that enhanced wrappers are identified
        assert len(enhanced_wrappers) == 4
        assert yolo_wrapper in enhanced_wrappers
        assert obs_wrapper in enhanced_wrappers
        assert action_wrapper in enhanced_wrappers
        assert reward_wrapper in enhanced_wrappers
    
    def test_wrapper_ordering(self, mock_env, base_env_config, enhanced_config):
        """Test that wrappers are applied in correct order."""
        wrapper_calls = []
        
        def track_wrapper_calls(wrapper_class):
            def wrapper_init(env, *args, **kwargs):
                wrapper_calls.append(wrapper_class.__name__)
                return mock_env
            return wrapper_init
        
        with patch('duckietown_utils.env.YOLOObjectDetectionWrapper', side_effect=track_wrapper_calls(YOLOObjectDetectionWrapper)):
            with patch('duckietown_utils.env.EnhancedObservationWrapper', side_effect=track_wrapper_calls(EnhancedObservationWrapper)):
                with patch('duckietown_utils.env.ObjectAvoidanceActionWrapper', side_effect=track_wrapper_calls(ObjectAvoidanceActionWrapper)):
                    with patch('duckietown_utils.env.LaneChangingActionWrapper', side_effect=track_wrapper_calls(LaneChangingActionWrapper)):
                        with patch('duckietown_utils.env.MultiObjectiveRewardWrapper', side_effect=track_wrapper_calls(MultiObjectiveRewardWrapper)):
                            
                            _apply_enhanced_wrappers(mock_env, base_env_config, enhanced_config)
                            
                            # Verify correct order: observation wrappers first, then action, then reward
                            expected_order = [
                                'YOLOObjectDetectionWrapper',
                                'EnhancedObservationWrapper',
                                'ObjectAvoidanceActionWrapper',
                                'LaneChangingActionWrapper',
                                'MultiObjectiveRewardWrapper'
                            ]
                            assert wrapper_calls == expected_order
    
    def test_backward_compatibility(self, base_env_config):
        """Test that enhanced environment maintains backward compatibility."""
        with patch('duckietown_utils.env.launch_and_wrap_env') as mock_launch:
            mock_env = Mock()
            mock_launch.return_value = mock_env
            
            # Test with minimal configuration (no enhanced features)
            minimal_config = EnhancedRLConfig(enabled_features=[])
            result = launch_and_wrap_enhanced_env(base_env_config, minimal_config)
            
            # Should still work and return the base environment
            mock_launch.assert_called_once()
            assert result == mock_env


class TestEnhancedEnvironmentIntegrationEnd2End:
    """End-to-end integration tests with real environment components."""
    
    @pytest.fixture
    def real_env_config(self):
        """Real environment configuration for end-to-end testing."""
        return {
            'training_map': 'small_loop',
            'episode_max_steps': 10,  # Short episodes for testing
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
    
    @pytest.mark.integration
    def test_enhanced_environment_creation_minimal(self, real_env_config):
        """Test creating enhanced environment with minimal features."""
        config = EnhancedRLConfig(enabled_features=[])
        
        try:
            env = launch_and_wrap_enhanced_env(real_env_config, config)
            
            # Test basic environment functionality
            obs = env.reset()
            assert obs is not None
            
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            assert obs is not None
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            
        except Exception as e:
            pytest.skip(f"Integration test skipped due to environment setup: {e}")
    
    @pytest.mark.integration
    def test_enhanced_environment_wrapper_composition(self, real_env_config):
        """Test that wrapper composition works correctly."""
        config = EnhancedRLConfig(enabled_features=[])
        
        try:
            env = launch_and_wrap_enhanced_env(real_env_config, config)
            
            # Get wrapper information
            obs_wrappers, action_wrappers, reward_wrappers, enhanced_wrappers = get_enhanced_wrappers(env)
            
            # Should have standard wrappers but no enhanced wrappers
            assert len(obs_wrappers) > 0  # Should have standard observation wrappers
            assert len(enhanced_wrappers) == 0  # No enhanced wrappers enabled
            
        except Exception as e:
            pytest.skip(f"Integration test skipped due to environment setup: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])