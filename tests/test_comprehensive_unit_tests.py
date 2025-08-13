"""
Comprehensive unit tests for all wrapper classes with mock environments.
Tests each wrapper in isolation to ensure proper functionality.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Enhanced Duckietown RL"

import pytest
import numpy as np
import gym
from unittest.mock import Mock, patch, MagicMock
import torch
from pathlib import Path
import tempfile
import json

from duckietown_utils.wrappers.yolo_detection_wrapper import YOLOObjectDetectionWrapper
from duckietown_utils.wrappers.enhanced_observation_wrapper import EnhancedObservationWrapper
from duckietown_utils.wrappers.object_avoidance_action_wrapper import ObjectAvoidanceActionWrapper
from duckietown_utils.wrappers.lane_changing_action_wrapper import LaneChangingActionWrapper
from duckietown_utils.wrappers.multi_objective_reward_wrapper import MultiObjectiveRewardWrapper
from config.enhanced_config import EnhancedRLConfig, YOLOConfig, ObjectAvoidanceConfig, LaneChangingConfig


class MockEnvironment(gym.Env):
    """Mock environment for testing wrappers."""
    
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        self.road_tile_size = 0.585
        self._step_count = 0
        self._episode_count = 0
        
    def reset(self):
        self._step_count = 0
        self._episode_count += 1
        return np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
    
    def step(self, action):
        self._step_count += 1
        obs = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
        reward = np.random.random()
        done = self._step_count >= 100
        info = {
            'Simulator': {
                'cur_pos': [1.0, 0.0, 1.0],
                'cur_angle': 0.0,
                'lane_position': {'dist': 0.1, 'dot_dir': 0.9, 'angle_deg': 5.0}
            }
        }
        return obs, reward, done, info
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass


class TestYOLOObjectDetectionWrapper:
    """Comprehensive unit tests for YOLO Object Detection Wrapper."""
    
    @pytest.fixture
    def mock_env(self):
        return MockEnvironment()
    
    @pytest.fixture
    def yolo_config(self):
        return YOLOConfig(
            model_path='yolov5s.pt',
            confidence_threshold=0.5,
            device='cpu',
            input_size=640,
            max_detections=100
        )
    
    @pytest.fixture
    def mock_yolo_model(self):
        """Mock YOLO model for testing."""
        model = Mock()
        
        # Mock detection results
        mock_results = Mock()
        mock_results.pandas.return_value.xyxy = [
            Mock(values=np.array([
                [100, 100, 200, 200, 0.8, 0, 'person'],
                [300, 150, 400, 250, 0.6, 2, 'car']
            ]))
        ]
        model.return_value = mock_results
        
        return model
    
    def test_wrapper_initialization(self, mock_env, yolo_config):
        """Test YOLO wrapper initialization."""
        with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_load:
            mock_load.return_value = Mock()
            
            wrapper = YOLOObjectDetectionWrapper(mock_env, yolo_config)
            
            assert wrapper.env == mock_env
            assert wrapper.config == yolo_config
            assert wrapper.model is not None
            mock_load.assert_called_once()
    
    def test_observation_processing(self, mock_env, yolo_config, mock_yolo_model):
        """Test observation processing with object detection."""
        with patch('duckietown_utils.yolo_utils.load_yolo_model', return_value=mock_yolo_model):
            wrapper = YOLOObjectDetectionWrapper(mock_env, yolo_config)
            
            # Test observation processing
            original_obs = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
            processed_obs = wrapper.observation(original_obs)
            
            # Verify observation structure
            assert isinstance(processed_obs, dict)
            assert 'image' in processed_obs
            assert 'detections' in processed_obs
            assert 'detection_count' in processed_obs
            assert 'safety_critical' in processed_obs
            
            # Verify detection data
            assert len(processed_obs['detections']) == 2
            assert processed_obs['detection_count'] == 2
            
            # Check detection structure
            detection = processed_obs['detections'][0]
            assert 'class' in detection
            assert 'confidence' in detection
            assert 'bbox' in detection
            assert 'distance' in detection
            assert 'relative_position' in detection
    
    def test_distance_estimation(self, mock_env, yolo_config, mock_yolo_model):
        """Test distance estimation from bounding boxes."""
        with patch('duckietown_utils.yolo_utils.load_yolo_model', return_value=mock_yolo_model):
            wrapper = YOLOObjectDetectionWrapper(mock_env, yolo_config)
            
            # Test distance calculation
            bbox = [100, 100, 200, 200]  # x1, y1, x2, y2
            distance = wrapper._estimate_distance(bbox, 'person')
            
            assert isinstance(distance, float)
            assert distance > 0
    
    def test_safety_critical_detection(self, mock_env, yolo_config):
        """Test safety critical object detection."""
        # Mock close object detection
        mock_model = Mock()
        mock_results = Mock()
        mock_results.pandas.return_value.xyxy = [
            Mock(values=np.array([
                [200, 200, 400, 400, 0.9, 0, 'person']  # Large bbox = close object
            ]))
        ]
        mock_model.return_value = mock_results
        
        with patch('duckietown_utils.yolo_utils.load_yolo_model', return_value=mock_model):
            wrapper = YOLOObjectDetectionWrapper(mock_env, yolo_config)
            
            original_obs = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
            processed_obs = wrapper.observation(original_obs)
            
            # Should detect safety critical situation
            assert processed_obs['safety_critical'] is True
    
    def test_error_handling(self, mock_env, yolo_config):
        """Test error handling in YOLO wrapper."""
        # Mock model that raises exception
        mock_model = Mock()
        mock_model.side_effect = Exception("YOLO inference failed")
        
        with patch('duckietown_utils.yolo_utils.load_yolo_model', return_value=mock_model):
            wrapper = YOLOObjectDetectionWrapper(mock_env, yolo_config)
            
            original_obs = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
            processed_obs = wrapper.observation(original_obs)
            
            # Should return safe fallback
            assert isinstance(processed_obs, dict)
            assert processed_obs['detection_count'] == 0
            assert processed_obs['safety_critical'] is False


class TestEnhancedObservationWrapper:
    """Comprehensive unit tests for Enhanced Observation Wrapper."""
    
    @pytest.fixture
    def mock_env_with_detections(self):
        """Mock environment that returns detection observations."""
        env = MockEnvironment()
        
        # Override observation space to include detections
        env.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            'detections': gym.spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
            'detection_count': gym.spaces.Discrete(100),
            'safety_critical': gym.spaces.Discrete(2)
        })
        
        return env
    
    def test_wrapper_initialization(self, mock_env_with_detections):
        """Test enhanced observation wrapper initialization."""
        wrapper = EnhancedObservationWrapper(
            mock_env_with_detections,
            include_detection_features=True,
            flatten_observations=True
        )
        
        assert wrapper.env == mock_env_with_detections
        assert wrapper.include_detection_features is True
        assert wrapper.flatten_observations is True
        
        # Check observation space transformation
        assert isinstance(wrapper.observation_space, gym.spaces.Box)
    
    def test_observation_flattening(self, mock_env_with_detections):
        """Test observation flattening functionality."""
        wrapper = EnhancedObservationWrapper(
            mock_env_with_detections,
            include_detection_features=True,
            flatten_observations=True
        )
        
        # Mock detection observation
        detection_obs = {
            'image': np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8),
            'detections': [
                {
                    'class': 'person',
                    'confidence': 0.8,
                    'bbox': [100, 100, 200, 200],
                    'distance': 2.5,
                    'relative_position': [0.1, 0.2]
                }
            ],
            'detection_count': 1,
            'safety_critical': True
        }
        
        flattened_obs = wrapper.observation(detection_obs)
        
        # Should return flattened numpy array
        assert isinstance(flattened_obs, np.ndarray)
        assert len(flattened_obs.shape) == 1  # Flattened
        assert flattened_obs.dtype == np.float32
    
    def test_feature_extraction(self, mock_env_with_detections):
        """Test detection feature extraction."""
        wrapper = EnhancedObservationWrapper(
            mock_env_with_detections,
            include_detection_features=True
        )
        
        # Test feature extraction from detections
        detections = [
            {
                'class': 'person',
                'confidence': 0.8,
                'distance': 2.5,
                'relative_position': [0.1, 0.2]
            },
            {
                'class': 'car',
                'confidence': 0.6,
                'distance': 5.0,
                'relative_position': [-0.3, 0.1]
            }
        ]
        
        features = wrapper._extract_detection_features(detections)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        
        # Should include closest object distance
        assert np.min([d['distance'] for d in detections]) in features
    
    def test_observation_normalization(self, mock_env_with_detections):
        """Test observation normalization."""
        wrapper = EnhancedObservationWrapper(
            mock_env_with_detections,
            normalize_observations=True
        )
        
        # Test with high-value image
        high_value_obs = {
            'image': np.full((84, 84, 3), 255, dtype=np.uint8),
            'detections': [],
            'detection_count': 0,
            'safety_critical': False
        }
        
        normalized_obs = wrapper.observation(high_value_obs)
        
        # Normalized values should be in [0, 1] range
        if isinstance(normalized_obs, np.ndarray):
            assert np.all(normalized_obs >= 0)
            assert np.all(normalized_obs <= 1)


class TestObjectAvoidanceActionWrapper:
    """Comprehensive unit tests for Object Avoidance Action Wrapper."""
    
    @pytest.fixture
    def mock_env_with_detections(self):
        """Mock environment with detection capabilities."""
        env = MockEnvironment()
        env._last_observation = {
            'detections': [],
            'safety_critical': False
        }
        return env
    
    @pytest.fixture
    def avoidance_config(self):
        return ObjectAvoidanceConfig(
            safety_distance=0.5,
            avoidance_strength=1.0,
            min_clearance=0.2,
            max_avoidance_angle=0.5,
            smoothing_factor=0.8
        )
    
    def test_wrapper_initialization(self, mock_env_with_detections, avoidance_config):
        """Test object avoidance wrapper initialization."""
        wrapper = ObjectAvoidanceActionWrapper(mock_env_with_detections, avoidance_config)
        
        assert wrapper.env == mock_env_with_detections
        assert wrapper.config == avoidance_config
        assert wrapper.action_space == mock_env_with_detections.action_space
    
    def test_no_objects_action_passthrough(self, mock_env_with_detections, avoidance_config):
        """Test action passthrough when no objects detected."""
        wrapper = ObjectAvoidanceActionWrapper(mock_env_with_detections, avoidance_config)
        
        # Set no detections
        wrapper.env._last_observation = {
            'detections': [],
            'safety_critical': False
        }
        
        original_action = np.array([0.5, 0.3])
        modified_action = wrapper.action(original_action)
        
        # Action should be unchanged
        np.testing.assert_array_almost_equal(modified_action, original_action)
    
    def test_object_avoidance_modification(self, mock_env_with_detections, avoidance_config):
        """Test action modification for object avoidance."""
        wrapper = ObjectAvoidanceActionWrapper(mock_env_with_detections, avoidance_config)
        
        # Set close object detection
        wrapper.env._last_observation = {
            'detections': [
                {
                    'class': 'person',
                    'confidence': 0.8,
                    'distance': 0.3,  # Within safety distance
                    'relative_position': [0.0, 0.2]  # Directly ahead
                }
            ],
            'safety_critical': True
        }
        
        original_action = np.array([0.5, 0.0])  # Forward motion
        modified_action = wrapper.action(original_action)
        
        # Action should be modified for avoidance
        assert not np.array_equal(modified_action, original_action)
        
        # Should have steering component for avoidance
        assert abs(modified_action[1]) > abs(original_action[1])
    
    def test_multiple_objects_prioritization(self, mock_env_with_detections, avoidance_config):
        """Test prioritization with multiple objects."""
        wrapper = ObjectAvoidanceActionWrapper(mock_env_with_detections, avoidance_config)
        
        # Set multiple object detections
        wrapper.env._last_observation = {
            'detections': [
                {
                    'class': 'person',
                    'distance': 0.8,  # Farther
                    'relative_position': [0.1, 0.2]
                },
                {
                    'class': 'car',
                    'distance': 0.3,  # Closer - should be prioritized
                    'relative_position': [-0.1, 0.1]
                }
            ],
            'safety_critical': True
        }
        
        original_action = np.array([0.5, 0.0])
        modified_action = wrapper.action(original_action)
        
        # Should prioritize closer object
        assert not np.array_equal(modified_action, original_action)
    
    def test_action_smoothing(self, mock_env_with_detections, avoidance_config):
        """Test action smoothing to prevent jerky movements."""
        wrapper = ObjectAvoidanceActionWrapper(mock_env_with_detections, avoidance_config)
        
        # Set previous action
        wrapper._previous_action = np.array([0.5, 0.1])
        
        # Set object detection
        wrapper.env._last_observation = {
            'detections': [
                {
                    'class': 'person',
                    'distance': 0.4,
                    'relative_position': [0.0, 0.2]
                }
            ],
            'safety_critical': True
        }
        
        original_action = np.array([0.5, -0.2])  # Sudden steering change
        modified_action = wrapper.action(original_action)
        
        # Should be smoothed relative to previous action
        steering_change = abs(modified_action[1] - wrapper._previous_action[1])
        raw_steering_change = abs(original_action[1] - wrapper._previous_action[1])
        
        # Smoothed change should be less dramatic
        assert steering_change <= raw_steering_change
    
    def test_safety_constraints(self, mock_env_with_detections, avoidance_config):
        """Test safety constraints on modified actions."""
        wrapper = ObjectAvoidanceActionWrapper(mock_env_with_detections, avoidance_config)
        
        # Set very close object
        wrapper.env._last_observation = {
            'detections': [
                {
                    'class': 'person',
                    'distance': 0.1,  # Very close
                    'relative_position': [0.0, 0.1]
                }
            ],
            'safety_critical': True
        }
        
        original_action = np.array([1.0, 0.0])  # Full speed ahead
        modified_action = wrapper.action(original_action)
        
        # Should reduce speed when very close to object
        assert modified_action[0] < original_action[0]
        
        # Action should still be within valid range
        assert np.all(modified_action >= wrapper.action_space.low)
        assert np.all(modified_action <= wrapper.action_space.high)


class TestLaneChangingActionWrapper:
    """Comprehensive unit tests for Lane Changing Action Wrapper."""
    
    @pytest.fixture
    def mock_env_with_lane_info(self):
        """Mock environment with lane information."""
        env = MockEnvironment()
        env._lane_info = {
            'current_lane': 0,
            'lane_count': 2,
            'lane_positions': [0.0, 0.585],  # Two lanes
            'lane_occupancy': [0.0, 0.0]  # Both lanes free
        }
        return env
    
    @pytest.fixture
    def lane_changing_config(self):
        return LaneChangingConfig(
            lane_change_threshold=0.3,
            safety_margin=2.0,
            max_lane_change_time=3.0,
            min_lane_width=0.4,
            evaluation_distance=5.0
        )
    
    def test_wrapper_initialization(self, mock_env_with_lane_info, lane_changing_config):
        """Test lane changing wrapper initialization."""
        wrapper = LaneChangingActionWrapper(mock_env_with_lane_info, lane_changing_config)
        
        assert wrapper.env == mock_env_with_lane_info
        assert wrapper.config == lane_changing_config
        assert wrapper.lane_change_state == 'following'
        assert wrapper.target_lane is None
    
    def test_lane_following_state(self, mock_env_with_lane_info, lane_changing_config):
        """Test normal lane following behavior."""
        wrapper = LaneChangingActionWrapper(mock_env_with_lane_info, lane_changing_config)
        
        # No obstacles, should maintain lane following
        original_action = np.array([0.5, 0.1])
        modified_action = wrapper.action(original_action)
        
        # Action should be unchanged in lane following mode
        np.testing.assert_array_almost_equal(modified_action, original_action)
        assert wrapper.lane_change_state == 'following'
    
    def test_lane_change_initiation(self, mock_env_with_lane_info, lane_changing_config):
        """Test lane change initiation when obstacle detected."""
        wrapper = LaneChangingActionWrapper(mock_env_with_lane_info, lane_changing_config)
        
        # Simulate obstacle in current lane
        wrapper._obstacle_in_current_lane = True
        wrapper.env._lane_info['lane_occupancy'] = [0.8, 0.0]  # Current lane blocked, adjacent free
        
        original_action = np.array([0.5, 0.0])
        modified_action = wrapper.action(original_action)
        
        # Should initiate lane change
        assert wrapper.lane_change_state in ['evaluating', 'initiating']
    
    def test_safe_lane_evaluation(self, mock_env_with_lane_info, lane_changing_config):
        """Test safe lane evaluation logic."""
        wrapper = LaneChangingActionWrapper(mock_env_with_lane_info, lane_changing_config)
        
        # Test safe lane detection
        lane_occupancy = [0.8, 0.1, 0.9]  # Lane 1 is safest
        safe_lane = wrapper._find_safest_lane(lane_occupancy, current_lane=0)
        
        assert safe_lane == 1  # Should choose lane with lowest occupancy
    
    def test_lane_change_execution(self, mock_env_with_lane_info, lane_changing_config):
        """Test lane change execution."""
        wrapper = LaneChangingActionWrapper(mock_env_with_lane_info, lane_changing_config)
        
        # Set up lane change state
        wrapper.lane_change_state = 'executing'
        wrapper.target_lane = 1
        wrapper.lane_change_start_time = 0.0
        wrapper._current_time = 1.0  # 1 second into lane change
        
        original_action = np.array([0.5, 0.0])
        modified_action = wrapper.action(original_action)
        
        # Should modify steering for lane change
        assert abs(modified_action[1]) > abs(original_action[1])
    
    def test_lane_change_completion(self, mock_env_with_lane_info, lane_changing_config):
        """Test lane change completion detection."""
        wrapper = LaneChangingActionWrapper(mock_env_with_lane_info, lane_changing_config)
        
        # Set up completed lane change
        wrapper.lane_change_state = 'executing'
        wrapper.target_lane = 1
        wrapper.env._lane_info['current_lane'] = 1  # Successfully changed lanes
        
        original_action = np.array([0.5, 0.0])
        modified_action = wrapper.action(original_action)
        
        # Should return to lane following
        assert wrapper.lane_change_state == 'following'
        assert wrapper.target_lane is None
    
    def test_lane_change_timeout(self, mock_env_with_lane_info, lane_changing_config):
        """Test lane change timeout handling."""
        wrapper = LaneChangingActionWrapper(mock_env_with_lane_info, lane_changing_config)
        
        # Set up timed out lane change
        wrapper.lane_change_state = 'executing'
        wrapper.target_lane = 1
        wrapper.lane_change_start_time = 0.0
        wrapper._current_time = 4.0  # Exceeded max_lane_change_time
        
        original_action = np.array([0.5, 0.0])
        modified_action = wrapper.action(original_action)
        
        # Should abort lane change
        assert wrapper.lane_change_state == 'following'
        assert wrapper.target_lane is None
    
    def test_unsafe_lane_change_prevention(self, mock_env_with_lane_info, lane_changing_config):
        """Test prevention of unsafe lane changes."""
        wrapper = LaneChangingActionWrapper(mock_env_with_lane_info, lane_changing_config)
        
        # All lanes occupied
        wrapper.env._lane_info['lane_occupancy'] = [0.8, 0.9]
        wrapper._obstacle_in_current_lane = True
        
        original_action = np.array([0.5, 0.0])
        modified_action = wrapper.action(original_action)
        
        # Should not initiate lane change if no safe lane available
        assert wrapper.lane_change_state == 'following'


class TestMultiObjectiveRewardWrapper:
    """Comprehensive unit tests for Multi-Objective Reward Wrapper."""
    
    @pytest.fixture
    def mock_env_with_info(self):
        """Mock environment that provides detailed info."""
        env = MockEnvironment()
        
        def step_with_info(action):
            obs, reward, done, info = env.step(action)
            info.update({
                'lane_following_reward': 0.5,
                'object_avoidance_reward': 0.3,
                'lane_change_reward': 0.1,
                'efficiency_reward': 0.2,
                'safety_penalty': -0.1,
                'collision': False,
                'lane_change_success': False
            })
            return obs, reward, done, info
        
        env.step = step_with_info
        return env
    
    @pytest.fixture
    def reward_config(self):
        return {
            'lane_following_weight': 1.0,
            'object_avoidance_weight': 0.5,
            'lane_change_weight': 0.3,
            'efficiency_weight': 0.2,
            'safety_penalty_weight': -2.0,
            'collision_penalty': -10.0
        }
    
    def test_wrapper_initialization(self, mock_env_with_info, reward_config):
        """Test multi-objective reward wrapper initialization."""
        wrapper = MultiObjectiveRewardWrapper(mock_env_with_info, reward_config)
        
        assert wrapper.env == mock_env_with_info
        assert wrapper.reward_weights == reward_config
    
    def test_reward_component_calculation(self, mock_env_with_info, reward_config):
        """Test individual reward component calculations."""
        wrapper = MultiObjectiveRewardWrapper(mock_env_with_info, reward_config)
        
        # Test step with reward components
        obs, reward, done, info = wrapper.step(np.array([0.5, 0.0]))
        
        # Should have calculated multi-objective reward
        assert 'reward_components' in info
        assert isinstance(reward, float)
        
        # Check reward components
        components = info['reward_components']
        assert 'lane_following' in components
        assert 'object_avoidance' in components
        assert 'efficiency' in components
    
    def test_reward_weighting(self, mock_env_with_info, reward_config):
        """Test reward component weighting."""
        wrapper = MultiObjectiveRewardWrapper(mock_env_with_info, reward_config)
        
        # Mock specific reward components
        wrapper._calculate_lane_following_reward = Mock(return_value=0.8)
        wrapper._calculate_object_avoidance_reward = Mock(return_value=0.6)
        wrapper._calculate_efficiency_reward = Mock(return_value=0.4)
        wrapper._calculate_safety_penalty = Mock(return_value=-0.2)
        
        # Calculate weighted reward
        info = {
            'lane_following_reward': 0.8,
            'object_avoidance_reward': 0.6,
            'efficiency_reward': 0.4,
            'safety_penalty': -0.2
        }
        
        weighted_reward = wrapper._calculate_weighted_reward(info)
        
        expected_reward = (
            0.8 * 1.0 +  # lane_following
            0.6 * 0.5 +  # object_avoidance
            0.4 * 0.2 +  # efficiency
            -0.2 * -2.0  # safety_penalty
        )
        
        assert abs(weighted_reward - expected_reward) < 1e-6
    
    def test_collision_penalty(self, mock_env_with_info, reward_config):
        """Test collision penalty application."""
        wrapper = MultiObjectiveRewardWrapper(mock_env_with_info, reward_config)
        
        # Mock collision
        def step_with_collision(action):
            obs, reward, done, info = wrapper.env.step(action)
            info['collision'] = True
            return obs, reward, done, info
        
        wrapper.env.step = step_with_collision
        
        obs, reward, done, info = wrapper.step(np.array([0.5, 0.0]))
        
        # Should apply collision penalty
        assert reward <= reward_config['collision_penalty']
        assert info['reward_components']['collision_penalty'] == reward_config['collision_penalty']
    
    def test_lane_change_success_bonus(self, mock_env_with_info, reward_config):
        """Test lane change success bonus."""
        wrapper = MultiObjectiveRewardWrapper(mock_env_with_info, reward_config)
        
        # Mock successful lane change
        def step_with_lane_change(action):
            obs, reward, done, info = wrapper.env.step(action)
            info['lane_change_success'] = True
            return obs, reward, done, info
        
        wrapper.env.step = step_with_lane_change
        
        obs, reward, done, info = wrapper.step(np.array([0.5, 0.0]))
        
        # Should include lane change bonus
        assert 'lane_change_bonus' in info['reward_components']
        assert info['reward_components']['lane_change_bonus'] > 0
    
    def test_reward_normalization(self, mock_env_with_info, reward_config):
        """Test reward normalization to prevent extreme values."""
        wrapper = MultiObjectiveRewardWrapper(mock_env_with_info, reward_config)
        
        # Test with extreme reward components
        info = {
            'lane_following_reward': 100.0,  # Extreme value
            'object_avoidance_reward': -50.0,  # Extreme negative
            'efficiency_reward': 0.5,
            'safety_penalty': 0.0
        }
        
        normalized_reward = wrapper._calculate_weighted_reward(info)
        
        # Should be normalized/clipped to reasonable range
        assert -20.0 <= normalized_reward <= 20.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])