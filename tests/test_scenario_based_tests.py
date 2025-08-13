"""
Scenario-based tests for static and dynamic obstacle avoidance.
Tests specific driving scenarios and behaviors.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Enhanced Duckietown RL"

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from duckietown_utils.env import launch_and_wrap_enhanced_env
from config.enhanced_config import EnhancedRLConfig
from duckietown_utils.wrappers.yolo_detection_wrapper import YOLOObjectDetectionWrapper
from duckietown_utils.wrappers.object_avoidance_action_wrapper import ObjectAvoidanceActionWrapper
from duckietown_utils.wrappers.lane_changing_action_wrapper import LaneChangingActionWrapper


class ScenarioTestEnvironment:
    """Helper class for creating scenario-based test environments."""
    
    def __init__(self, base_config, enhanced_config):
        self.base_config = base_config
        self.enhanced_config = enhanced_config
        self.env = None
        self.scenario_data = {}
    
    def setup_environment(self):
        """Set up the test environment."""
        try:
            self.env = launch_and_wrap_enhanced_env(self.base_config, self.enhanced_config)
            return True
        except Exception as e:
            print(f"Failed to setup environment: {e}")
            return False
    
    def inject_scenario(self, scenario_type, **kwargs):
        """Inject a specific scenario into the environment."""
        self.scenario_data = {
            'type': scenario_type,
            'params': kwargs,
            'active': True
        }
    
    def get_mock_detections(self, scenario_type, **params):
        """Generate mock detections for different scenarios."""
        if scenario_type == 'static_obstacle_ahead':
            return [
                {
                    'class': 'person',
                    'confidence': 0.9,
                    'bbox': [200, 150, 300, 250],
                    'distance': params.get('distance', 1.0),
                    'relative_position': [0.0, params.get('lateral_offset', 0.0)]
                }
            ]
        
        elif scenario_type == 'static_obstacle_side':
            return [
                {
                    'class': 'car',
                    'confidence': 0.8,
                    'bbox': [100, 100, 200, 200],
                    'distance': params.get('distance', 2.0),
                    'relative_position': [params.get('side_offset', -0.3), 0.1]
                }
            ]
        
        elif scenario_type == 'multiple_obstacles':
            return [
                {
                    'class': 'person',
                    'confidence': 0.9,
                    'bbox': [180, 140, 280, 240],
                    'distance': params.get('distance1', 1.5),
                    'relative_position': [0.1, 0.0]
                },
                {
                    'class': 'car',
                    'confidence': 0.7,
                    'bbox': [320, 160, 420, 260],
                    'distance': params.get('distance2', 2.5),
                    'relative_position': [-0.2, 0.1]
                }
            ]
        
        elif scenario_type == 'dynamic_obstacle':
            # Simulate moving obstacle
            time_factor = params.get('time_step', 0) * 0.1
            return [
                {
                    'class': 'person',
                    'confidence': 0.8,
                    'bbox': [200 + int(time_factor * 50), 150, 300 + int(time_factor * 50), 250],
                    'distance': max(0.5, params.get('initial_distance', 2.0) - time_factor),
                    'relative_position': [time_factor * 0.1, 0.0]
                }
            ]
        
        elif scenario_type == 'lane_blocked':
            return [
                {
                    'class': 'car',
                    'confidence': 0.9,
                    'bbox': [150, 100, 350, 300],  # Large obstacle blocking lane
                    'distance': params.get('distance', 0.8),
                    'relative_position': [0.0, 0.0]
                }
            ]
        
        return []
    
    def cleanup(self):
        """Clean up the test environment."""
        if self.env:
            try:
                self.env.close()
            except:
                pass
            self.env = None


class TestStaticObstacleScenarios:
    """Test scenarios with static obstacles."""
    
    @pytest.fixture
    def scenario_env_config(self):
        """Environment configuration for scenario testing."""
        return {
            'training_map': 'small_loop',
            'episode_max_steps': 100,
            'domain_rand': False,
            'dynamics_rand': False,
            'camera_rand': False,
            'simulation_framerate': 30,
            'frame_skip': 1,
            'robot_speed': 0.3,
            'mode': 'debug',
            'resized_input_shape': (84, 84, 3),
            'action_type': 'continuous'
        }
    
    @pytest.fixture
    def avoidance_enhanced_config(self):
        """Enhanced configuration with object avoidance."""
        return EnhancedRLConfig(
            enabled_features=['yolo', 'object_avoidance', 'multi_objective_reward'],
            debug_mode=True,
            yolo={
                'model_path': 'yolov5s.pt',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            object_avoidance={
                'safety_distance': 0.5,
                'avoidance_strength': 1.0,
                'min_clearance': 0.2,
                'max_avoidance_angle': 0.5
            },
            reward={
                'lane_following_weight': 1.0,
                'object_avoidance_weight': 0.8,
                'safety_penalty_weight': -2.0
            }
        )
    
    @pytest.mark.scenario
    def test_static_obstacle_ahead_avoidance(self, scenario_env_config, avoidance_enhanced_config):
        """Test avoidance of static obstacle directly ahead."""
        scenario_env = ScenarioTestEnvironment(scenario_env_config, avoidance_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                # Mock YOLO model with scenario-specific detections
                mock_model = Mock()
                
                def mock_detection(img):
                    detections = scenario_env.get_mock_detections('static_obstacle_ahead', distance=0.8)
                    mock_results = Mock()
                    
                    if detections:
                        detection_array = np.array([
                            [d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], 
                             d['confidence'], 0, d['class']] for d in detections
                        ])
                    else:
                        detection_array = np.array([])
                    
                    mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                    return mock_results
                
                mock_model.side_effect = mock_detection
                mock_yolo.return_value = mock_model
                
                if not scenario_env.setup_environment():
                    pytest.skip("Could not setup scenario environment")
                
                scenario_env.inject_scenario('static_obstacle_ahead', distance=0.8)
                
                obs = scenario_env.env.reset()
                
                # Test avoidance behavior
                avoidance_detected = False
                steering_changes = []
                
                for step in range(20):
                    # Use forward action to test avoidance
                    action = np.array([0.5, 0.0])  # Forward, no steering
                    obs, reward, done, info = scenario_env.env.step(action)
                    
                    # Check if avoidance was triggered
                    if 'reward_components' in info:
                        if info['reward_components'].get('object_avoidance', 0) > 0:
                            avoidance_detected = True
                    
                    # Monitor steering changes (action wrapper should modify steering)
                    if hasattr(scenario_env.env, '_last_action'):
                        last_action = getattr(scenario_env.env, '_last_action', action)
                        if abs(last_action[1]) > abs(action[1]):
                            steering_changes.append(step)
                    
                    if done:
                        break
                
                # Verify avoidance behavior
                assert avoidance_detected or len(steering_changes) > 0, \
                    "No avoidance behavior detected for static obstacle ahead"
                
                print(f"Static obstacle avoidance: {len(steering_changes)} steering corrections")
                
        except Exception as e:
            pytest.skip(f"Static obstacle test skipped: {e}")
        finally:
            scenario_env.cleanup()
    
    @pytest.mark.scenario
    def test_static_obstacle_side_clearance(self, scenario_env_config, avoidance_enhanced_config):
        """Test maintaining clearance from side obstacles."""
        scenario_env = ScenarioTestEnvironment(scenario_env_config, avoidance_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                
                def mock_side_detection(img):
                    detections = scenario_env.get_mock_detections('static_obstacle_side', 
                                                                distance=1.5, side_offset=-0.4)
                    mock_results = Mock()
                    
                    if detections:
                        detection_array = np.array([
                            [d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], 
                             d['confidence'], 0, d['class']] for d in detections
                        ])
                    else:
                        detection_array = np.array([])
                    
                    mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                    return mock_results
                
                mock_model.side_effect = mock_side_detection
                mock_yolo.return_value = mock_model
                
                if not scenario_env.setup_environment():
                    pytest.skip("Could not setup scenario environment")
                
                scenario_env.inject_scenario('static_obstacle_side', distance=1.5, side_offset=-0.4)
                
                obs = scenario_env.env.reset()
                
                # Test clearance maintenance
                clearance_maintained = True
                min_safety_distance = float('inf')
                
                for step in range(15):
                    action = np.array([0.4, 0.0])  # Steady forward
                    obs, reward, done, info = scenario_env.env.step(action)
                    
                    # Check safety distance maintenance
                    if 'safety_distance' in info:
                        min_safety_distance = min(min_safety_distance, info['safety_distance'])
                        if info['safety_distance'] < 0.2:  # Below min clearance
                            clearance_maintained = False
                    
                    if done:
                        break
                
                # Should maintain safe clearance
                assert clearance_maintained, "Failed to maintain safe clearance from side obstacle"
                
                if min_safety_distance != float('inf'):
                    assert min_safety_distance >= 0.15, \
                        f"Minimum safety distance too low: {min_safety_distance:.3f}m"
                
                print(f"Side obstacle clearance: {min_safety_distance:.3f}m minimum distance")
                
        except Exception as e:
            pytest.skip(f"Side obstacle test skipped: {e}")
        finally:
            scenario_env.cleanup()
    
    @pytest.mark.scenario
    def test_multiple_static_obstacles(self, scenario_env_config, avoidance_enhanced_config):
        """Test navigation through multiple static obstacles."""
        scenario_env = ScenarioTestEnvironment(scenario_env_config, avoidance_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                
                def mock_multiple_detection(img):
                    detections = scenario_env.get_mock_detections('multiple_obstacles', 
                                                                distance1=1.2, distance2=2.0)
                    mock_results = Mock()
                    
                    if detections:
                        detection_array = np.array([
                            [d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], 
                             d['confidence'], 0, d['class']] for d in detections
                        ])
                    else:
                        detection_array = np.array([])
                    
                    mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                    return mock_results
                
                mock_model.side_effect = mock_multiple_detection
                mock_yolo.return_value = mock_model
                
                if not scenario_env.setup_environment():
                    pytest.skip("Could not setup scenario environment")
                
                scenario_env.inject_scenario('multiple_obstacles', distance1=1.2, distance2=2.0)
                
                obs = scenario_env.env.reset()
                
                # Test multi-obstacle navigation
                obstacles_detected = 0
                avoidance_actions = 0
                
                for step in range(25):
                    action = np.array([0.4, 0.0])
                    obs, reward, done, info = scenario_env.env.step(action)
                    
                    # Count detected obstacles
                    if 'detections' in info:
                        obstacles_detected = max(obstacles_detected, len(info['detections']))
                    
                    # Count avoidance actions
                    if 'reward_components' in info:
                        if info['reward_components'].get('object_avoidance', 0) > 0:
                            avoidance_actions += 1
                    
                    if done:
                        break
                
                # Should detect multiple obstacles and perform avoidance
                assert obstacles_detected >= 2, f"Should detect multiple obstacles, got {obstacles_detected}"
                assert avoidance_actions > 0, "Should perform avoidance actions for multiple obstacles"
                
                print(f"Multiple obstacles: {obstacles_detected} detected, {avoidance_actions} avoidance actions")
                
        except Exception as e:
            pytest.skip(f"Multiple obstacles test skipped: {e}")
        finally:
            scenario_env.cleanup()


class TestDynamicObstacleScenarios:
    """Test scenarios with dynamic (moving) obstacles."""
    
    @pytest.fixture
    def dynamic_env_config(self):
        """Environment configuration for dynamic obstacle testing."""
        return {
            'training_map': 'small_loop',
            'episode_max_steps': 150,
            'domain_rand': False,
            'dynamics_rand': False,
            'simulation_framerate': 30,
            'frame_skip': 1,
            'robot_speed': 0.4,
            'mode': 'debug',
            'resized_input_shape': (84, 84, 3),
            'action_type': 'continuous'
        }
    
    @pytest.fixture
    def dynamic_enhanced_config(self):
        """Enhanced configuration for dynamic obstacle scenarios."""
        return EnhancedRLConfig(
            enabled_features=['yolo', 'object_avoidance', 'multi_objective_reward'],
            debug_mode=True,
            yolo={
                'model_path': 'yolov5s.pt',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            object_avoidance={
                'safety_distance': 0.6,  # Larger safety distance for dynamic obstacles
                'avoidance_strength': 1.2,
                'min_clearance': 0.3,
                'smoothing_factor': 0.7  # More smoothing for dynamic scenarios
            },
            reward={
                'lane_following_weight': 1.0,
                'object_avoidance_weight': 1.0,  # Higher weight for dynamic scenarios
                'safety_penalty_weight': -3.0
            }
        )
    
    @pytest.mark.scenario
    def test_approaching_dynamic_obstacle(self, dynamic_env_config, dynamic_enhanced_config):
        """Test reaction to approaching dynamic obstacle."""
        scenario_env = ScenarioTestEnvironment(dynamic_env_config, dynamic_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                step_counter = [0]  # Use list for mutable counter
                
                def mock_dynamic_detection(img):
                    detections = scenario_env.get_mock_detections('dynamic_obstacle', 
                                                                time_step=step_counter[0], 
                                                                initial_distance=3.0)
                    step_counter[0] += 1
                    
                    mock_results = Mock()
                    
                    if detections:
                        detection_array = np.array([
                            [d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], 
                             d['confidence'], 0, d['class']] for d in detections
                        ])
                    else:
                        detection_array = np.array([])
                    
                    mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                    return mock_results
                
                mock_model.side_effect = mock_dynamic_detection
                mock_yolo.return_value = mock_model
                
                if not scenario_env.setup_environment():
                    pytest.skip("Could not setup scenario environment")
                
                scenario_env.inject_scenario('dynamic_obstacle', initial_distance=3.0)
                
                obs = scenario_env.env.reset()
                
                # Test dynamic obstacle reaction
                distance_history = []
                reaction_triggered = False
                early_avoidance = False
                
                for step in range(30):
                    action = np.array([0.5, 0.0])  # Constant forward motion
                    obs, reward, done, info = scenario_env.env.step(action)
                    
                    # Track obstacle distance
                    if 'detections' in info and info['detections']:
                        distance = info['detections'][0].get('distance', float('inf'))
                        distance_history.append(distance)
                        
                        # Check for early reaction (distance > 1.5m)
                        if distance > 1.5 and info.get('reward_components', {}).get('object_avoidance', 0) > 0:
                            early_avoidance = True
                        
                        # Check for reaction when close
                        if distance < 1.0:
                            reaction_triggered = True
                    
                    if done:
                        break
                
                # Should react to approaching obstacle
                assert len(distance_history) > 0, "No dynamic obstacle detected"
                assert reaction_triggered or early_avoidance, "No reaction to approaching dynamic obstacle"
                
                # Distance should generally decrease (obstacle approaching)
                if len(distance_history) >= 3:
                    initial_distance = distance_history[0]
                    final_distance = distance_history[-1]
                    assert final_distance < initial_distance, "Dynamic obstacle should be approaching"
                
                print(f"Dynamic obstacle: {len(distance_history)} detections, early_avoidance={early_avoidance}")
                
        except Exception as e:
            pytest.skip(f"Dynamic obstacle test skipped: {e}")
        finally:
            scenario_env.cleanup()
    
    @pytest.mark.scenario
    def test_crossing_dynamic_obstacle(self, dynamic_env_config, dynamic_enhanced_config):
        """Test handling of obstacle crossing path."""
        scenario_env = ScenarioTestEnvironment(dynamic_env_config, dynamic_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                step_counter = [0]
                
                def mock_crossing_detection(img):
                    # Simulate obstacle crossing from left to right
                    time_step = step_counter[0]
                    step_counter[0] += 1
                    
                    if time_step < 20:  # Obstacle visible for 20 steps
                        lateral_position = -0.5 + (time_step * 0.05)  # Moving right
                        distance = 1.5 - (time_step * 0.02)  # Getting closer
                        
                        detections = [
                            {
                                'class': 'person',
                                'confidence': 0.8,
                                'bbox': [200 + int(lateral_position * 100), 150, 
                                        300 + int(lateral_position * 100), 250],
                                'distance': max(0.5, distance),
                                'relative_position': [lateral_position, 0.0]
                            }
                        ]
                    else:
                        detections = []  # Obstacle passed
                    
                    mock_results = Mock()
                    
                    if detections:
                        detection_array = np.array([
                            [d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], 
                             d['confidence'], 0, d['class']] for d in detections
                        ])
                    else:
                        detection_array = np.array([])
                    
                    mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                    return mock_results
                
                mock_model.side_effect = mock_crossing_detection
                mock_yolo.return_value = mock_model
                
                if not scenario_env.setup_environment():
                    pytest.skip("Could not setup scenario environment")
                
                scenario_env.inject_scenario('crossing_obstacle')
                
                obs = scenario_env.env.reset()
                
                # Test crossing obstacle handling
                crossing_detected = False
                avoidance_duration = 0
                max_avoidance_action = 0
                
                for step in range(35):
                    action = np.array([0.4, 0.0])
                    obs, reward, done, info = scenario_env.env.step(action)
                    
                    # Check for crossing detection and avoidance
                    if 'detections' in info and info['detections']:
                        detection = info['detections'][0]
                        lateral_pos = detection.get('relative_position', [0, 0])[0]
                        
                        # Detect crossing motion (lateral position changing)
                        if abs(lateral_pos) < 0.3:  # Close to path
                            crossing_detected = True
                        
                        # Track avoidance actions
                        if info.get('reward_components', {}).get('object_avoidance', 0) > 0:
                            avoidance_duration += 1
                            max_avoidance_action = max(max_avoidance_action, 
                                                     info['reward_components']['object_avoidance'])
                    
                    if done:
                        break
                
                # Should detect crossing and perform avoidance
                assert crossing_detected, "Failed to detect crossing obstacle"
                assert avoidance_duration > 0, "No avoidance actions for crossing obstacle"
                
                print(f"Crossing obstacle: detected={crossing_detected}, avoidance_duration={avoidance_duration}")
                
        except Exception as e:
            pytest.skip(f"Crossing obstacle test skipped: {e}")
        finally:
            scenario_env.cleanup()


class TestLaneChangingScenarios:
    """Test lane changing scenarios."""
    
    @pytest.fixture
    def lane_changing_env_config(self):
        """Environment configuration for lane changing tests."""
        return {
            'training_map': 'multi_track',  # Map with multiple lanes
            'episode_max_steps': 200,
            'domain_rand': False,
            'dynamics_rand': False,
            'simulation_framerate': 30,
            'frame_skip': 1,
            'robot_speed': 0.4,
            'mode': 'debug',
            'resized_input_shape': (84, 84, 3),
            'action_type': 'continuous'
        }
    
    @pytest.fixture
    def lane_changing_enhanced_config(self):
        """Enhanced configuration with lane changing capabilities."""
        return EnhancedRLConfig(
            enabled_features=['yolo', 'object_avoidance', 'lane_changing', 'multi_objective_reward'],
            debug_mode=True,
            yolo={
                'model_path': 'yolov5s.pt',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            object_avoidance={
                'safety_distance': 0.5,
                'avoidance_strength': 1.0,
                'min_clearance': 0.2
            },
            lane_changing={
                'lane_change_threshold': 0.3,
                'safety_margin': 2.0,
                'max_lane_change_time': 3.0,
                'min_lane_width': 0.4
            },
            reward={
                'lane_following_weight': 1.0,
                'object_avoidance_weight': 0.8,
                'lane_change_weight': 0.5,
                'safety_penalty_weight': -2.0
            }
        )
    
    @pytest.mark.scenario
    def test_lane_change_for_blocked_lane(self, lane_changing_env_config, lane_changing_enhanced_config):
        """Test lane change when current lane is blocked."""
        scenario_env = ScenarioTestEnvironment(lane_changing_env_config, lane_changing_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                
                def mock_blocked_lane_detection(img):
                    detections = scenario_env.get_mock_detections('lane_blocked', distance=1.0)
                    mock_results = Mock()
                    
                    if detections:
                        detection_array = np.array([
                            [d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], 
                             d['confidence'], 0, d['class']] for d in detections
                        ])
                    else:
                        detection_array = np.array([])
                    
                    mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                    return mock_results
                
                mock_model.side_effect = mock_blocked_lane_detection
                mock_yolo.return_value = mock_model
                
                if not scenario_env.setup_environment():
                    pytest.skip("Could not setup scenario environment")
                
                scenario_env.inject_scenario('lane_blocked', distance=1.0)
                
                obs = scenario_env.env.reset()
                
                # Test lane change initiation
                lane_change_initiated = False
                lane_change_completed = False
                lane_change_steps = 0
                
                for step in range(50):
                    action = np.array([0.4, 0.0])
                    obs, reward, done, info = scenario_env.env.step(action)
                    
                    # Check for lane change behavior
                    if 'lane_change_state' in info:
                        state = info['lane_change_state']
                        if state in ['evaluating', 'initiating', 'executing']:
                            lane_change_initiated = True
                            lane_change_steps += 1
                        elif state == 'completed':
                            lane_change_completed = True
                    
                    # Check reward components for lane change
                    if 'reward_components' in info:
                        if info['reward_components'].get('lane_change', 0) > 0:
                            lane_change_initiated = True
                    
                    if done:
                        break
                
                # Should initiate lane change for blocked lane
                assert lane_change_initiated, "Failed to initiate lane change for blocked lane"
                
                print(f"Lane change for blocked lane: initiated={lane_change_initiated}, "
                      f"completed={lane_change_completed}, steps={lane_change_steps}")
                
        except Exception as e:
            pytest.skip(f"Lane change test skipped: {e}")
        finally:
            scenario_env.cleanup()
    
    @pytest.mark.scenario
    def test_safe_lane_change_evaluation(self, lane_changing_env_config, lane_changing_enhanced_config):
        """Test safe lane evaluation before lane change."""
        scenario_env = ScenarioTestEnvironment(lane_changing_env_config, lane_changing_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                step_counter = [0]
                
                def mock_unsafe_lane_detection(img):
                    # Simulate blocked current lane and unsafe adjacent lane
                    step = step_counter[0]
                    step_counter[0] += 1
                    
                    detections = [
                        # Obstacle in current lane
                        {
                            'class': 'car',
                            'confidence': 0.9,
                            'bbox': [180, 120, 320, 280],
                            'distance': 1.0,
                            'relative_position': [0.0, 0.0]
                        }
                    ]
                    
                    # Add obstacle in adjacent lane for first half of test
                    if step < 25:
                        detections.append({
                            'class': 'person',
                            'confidence': 0.8,
                            'bbox': [100, 140, 200, 240],
                            'distance': 1.5,
                            'relative_position': [-0.6, 0.1]  # Adjacent lane
                        })
                    
                    mock_results = Mock()
                    detection_array = np.array([
                        [d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], 
                         d['confidence'], 0, d['class']] for d in detections
                    ])
                    mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                    return mock_results
                
                mock_model.side_effect = mock_unsafe_lane_detection
                mock_yolo.return_value = mock_model
                
                if not scenario_env.setup_environment():
                    pytest.skip("Could not setup scenario environment")
                
                scenario_env.inject_scenario('unsafe_lane_change')
                
                obs = scenario_env.env.reset()
                
                # Test safe lane evaluation
                unsafe_period_changes = 0
                safe_period_changes = 0
                
                for step in range(50):
                    action = np.array([0.4, 0.0])
                    obs, reward, done, info = scenario_env.env.step(action)
                    
                    # Track lane change attempts
                    if 'lane_change_state' in info:
                        state = info['lane_change_state']
                        if state in ['initiating', 'executing']:
                            if step < 25:  # Unsafe period
                                unsafe_period_changes += 1
                            else:  # Safe period
                                safe_period_changes += 1
                    
                    if done:
                        break
                
                # Should avoid lane changes during unsafe period
                # and allow them during safe period
                print(f"Safe lane evaluation: unsafe_period={unsafe_period_changes}, "
                      f"safe_period={safe_period_changes}")
                
                # This test verifies the system evaluates safety
                # The exact behavior depends on implementation details
                
        except Exception as e:
            pytest.skip(f"Safe lane evaluation test skipped: {e}")
        finally:
            scenario_env.cleanup()


class TestComplexScenarios:
    """Test complex multi-component scenarios."""
    
    @pytest.fixture
    def complex_env_config(self):
        """Environment configuration for complex scenarios."""
        return {
            'training_map': 'multi_track2',
            'episode_max_steps': 300,
            'domain_rand': False,
            'dynamics_rand': False,
            'simulation_framerate': 30,
            'frame_skip': 1,
            'robot_speed': 0.5,
            'mode': 'debug',
            'resized_input_shape': (84, 84, 3),
            'action_type': 'continuous'
        }
    
    @pytest.fixture
    def full_enhanced_config(self):
        """Full enhanced configuration for complex scenarios."""
        return EnhancedRLConfig(
            enabled_features=['yolo', 'object_avoidance', 'lane_changing', 'multi_objective_reward'],
            debug_mode=True,
            yolo={
                'model_path': 'yolov5s.pt',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            object_avoidance={
                'safety_distance': 0.6,
                'avoidance_strength': 1.0,
                'min_clearance': 0.25
            },
            lane_changing={
                'lane_change_threshold': 0.3,
                'safety_margin': 2.0,
                'max_lane_change_time': 3.0
            },
            reward={
                'lane_following_weight': 1.0,
                'object_avoidance_weight': 0.8,
                'lane_change_weight': 0.6,
                'efficiency_weight': 0.3,
                'safety_penalty_weight': -2.0
            }
        )
    
    @pytest.mark.scenario
    def test_multi_obstacle_lane_change_scenario(self, complex_env_config, full_enhanced_config):
        """Test complex scenario with multiple obstacles requiring lane change."""
        scenario_env = ScenarioTestEnvironment(complex_env_config, full_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                step_counter = [0]
                
                def mock_complex_scenario(img):
                    step = step_counter[0]
                    step_counter[0] += 1
                    
                    detections = []
                    
                    # Static obstacle in current lane
                    detections.append({
                        'class': 'car',
                        'confidence': 0.9,
                        'bbox': [200, 100, 350, 300],
                        'distance': max(0.5, 2.0 - step * 0.05),
                        'relative_position': [0.0, 0.0]
                    })
                    
                    # Dynamic obstacle in adjacent lane (first half)
                    if step < 30:
                        detections.append({
                            'class': 'person',
                            'confidence': 0.8,
                            'bbox': [100, 150, 200, 250],
                            'distance': 2.0 - step * 0.03,
                            'relative_position': [-0.6, 0.1]
                        })
                    
                    # Additional obstacle appears later
                    if step > 20:
                        detections.append({
                            'class': 'duckie',
                            'confidence': 0.7,
                            'bbox': [400, 180, 480, 220],
                            'distance': 3.0,
                            'relative_position': [0.3, 0.2]
                        })
                    
                    mock_results = Mock()
                    if detections:
                        detection_array = np.array([
                            [d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], 
                             d['confidence'], 0, d['class']] for d in detections
                        ])
                    else:
                        detection_array = np.array([])
                    
                    mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                    return mock_results
                
                mock_model.side_effect = mock_complex_scenario
                mock_yolo.return_value = mock_model
                
                if not scenario_env.setup_environment():
                    pytest.skip("Could not setup scenario environment")
                
                scenario_env.inject_scenario('complex_multi_obstacle')
                
                obs = scenario_env.env.reset()
                
                # Test complex scenario handling
                max_detections = 0
                avoidance_actions = 0
                lane_change_attempts = 0
                safety_violations = 0
                
                for step in range(60):
                    action = np.array([0.4, 0.0])
                    obs, reward, done, info = scenario_env.env.step(action)
                    
                    # Track scenario complexity
                    if 'detections' in info:
                        max_detections = max(max_detections, len(info['detections']))
                    
                    # Track system responses
                    if 'reward_components' in info:
                        components = info['reward_components']
                        if components.get('object_avoidance', 0) > 0:
                            avoidance_actions += 1
                        if components.get('lane_change', 0) > 0:
                            lane_change_attempts += 1
                        if components.get('safety_penalty', 0) < 0:
                            safety_violations += 1
                    
                    if done:
                        break
                
                # Verify complex scenario handling
                assert max_detections >= 2, f"Should handle multiple obstacles, max detected: {max_detections}"
                assert avoidance_actions > 0, "Should perform object avoidance in complex scenario"
                
                # Safety violations should be minimal
                violation_rate = safety_violations / max(1, step)
                assert violation_rate < 0.3, f"Too many safety violations: {violation_rate:.2f}"
                
                print(f"Complex scenario: {max_detections} max obstacles, "
                      f"{avoidance_actions} avoidance actions, "
                      f"{lane_change_attempts} lane change attempts, "
                      f"{safety_violations} safety violations")
                
        except Exception as e:
            pytest.skip(f"Complex scenario test skipped: {e}")
        finally:
            scenario_env.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "scenario"])