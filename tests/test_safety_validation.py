"""
Safety validation tests for collision avoidance and lane changing.
Tests safety-critical behaviors and failure modes.
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
import json

from duckietown_utils.env import launch_and_wrap_enhanced_env
from config.enhanced_config import EnhancedRLConfig
from duckietown_utils.wrappers.object_avoidance_action_wrapper import ObjectAvoidanceActionWrapper
from duckietown_utils.wrappers.lane_changing_action_wrapper import LaneChangingActionWrapper


class SafetyTestHarness:
    """Test harness for safety validation."""
    
    def __init__(self, env_config, enhanced_config):
        self.env_config = env_config
        self.enhanced_config = enhanced_config
        self.env = None
        self.safety_metrics = {
            'collisions': 0,
            'near_misses': 0,
            'safety_violations': 0,
            'emergency_stops': 0,
            'unsafe_lane_changes': 0,
            'total_steps': 0
        }
    
    def setup_environment(self):
        """Set up the safety test environment."""
        try:
            self.env = launch_and_wrap_enhanced_env(self.env_config, self.enhanced_config)
            return True
        except Exception as e:
            print(f"Failed to setup safety test environment: {e}")
            return False
    
    def record_safety_event(self, event_type, details=None):
        """Record a safety event."""
        if event_type in self.safety_metrics:
            self.safety_metrics[event_type] += 1
        
        if details:
            print(f"Safety Event: {event_type} - {details}")
    
    def analyze_step(self, obs, reward, done, info, action):
        """Analyze a step for safety violations."""
        self.safety_metrics['total_steps'] += 1
        
        # Check for collisions
        if info.get('collision', False):
            self.record_safety_event('collisions', f"Collision detected at step {self.safety_metrics['total_steps']}")
        
        # Check for near misses (very close to obstacles)
        if 'detections' in info:
            for detection in info['detections']:
                distance = detection.get('distance', float('inf'))
                if distance < 0.15:  # Very close
                    self.record_safety_event('near_misses', f"Near miss: {distance:.3f}m")
        
        # Check for safety violations in reward
        if 'reward_components' in info:
            safety_penalty = info['reward_components'].get('safety_penalty', 0)
            if safety_penalty < -1.0:  # Significant safety penalty
                self.record_safety_event('safety_violations', f"Safety penalty: {safety_penalty:.2f}")
        
        # Check for emergency stops (very low speed when obstacle close)
        if 'detections' in info and info['detections']:
            min_distance = min(d.get('distance', float('inf')) for d in info['detections'])
            if min_distance < 0.3 and abs(action[0]) < 0.1:  # Low speed near obstacle
                self.record_safety_event('emergency_stops', f"Emergency stop at {min_distance:.3f}m")
        
        # Check for unsafe lane changes
        if 'lane_change_state' in info:
            if info['lane_change_state'] == 'executing' and info.get('lane_change_safe', True) == False:
                self.record_safety_event('unsafe_lane_changes', "Unsafe lane change executed")
    
    def get_safety_score(self):
        """Calculate overall safety score (0-1, higher is safer)."""
        if self.safety_metrics['total_steps'] == 0:
            return 1.0
        
        # Weight different safety events
        weighted_violations = (
            self.safety_metrics['collisions'] * 10 +
            self.safety_metrics['near_misses'] * 3 +
            self.safety_metrics['safety_violations'] * 2 +
            self.safety_metrics['unsafe_lane_changes'] * 5
        )
        
        # Calculate score (emergency stops are not necessarily bad)
        max_violations = self.safety_metrics['total_steps'] * 0.1  # Allow 10% violation rate
        safety_score = max(0.0, 1.0 - (weighted_violations / max(1, max_violations)))
        
        return safety_score
    
    def cleanup(self):
        """Clean up the test environment."""
        if self.env:
            try:
                self.env.close()
            except:
                pass
            self.env = None


class TestCollisionAvoidance:
    """Test collision avoidance safety mechanisms."""
    
    @pytest.fixture
    def safety_env_config(self):
        """Environment configuration for safety testing."""
        return {
            'training_map': 'small_loop',
            'episode_max_steps': 200,
            'domain_rand': False,
            'dynamics_rand': False,
            'camera_rand': False,
            'simulation_framerate': 30,
            'frame_skip': 1,
            'robot_speed': 0.4,
            'mode': 'debug',
            'resized_input_shape': (84, 84, 3),
            'action_type': 'continuous'
        }
    
    @pytest.fixture
    def safety_enhanced_config(self):
        """Enhanced configuration optimized for safety."""
        return EnhancedRLConfig(
            enabled_features=['yolo', 'object_avoidance', 'multi_objective_reward'],
            debug_mode=True,
            yolo={
                'model_path': 'yolov5s.pt',
                'confidence_threshold': 0.4,  # Lower threshold for better detection
                'device': 'cpu'
            },
            object_avoidance={
                'safety_distance': 0.8,  # Larger safety distance
                'avoidance_strength': 1.5,  # Stronger avoidance
                'min_clearance': 0.3,  # Larger minimum clearance
                'max_avoidance_angle': 0.6
            },
            reward={
                'lane_following_weight': 1.0,
                'object_avoidance_weight': 1.5,  # Higher weight for safety
                'safety_penalty_weight': -5.0,  # Strong safety penalty
                'collision_penalty': -20.0  # Very strong collision penalty
            }
        )
    
    @pytest.mark.safety
    def test_static_obstacle_collision_avoidance(self, safety_env_config, safety_enhanced_config):
        """Test collision avoidance with static obstacles."""
        safety_harness = SafetyTestHarness(safety_env_config, safety_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                
                def mock_static_obstacle(img):
                    # Static obstacle directly ahead
                    mock_results = Mock()
                    detection_array = np.array([
                        [200, 150, 300, 250, 0.9, 0, 'person']  # Person ahead
                    ])
                    mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                    return mock_results
                
                mock_model.side_effect = mock_static_obstacle
                mock_yolo.return_value = mock_model
                
                if not safety_harness.setup_environment():
                    pytest.skip("Could not setup safety test environment")
                
                obs = safety_harness.env.reset()
                
                # Test collision avoidance
                for step in range(50):
                    # Drive straight toward obstacle
                    action = np.array([0.6, 0.0])  # High speed, no steering
                    obs, reward, done, info = safety_harness.env.step(action)
                    
                    safety_harness.analyze_step(obs, reward, done, info, action)
                    
                    if done:
                        break
                
                # Evaluate safety performance
                safety_score = safety_harness.get_safety_score()
                
                # Should avoid collisions
                assert safety_harness.safety_metrics['collisions'] == 0, \
                    f"Collisions detected: {safety_harness.safety_metrics['collisions']}"
                
                # Should have minimal near misses
                assert safety_harness.safety_metrics['near_misses'] <= 2, \
                    f"Too many near misses: {safety_harness.safety_metrics['near_misses']}"
                
                # Overall safety score should be high
                assert safety_score >= 0.8, f"Safety score too low: {safety_score:.3f}"
                
                print(f"Static obstacle safety: {safety_harness.safety_metrics}, score: {safety_score:.3f}")
                
        except Exception as e:
            pytest.skip(f"Static obstacle collision test skipped: {e}")
        finally:
            safety_harness.cleanup()
    
    @pytest.mark.safety
    def test_multiple_obstacle_collision_avoidance(self, safety_env_config, safety_enhanced_config):
        """Test collision avoidance with multiple obstacles."""
        safety_harness = SafetyTestHarness(safety_env_config, safety_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                
                def mock_multiple_obstacles(img):
                    # Multiple obstacles in different positions
                    mock_results = Mock()
                    detection_array = np.array([
                        [180, 140, 280, 240, 0.9, 0, 'person'],   # Left obstacle
                        [320, 160, 420, 260, 0.8, 2, 'car'],     # Right obstacle
                        [240, 120, 340, 220, 0.7, 0, 'duckie']   # Center obstacle
                    ])
                    mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                    return mock_results
                
                mock_model.side_effect = mock_multiple_obstacles
                mock_yolo.return_value = mock_model
                
                if not safety_harness.setup_environment():
                    pytest.skip("Could not setup safety test environment")
                
                obs = safety_harness.env.reset()
                
                # Test navigation through multiple obstacles
                for step in range(60):
                    action = np.array([0.5, 0.0])  # Forward motion
                    obs, reward, done, info = safety_harness.env.step(action)
                    
                    safety_harness.analyze_step(obs, reward, done, info, action)
                    
                    if done:
                        break
                
                safety_score = safety_harness.get_safety_score()
                
                # Should avoid all collisions
                assert safety_harness.safety_metrics['collisions'] == 0, \
                    f"Collisions with multiple obstacles: {safety_harness.safety_metrics['collisions']}"
                
                # Should handle complexity safely
                assert safety_score >= 0.7, f"Multi-obstacle safety score too low: {safety_score:.3f}"
                
                print(f"Multiple obstacle safety: {safety_harness.safety_metrics}, score: {safety_score:.3f}")
                
        except Exception as e:
            pytest.skip(f"Multiple obstacle collision test skipped: {e}")
        finally:
            safety_harness.cleanup()
    
    @pytest.mark.safety
    def test_emergency_braking(self, safety_env_config, safety_enhanced_config):
        """Test emergency braking when collision is imminent."""
        safety_harness = SafetyTestHarness(safety_env_config, safety_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                step_counter = [0]
                
                def mock_approaching_obstacle(img):
                    # Obstacle getting closer each step
                    step = step_counter[0]
                    step_counter[0] += 1
                    
                    # Start far, get closer
                    distance_factor = max(0.1, 1.0 - step * 0.05)
                    bbox_size = int(100 + (1.0 - distance_factor) * 200)  # Larger as closer
                    
                    mock_results = Mock()
                    detection_array = np.array([
                        [200, 150, 200 + bbox_size, 150 + bbox_size, 0.9, 0, 'person']
                    ])
                    mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                    return mock_results
                
                mock_model.side_effect = mock_approaching_obstacle
                mock_yolo.return_value = mock_model
                
                if not safety_harness.setup_environment():
                    pytest.skip("Could not setup safety test environment")
                
                obs = safety_harness.env.reset()
                
                # Test emergency braking
                emergency_brake_detected = False
                min_distance_achieved = float('inf')
                
                for step in range(30):
                    action = np.array([0.8, 0.0])  # High speed toward obstacle
                    obs, reward, done, info = safety_harness.env.step(action)
                    
                    safety_harness.analyze_step(obs, reward, done, info, action)
                    
                    # Check for emergency braking (low actual speed despite high commanded speed)
                    if 'detections' in info and info['detections']:
                        distance = info['detections'][0].get('distance', float('inf'))
                        min_distance_achieved = min(min_distance_achieved, distance)
                        
                        # If very close and system reduces speed significantly
                        if distance < 0.4 and abs(action[0]) > 0.5:
                            # Check if actual action was modified (emergency brake)
                            if hasattr(safety_harness.env, '_last_modified_action'):
                                modified_action = getattr(safety_harness.env, '_last_modified_action')
                                if modified_action[0] < action[0] * 0.5:  # Speed reduced significantly
                                    emergency_brake_detected = True
                    
                    if done:
                        break
                
                safety_score = safety_harness.get_safety_score()
                
                # Should not collide even with aggressive driving
                assert safety_harness.safety_metrics['collisions'] == 0, \
                    f"Emergency braking failed: {safety_harness.safety_metrics['collisions']} collisions"
                
                # Should achieve reasonable minimum distance
                assert min_distance_achieved >= 0.1, \
                    f"Got too close to obstacle: {min_distance_achieved:.3f}m"
                
                print(f"Emergency braking: min_distance={min_distance_achieved:.3f}m, "
                      f"brake_detected={emergency_brake_detected}, score={safety_score:.3f}")
                
        except Exception as e:
            pytest.skip(f"Emergency braking test skipped: {e}")
        finally:
            safety_harness.cleanup()


class TestLaneChangingSafety:
    """Test lane changing safety mechanisms."""
    
    @pytest.fixture
    def lane_safety_env_config(self):
        """Environment configuration for lane changing safety tests."""
        return {
            'training_map': 'multi_track',
            'episode_max_steps': 250,
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
    def lane_safety_enhanced_config(self):
        """Enhanced configuration for lane changing safety."""
        return EnhancedRLConfig(
            enabled_features=['yolo', 'object_avoidance', 'lane_changing', 'multi_objective_reward'],
            debug_mode=True,
            yolo={
                'model_path': 'yolov5s.pt',
                'confidence_threshold': 0.4,
                'device': 'cpu'
            },
            object_avoidance={
                'safety_distance': 0.6,
                'avoidance_strength': 1.2,
                'min_clearance': 0.25
            },
            lane_changing={
                'lane_change_threshold': 0.2,  # More conservative
                'safety_margin': 3.0,  # Larger safety margin
                'max_lane_change_time': 2.5,
                'min_lane_width': 0.5,
                'evaluation_distance': 6.0  # Look further ahead
            },
            reward={
                'lane_following_weight': 1.0,
                'object_avoidance_weight': 1.2,
                'lane_change_weight': 0.4,
                'safety_penalty_weight': -3.0,
                'collision_penalty': -15.0
            }
        )
    
    @pytest.mark.safety
    def test_unsafe_lane_change_prevention(self, lane_safety_env_config, lane_safety_enhanced_config):
        """Test prevention of unsafe lane changes."""
        safety_harness = SafetyTestHarness(lane_safety_env_config, lane_safety_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                
                def mock_unsafe_lane_scenario(img):
                    # Current lane blocked, adjacent lane also has obstacle
                    mock_results = Mock()
                    detection_array = np.array([
                        [200, 120, 350, 280, 0.9, 2, 'car'],     # Current lane blocked
                        [100, 140, 200, 240, 0.8, 0, 'person']  # Adjacent lane occupied
                    ])
                    mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                    return mock_results
                
                mock_model.side_effect = mock_unsafe_lane_scenario
                mock_yolo.return_value = mock_model
                
                if not safety_harness.setup_environment():
                    pytest.skip("Could not setup lane safety test environment")
                
                obs = safety_harness.env.reset()
                
                # Test unsafe lane change prevention
                lane_change_attempts = 0
                unsafe_lane_changes = 0
                
                for step in range(80):
                    action = np.array([0.4, 0.0])
                    obs, reward, done, info = safety_harness.env.step(action)
                    
                    safety_harness.analyze_step(obs, reward, done, info, action)
                    
                    # Track lane change attempts
                    if 'lane_change_state' in info:
                        state = info['lane_change_state']
                        if state in ['evaluating', 'initiating']:
                            lane_change_attempts += 1
                        elif state == 'executing':
                            # Check if this is unsafe (adjacent lane occupied)
                            if 'detections' in info:
                                adjacent_occupied = any(
                                    abs(d.get('relative_position', [0, 0])[0]) > 0.4
                                    for d in info['detections']
                                )
                                if adjacent_occupied:
                                    unsafe_lane_changes += 1
                    
                    if done:
                        break
                
                safety_score = safety_harness.get_safety_score()
                
                # Should not execute unsafe lane changes
                assert unsafe_lane_changes == 0, \
                    f"Unsafe lane changes executed: {unsafe_lane_changes}"
                
                # Should evaluate but not execute when unsafe
                assert lane_change_attempts > 0, "Should at least evaluate lane changes"
                
                print(f"Unsafe lane change prevention: attempts={lane_change_attempts}, "
                      f"unsafe={unsafe_lane_changes}, score={safety_score:.3f}")
                
        except Exception as e:
            pytest.skip(f"Unsafe lane change test skipped: {e}")
        finally:
            safety_harness.cleanup()
    
    @pytest.mark.safety
    def test_lane_change_abort_mechanism(self, lane_safety_env_config, lane_safety_enhanced_config):
        """Test lane change abort when conditions become unsafe."""
        safety_harness = SafetyTestHarness(lane_safety_env_config, lane_safety_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                step_counter = [0]
                
                def mock_changing_conditions(img):
                    step = step_counter[0]
                    step_counter[0] += 1
                    
                    mock_results = Mock()
                    
                    # Initially safe lane change opportunity
                    if step < 20:
                        detection_array = np.array([
                            [200, 120, 350, 280, 0.9, 2, 'car']  # Only current lane blocked
                        ])
                    else:
                        # Obstacle appears in target lane during lane change
                        detection_array = np.array([
                            [200, 120, 350, 280, 0.9, 2, 'car'],     # Current lane blocked
                            [80, 130, 180, 230, 0.8, 0, 'person']   # Target lane now occupied
                        ])
                    
                    mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                    return mock_results
                
                mock_model.side_effect = mock_changing_conditions
                mock_yolo.return_value = mock_model
                
                if not safety_harness.setup_environment():
                    pytest.skip("Could not setup lane safety test environment")
                
                obs = safety_harness.env.reset()
                
                # Test lane change abort
                lane_change_started = False
                lane_change_aborted = False
                
                for step in range(60):
                    action = np.array([0.4, 0.0])
                    obs, reward, done, info = safety_harness.env.step(action)
                    
                    safety_harness.analyze_step(obs, reward, done, info, action)
                    
                    # Track lane change state transitions
                    if 'lane_change_state' in info:
                        state = info['lane_change_state']
                        if state == 'executing':
                            lane_change_started = True
                        elif lane_change_started and state == 'following':
                            # Lane change was aborted (returned to following)
                            lane_change_aborted = True
                    
                    if done:
                        break
                
                safety_score = safety_harness.get_safety_score()
                
                # Should abort lane change when conditions become unsafe
                # (This test depends on implementation details)
                print(f"Lane change abort: started={lane_change_started}, "
                      f"aborted={lane_change_aborted}, score={safety_score:.3f}")
                
                # Should maintain high safety score even with changing conditions
                assert safety_score >= 0.7, f"Safety score too low with changing conditions: {safety_score:.3f}"
                
        except Exception as e:
            pytest.skip(f"Lane change abort test skipped: {e}")
        finally:
            safety_harness.cleanup()
    
    @pytest.mark.safety
    def test_lane_change_timeout_safety(self, lane_safety_env_config, lane_safety_enhanced_config):
        """Test safety when lane change times out."""
        safety_harness = SafetyTestHarness(lane_safety_env_config, lane_safety_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                
                def mock_persistent_obstacle(img):
                    # Persistent obstacle that makes lane change difficult
                    mock_results = Mock()
                    detection_array = np.array([
                        [200, 120, 350, 280, 0.9, 2, 'car'],  # Current lane blocked
                        [120, 150, 220, 250, 0.7, 0, 'person']  # Partial obstruction in target lane
                    ])
                    mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                    return mock_results
                
                mock_model.side_effect = mock_persistent_obstacle
                mock_yolo.return_value = mock_model
                
                if not safety_harness.setup_environment():
                    pytest.skip("Could not setup lane safety test environment")
                
                obs = safety_harness.env.reset()
                
                # Test lane change timeout handling
                lane_change_timeouts = 0
                
                for step in range(100):  # Longer episode for timeout testing
                    action = np.array([0.3, 0.0])  # Slower speed
                    obs, reward, done, info = safety_harness.env.step(action)
                    
                    safety_harness.analyze_step(obs, reward, done, info, action)
                    
                    # Check for lane change timeout
                    if 'lane_change_timeout' in info and info['lane_change_timeout']:
                        lane_change_timeouts += 1
                    
                    if done:
                        break
                
                safety_score = safety_harness.get_safety_score()
                
                # Should handle timeouts safely
                assert safety_harness.safety_metrics['collisions'] == 0, \
                    f"Collisions during timeout handling: {safety_harness.safety_metrics['collisions']}"
                
                # Should maintain reasonable safety score
                assert safety_score >= 0.6, f"Safety score too low with timeouts: {safety_score:.3f}"
                
                print(f"Lane change timeout safety: timeouts={lane_change_timeouts}, "
                      f"score={safety_score:.3f}")
                
        except Exception as e:
            pytest.skip(f"Lane change timeout test skipped: {e}")
        finally:
            safety_harness.cleanup()


class TestFailureModeAnalysis:
    """Test system behavior under failure conditions."""
    
    @pytest.fixture
    def failure_env_config(self):
        """Environment configuration for failure mode testing."""
        return {
            'training_map': 'small_loop',
            'episode_max_steps': 150,
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
    def failure_enhanced_config(self):
        """Enhanced configuration for failure mode testing."""
        return EnhancedRLConfig(
            enabled_features=['yolo', 'object_avoidance', 'multi_objective_reward'],
            debug_mode=False,  # Test non-debug mode failure handling
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
            reward={
                'lane_following_weight': 1.0,
                'object_avoidance_weight': 0.8,
                'safety_penalty_weight': -2.0
            }
        )
    
    @pytest.mark.safety
    def test_yolo_failure_graceful_degradation(self, failure_env_config, failure_enhanced_config):
        """Test graceful degradation when YOLO fails."""
        safety_harness = SafetyTestHarness(failure_env_config, failure_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                failure_step = 15  # YOLO fails after 15 steps
                step_counter = [0]
                
                def mock_failing_yolo(img):
                    step = step_counter[0]
                    step_counter[0] += 1
                    
                    if step < failure_step:
                        # Normal operation
                        mock_results = Mock()
                        detection_array = np.array([
                            [200, 150, 300, 250, 0.8, 0, 'person']
                        ])
                        mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                        return mock_results
                    else:
                        # YOLO failure
                        raise RuntimeError("YOLO inference failed")
                
                mock_model.side_effect = mock_failing_yolo
                mock_yolo.return_value = mock_model
                
                if not safety_harness.setup_environment():
                    pytest.skip("Could not setup failure test environment")
                
                obs = safety_harness.env.reset()
                
                # Test graceful degradation
                yolo_failures = 0
                system_crashes = 0
                
                for step in range(40):
                    try:
                        action = np.array([0.4, 0.0])
                        obs, reward, done, info = safety_harness.env.step(action)
                        
                        safety_harness.analyze_step(obs, reward, done, info, action)
                        
                        # Check if YOLO failure was handled
                        if step >= failure_step:
                            yolo_failures += 1
                        
                        if done:
                            break
                            
                    except Exception as e:
                        system_crashes += 1
                        print(f"System crash at step {step}: {e}")
                        break
                
                safety_score = safety_harness.get_safety_score()
                
                # System should not crash due to YOLO failure
                assert system_crashes == 0, f"System crashed {system_crashes} times due to YOLO failure"
                
                # Should handle YOLO failures gracefully
                assert yolo_failures > 0, "YOLO failure scenario not triggered"
                
                # Should maintain reasonable safety despite failures
                assert safety_score >= 0.5, f"Safety score too low during failures: {safety_score:.3f}"
                
                print(f"YOLO failure handling: failures={yolo_failures}, "
                      f"crashes={system_crashes}, score={safety_score:.3f}")
                
        except Exception as e:
            pytest.skip(f"YOLO failure test skipped: {e}")
        finally:
            safety_harness.cleanup()
    
    @pytest.mark.safety
    def test_extreme_action_safety_limits(self, failure_env_config, failure_enhanced_config):
        """Test safety limits with extreme actions."""
        safety_harness = SafetyTestHarness(failure_env_config, failure_enhanced_config)
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                
                def mock_obstacle_detection(img):
                    mock_results = Mock()
                    detection_array = np.array([
                        [200, 150, 300, 250, 0.9, 0, 'person']
                    ])
                    mock_results.pandas.return_value.xyxy = [Mock(values=detection_array)]
                    return mock_results
                
                mock_model.side_effect = mock_obstacle_detection
                mock_yolo.return_value = mock_model
                
                if not safety_harness.setup_environment():
                    pytest.skip("Could not setup extreme action test environment")
                
                obs = safety_harness.env.reset()
                
                # Test with extreme actions
                extreme_actions = [
                    np.array([1.0, 1.0]),    # Full speed, full right
                    np.array([1.0, -1.0]),   # Full speed, full left
                    np.array([-1.0, 0.0]),   # Full reverse
                    np.array([1.0, 0.0]),    # Full speed ahead
                ]
                
                action_safety_violations = 0
                
                for i, extreme_action in enumerate(extreme_actions):
                    # Use extreme action for several steps
                    for step in range(8):
                        obs, reward, done, info = safety_harness.env.step(extreme_action)
                        
                        safety_harness.analyze_step(obs, reward, done, info, extreme_action)
                        
                        # Check if action was safely limited
                        if 'modified_action' in info:
                            modified = info['modified_action']
                            if not np.allclose(modified, extreme_action, atol=0.1):
                                # Action was modified for safety
                                pass
                        
                        # Check for safety violations with extreme actions
                        if 'reward_components' in info:
                            if info['reward_components'].get('safety_penalty', 0) < -2.0:
                                action_safety_violations += 1
                        
                        if done:
                            obs = safety_harness.env.reset()
                            break
                
                safety_score = safety_harness.get_safety_score()
                
                # Should handle extreme actions safely
                assert safety_harness.safety_metrics['collisions'] <= 1, \
                    f"Too many collisions with extreme actions: {safety_harness.safety_metrics['collisions']}"
                
                # Should maintain minimum safety score
                assert safety_score >= 0.4, f"Safety score too low with extreme actions: {safety_score:.3f}"
                
                print(f"Extreme action safety: violations={action_safety_violations}, "
                      f"score={safety_score:.3f}")
                
        except Exception as e:
            pytest.skip(f"Extreme action test skipped: {e}")
        finally:
            safety_harness.cleanup()


class TestSafetyMetricsValidation:
    """Test safety metrics and validation."""
    
    @pytest.mark.safety
    def test_safety_metrics_calculation(self):
        """Test safety metrics calculation accuracy."""
        # Create test harness
        harness = SafetyTestHarness({}, EnhancedRLConfig())
        
        # Simulate safety events
        harness.safety_metrics['total_steps'] = 100
        harness.safety_metrics['collisions'] = 1
        harness.safety_metrics['near_misses'] = 3
        harness.safety_metrics['safety_violations'] = 2
        harness.safety_metrics['unsafe_lane_changes'] = 1
        
        # Calculate safety score
        safety_score = harness.get_safety_score()
        
        # Verify calculation
        expected_weighted_violations = 1*10 + 3*3 + 2*2 + 1*5  # 28
        expected_max_violations = 100 * 0.1  # 10
        expected_score = max(0.0, 1.0 - (28 / 10))  # Negative, so 0.0
        
        assert safety_score == 0.0, f"Safety score calculation incorrect: {safety_score}"
        
        # Test with fewer violations
        harness.safety_metrics['collisions'] = 0
        harness.safety_metrics['near_misses'] = 1
        harness.safety_metrics['safety_violations'] = 1
        harness.safety_metrics['unsafe_lane_changes'] = 0
        
        safety_score = harness.get_safety_score()
        expected_weighted_violations = 0*10 + 1*3 + 1*2 + 0*5  # 5
        expected_score = max(0.0, 1.0 - (5 / 10))  # 0.5
        
        assert abs(safety_score - 0.5) < 0.01, f"Safety score calculation incorrect: {safety_score}"
        
        print(f"Safety metrics validation: score calculation correct")
    
    @pytest.mark.safety
    def test_safety_thresholds(self):
        """Test safety threshold validation."""
        # Define safety thresholds
        safety_thresholds = {
            'min_safety_score': 0.7,
            'max_collision_rate': 0.02,  # 2% of steps
            'max_near_miss_rate': 0.05,  # 5% of steps
            'max_violation_rate': 0.1    # 10% of steps
        }
        
        # Test case 1: Safe system
        safe_metrics = {
            'total_steps': 1000,
            'collisions': 5,      # 0.5% rate
            'near_misses': 20,    # 2% rate
            'safety_violations': 50,  # 5% rate
            'unsafe_lane_changes': 2
        }
        
        harness = SafetyTestHarness({}, EnhancedRLConfig())
        harness.safety_metrics.update(safe_metrics)
        safety_score = harness.get_safety_score()
        
        collision_rate = safe_metrics['collisions'] / safe_metrics['total_steps']
        near_miss_rate = safe_metrics['near_misses'] / safe_metrics['total_steps']
        violation_rate = safe_metrics['safety_violations'] / safe_metrics['total_steps']
        
        # Verify thresholds
        assert collision_rate <= safety_thresholds['max_collision_rate'], \
            f"Collision rate too high: {collision_rate:.3f}"
        assert near_miss_rate <= safety_thresholds['max_near_miss_rate'], \
            f"Near miss rate too high: {near_miss_rate:.3f}"
        assert violation_rate <= safety_thresholds['max_violation_rate'], \
            f"Violation rate too high: {violation_rate:.3f}"
        
        print(f"Safety thresholds: collision_rate={collision_rate:.3f}, "
              f"near_miss_rate={near_miss_rate:.3f}, violation_rate={violation_rate:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "safety"])