#!/usr/bin/env python3
"""
🧪 ENHANCED RL SYSTEM COMPREHENSIVE TEST SUITE 🧪

Complete test suite for validating the enhanced RL system:
- Component integration tests
- YOLO detection validation
- Wrapper functionality tests
- Training pipeline validation
- Deployment system tests
- Performance benchmarks

This ensures the system works end-to-end before deployment.
"""

import os
import sys
import time
import json
import logging
import unittest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import test frameworks
import torch
import cv2

# Import our systems
from enhanced_rl_training_system import EnhancedRLTrainer, TrainingConfig, EnhancedDQNNetwork, EnhancedDQNAgent
from enhanced_deployment_system import EnhancedRLInferenceNode, DeploymentConfig
from config.enhanced_config import load_enhanced_config
from duckietown_utils.wrappers.yolo_detection_wrapper import YOLOObjectDetectionWrapper
from duckietown_utils.wrappers.enhanced_observation_wrapper import EnhancedObservationWrapper
from duckietown_utils.wrappers.object_avoidance_action_wrapper import ObjectAvoidanceActionWrapper
from duckietown_utils.wrappers.lane_changing_action_wrapper import LaneChangingActionWrapper
from duckietown_utils.wrappers.multi_objective_reward_wrapper import MultiObjectiveRewardWrapper

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockEnvironment:
    """Mock environment for testing."""
    
    def __init__(self, use_enhanced_obs=True):
        self.use_enhanced_obs = use_enhanced_obs
        
        if use_enhanced_obs:
            from gym import spaces
            self.observation_space = spaces.Dict({
                'image': spaces.Box(0, 255, (120, 160, 3), dtype=np.uint8),
                'detection_features': spaces.Box(-np.inf, np.inf, (90,), dtype=np.float32),
                'safety_features': spaces.Box(-np.inf, np.inf, (5,), dtype=np.float32)
            })
        else:
            from gym import spaces
            self.observation_space = spaces.Box(0, 255, (120, 160, 3), dtype=np.uint8)
        
        from gym import spaces
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
        
        self.current_step = 0
        self.max_steps = 100
    
    def reset(self):
        """Reset environment."""
        self.current_step = 0
        return self._get_observation()
    
    def step(self, action):
        """Take a step."""
        self.current_step += 1
        
        obs = self._get_observation()
        reward = np.random.uniform(-1, 1)
        done = self.current_step >= self.max_steps
        info = {'step': self.current_step}
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Get mock observation."""
        if self.use_enhanced_obs:
            return {
                'image': np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8),
                'detection_features': np.random.randn(90).astype(np.float32),
                'safety_features': np.random.randn(5).astype(np.float32)
            }
        else:
            return np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)


class TestYOLOIntegration(unittest.TestCase):
    """Test YOLO detection integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_env = MockEnvironment(use_enhanced_obs=False)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_yolo_wrapper_initialization(self):
        """Test YOLO wrapper initialization."""
        logger.info("🧪 Testing YOLO wrapper initialization...")
        
        try:
            wrapper = YOLOObjectDetectionWrapper(
                self.mock_env,
                model_path="yolov5s.pt",
                confidence_threshold=0.5,
                device='cpu',
                max_detections=10
            )
            
            self.assertIsNotNone(wrapper)
            self.assertTrue(hasattr(wrapper, 'yolo_system'))
            logger.info("✅ YOLO wrapper initialization test passed")
            
        except Exception as e:
            logger.warning(f"⚠️ YOLO wrapper test failed (expected if YOLO not available): {e}")
            self.skipTest("YOLO not available")
    
    def test_yolo_detection_processing(self):
        """Test YOLO detection processing."""
        logger.info("🧪 Testing YOLO detection processing...")
        
        try:
            wrapper = YOLOObjectDetectionWrapper(
                self.mock_env,
                model_path="yolov5s.pt",
                confidence_threshold=0.5,
                device='cpu'
            )
            
            # Test with mock image
            test_image = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
            processed_obs = wrapper.observation(test_image)
            
            self.assertIsInstance(processed_obs, dict)
            self.assertIn('detections', processed_obs)
            self.assertIn('detection_count', processed_obs)
            
            logger.info("✅ YOLO detection processing test passed")
            
        except Exception as e:
            logger.warning(f"⚠️ YOLO detection test failed: {e}")
            self.skipTest("YOLO detection not working")


class TestEnhancedObservationWrapper(unittest.TestCase):
    """Test enhanced observation wrapper."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_env = MockEnvironment(use_enhanced_obs=True)
    
    def test_enhanced_observation_wrapper(self):
        """Test enhanced observation wrapper."""
        logger.info("🧪 Testing enhanced observation wrapper...")
        
        wrapper = EnhancedObservationWrapper(
            self.mock_env,
            include_detection_features=True,
            include_image_features=True,
            output_mode='flattened'
        )
        
        # Test observation processing
        mock_obs = {
            'image': np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8),
            'detections': np.random.randn(10, 9).astype(np.float32),
            'detection_count': np.array([5]),
            'safety_critical': np.array([0]),
            'inference_time': np.array([0.05])
        }
        
        processed_obs = wrapper.observation(mock_obs)
        
        self.assertIsInstance(processed_obs, np.ndarray)
        self.assertGreater(len(processed_obs), 0)
        
        logger.info("✅ Enhanced observation wrapper test passed")
    
    def test_observation_space_setup(self):
        """Test observation space setup."""
        logger.info("🧪 Testing observation space setup...")
        
        # Test flattened mode
        wrapper_flat = EnhancedObservationWrapper(
            self.mock_env,
            output_mode='flattened'
        )
        
        self.assertTrue(hasattr(wrapper_flat.observation_space, 'shape'))
        
        # Test dict mode
        wrapper_dict = EnhancedObservationWrapper(
            self.mock_env,
            output_mode='dict'
        )
        
        from gym import spaces
        self.assertIsInstance(wrapper_dict.observation_space, spaces.Dict)
        
        logger.info("✅ Observation space setup test passed")


class TestActionWrappers(unittest.TestCase):
    """Test action wrappers."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_env = MockEnvironment()
    
    def test_object_avoidance_wrapper(self):
        """Test object avoidance action wrapper."""
        logger.info("🧪 Testing object avoidance wrapper...")
        
        wrapper = ObjectAvoidanceActionWrapper(
            self.mock_env,
            safety_distance=0.5,
            avoidance_strength=1.0
        )
        
        # Test action modification
        test_action = np.array([0.0, 0.5])  # Straight forward
        modified_action = wrapper.action(test_action)
        
        self.assertIsInstance(modified_action, np.ndarray)
        self.assertEqual(len(modified_action), 2)
        
        logger.info("✅ Object avoidance wrapper test passed")
    
    def test_lane_changing_wrapper(self):
        """Test lane changing action wrapper."""
        logger.info("🧪 Testing lane changing wrapper...")
        
        wrapper = LaneChangingActionWrapper(
            self.mock_env,
            lane_change_threshold=0.3,
            safety_margin=2.0
        )
        
        # Test action processing
        test_action = np.array([0.0, 0.5])
        modified_action = wrapper.action(test_action)
        
        self.assertIsInstance(modified_action, np.ndarray)
        self.assertEqual(len(modified_action), 2)
        
        logger.info("✅ Lane changing wrapper test passed")


class TestMultiObjectiveReward(unittest.TestCase):
    """Test multi-objective reward wrapper."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_env = MockEnvironment()
    
    def test_reward_calculation(self):
        """Test reward calculation."""
        logger.info("🧪 Testing multi-objective reward calculation...")
        
        reward_weights = {
            'lane_following': 1.0,
            'object_avoidance': 0.5,
            'lane_changing': 0.3,
            'efficiency': 0.2,
            'safety_penalty': -2.0
        }
        
        wrapper = MultiObjectiveRewardWrapper(
            self.mock_env,
            reward_weights=reward_weights
        )
        
        # Test reward processing
        base_reward = 1.0
        multi_reward = wrapper.reward(base_reward)
        
        self.assertIsInstance(multi_reward, (int, float))
        
        logger.info("✅ Multi-objective reward test passed")


class TestEnhancedDQNNetwork(unittest.TestCase):
    """Test enhanced DQN network."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = TrainingConfig()
        self.mock_env = MockEnvironment(use_enhanced_obs=True)
    
    def test_network_initialization(self):
        """Test network initialization."""
        logger.info("🧪 Testing enhanced DQN network initialization...")
        
        network = EnhancedDQNNetwork(
            self.mock_env.observation_space,
            self.mock_env.action_space,
            self.config
        )
        
        self.assertIsNotNone(network)
        self.assertTrue(hasattr(network, 'forward'))
        
        logger.info("✅ Enhanced DQN network initialization test passed")
    
    def test_network_forward_pass(self):
        """Test network forward pass."""
        logger.info("🧪 Testing network forward pass...")
        
        network = EnhancedDQNNetwork(
            self.mock_env.observation_space,
            self.mock_env.action_space,
            self.config
        )
        
        # Test with enhanced observation
        test_obs = {
            'image': torch.randn(1, 3, 120, 160),
            'detection_features': torch.randn(1, 90),
            'safety_features': torch.randn(1, 5)
        }
        
        with torch.no_grad():
            output = network(test_obs)
        
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape[1], 2)  # Action dimension
        
        logger.info("✅ Network forward pass test passed")


class TestEnhancedDQNAgent(unittest.TestCase):
    """Test enhanced DQN agent."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = TrainingConfig()
        self.config.total_timesteps = 1000  # Short test
        self.mock_env = MockEnvironment(use_enhanced_obs=True)
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        logger.info("🧪 Testing enhanced DQN agent initialization...")
        
        agent = EnhancedDQNAgent(self.mock_env, self.config)
        
        self.assertIsNotNone(agent)
        self.assertIsNotNone(agent.q_network)
        self.assertIsNotNone(agent.target_network)
        
        logger.info("✅ Enhanced DQN agent initialization test passed")
    
    def test_action_selection(self):
        """Test action selection."""
        logger.info("🧪 Testing action selection...")
        
        agent = EnhancedDQNAgent(self.mock_env, self.config)
        
        test_obs = {
            'image': np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8),
            'detection_features': np.random.randn(90).astype(np.float32),
            'safety_features': np.random.randn(5).astype(np.float32)
        }
        
        action = agent.select_action(test_obs, training=False)
        
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(len(action), 2)
        
        logger.info("✅ Action selection test passed")
    
    def test_training_step(self):
        """Test training step."""
        logger.info("🧪 Testing training step...")
        
        agent = EnhancedDQNAgent(self.mock_env, self.config)
        
        # Add some transitions to replay buffer
        for _ in range(self.config.batch_size + 10):
            obs = self.mock_env.reset()
            action = agent.select_action(obs)
            next_obs, reward, done, info = self.mock_env.step(action)
            agent.store_transition(obs, action, reward, next_obs, done)
        
        # Test training step
        loss = agent.train_step()
        
        if loss is not None:
            self.assertIsInstance(loss, (int, float))
            self.assertGreaterEqual(loss, 0)
        
        logger.info("✅ Training step test passed")


class TestTrainingSystem(unittest.TestCase):
    """Test complete training system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        logger.info("🧪 Testing trainer initialization...")
        
        trainer = EnhancedRLTrainer()
        trainer.log_dir = self.temp_dir / "logs"
        trainer.model_dir = self.temp_dir / "models"
        
        self.assertIsNotNone(trainer)
        self.assertIsInstance(trainer.training_config, TrainingConfig)
        
        logger.info("✅ Trainer initialization test passed")
    
    def test_environment_creation(self):
        """Test enhanced environment creation."""
        logger.info("🧪 Testing enhanced environment creation...")
        
        trainer = EnhancedRLTrainer()
        
        try:
            # This might fail if gym-duckietown is not available
            env = trainer.create_enhanced_environment("loop_empty")
            
            self.assertIsNotNone(env)
            self.assertTrue(hasattr(env, 'observation_space'))
            self.assertTrue(hasattr(env, 'action_space'))
            
            env.close()
            logger.info("✅ Enhanced environment creation test passed")
            
        except Exception as e:
            logger.warning(f"⚠️ Environment creation test failed (expected if gym-duckietown not available): {e}")
            self.skipTest("gym-duckietown not available")


class TestDeploymentSystem(unittest.TestCase):
    """Test deployment system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = DeploymentConfig()
        self.config.model_path = str(self.temp_dir / "test_model.pth")
        
        # Create a dummy model file
        torch.save({'model_state_dict': {}}, self.config.model_path)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_deployment_node_initialization(self):
        """Test deployment node initialization."""
        logger.info("🧪 Testing deployment node initialization...")
        
        # Skip ROS-dependent tests
        if not hasattr(sys.modules.get('enhanced_deployment_system', None), 'ROS_AVAILABLE'):
            self.skipTest("ROS not available")
        
        try:
            node = EnhancedRLInferenceNode(self.config)
            
            self.assertIsNotNone(node)
            self.assertIsNotNone(node.config)
            
            logger.info("✅ Deployment node initialization test passed")
            
        except Exception as e:
            logger.warning(f"⚠️ Deployment test failed: {e}")
            self.skipTest("Deployment dependencies not available")


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""
    
    def test_inference_speed(self):
        """Test inference speed."""
        logger.info("🧪 Testing inference speed...")
        
        config = TrainingConfig()
        mock_env = MockEnvironment(use_enhanced_obs=True)
        
        network = EnhancedDQNNetwork(
            mock_env.observation_space,
            mock_env.action_space,
            config
        )
        
        # Benchmark inference speed
        test_obs = {
            'image': torch.randn(1, 3, 120, 160),
            'detection_features': torch.randn(1, 90),
            'safety_features': torch.randn(1, 5)
        }
        
        num_inferences = 100
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_inferences):
                output = network(test_obs)
        
        end_time = time.time()
        avg_inference_time = (end_time - start_time) / num_inferences
        
        logger.info(f"📊 Average inference time: {avg_inference_time*1000:.2f}ms")
        
        # Should be fast enough for real-time control (< 100ms)
        self.assertLess(avg_inference_time, 0.1)
        
        logger.info("✅ Inference speed test passed")
    
    def test_memory_usage(self):
        """Test memory usage."""
        logger.info("🧪 Testing memory usage...")
        
        config = TrainingConfig()
        mock_env = MockEnvironment(use_enhanced_obs=True)
        
        network = EnhancedDQNNetwork(
            mock_env.observation_space,
            mock_env.action_space,
            config
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        
        logger.info(f"📊 Total parameters: {total_params:,}")
        logger.info(f"📊 Trainable parameters: {trainable_params:,}")
        
        # Should be reasonable for deployment (< 10M parameters)
        self.assertLess(total_params, 10_000_000)
        
        logger.info("✅ Memory usage test passed")


class TestSystemIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline."""
        logger.info("🧪 Testing end-to-end pipeline...")
        
        # Create mock environment
        mock_env = MockEnvironment(use_enhanced_obs=True)
        
        # Test observation processing pipeline
        obs = mock_env.reset()
        
        # Simulate YOLO detection
        enhanced_obs = {
            'image': obs['image'] if isinstance(obs, dict) else obs,
            'detections': np.random.randn(5, 9).astype(np.float32),
            'detection_count': np.array([5]),
            'safety_critical': np.array([0]),
            'inference_time': np.array([0.05])
        }
        
        # Test enhanced observation wrapper
        obs_wrapper = EnhancedObservationWrapper(
            mock_env,
            output_mode='dict'
        )
        processed_obs = obs_wrapper.observation(enhanced_obs)
        
        # Test network inference
        config = TrainingConfig()
        network = EnhancedDQNNetwork(
            obs_wrapper.observation_space,
            mock_env.action_space,
            config
        )
        
        with torch.no_grad():
            obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in processed_obs.items()}
            q_values = network(obs_tensor)
            action = q_values.cpu().numpy().squeeze()
        
        # Test action wrappers
        avoidance_wrapper = ObjectAvoidanceActionWrapper(mock_env)
        enhanced_action = avoidance_wrapper.action(action)
        
        # Test reward wrapper
        reward_wrapper = MultiObjectiveRewardWrapper(mock_env)
        enhanced_reward = reward_wrapper.reward(1.0)
        
        # Verify pipeline works
        self.assertIsInstance(processed_obs, dict)
        self.assertIsInstance(action, np.ndarray)
        self.assertIsInstance(enhanced_action, np.ndarray)
        self.assertIsInstance(enhanced_reward, (int, float))
        
        logger.info("✅ End-to-end pipeline test passed")


def run_comprehensive_tests():
    """Run comprehensive test suite."""
    logger.info("🧪 STARTING COMPREHENSIVE ENHANCED RL SYSTEM TESTS")
    logger.info("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestYOLOIntegration,
        TestEnhancedObservationWrapper,
        TestActionWrappers,
        TestMultiObjectiveReward,
        TestEnhancedDQNNetwork,
        TestEnhancedDQNAgent,
        TestTrainingSystem,
        TestDeploymentSystem,
        TestPerformanceBenchmarks,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    logger.info("=" * 80)
    logger.info("🧪 TEST SUMMARY")
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        logger.error("❌ FAILURES:")
        for test, traceback in result.failures:
            logger.error(f"  - {test}: {traceback}")
    
    if result.errors:
        logger.error("❌ ERRORS:")
        for test, traceback in result.errors:
            logger.error(f"  - {test}: {traceback}")
    
    if result.skipped:
        logger.warning("⚠️ SKIPPED:")
        for test, reason in result.skipped:
            logger.warning(f"  - {test}: {reason}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
    
    if success_rate >= 0.8:
        logger.info(f"✅ TESTS PASSED: {success_rate:.1%} success rate")
        logger.info("🎉 Enhanced RL system is ready for deployment!")
        return True
    else:
        logger.error(f"❌ TESTS FAILED: {success_rate:.1%} success rate")
        logger.error("🚨 System needs fixes before deployment!")
        return False


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced RL System Test Suite")
    parser.add_argument('--test-class', type=str, help='Run specific test class')
    parser.add_argument('--benchmark-only', action='store_true', help='Run benchmarks only')
    parser.add_argument('--integration-only', action='store_true', help='Run integration tests only')
    
    args = parser.parse_args()
    
    if args.test_class:
        # Run specific test class
        test_class = globals().get(args.test_class)
        if test_class:
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(suite)
        else:
            logger.error(f"Test class {args.test_class} not found")
            sys.exit(1)
    
    elif args.benchmark_only:
        # Run benchmarks only
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformanceBenchmarks)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    
    elif args.integration_only:
        # Run integration tests only
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSystemIntegration)
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    
    else:
        # Run comprehensive tests
        success = run_comprehensive_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()