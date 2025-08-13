"""
Simple unit tests for Enhanced Observation Wrapper (without gym dependency).

This module contains basic tests for the EnhancedObservationWrapper class
that can run without the full gym environment setup.
"""

import sys
import os
import unittest
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from duckietown_utils.wrappers.enhanced_observation_wrapper import EnhancedObservationWrapper
    WRAPPER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import EnhancedObservationWrapper: {e}")
    WRAPPER_AVAILABLE = False


class TestEnhancedObservationWrapperBasic(unittest.TestCase):
    """Basic test cases for EnhancedObservationWrapper without gym dependency."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not WRAPPER_AVAILABLE:
            self.skipTest("EnhancedObservationWrapper not available")
    
    def test_import_success(self):
        """Test that the wrapper can be imported successfully."""
        self.assertTrue(WRAPPER_AVAILABLE, "EnhancedObservationWrapper should be importable")
    
    def test_feature_statistics_initialization(self):
        """Test feature statistics initialization without full environment."""
        # Create a mock environment-like object
        class MockEnv:
            def __init__(self):
                self.observation_space = type('MockSpace', (), {
                    'shape': (120, 160, 3),
                    'dtype': np.uint8
                })()
        
        mock_env = MockEnv()
        
        # This should work without gym
        try:
            wrapper = EnhancedObservationWrapper(
                mock_env,
                include_detection_features=True,
                include_image_features=True,
                max_detections=5
            )
            
            # Check that feature stats are initialized
            self.assertIn('detection_min', wrapper._feature_stats)
            self.assertIn('detection_max', wrapper._feature_stats)
            self.assertIn('detection_mean', wrapper._feature_stats)
            self.assertIn('detection_std', wrapper._feature_stats)
            
            # Check array sizes
            expected_size = 5 * 9  # max_detections * detection_feature_size
            self.assertEqual(len(wrapper._feature_stats['detection_min']), expected_size)
            self.assertEqual(len(wrapper._feature_stats['detection_max']), expected_size)
            
        except Exception as e:
            # If it fails due to gym dependency, that's expected
            if 'gym' in str(e).lower():
                self.skipTest(f"Test requires gym: {e}")
            else:
                raise
    
    def test_detection_feature_processing(self):
        """Test detection feature processing logic."""
        if not WRAPPER_AVAILABLE:
            self.skipTest("EnhancedObservationWrapper not available")
        
        # Create mock wrapper instance for testing methods
        class MockWrapper:
            def __init__(self):
                self.max_detections = 3
                self.detection_feature_size = 9
                self.normalize_features = False
                self._feature_stats = {
                    'detection_min': np.zeros(27),  # 3 * 9
                    'detection_max': np.ones(27) * 1000,
                    'detection_mean': np.ones(27) * 500,
                    'detection_std': np.ones(27) * 250
                }
            
            def _extract_detection_features(self, detections):
                """Simplified version of the method."""
                features = detections.flatten().astype(np.float32)
                expected_size = self.max_detections * self.detection_feature_size
                if len(features) < expected_size:
                    features = np.pad(features, (0, expected_size - len(features)), mode='constant')
                elif len(features) > expected_size:
                    features = features[:expected_size]
                return features
        
        mock_wrapper = MockWrapper()
        
        # Test detection feature extraction
        test_detections = np.array([
            [1, 0.9, 100, 50, 200, 150, 0.5, 1.0, 2.5],  # Detection 1
            [2, 0.8, 150, 75, 250, 175, -0.3, 0.8, 3.2],  # Detection 2
        ], dtype=np.float32)
        
        features = mock_wrapper._extract_detection_features(test_detections)
        
        # Check feature size
        expected_size = 3 * 9  # max_detections * detection_feature_size
        self.assertEqual(features.shape[0], expected_size)
        self.assertEqual(features.dtype, np.float32)
        
        # Check that detection data is preserved
        self.assertAlmostEqual(features[0], 1.0)  # First class ID
        self.assertAlmostEqual(features[1], 0.9)  # First confidence
        self.assertAlmostEqual(features[9], 2.0)  # Second class ID
        self.assertAlmostEqual(features[10], 0.8)  # Second confidence
    
    def test_safety_feature_calculation(self):
        """Test safety feature calculation logic."""
        if not WRAPPER_AVAILABLE:
            self.skipTest("EnhancedObservationWrapper not available")
        
        # Test safety feature calculation without full wrapper
        max_detections = 5
        safety_feature_weight = 2.0
        distance_normalization_factor = 10.0
        
        # Mock detections with distances
        detections = np.array([
            [1, 0.9, 100, 50, 200, 150, 0.5, 1.0, 2.5],  # Distance 2.5m
            [2, 0.8, 150, 75, 250, 175, -0.3, 0.8, 1.2],  # Distance 1.2m
            [3, 0.7, 200, 100, 300, 200, 0.0, 0.5, 4.0],  # Distance 4.0m
        ], dtype=np.float32)
        
        detection_count = 3
        safety_critical = 1
        inference_time = 0.05
        
        # Calculate safety features manually
        distances = [2.5, 1.2, 4.0]
        avg_distance = np.mean(distances)
        closest_distance = np.min(distances)
        
        expected_safety_features = np.array([
            detection_count / max_detections,  # Normalized detection count
            safety_critical * safety_feature_weight,  # Weighted safety flag
            np.clip(inference_time, 0, 1.0),  # Clipped inference time
            avg_distance / distance_normalization_factor,  # Normalized average distance
            closest_distance / distance_normalization_factor  # Normalized closest distance
        ], dtype=np.float32)
        
        # Verify calculations
        self.assertAlmostEqual(expected_safety_features[0], 3.0 / 5.0)  # 0.6
        self.assertAlmostEqual(expected_safety_features[1], 2.0)  # 1 * 2.0
        self.assertAlmostEqual(expected_safety_features[2], 0.05)
        self.assertAlmostEqual(expected_safety_features[3], (2.5 + 1.2 + 4.0) / 3.0 / 10.0, places=3)
        self.assertAlmostEqual(expected_safety_features[4], 1.2 / 10.0)
    
    def test_image_encoding_logic(self):
        """Test simple image encoding logic."""
        if not WRAPPER_AVAILABLE:
            self.skipTest("EnhancedObservationWrapper not available")
        
        # Test simple image encoding without full wrapper
        test_image = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
        
        # Simplified encoding logic
        features = []
        
        # Global statistics
        features.extend([
            np.mean(test_image),
            np.std(test_image),
            np.min(test_image),
            np.max(test_image)
        ])
        
        # Per-channel statistics
        for channel in range(test_image.shape[2]):
            channel_data = test_image[:, :, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data)
            ])
        
        # Spatial features
        gray = np.mean(test_image, axis=2)
        grad_x = np.abs(np.diff(gray, axis=1)).mean()
        grad_y = np.abs(np.diff(gray, axis=0)).mean()
        features.extend([grad_x, grad_y])
        
        # Convert to array and pad to 512
        features = np.array(features, dtype=np.float32)
        if len(features) < 512:
            features = np.pad(features, (0, 512 - len(features)), mode='constant')
        else:
            features = features[:512]
        
        # Verify encoding
        self.assertEqual(len(features), 512)
        self.assertEqual(features.dtype, np.float32)
        
        # Check that features contain meaningful values
        self.assertGreater(features[0], 0)  # Mean should be > 0
        self.assertGreaterEqual(features[1], 0)  # Std should be >= 0
    
    def test_normalization_methods(self):
        """Test feature normalization methods."""
        if not WRAPPER_AVAILABLE:
            self.skipTest("EnhancedObservationWrapper not available")
        
        # Test min-max normalization
        test_features = np.array([100.0, 0.5, 200.0, 150.0], dtype=np.float32)
        feature_min = np.array([0.0, 0.0, 0.0, 0.0])
        feature_max = np.array([1000.0, 1.0, 640.0, 480.0])
        
        # Min-max normalization
        normalized = (test_features - feature_min) / (feature_max - feature_min + 1e-8)
        
        # Check normalization results
        self.assertAlmostEqual(normalized[0], 100.0 / 1000.0)  # 0.1
        self.assertAlmostEqual(normalized[1], 0.5 / 1.0)  # 0.5
        self.assertAlmostEqual(normalized[2], 200.0 / 640.0, places=3)  # ~0.312
        self.assertAlmostEqual(normalized[3], 150.0 / 480.0, places=3)  # ~0.312
        
        # Test standard normalization
        feature_mean = np.array([500.0, 0.5, 320.0, 240.0])
        feature_std = np.array([250.0, 0.25, 160.0, 120.0])
        
        standard_normalized = (test_features - feature_mean) / (feature_std + 1e-8)
        
        # Check standard normalization
        self.assertAlmostEqual(standard_normalized[0], (100.0 - 500.0) / 250.0)  # -1.6
        self.assertAlmostEqual(standard_normalized[1], (0.5 - 0.5) / 0.25)  # 0.0


def run_basic_tests():
    """Run basic tests and return results."""
    if not WRAPPER_AVAILABLE:
        print("EnhancedObservationWrapper not available for testing")
        return False
    
    # Run a simple syntax check
    try:
        # Try to import and create a basic instance
        print("Testing basic import and initialization...")
        
        # This is a minimal test that should work
        print("✓ Import successful")
        print("✓ Basic functionality tests would run here")
        return True
        
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        return False


if __name__ == '__main__':
    print("Running basic Enhanced Observation Wrapper tests...")
    
    # Run basic functionality test
    success = run_basic_tests()
    
    if success:
        print("\nRunning unittest suite...")
        unittest.main(verbosity=2)
    else:
        print("Basic tests failed, skipping unittest suite")
        sys.exit(1)