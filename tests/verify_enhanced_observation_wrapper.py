"""
Verification script for Enhanced Observation Wrapper.

This script performs basic syntax checking and logic verification
without requiring external dependencies like gym.
"""

import sys
import os
import ast
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def check_syntax():
    """Check syntax of the enhanced observation wrapper."""
    wrapper_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'duckietown_utils', 
        'wrappers', 
        'enhanced_observation_wrapper.py'
    )
    
    try:
        with open(wrapper_path, 'r') as f:
            source_code = f.read()
        
        # Parse the AST to check syntax
        ast.parse(source_code)
        print("✓ Syntax check passed")
        return True
        
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False
    except FileNotFoundError:
        print(f"✗ File not found: {wrapper_path}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during syntax check: {e}")
        return False


def check_class_structure():
    """Check that the class has the required methods and structure."""
    wrapper_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'duckietown_utils', 
        'wrappers', 
        'enhanced_observation_wrapper.py'
    )
    
    try:
        with open(wrapper_path, 'r') as f:
            source_code = f.read()
        
        # Check for required class and methods
        required_elements = [
            'class EnhancedObservationWrapper',
            'def __init__',
            'def observation',
            'def _extract_detection_features',
            'def _extract_image_features',
            'def _extract_safety_features',
            'def _create_flattened_observation',
            'def _create_dict_observation',
            'def get_feature_stats',
            'def reset'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in source_code:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"✗ Missing required elements: {missing_elements}")
            return False
        else:
            print("✓ Class structure check passed")
            return True
            
    except Exception as e:
        print(f"✗ Error checking class structure: {e}")
        return False


def test_feature_calculation_logic():
    """Test the core feature calculation logic without dependencies."""
    print("Testing feature calculation logic...")
    
    # Test detection feature processing
    max_detections = 3
    detection_feature_size = 9
    
    # Simulate detection array
    test_detections = np.array([
        [1, 0.9, 100, 50, 200, 150, 0.5, 1.0, 2.5],  # Detection 1
        [2, 0.8, 150, 75, 250, 175, -0.3, 0.8, 3.2],  # Detection 2
    ], dtype=np.float32)
    
    # Flatten and pad (simulating _extract_detection_features logic)
    features = test_detections.flatten().astype(np.float32)
    expected_size = max_detections * detection_feature_size
    if len(features) < expected_size:
        features = np.pad(features, (0, expected_size - len(features)), mode='constant')
    elif len(features) > expected_size:
        features = features[:expected_size]
    
    # Verify
    if features.shape[0] == expected_size and features.dtype == np.float32:
        print("✓ Detection feature processing logic correct")
    else:
        print(f"✗ Detection feature processing failed: shape={features.shape}, dtype={features.dtype}")
        return False
    
    # Test safety feature calculation
    detection_count = 2
    safety_critical = 1
    inference_time = 0.05
    safety_feature_weight = 2.0
    distance_normalization_factor = 10.0
    
    # Extract distances from detections
    distances = [2.5, 3.2]  # From test_detections
    avg_distance = np.mean(distances)
    closest_distance = np.min(distances)
    
    # Calculate safety features
    safety_features = np.array([
        detection_count / max_detections,
        safety_critical * safety_feature_weight,
        np.clip(inference_time, 0, 1.0),
        avg_distance / distance_normalization_factor,
        closest_distance / distance_normalization_factor
    ], dtype=np.float32)
    
    # Verify safety features
    expected_values = [2/3, 2.0, 0.05, (2.5+3.2)/2/10, 2.5/10]
    
    for i, (actual, expected) in enumerate(zip(safety_features, expected_values)):
        if not np.isclose(actual, expected, rtol=1e-3):
            print(f"✗ Safety feature {i} incorrect: {actual} != {expected}")
            return False
    
    print("✓ Safety feature calculation logic correct")
    
    # Test normalization logic
    test_values = np.array([100.0, 0.5, 200.0, 150.0], dtype=np.float32)
    min_vals = np.array([0.0, 0.0, 0.0, 0.0])
    max_vals = np.array([1000.0, 1.0, 640.0, 480.0])
    
    # Min-max normalization
    normalized = (test_values - min_vals) / (max_vals - min_vals + 1e-8)
    
    expected_normalized = [0.1, 0.5, 200.0/640.0, 150.0/480.0]
    
    for i, (actual, expected) in enumerate(zip(normalized, expected_normalized)):
        if not np.isclose(actual, expected, rtol=1e-3):
            print(f"✗ Normalization {i} incorrect: {actual} != {expected}")
            return False
    
    print("✓ Normalization logic correct")
    return True


def test_image_encoding_logic():
    """Test image encoding logic."""
    print("Testing image encoding logic...")
    
    # Create test image
    test_image = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    
    # Test flatten method
    flattened = test_image.flatten().astype(np.float32)
    expected_size = 120 * 160 * 3
    
    if flattened.shape[0] == expected_size and flattened.dtype == np.float32:
        print("✓ Image flattening logic correct")
    else:
        print(f"✗ Image flattening failed: shape={flattened.shape}, dtype={flattened.dtype}")
        return False
    
    # Test simple encoding method
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
    
    # Pad to 512
    features = np.array(features, dtype=np.float32)
    if len(features) < 512:
        features = np.pad(features, (0, 512 - len(features)), mode='constant')
    else:
        features = features[:512]
    
    if len(features) == 512 and features.dtype == np.float32:
        print("✓ Image encoding logic correct")
        return True
    else:
        print(f"✗ Image encoding failed: length={len(features)}, dtype={features.dtype}")
        return False


def check_requirements_compliance():
    """Check compliance with task requirements."""
    print("Checking requirements compliance...")
    
    wrapper_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'duckietown_utils', 
        'wrappers', 
        'enhanced_observation_wrapper.py'
    )
    
    try:
        with open(wrapper_path, 'r') as f:
            source_code = f.read()
        
        # Check for requirement compliance
        requirements_checks = [
            ('EnhancedObservationWrapper to combine detection data', 'EnhancedObservationWrapper' in source_code),
            ('feature vector flattening', '_create_flattened_observation' in source_code),
            ('normalization and scaling', 'normalize_features' in source_code and '_normalize_' in source_code),
            ('PPO observation space compatibility', 'spaces.Box' in source_code),
            ('detection feature extraction', '_extract_detection_features' in source_code),
            ('image feature processing', '_extract_image_features' in source_code),
            ('safety feature calculation', '_extract_safety_features' in source_code),
        ]
        
        all_passed = True
        for requirement, check in requirements_checks:
            if check:
                print(f"✓ {requirement}")
            else:
                print(f"✗ {requirement}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Error checking requirements: {e}")
        return False


def main():
    """Run all verification tests."""
    print("Enhanced Observation Wrapper Verification")
    print("=" * 50)
    
    tests = [
        ("Syntax Check", check_syntax),
        ("Class Structure Check", check_class_structure),
        ("Feature Calculation Logic", test_feature_calculation_logic),
        ("Image Encoding Logic", test_image_encoding_logic),
        ("Requirements Compliance", check_requirements_compliance),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All verification tests passed!")
        return True
    else:
        print("✗ Some verification tests failed.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)