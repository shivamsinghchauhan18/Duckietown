#!/usr/bin/env python3
"""
Direct test of MultiObjectiveRewardWrapper without going through __init__.py
"""

import sys
import numpy as np
from unittest.mock import Mock

# Add current directory to path
sys.path.insert(0, '.')

def test_multi_objective_reward_wrapper():
    """Test the MultiObjectiveRewardWrapper directly."""
    print("Testing MultiObjectiveRewardWrapper directly...")
    
    try:
        # Import directly from the module file
        from duckietown_utils.wrappers.multi_objective_reward_wrapper import MultiObjectiveRewardWrapper
        print("✓ Successfully imported MultiObjectiveRewardWrapper")
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        return False
    
    # Test basic initialization
    try:
        mock_env = Mock()
        mock_env.unwrapped = Mock()
        wrapper = MultiObjectiveRewardWrapper(mock_env)
        print("✓ Basic initialization successful")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False
    
    # Test default weights
    expected_weights = {
        'lane_following': 1.0,
        'object_avoidance': 0.5,
        'lane_changing': 0.3,
        'efficiency': 0.2,
        'safety_penalty': -2.0
    }
    
    if wrapper.reward_weights == expected_weights:
        print("✓ Default weights are correct")
    else:
        print(f"✗ Default weights incorrect. Expected: {expected_weights}, Got: {wrapper.reward_weights}")
        return False
    
    # Test custom weights
    try:
        custom_weights = {
            'lane_following': 2.0,
            'object_avoidance': 1.0,
            'lane_changing': 0.5,
            'efficiency': 0.1,
            'safety_penalty': -3.0
        }
        wrapper_custom = MultiObjectiveRewardWrapper(mock_env, custom_weights)
        if wrapper_custom.reward_weights == custom_weights:
            print("✓ Custom weights initialization works")
        else:
            print("✗ Custom weights not set correctly")
            return False
    except Exception as e:
        print(f"✗ Custom weights test failed: {e}")
        return False
    
    # Test weight validation
    try:
        invalid_weights = {'lane_following': 'invalid'}
        try:
            MultiObjectiveRewardWrapper(mock_env, invalid_weights)
            print("✗ Should have failed with invalid weights")
            return False
        except ValueError:
            print("✓ Weight validation works correctly")
    except Exception as e:
        print(f"✗ Weight validation test failed: {e}")
        return False
    
    # Test reward calculation
    try:
        # Mock the individual reward methods
        wrapper._calculate_lane_following_reward = Mock(return_value=0.8)
        wrapper._calculate_object_avoidance_reward = Mock(return_value=0.2)
        wrapper._calculate_lane_changing_reward = Mock(return_value=0.1)
        wrapper._calculate_efficiency_reward = Mock(return_value=0.3)
        wrapper._calculate_safety_penalty = Mock(return_value=-0.1)
        
        reward = wrapper.reward(1.0)
        
        # Expected: 1.0*0.8 + 0.5*0.2 + 0.3*0.1 + 0.2*0.3 + (-2.0)*(-0.1) = 1.19
        expected = 1.19
        
        if abs(reward - expected) < 0.001:
            print("✓ Reward calculation is correct")
        else:
            print(f"✗ Reward calculation wrong. Expected: {expected}, Got: {reward}")
            return False
    except Exception as e:
        print(f"✗ Reward calculation test failed: {e}")
        return False
    
    # Test getter methods
    try:
        weights = wrapper.get_reward_weights()
        components = wrapper.get_reward_components()
        
        if isinstance(weights, dict) and isinstance(components, dict):
            print("✓ Getter methods work correctly")
        else:
            print("✗ Getter methods don't return dicts")
            return False
    except Exception as e:
        print(f"✗ Getter methods test failed: {e}")
        return False
    
    # Test weight updates
    try:
        wrapper.update_weights({'lane_following': 5.0})
        if wrapper.reward_weights['lane_following'] == 5.0:
            print("✓ Weight update works correctly")
        else:
            print("✗ Weight update failed")
            return False
    except Exception as e:
        print(f"✗ Weight update test failed: {e}")
        return False
    
    print("🎉 All tests passed!")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("Direct MultiObjectiveRewardWrapper Test")
    print("=" * 50)
    
    success = test_multi_objective_reward_wrapper()
    
    print("=" * 50)
    if success:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 50)
    
    sys.exit(0 if success else 1)