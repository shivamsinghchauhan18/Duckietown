#!/usr/bin/env python3
"""
Validation script for MultiObjectiveRewardWrapper.

This script validates the core functionality without requiring gym dependencies.
"""

import sys
import numpy as np
from unittest.mock import Mock


def validate_reward_wrapper():
    """Validate the MultiObjectiveRewardWrapper implementation."""
    print("Validating MultiObjectiveRewardWrapper...")
    
    try:
        # Import the wrapper
        sys.path.append('.')
        from duckietown_utils.wrappers.multi_objective_reward_wrapper import MultiObjectiveRewardWrapper
        print("✓ Successfully imported MultiObjectiveRewardWrapper")
    except ImportError as e:
        print(f"✗ Failed to import MultiObjectiveRewardWrapper: {e}")
        return False
    
    # Test 1: Basic initialization
    try:
        mock_env = Mock()
        mock_env.unwrapped = Mock()
        wrapper = MultiObjectiveRewardWrapper(mock_env)
        print("✓ Basic initialization successful")
    except Exception as e:
        print(f"✗ Basic initialization failed: {e}")
        return False
    
    # Test 2: Default weights validation
    try:
        expected_keys = {'lane_following', 'object_avoidance', 'lane_changing', 'efficiency', 'safety_penalty'}
        actual_keys = set(wrapper.reward_weights.keys())
        if expected_keys == actual_keys:
            print("✓ Default weights have correct keys")
        else:
            print(f"✗ Default weights missing keys: {expected_keys - actual_keys}")
            return False
    except Exception as e:
        print(f"✗ Default weights validation failed: {e}")
        return False
    
    # Test 3: Custom weights initialization
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
            print("✓ Custom weights initialization successful")
        else:
            print("✗ Custom weights not set correctly")
            return False
    except Exception as e:
        print(f"✗ Custom weights initialization failed: {e}")
        return False
    
    # Test 4: Weight validation
    try:
        invalid_weights = {
            'lane_following': 1.0,
            'object_avoidance': 'invalid'  # Invalid type
        }
        try:
            MultiObjectiveRewardWrapper(mock_env, invalid_weights)
            print("✗ Weight validation should have failed for invalid types")
            return False
        except ValueError:
            print("✓ Weight validation correctly rejects invalid types")
    except Exception as e:
        print(f"✗ Weight validation test failed: {e}")
        return False
    
    # Test 5: Reward components structure
    try:
        components = wrapper.get_reward_components()
        expected_components = {'lane_following', 'object_avoidance', 'lane_changing', 'efficiency', 'safety_penalty', 'total'}
        actual_components = set(components.keys())
        if expected_components == actual_components:
            print("✓ Reward components have correct structure")
        else:
            print(f"✗ Reward components missing: {expected_components - actual_components}")
            return False
    except Exception as e:
        print(f"✗ Reward components test failed: {e}")
        return False
    
    # Test 6: Weight update functionality
    try:
        original_weight = wrapper.reward_weights['lane_following']
        wrapper.update_weights({'lane_following': 5.0})
        if wrapper.reward_weights['lane_following'] == 5.0:
            print("✓ Weight update functionality works")
        else:
            print("✗ Weight update did not work correctly")
            return False
    except Exception as e:
        print(f"✗ Weight update test failed: {e}")
        return False
    
    # Test 7: Getter methods return copies
    try:
        weights = wrapper.get_reward_weights()
        components = wrapper.get_reward_components()
        
        # Modify returned dicts
        weights['lane_following'] = 999
        components['total'] = 999
        
        # Check that original values are unchanged
        if (wrapper.reward_weights['lane_following'] != 999 and 
            wrapper.reward_components['total'] != 999):
            print("✓ Getter methods return copies, not references")
        else:
            print("✗ Getter methods return references instead of copies")
            return False
    except Exception as e:
        print(f"✗ Getter methods test failed: {e}")
        return False
    
    print("\n🎉 All validation tests passed!")
    return True


def validate_reward_calculation_logic():
    """Validate the reward calculation logic with mocked methods."""
    print("\nValidating reward calculation logic...")
    
    try:
        from duckietown_utils.wrappers.multi_objective_reward_wrapper import MultiObjectiveRewardWrapper
        
        mock_env = Mock()
        mock_env.unwrapped = Mock()
        wrapper = MultiObjectiveRewardWrapper(mock_env)
        
        # Mock individual reward calculation methods
        wrapper._calculate_lane_following_reward = Mock(return_value=0.8)
        wrapper._calculate_object_avoidance_reward = Mock(return_value=0.2)
        wrapper._calculate_lane_changing_reward = Mock(return_value=0.1)
        wrapper._calculate_efficiency_reward = Mock(return_value=0.3)
        wrapper._calculate_safety_penalty = Mock(return_value=-0.1)
        
        # Calculate reward
        reward = wrapper.reward(1.0)
        
        # Expected calculation with default weights:
        # 1.0 * 0.8 + 0.5 * 0.2 + 0.3 * 0.1 + 0.2 * 0.3 + (-2.0) * (-0.1)
        # = 0.8 + 0.1 + 0.03 + 0.06 + 0.2 = 1.19
        expected = 1.19
        
        if abs(reward - expected) < 0.001:
            print("✓ Reward calculation logic is correct")
        else:
            print(f"✗ Reward calculation incorrect. Expected: {expected}, Got: {reward}")
            return False
            
        # Check that components were stored
        if (wrapper.reward_components['lane_following'] == 0.8 and
            wrapper.reward_components['total'] == reward):
            print("✓ Reward components are stored correctly")
        else:
            print("✗ Reward components not stored correctly")
            return False
            
    except Exception as e:
        print(f"✗ Reward calculation validation failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("MultiObjectiveRewardWrapper Validation")
    print("=" * 60)
    
    success = True
    success &= validate_reward_wrapper()
    success &= validate_reward_calculation_logic()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL VALIDATIONS PASSED! 🎉")
        print("MultiObjectiveRewardWrapper is ready for use.")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("Please check the implementation.")
    print("=" * 60)
    
    sys.exit(0 if success else 1)