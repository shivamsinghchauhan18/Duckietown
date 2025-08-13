#!/usr/bin/env python3
"""
Test the syntax and basic structure of the MultiObjectiveRewardWrapper
"""

import ast
import sys

def test_syntax():
    """Test that the wrapper file has valid Python syntax."""
    print("Testing MultiObjectiveRewardWrapper syntax...")
    
    try:
        with open('duckietown_utils/wrappers/multi_objective_reward_wrapper.py', 'r') as f:
            content = f.read()
        
        # Parse the AST to check syntax
        tree = ast.parse(content)
        print("✓ File has valid Python syntax")
        
        # Check for required class
        class_found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'MultiObjectiveRewardWrapper':
                class_found = True
                print("✓ MultiObjectiveRewardWrapper class found")
                
                # Check for required methods
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                required_methods = ['__init__', 'reward', 'step', 'reset', 'update_weights', 
                                  'get_reward_components', 'get_reward_weights']
                
                missing_methods = [m for m in required_methods if m not in methods]
                if not missing_methods:
                    print("✓ All required methods are present")
                else:
                    print(f"✗ Missing methods: {missing_methods}")
                    return False
                break
        
        if not class_found:
            print("✗ MultiObjectiveRewardWrapper class not found")
            return False
            
        return True
        
    except FileNotFoundError:
        print("✗ MultiObjectiveRewardWrapper file not found")
        return False
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking syntax: {e}")
        return False


def test_structure():
    """Test the structure and content of the wrapper."""
    print("\nTesting wrapper structure...")
    
    try:
        with open('duckietown_utils/wrappers/multi_objective_reward_wrapper.py', 'r') as f:
            content = f.read()
        
        # Check for key components
        checks = [
            ('class MultiObjectiveRewardWrapper', 'Main class definition'),
            ('def __init__', 'Constructor method'),
            ('def reward', 'Reward calculation method'),
            ('def step', 'Step method'),
            ('def reset', 'Reset method'),
            ('def _calculate_lane_following_reward', 'Lane following reward method'),
            ('def _calculate_object_avoidance_reward', 'Object avoidance reward method'),
            ('def _calculate_lane_changing_reward', 'Lane changing reward method'),
            ('def _calculate_efficiency_reward', 'Efficiency reward method'),
            ('def _calculate_safety_penalty', 'Safety penalty method'),
            ('reward_weights', 'Reward weights attribute'),
            ('reward_components', 'Reward components attribute'),
        ]
        
        for check, description in checks:
            if check in content:
                print(f"✓ {description} found")
            else:
                print(f"✗ {description} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error checking structure: {e}")
        return False


def test_init_file():
    """Test that the __init__.py file includes the new wrapper."""
    print("\nTesting __init__.py integration...")
    
    try:
        with open('duckietown_utils/wrappers/__init__.py', 'r') as f:
            content = f.read()
        
        if 'MultiObjectiveRewardWrapper' in content:
            print("✓ MultiObjectiveRewardWrapper is exported in __init__.py")
            return True
        else:
            print("✗ MultiObjectiveRewardWrapper not found in __init__.py")
            return False
            
    except Exception as e:
        print(f"✗ Error checking __init__.py: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("MultiObjectiveRewardWrapper Structure Test")
    print("=" * 60)
    
    success = True
    success &= test_syntax()
    success &= test_structure()
    success &= test_init_file()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ ALL STRUCTURE TESTS PASSED!")
        print("The MultiObjectiveRewardWrapper implementation looks good.")
    else:
        print("❌ SOME STRUCTURE TESTS FAILED")
    print("=" * 60)
    
    sys.exit(0 if success else 1)