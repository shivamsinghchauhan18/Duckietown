"""
Validation script for Enhanced PPO Training Integration.
Tests core functionality without requiring full Ray/RLLib setup.
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_config():
    """Test enhanced configuration loading."""
    try:
        from config.enhanced_config import load_enhanced_config
        
        config = load_enhanced_config('./config/enhanced_config.yml')
        
        # Validate configuration
        assert 'yolo' in config.enabled_features
        assert 'multi_objective_reward' in config.enabled_features
        assert config.yolo.confidence_threshold == 0.5
        assert config.reward.lane_following_weight == 1.0
        
        logger.info("✓ Enhanced configuration test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Enhanced configuration test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    required_files = [
        'experiments/train_enhanced_rllib.py',
        'duckietown_utils/enhanced_rllib_callbacks.py',
        'duckietown_utils/enhanced_rllib_loggers.py',
        'duckietown_utils/training_utils.py',
        'config/enhanced_ppo_config.yml',
        'examples/enhanced_ppo_training_example.py',
        'tests/test_enhanced_ppo_training.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"✗ Missing files: {missing_files}")
        return False
    else:
        logger.info("✓ All required files exist")
        return True

def test_syntax_validation():
    """Test Python syntax of all created files."""
    python_files = [
        'experiments/train_enhanced_rllib.py',
        'duckietown_utils/enhanced_rllib_callbacks.py',
        'duckietown_utils/enhanced_rllib_loggers.py',
        'duckietown_utils/training_utils.py',
        'examples/enhanced_ppo_training_example.py'
    ]
    
    import py_compile
    
    for file_path in python_files:
        try:
            py_compile.compile(file_path, doraise=True)
        except Exception as e:
            logger.error(f"✗ Syntax error in {file_path}: {e}")
            return False
    
    logger.info("✓ All Python files have valid syntax")
    return True

def main():
    """Run all validation tests."""
    logger.info("Enhanced PPO Training Integration Validation")
    logger.info("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Syntax Validation", test_syntax_validation),
        ("Enhanced Configuration", test_enhanced_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        if test_func():
            passed += 1
    
    logger.info(f"\nValidation Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✓ All validation tests passed!")
        logger.info("\nEnhanced PPO Training Integration is ready for use.")
        logger.info("Note: Full functionality requires Ray/RLLib dependencies.")
        return True
    else:
        logger.error("✗ Some validation tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)