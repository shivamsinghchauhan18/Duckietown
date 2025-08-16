#!/usr/bin/env python3
"""
🧪 TEST SUITES EXAMPLE 🧪
Example demonstrating the evaluation test suites implementation

This example shows how to use the different evaluation test suites
for comprehensive model testing across various environmental conditions.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from duckietown_utils.test_suites import (
    TestSuiteFactory, create_all_suite_configs,
    BaseSuite, HardRandomizationSuite, LawIntersectionSuite,
    OutOfDistributionSuite, StressAdversarialSuite
)
from duckietown_utils.suite_manager import SuiteManager


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demonstrate_individual_suites():
    """Demonstrate individual test suite configurations."""
    print("🧪 INDIVIDUAL TEST SUITE CONFIGURATIONS")
    print("=" * 60)
    
    # Configuration for all suites
    config = {
        'episodes_per_map': 10,
        'timeout_per_episode': 120.0
    }
    
    suite_names = ['base', 'hard_randomization', 'law_intersection', 
                   'out_of_distribution', 'stress_adversarial']
    
    for suite_name in suite_names:
        print(f"\n📋 {suite_name.upper().replace('_', ' ')} SUITE:")
        print("-" * 40)
        
        # Create suite instance
        suite = TestSuiteFactory.create_suite(suite_name, config)
        
        # Display basic information
        print(f"Description: {suite.get_description()}")
        print(f"Maps: {len(suite.get_maps())} maps")
        print(f"Episodes per map: {suite.get_episodes_per_map()}")
        print(f"Total episodes: {len(suite.get_maps()) * suite.get_episodes_per_map()}")
        print(f"Timeout per episode: {suite.get_timeout_per_episode()}s")
        
        # Show some key environment configurations
        env_config = suite.get_environment_config()
        print(f"\nKey Environment Settings:")
        
        if suite_name == 'base':
            print(f"  - Lighting variation: {env_config.get('lighting_variation', 'N/A')}")
            print(f"  - Texture variation: {env_config.get('texture_variation', 'N/A')}")
            print(f"  - Camera noise: {env_config.get('camera_noise', 'N/A')}")
            print(f"  - Traffic density: {env_config.get('traffic_density', 'N/A')}")
        
        elif suite_name == 'hard_randomization':
            print(f"  - Lighting variation: {env_config.get('lighting_variation', 'N/A')}")
            print(f"  - Texture variation: {env_config.get('texture_variation', 'N/A')}")
            print(f"  - Camera noise: {env_config.get('camera_noise', 'N/A')}")
            print(f"  - Weather effects: {env_config.get('weather_effects', 'N/A')}")
        
        elif suite_name == 'law_intersection':
            print(f"  - Stop signs: {env_config.get('stop_signs', 'N/A')}")
            print(f"  - Traffic lights: {env_config.get('traffic_lights', 'N/A')}")
            print(f"  - Right-of-way scenarios: {env_config.get('right_of_way_scenarios', 'N/A')}")
            print(f"  - Intersection complexity: {env_config.get('intersection_complexity', 'N/A')}")
        
        elif suite_name == 'out_of_distribution':
            print(f"  - Unseen textures: {env_config.get('unseen_textures', 'N/A')}")
            print(f"  - Night conditions: {env_config.get('night_conditions', 'N/A')}")
            print(f"  - Sensor noise: {env_config.get('sensor_noise', 'N/A')}")
            print(f"  - Novel obstacles: {env_config.get('novel_obstacles', 'N/A')}")
        
        elif suite_name == 'stress_adversarial':
            print(f"  - Sensor dropouts: {env_config.get('sensor_dropouts', 'N/A')}")
            print(f"  - Wheel bias: {env_config.get('wheel_bias', 'N/A')}")
            print(f"  - Moving obstacles: {env_config.get('moving_obstacles', 'N/A')}")
            print(f"  - Extreme lighting: {env_config.get('extreme_lighting', 'N/A')}")


def demonstrate_suite_manager_integration():
    """Demonstrate integration with SuiteManager."""
    print("\n\n🔧 SUITE MANAGER INTEGRATION")
    print("=" * 60)
    
    # Create suite manager configuration
    config = {
        'results_dir': 'logs/suite_results',
        'base_config': {
            'episodes_per_map': 5,
            'maps': ['LF-norm-loop', 'LF-norm-small_loop']
        },
        'hard_randomization_config': {
            'episodes_per_map': 4,
            'maps': ['LF-norm-loop', 'huge_loop']
        },
        'law_intersection_config': {
            'episodes_per_map': 3,
            'maps': ['ETHZ_autolab_technical_track']
        }
    }
    
    # Create suite manager
    suite_manager = SuiteManager(config)
    
    print(f"Available suites: {suite_manager.list_suites()}")
    
    # Show detailed information for each suite
    for suite_name in suite_manager.list_suites():
        info = suite_manager.get_suite_info(suite_name)
        if info:
            print(f"\n📊 {suite_name.upper()}:")
            print(f"  Type: {info['type']}")
            print(f"  Description: {info['description']}")
            print(f"  Maps: {len(info['maps'])}")
            print(f"  Episodes per map: {info['episodes_per_map']}")
            print(f"  Total episodes: {info['total_episodes']}")
            print(f"  Estimated runtime: {info.get('timeout_per_episode', 120) * info['total_episodes'] / 60:.1f} minutes")


def demonstrate_configuration_customization():
    """Demonstrate how to customize suite configurations."""
    print("\n\n⚙️ CONFIGURATION CUSTOMIZATION")
    print("=" * 60)
    
    # Custom configuration for specific research needs
    custom_config = {
        'episodes_per_map': 20,
        'timeout_per_episode': 180.0,
        'maps': ['custom_research_map_1', 'custom_research_map_2']
    }
    
    print("Creating custom Base Suite configuration:")
    custom_base = BaseSuite(custom_config)
    suite_config = custom_base.create_suite_config()
    
    print(f"  Custom maps: {suite_config.maps}")
    print(f"  Episodes per map: {suite_config.episodes_per_map}")
    print(f"  Total episodes: {len(suite_config.maps) * suite_config.episodes_per_map}")
    print(f"  Timeout: {suite_config.timeout_per_episode}s")
    
    # Show how environment config can be accessed
    env_config = custom_base.get_environment_config()
    print(f"\nEnvironment configuration keys: {len(env_config)} parameters")
    print(f"Sample parameters: {list(env_config.keys())[:5]}...")


def demonstrate_suite_comparison():
    """Demonstrate comparing different suites."""
    print("\n\n📊 SUITE COMPARISON")
    print("=" * 60)
    
    config = {'episodes_per_map': 10}
    
    # Create different suites
    suites = {
        'base': TestSuiteFactory.create_suite('base', config),
        'hard': TestSuiteFactory.create_suite('hard_randomization', config),
        'stress': TestSuiteFactory.create_suite('stress_adversarial', config)
    }
    
    print("Difficulty progression comparison:")
    print(f"{'Suite':<20} {'Maps':<6} {'Episodes':<9} {'Timeout':<8} {'Difficulty'}")
    print("-" * 60)
    
    difficulty_indicators = {
        'base': 'Low',
        'hard': 'High', 
        'stress': 'Extreme'
    }
    
    for name, suite in suites.items():
        maps_count = len(suite.get_maps())
        episodes = suite.get_episodes_per_map()
        timeout = suite.get_timeout_per_episode()
        difficulty = difficulty_indicators.get(name, 'Unknown')
        
        print(f"{name:<20} {maps_count:<6} {episodes:<9} {timeout:<8.0f} {difficulty}")


def demonstrate_json_serialization():
    """Demonstrate JSON serialization of suite configurations."""
    print("\n\n💾 JSON SERIALIZATION")
    print("=" * 60)
    
    # Create all suite configurations
    config = {
        'base_config': {'episodes_per_map': 5},
        'hard_randomization_config': {'episodes_per_map': 4},
        'law_intersection_config': {'episodes_per_map': 3}
    }
    
    suite_configs = create_all_suite_configs(config)
    
    # Convert to JSON-serializable format
    serializable_configs = {}
    for name, suite_config in suite_configs.items():
        serializable_configs[name] = {
            'suite_name': suite_config.suite_name,
            'suite_type': suite_config.suite_type.value,
            'description': suite_config.description,
            'maps': suite_config.maps,
            'episodes_per_map': suite_config.episodes_per_map,
            'timeout_per_episode': suite_config.timeout_per_episode,
            'environment_config_keys': len(suite_config.environment_config),
            'evaluation_config_keys': len(suite_config.evaluation_config)
        }
    
    # Save to JSON file
    output_file = Path('logs/suite_configs_example.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_configs, f, indent=2)
    
    print(f"Suite configurations saved to: {output_file}")
    print(f"Total suites: {len(serializable_configs)}")
    
    # Show sample of the JSON structure
    print("\nSample JSON structure:")
    sample_suite = list(serializable_configs.values())[0]
    for key, value in sample_suite.items():
        print(f"  {key}: {value}")


def main():
    """Main example function."""
    setup_logging()
    
    print("🧪 EVALUATION TEST SUITES EXAMPLE")
    print("=" * 80)
    print("This example demonstrates the comprehensive evaluation test suites")
    print("for rigorous model testing across different environmental conditions.")
    print("=" * 80)
    
    try:
        # Run all demonstrations
        demonstrate_individual_suites()
        demonstrate_suite_manager_integration()
        demonstrate_configuration_customization()
        demonstrate_suite_comparison()
        demonstrate_json_serialization()
        
        print("\n\n✅ EXAMPLE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("The evaluation test suites are ready for use!")
        print("\nNext steps:")
        print("1. Integrate with your model evaluation pipeline")
        print("2. Customize suite configurations for your research needs")
        print("3. Run comprehensive evaluations across all suites")
        print("4. Analyze results to identify model strengths and weaknesses")
        
    except Exception as e:
        print(f"\n❌ Error during example execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())