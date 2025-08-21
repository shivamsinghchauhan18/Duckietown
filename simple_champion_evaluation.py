#!/usr/bin/env python3
"""
Simple but comprehensive evaluation of the working champion model.
Direct evaluation without complex dependencies.
"""

import sys
import numpy as np
import torch
import json
import time
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
sys.path.append('.')

# Import model loading utilities
from duckiebot_deployment.model_loader import load_model_for_deployment

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleChampionEvaluator:
    """Simple but thorough evaluator for the champion model."""
    
    def __init__(self, model_path: str = "champion_model.pth"):
        self.model_path = model_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"evaluation_results/simple_evaluation_{self.timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🚀 Initializing Simple Champion Model Evaluator")
        logger.info(f"Model: {model_path}")
        logger.info(f"Results: {self.results_dir}")
        
        # Load the model
        self.model_wrapper = load_model_for_deployment(self.model_path)
        logger.info(f"✅ Model loaded: {self.model_wrapper.get_model_info()}")
    
    def run_evaluation(self) -> dict:
        """Run comprehensive evaluation."""
        
        print("\n🧪 Running Champion Model Evaluation...")
        print("=" * 50)
        
        results = {
            'model_path': self.model_path,
            'timestamp': self.timestamp,
            'model_info': self.model_wrapper.get_model_info(),
            'tests': {}
        }
        
        # Test 1: Basic Functionality
        print("\n1️⃣ Basic Functionality Test")
        results['tests']['basic_functionality'] = self.test_basic_functionality()
        
        # Test 2: Performance Benchmarking
        print("\n2️⃣ Performance Benchmarking")
        results['tests']['performance'] = self.test_performance()
        
        # Test 3: Robustness Testing
        print("\n3️⃣ Robustness Testing")
        results['tests']['robustness'] = self.test_robustness()
        
        # Test 4: Scenario-Based Evaluation
        print("\n4️⃣ Scenario-Based Evaluation")
        results['tests']['scenarios'] = self.test_scenarios()
        
        # Test 5: Safety Assessment
        print("\n5️⃣ Safety Assessment")
        results['tests']['safety'] = self.test_safety()
        
        # Test 6: Deployment Readiness
        print("\n6️⃣ Deployment Readiness")
        results['tests']['deployment'] = self.test_deployment_readiness()
        
        # Calculate overall assessment
        print("\n🏆 Overall Assessment")
        results['overall_assessment'] = self.calculate_overall_assessment(results)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def test_basic_functionality(self) -> dict:
        """Test basic model functionality."""
        
        test_results = {
            'input_format_tests': {},
            'output_validation': {},
            'error_handling': {}
        }
        
        print("   Testing input formats...")
        
        # Test different input formats
        formats_to_test = [
            ('HWC_uint8', np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)),
            ('CHW_float32', np.random.rand(3, 120, 160).astype(np.float32)),
            ('HWC_float32', np.random.rand(120, 160, 3).astype(np.float32)),
            ('normalized', np.random.rand(120, 160, 3).astype(np.float32)),
            ('high_values', np.random.randint(200, 255, (120, 160, 3), dtype=np.uint8)),
            ('low_values', np.random.randint(0, 50, (120, 160, 3), dtype=np.uint8))
        ]
        
        for format_name, test_input in formats_to_test:
            try:
                action = self.model_wrapper.compute_action(test_input)
                
                # Validate output
                valid_output = (
                    isinstance(action, np.ndarray) and
                    len(action) == 2 and
                    -1 <= action[0] <= 1 and
                    -1 <= action[1] <= 1
                )
                
                test_results['input_format_tests'][format_name] = {
                    'success': True,
                    'action': action.tolist(),
                    'valid_output': valid_output
                }
                
                print(f"      ✅ {format_name}: {action}")
                
            except Exception as e:
                test_results['input_format_tests'][format_name] = {
                    'success': False,
                    'error': str(e),
                    'valid_output': False
                }
                print(f"      ❌ {format_name}: {e}")
        
        # Test output consistency
        print("   Testing output consistency...")
        base_obs = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        actions = []
        
        for i in range(10):
            action = self.model_wrapper.compute_action(base_obs)
            actions.append(action)
        
        actions = np.array(actions)
        steering_std = np.std(actions[:, 0])
        throttle_std = np.std(actions[:, 1])
        
        test_results['output_validation'] = {
            'consistency_test': {
                'steering_std': steering_std,
                'throttle_std': throttle_std,
                'consistent': steering_std < 0.001 and throttle_std < 0.001
            },
            'range_validation': {
                'steering_in_range': all(-1 <= a[0] <= 1 for a in actions),
                'throttle_in_range': all(-1 <= a[1] <= 1 for a in actions)
            }
        }
        
        print(f"      Steering consistency: {steering_std:.6f}")
        print(f"      Throttle consistency: {throttle_std:.6f}")
        
        return test_results
    
    def test_performance(self) -> dict:
        """Test model performance metrics."""
        
        print("   Measuring inference speed...")
        
        # Inference speed test
        test_obs = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        inference_times = []
        
        # Warmup
        for _ in range(10):
            self.model_wrapper.compute_action(test_obs)
        
        # Actual measurement
        for _ in range(100):
            start_time = time.time()
            self.model_wrapper.compute_action(test_obs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
        
        # Memory usage estimation
        param_count = 0
        if hasattr(self.model_wrapper.model, 'parameters'):
            param_count = sum(p.numel() for p in self.model_wrapper.model.parameters())
        
        memory_mb = (param_count * 4) / (1024 * 1024)  # 4 bytes per float32
        
        performance_results = {
            'inference_speed': {
                'mean_time_ms': np.mean(inference_times) * 1000,
                'std_time_ms': np.std(inference_times) * 1000,
                'min_time_ms': np.min(inference_times) * 1000,
                'max_time_ms': np.max(inference_times) * 1000,
                'fps_estimate': 1.0 / np.mean(inference_times)
            },
            'model_size': {
                'parameters': param_count,
                'estimated_memory_mb': memory_mb
            },
            'real_time_capable': np.mean(inference_times) < 0.1  # 10 FPS minimum
        }
        
        fps = performance_results['inference_speed']['fps_estimate']
        avg_time = performance_results['inference_speed']['mean_time_ms']
        
        print(f"      Average inference time: {avg_time:.2f} ms")
        print(f"      Estimated FPS: {fps:.1f}")
        print(f"      Model parameters: {param_count:,}")
        print(f"      Estimated memory: {memory_mb:.1f} MB")
        
        return performance_results
    
    def test_robustness(self) -> dict:
        """Test model robustness to various conditions."""
        
        print("   Testing robustness to noise and variations...")
        
        base_obs = np.random.randint(50, 200, (120, 160, 3), dtype=np.uint8)
        base_action = self.model_wrapper.compute_action(base_obs)
        
        robustness_results = {
            'noise_robustness': {},
            'lighting_robustness': {},
            'blur_robustness': {}
        }
        
        # Noise robustness
        noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        for noise_level in noise_levels:
            deviations = []
            for _ in range(5):
                noise = np.random.normal(0, noise_level * 255, base_obs.shape)
                noisy_obs = np.clip(base_obs + noise, 0, 255).astype(np.uint8)
                noisy_action = self.model_wrapper.compute_action(noisy_obs)
                deviation = np.linalg.norm(noisy_action - base_action)
                deviations.append(deviation)
            
            robustness_results['noise_robustness'][f'level_{noise_level}'] = {
                'mean_deviation': np.mean(deviations),
                'max_deviation': np.max(deviations)
            }
        
        # Lighting robustness
        brightness_levels = [0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0]
        for brightness in brightness_levels:
            bright_obs = np.clip(base_obs * brightness, 0, 255).astype(np.uint8)
            bright_action = self.model_wrapper.compute_action(bright_obs)
            deviation = np.linalg.norm(bright_action - base_action)
            
            robustness_results['lighting_robustness'][f'brightness_{brightness}'] = {
                'deviation': deviation,
                'action': bright_action.tolist()
            }
        
        print(f"      Noise robustness: {len(noise_levels)} levels tested")
        print(f"      Lighting robustness: {len(brightness_levels)} levels tested")
        
        return robustness_results
    
    def test_scenarios(self) -> dict:
        """Test model behavior in different driving scenarios."""
        
        print("   Testing scenario-specific behaviors...")
        
        scenarios = {
            'straight_road': self.create_straight_road_scenario(),
            'left_curve': self.create_curve_scenario('left'),
            'right_curve': self.create_curve_scenario('right'),
            'intersection': self.create_intersection_scenario(),
            'narrow_lane': self.create_narrow_lane_scenario(),
            'wide_road': self.create_wide_road_scenario()
        }
        
        scenario_results = {}
        
        for scenario_name, scenario_obs in scenarios.items():
            action = self.model_wrapper.compute_action(scenario_obs)
            
            # Analyze action appropriateness
            steering, throttle = action[0], action[1]
            
            analysis = {
                'action': action.tolist(),
                'steering_magnitude': abs(steering),
                'throttle_value': throttle,
                'appropriate_response': self.analyze_scenario_response(
                    scenario_name, steering, throttle
                )
            }
            
            scenario_results[scenario_name] = analysis
            
            print(f"      {scenario_name}: steering={steering:.3f}, throttle={throttle:.3f}")
        
        return scenario_results
    
    def create_straight_road_scenario(self) -> np.ndarray:
        """Create a straight road scenario."""
        obs = np.full((120, 160, 3), 100, dtype=np.uint8)  # Gray background
        
        # Road surface
        obs[60:120, 40:120, :] = [150, 150, 150]  # Road
        
        # Lane markings
        obs[60:120, 75:85, :] = [255, 255, 255]  # Center line
        obs[60:120, 45:50, :] = [255, 255, 255]  # Left edge
        obs[60:120, 115:120, :] = [255, 255, 255]  # Right edge
        
        return obs
    
    def create_curve_scenario(self, direction: str) -> np.ndarray:
        """Create a curved road scenario."""
        obs = np.full((120, 160, 3), 100, dtype=np.uint8)
        
        # Create curved road
        for y in range(60, 120):
            if direction == 'left':
                offset = int((y - 60) * 0.5)  # Curve left
            else:
                offset = -int((y - 60) * 0.5)  # Curve right
            
            road_left = max(0, 40 + offset)
            road_right = min(160, 120 + offset)
            
            if road_right > road_left:
                obs[y, road_left:road_right, :] = [150, 150, 150]
                
                # Lane markings
                center = (road_left + road_right) // 2
                obs[y, max(0, center-2):min(160, center+3), :] = [255, 255, 255]
        
        return obs
    
    def create_intersection_scenario(self) -> np.ndarray:
        """Create an intersection scenario."""
        obs = np.full((120, 160, 3), 100, dtype=np.uint8)
        
        # Main road
        obs[60:120, 40:120, :] = [150, 150, 150]
        
        # Cross road
        obs[80:100, 20:140, :] = [150, 150, 150]
        
        # Stop line
        obs[78:80, 40:120, :] = [255, 255, 255]
        
        return obs
    
    def create_narrow_lane_scenario(self) -> np.ndarray:
        """Create a narrow lane scenario."""
        obs = np.full((120, 160, 3), 100, dtype=np.uint8)
        
        # Narrow road
        obs[60:120, 60:100, :] = [150, 150, 150]
        
        # Lane markings
        obs[60:120, 78:82, :] = [255, 255, 255]  # Center
        obs[60:120, 58:62, :] = [255, 255, 255]  # Left
        obs[60:120, 98:102, :] = [255, 255, 255]  # Right
        
        return obs
    
    def create_wide_road_scenario(self) -> np.ndarray:
        """Create a wide road scenario."""
        obs = np.full((120, 160, 3), 100, dtype=np.uint8)
        
        # Wide road
        obs[60:120, 20:140, :] = [150, 150, 150]
        
        # Multiple lanes
        obs[60:120, 60:65, :] = [255, 255, 255]  # Lane 1
        obs[60:120, 95:100, :] = [255, 255, 255]  # Lane 2
        obs[60:120, 78:82, :] = [255, 255, 0]  # Center divider
        
        return obs
    
    def analyze_scenario_response(self, scenario: str, steering: float, throttle: float) -> dict:
        """Analyze if the model's response is appropriate for the scenario."""
        
        analysis = {'appropriate': True, 'reasons': []}
        
        if scenario == 'straight_road':
            if abs(steering) > 0.2:
                analysis['appropriate'] = False
                analysis['reasons'].append("Excessive steering for straight road")
            if throttle < 0:
                analysis['appropriate'] = False
                analysis['reasons'].append("Reverse throttle on straight road")
        
        elif 'curve' in scenario:
            if abs(steering) < 0.1:
                analysis['appropriate'] = False
                analysis['reasons'].append("Insufficient steering for curve")
            
            direction = scenario.split('_')[0]
            if direction == 'left' and steering > 0:
                analysis['appropriate'] = False
                analysis['reasons'].append("Wrong steering direction for left curve")
            elif direction == 'right' and steering < 0:
                analysis['appropriate'] = False
                analysis['reasons'].append("Wrong steering direction for right curve")
        
        elif scenario == 'intersection':
            if throttle > 0.5:
                analysis['appropriate'] = False
                analysis['reasons'].append("Too fast approaching intersection")
        
        elif scenario == 'narrow_lane':
            if abs(steering) > 0.3:
                analysis['appropriate'] = False
                analysis['reasons'].append("Excessive steering for narrow lane")
        
        if not analysis['reasons']:
            analysis['reasons'].append("Response appears appropriate")
        
        return analysis
    
    def test_safety(self) -> dict:
        """Test safety-related behaviors."""
        
        print("   Testing safety behaviors...")
        
        safety_results = {
            'emergency_scenarios': {},
            'speed_control': {},
            'stability': {}
        }
        
        # Test emergency scenarios
        emergency_scenarios = {
            'obstacle_ahead': self.create_obstacle_scenario(),
            'sharp_turn': self.create_sharp_turn_scenario(),
            'poor_visibility': self.create_poor_visibility_scenario()
        }
        
        for scenario_name, scenario_obs in emergency_scenarios.items():
            action = self.model_wrapper.compute_action(scenario_obs)
            steering, throttle = action[0], action[1]
            
            # Evaluate safety response
            safe_response = True
            safety_notes = []
            
            if scenario_name == 'obstacle_ahead' and throttle > 0.3:
                safe_response = False
                safety_notes.append("Should reduce speed near obstacles")
            
            if scenario_name == 'sharp_turn' and abs(steering) < 0.2:
                safe_response = False
                safety_notes.append("Insufficient steering for sharp turn")
            
            if abs(steering) > 0.9 or abs(throttle) > 0.9:
                safe_response = False
                safety_notes.append("Extreme control inputs detected")
            
            safety_results['emergency_scenarios'][scenario_name] = {
                'action': action.tolist(),
                'safe_response': safe_response,
                'notes': safety_notes
            }
        
        # Test speed control
        speed_test_obs = self.create_straight_road_scenario()
        actions_over_time = []
        
        for _ in range(20):
            action = self.model_wrapper.compute_action(speed_test_obs)
            actions_over_time.append(action[1])  # Throttle values
        
        throttle_stability = np.std(actions_over_time)
        avg_throttle = np.mean(actions_over_time)
        
        safety_results['speed_control'] = {
            'average_throttle': avg_throttle,
            'throttle_stability': throttle_stability,
            'reasonable_speed': 0.1 <= avg_throttle <= 0.7,
            'stable_control': throttle_stability < 0.1
        }
        
        print(f"      Emergency scenarios: {len(emergency_scenarios)} tested")
        print(f"      Average throttle: {avg_throttle:.3f}")
        print(f"      Throttle stability: {throttle_stability:.3f}")
        
        return safety_results
    
    def create_obstacle_scenario(self) -> np.ndarray:
        """Create scenario with obstacle ahead."""
        obs = self.create_straight_road_scenario()
        
        # Add obstacle
        obs[70:90, 70:90, :] = [50, 50, 50]  # Dark obstacle
        
        return obs
    
    def create_sharp_turn_scenario(self) -> np.ndarray:
        """Create sharp turn scenario."""
        obs = np.full((120, 160, 3), 100, dtype=np.uint8)
        
        # Sharp left turn
        for y in range(60, 120):
            offset = int((y - 60) * 1.5)  # Sharp curve
            road_left = max(0, 40 + offset)
            road_right = min(160, 100 + offset)
            
            if road_right > road_left:
                obs[y, road_left:road_right, :] = [150, 150, 150]
        
        return obs
    
    def create_poor_visibility_scenario(self) -> np.ndarray:
        """Create poor visibility scenario."""
        obs = self.create_straight_road_scenario()
        
        # Add fog/darkness effect
        obs = obs * 0.3 + np.random.randint(0, 50, obs.shape)
        obs = np.clip(obs, 0, 255).astype(np.uint8)
        
        return obs
    
    def test_deployment_readiness(self) -> dict:
        """Test deployment readiness."""
        
        print("   Testing deployment readiness...")
        
        deployment_results = {
            'model_loading': True,
            'inference_interface': True,
            'performance_requirements': {},
            'compatibility': {}
        }
        
        # Performance requirements for real-time deployment
        perf_data = self.test_performance()
        fps = perf_data['inference_speed']['fps_estimate']
        memory_mb = perf_data['model_size']['estimated_memory_mb']
        
        deployment_results['performance_requirements'] = {
            'real_time_capable': fps >= 10,  # Minimum 10 FPS
            'memory_efficient': memory_mb <= 500,  # Max 500 MB
            'fps': fps,
            'memory_mb': memory_mb
        }
        
        # Test compatibility with deployment interface
        try:
            # Test the exact interface used in deployment
            test_obs = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
            action = self.model_wrapper.compute_action(test_obs, explore=False)
            
            deployment_results['compatibility']['compute_action_interface'] = True
            deployment_results['compatibility']['action_format'] = len(action) == 2
            deployment_results['compatibility']['action_range'] = (
                -1 <= action[0] <= 1 and -1 <= action[1] <= 1
            )
            
        except Exception as e:
            deployment_results['compatibility']['compute_action_interface'] = False
            deployment_results['compatibility']['error'] = str(e)
        
        print(f"      Real-time capable: {deployment_results['performance_requirements']['real_time_capable']}")
        print(f"      Memory efficient: {deployment_results['performance_requirements']['memory_efficient']}")
        print(f"      Interface compatible: {deployment_results['compatibility']['compute_action_interface']}")
        
        return deployment_results
    
    def calculate_overall_assessment(self, results: dict) -> dict:
        """Calculate overall model assessment."""
        
        # Extract key metrics
        basic_tests = results['tests']['basic_functionality']
        performance = results['tests']['performance']
        scenarios = results['tests']['scenarios']
        safety = results['tests']['safety']
        deployment = results['tests']['deployment']
        
        # Calculate component scores (0-100)
        functionality_score = self.score_functionality(basic_tests)
        performance_score = self.score_performance(performance)
        scenario_score = self.score_scenarios(scenarios)
        safety_score = self.score_safety(safety)
        deployment_score = self.score_deployment(deployment)
        
        # Overall weighted score
        overall_score = (
            functionality_score * 0.25 +
            performance_score * 0.20 +
            scenario_score * 0.25 +
            safety_score * 0.20 +
            deployment_score * 0.10
        )
        
        # Determine rating
        if overall_score >= 90:
            rating = "EXCELLENT"
            emoji = "🏆"
        elif overall_score >= 80:
            rating = "GOOD"
            emoji = "✅"
        elif overall_score >= 70:
            rating = "ACCEPTABLE"
            emoji = "👍"
        elif overall_score >= 60:
            rating = "NEEDS_IMPROVEMENT"
            emoji = "⚠️"
        else:
            rating = "POOR"
            emoji = "❌"
        
        assessment = {
            'overall_score': round(overall_score, 1),
            'rating': rating,
            'emoji': emoji,
            'component_scores': {
                'functionality': round(functionality_score, 1),
                'performance': round(performance_score, 1),
                'scenarios': round(scenario_score, 1),
                'safety': round(safety_score, 1),
                'deployment': round(deployment_score, 1)
            },
            'summary': self.generate_summary(overall_score, results),
            'recommendations': self.generate_recommendations(results)
        }
        
        return assessment
    
    def score_functionality(self, basic_tests: dict) -> float:
        """Score basic functionality tests."""
        input_tests = basic_tests['input_format_tests']
        output_tests = basic_tests['output_validation']
        
        # Count successful input format tests
        successful_inputs = sum(1 for test in input_tests.values() if test['success'])
        total_inputs = len(input_tests)
        input_score = (successful_inputs / total_inputs) * 100
        
        # Check output validation
        consistency_score = 100 if output_tests['consistency_test']['consistent'] else 50
        range_score = 100 if (output_tests['range_validation']['steering_in_range'] and 
                             output_tests['range_validation']['throttle_in_range']) else 0
        
        return (input_score + consistency_score + range_score) / 3
    
    def score_performance(self, performance: dict) -> float:
        """Score performance metrics."""
        fps = performance['inference_speed']['fps_estimate']
        memory_mb = performance['model_size']['estimated_memory_mb']
        
        # FPS score (target: 20+ FPS = 100, 10 FPS = 50, <5 FPS = 0)
        fps_score = min(100, max(0, (fps - 5) / 15 * 100))
        
        # Memory score (target: <100MB = 100, <500MB = 50, >1GB = 0)
        if memory_mb < 100:
            memory_score = 100
        elif memory_mb < 500:
            memory_score = 50
        elif memory_mb < 1000:
            memory_score = 25
        else:
            memory_score = 0
        
        return (fps_score + memory_score) / 2
    
    def score_scenarios(self, scenarios: dict) -> float:
        """Score scenario-based tests."""
        appropriate_responses = sum(
            1 for scenario in scenarios.values() 
            if scenario['appropriate_response']['appropriate']
        )
        total_scenarios = len(scenarios)
        
        return (appropriate_responses / total_scenarios) * 100
    
    def score_safety(self, safety: dict) -> float:
        """Score safety tests."""
        emergency_tests = safety['emergency_scenarios']
        speed_control = safety['speed_control']
        
        # Emergency response score
        safe_responses = sum(
            1 for test in emergency_tests.values() 
            if test['safe_response']
        )
        emergency_score = (safe_responses / len(emergency_tests)) * 100
        
        # Speed control score
        speed_score = 100 if (speed_control['reasonable_speed'] and 
                             speed_control['stable_control']) else 50
        
        return (emergency_score + speed_score) / 2
    
    def score_deployment(self, deployment: dict) -> float:
        """Score deployment readiness."""
        perf_req = deployment['performance_requirements']
        compatibility = deployment['compatibility']
        
        # Performance requirements score
        perf_score = 100 if (perf_req['real_time_capable'] and 
                            perf_req['memory_efficient']) else 50
        
        # Compatibility score
        compat_score = 100 if (compatibility.get('compute_action_interface', False) and
                              compatibility.get('action_format', False) and
                              compatibility.get('action_range', False)) else 0
        
        return (perf_score + compat_score) / 2
    
    def generate_summary(self, overall_score: float, results: dict) -> str:
        """Generate assessment summary."""
        if overall_score >= 90:
            return "Excellent model ready for production deployment with outstanding performance across all metrics."
        elif overall_score >= 80:
            return "Good model suitable for deployment with strong performance in most areas."
        elif overall_score >= 70:
            return "Acceptable model that can be deployed with some limitations or improvements needed."
        elif overall_score >= 60:
            return "Model needs improvement before deployment, several issues identified."
        else:
            return "Poor model performance, significant improvements required before deployment."
    
    def generate_recommendations(self, results: dict) -> list:
        """Generate improvement recommendations."""
        recommendations = []
        
        performance = results['tests']['performance']
        safety = results['tests']['safety']
        deployment = results['tests']['deployment']
        
        # Performance recommendations
        fps = performance['inference_speed']['fps_estimate']
        if fps < 10:
            recommendations.append("Optimize model for faster inference (current: {:.1f} FPS)".format(fps))
        
        # Safety recommendations
        emergency_tests = safety['emergency_scenarios']
        unsafe_responses = [name for name, test in emergency_tests.items() if not test['safe_response']]
        if unsafe_responses:
            recommendations.append(f"Improve safety responses for: {', '.join(unsafe_responses)}")
        
        # Deployment recommendations
        if not deployment['performance_requirements']['real_time_capable']:
            recommendations.append("Improve inference speed for real-time deployment")
        
        if not deployment['performance_requirements']['memory_efficient']:
            recommendations.append("Reduce model size for memory efficiency")
        
        if not recommendations:
            recommendations.append("Model performs well across all tested areas")
        
        return recommendations
    
    def save_results(self, results: dict):
        """Save evaluation results."""
        
        # Save JSON results
        json_file = self.results_dir / "evaluation_results.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save human-readable report
        report_file = self.results_dir / "evaluation_report.md"
        self.generate_report(results, report_file)
        
        logger.info(f"Results saved to: {self.results_dir}")
    
    def generate_report(self, results: dict, report_file: Path):
        """Generate human-readable report."""
        
        assessment = results['overall_assessment']
        
        with open(report_file, 'w') as f:
            f.write("# Champion Model Evaluation Report\n\n")
            f.write(f"**Model**: {results['model_path']}\n")
            f.write(f"**Evaluation Date**: {results['timestamp']}\n")
            f.write(f"**Model Type**: {results['model_info']['model_type']}\n\n")
            
            # Overall Assessment
            f.write("## Overall Assessment\n\n")
            f.write(f"{assessment['emoji']} **Overall Score**: {assessment['overall_score']}/100\n")
            f.write(f"**Rating**: {assessment['rating']}\n\n")
            f.write(f"**Summary**: {assessment['summary']}\n\n")
            
            # Component Scores
            f.write("### Component Scores\n\n")
            for component, score in assessment['component_scores'].items():
                f.write(f"- **{component.title()}**: {score}/100\n")
            f.write("\n")
            
            # Recommendations
            f.write("### Recommendations\n\n")
            for rec in assessment['recommendations']:
                f.write(f"- {rec}\n")
            f.write("\n")
            
            # Detailed Results
            performance = results['tests']['performance']
            f.write("## Performance Details\n\n")
            f.write(f"- **Inference Speed**: {performance['inference_speed']['mean_time_ms']:.2f} ms\n")
            f.write(f"- **FPS Estimate**: {performance['inference_speed']['fps_estimate']:.1f}\n")
            f.write(f"- **Model Parameters**: {performance['model_size']['parameters']:,}\n")
            f.write(f"- **Memory Usage**: {performance['model_size']['estimated_memory_mb']:.1f} MB\n\n")
            
            # Safety Results
            safety = results['tests']['safety']
            f.write("## Safety Assessment\n\n")
            f.write(f"- **Speed Control**: {'✅' if safety['speed_control']['reasonable_speed'] else '❌'}\n")
            f.write(f"- **Control Stability**: {'✅' if safety['speed_control']['stable_control'] else '❌'}\n")
            f.write(f"- **Average Throttle**: {safety['speed_control']['average_throttle']:.3f}\n\n")
            
            # Deployment Readiness
            deployment = results['tests']['deployment']
            f.write("## Deployment Readiness\n\n")
            f.write(f"- **Real-time Capable**: {'✅' if deployment['performance_requirements']['real_time_capable'] else '❌'}\n")
            f.write(f"- **Memory Efficient**: {'✅' if deployment['performance_requirements']['memory_efficient'] else '❌'}\n")
            f.write(f"- **Interface Compatible**: {'✅' if deployment['compatibility']['compute_action_interface'] else '❌'}\n")


def main():
    """Main evaluation function."""
    
    print("🏆 Simple Champion Model Evaluation")
    print("=" * 60)
    
    # Check if model exists
    model_path = "champion_model.pth"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("   Run: python create_working_champion_model.py")
        return False
    
    try:
        # Create evaluator and run evaluation
        evaluator = SimpleChampionEvaluator(model_path)
        results = evaluator.run_evaluation()
        
        # Print final summary
        assessment = results['overall_assessment']
        
        print("\n" + "=" * 60)
        print("🎉 EVALUATION COMPLETED!")
        print(f"\n{assessment['emoji']} **OVERALL SCORE: {assessment['overall_score']}/100**")
        print(f"📊 **RATING: {assessment['rating']}**")
        
        print(f"\n📈 Component Breakdown:")
        for component, score in assessment['component_scores'].items():
            emoji = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            print(f"   {emoji} {component.title()}: {score}/100")
        
        print(f"\n📝 Summary:")
        print(f"   {assessment['summary']}")
        
        if assessment['recommendations']:
            print(f"\n💡 Key Recommendations:")
            for rec in assessment['recommendations'][:3]:  # Show top 3
                print(f"   • {rec}")
        
        print(f"\n📁 Detailed results: {evaluator.results_dir}")
        
        # Deployment readiness check
        deployment_ready = (
            assessment['overall_score'] >= 70 and
            assessment['component_scores']['deployment'] >= 70
        )
        
        if deployment_ready:
            print(f"\n🚀 **MODEL IS READY FOR DEPLOYMENT!**")
            print(f"   Use: python duckiebot_deployment/deploy_to_duckiebot.py")
        else:
            print(f"\n⚠️ **MODEL NEEDS IMPROVEMENT BEFORE DEPLOYMENT**")
            print(f"   Address the recommendations above first")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)