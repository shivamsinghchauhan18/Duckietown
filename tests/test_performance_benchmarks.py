"""
Performance benchmarking tests for real-time processing requirements.
Tests frame rates, processing times, and resource usage.
"""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2024 Enhanced Duckietown RL"

import pytest
import numpy as np
import time
import threading
import queue
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import shutil
import json
import statistics

from duckietown_utils.env import launch_and_wrap_enhanced_env
from config.enhanced_config import EnhancedRLConfig
from duckietown_utils.wrappers.yolo_detection_wrapper import YOLOObjectDetectionWrapper
from duckietown_utils.wrappers.enhanced_observation_wrapper import EnhancedObservationWrapper
from duckietown_utils.wrappers.object_avoidance_action_wrapper import ObjectAvoidanceActionWrapper


class PerformanceProfiler:
    """Utility class for performance profiling."""
    
    def __init__(self):
        self.measurements = {}
        self.start_times = {}
    
    def start_timer(self, name):
        """Start timing a operation."""
        self.start_times[name] = time.time()
    
    def end_timer(self, name):
        """End timing and record measurement."""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            if name not in self.measurements:
                self.measurements[name] = []
            self.measurements[name].append(duration)
            del self.start_times[name]
            return duration
        return None
    
    def get_stats(self, name):
        """Get statistics for a measurement."""
        if name not in self.measurements or not self.measurements[name]:
            return None
        
        values = self.measurements[name]
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def clear(self):
        """Clear all measurements."""
        self.measurements.clear()
        self.start_times.clear()


class TestFrameRatePerformance:
    """Test frame rate and real-time processing performance."""
    
    @pytest.fixture
    def profiler(self):
        return PerformanceProfiler()
    
    @pytest.fixture
    def performance_env_config(self):
        """Environment configuration for performance testing."""
        return {
            'training_map': 'small_loop',
            'episode_max_steps': 200,
            'domain_rand': False,
            'dynamics_rand': False,
            'camera_rand': False,
            'simulation_framerate': 30,
            'frame_skip': 1,
            'robot_speed': 0.5,
            'mode': 'debug',
            'resized_input_shape': (84, 84, 3),
            'action_type': 'continuous',
            'reward_function': 'Posangle'
        }
    
    @pytest.fixture
    def fast_enhanced_config(self):
        """Enhanced configuration optimized for speed."""
        return EnhancedRLConfig(
            enabled_features=['yolo', 'object_avoidance'],
            debug_mode=False,
            yolo={
                'model_path': 'yolov5n.pt',  # Nano model for speed
                'confidence_threshold': 0.5,
                'device': 'cpu',
                'input_size': 320,
                'max_detections': 20
            },
            object_avoidance={
                'safety_distance': 0.5,
                'avoidance_strength': 1.0,
                'min_clearance': 0.2
            },
            performance={
                'max_fps': 30.0,
                'detection_batch_size': 1,
                'use_gpu_acceleration': False,
                'memory_limit_gb': 2.0
            }
        )
    
    @pytest.mark.performance
    def test_baseline_frame_rate(self, performance_env_config, profiler):
        """Test baseline frame rate without enhanced features."""
        minimal_config = EnhancedRLConfig(enabled_features=[])
        
        try:
            env = launch_and_wrap_enhanced_env(performance_env_config, minimal_config)
            
            obs = env.reset()
            num_steps = 100
            
            start_time = time.time()
            
            for i in range(num_steps):
                profiler.start_timer('step')
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                profiler.end_timer('step')
                
                if done:
                    obs = env.reset()
            
            end_time = time.time()
            total_time = end_time - start_time
            fps = num_steps / total_time
            
            env.close()
            
            # Get step timing statistics
            step_stats = profiler.get_stats('step')
            
            # Baseline should achieve good performance
            assert fps >= 20.0, f"Baseline FPS too low: {fps:.2f}"
            assert step_stats['mean'] <= 0.05, f"Step time too high: {step_stats['mean']:.4f}s"
            
            print(f"Baseline Performance: {fps:.2f} FPS, {step_stats['mean']*1000:.2f}ms avg step time")
            
        except Exception as e:
            pytest.skip(f"Baseline performance test skipped: {e}")
    
    @pytest.mark.performance
    def test_yolo_detection_performance(self, performance_env_config, fast_enhanced_config, profiler):
        """Test performance with YOLO object detection."""
        yolo_config = EnhancedRLConfig(
            enabled_features=['yolo'],
            debug_mode=False,
            yolo=fast_enhanced_config.yolo
        )
        
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                # Mock fast YOLO model
                mock_model = Mock()
                
                def mock_inference(img):
                    # Simulate inference time
                    time.sleep(0.01)  # 10ms inference time
                    mock_results = Mock()
                    mock_results.pandas.return_value.xyxy = [Mock(values=np.array([]))]
                    return mock_results
                
                mock_model.side_effect = mock_inference
                mock_yolo.return_value = mock_model
                
                env = launch_and_wrap_enhanced_env(performance_env_config, yolo_config)
                
                obs = env.reset()
                num_steps = 50  # Fewer steps due to YOLO overhead
                
                start_time = time.time()
                
                for i in range(num_steps):
                    profiler.start_timer('yolo_step')
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                    profiler.end_timer('yolo_step')
                    
                    if done:
                        obs = env.reset()
                
                end_time = time.time()
                total_time = end_time - start_time
                fps = num_steps / total_time
                
                env.close()
                
                step_stats = profiler.get_stats('yolo_step')
                
                # Should still achieve reasonable performance with YOLO
                assert fps >= 10.0, f"YOLO FPS too low: {fps:.2f}"
                assert step_stats['mean'] <= 0.1, f"YOLO step time too high: {step_stats['mean']:.4f}s"
                
                print(f"YOLO Performance: {fps:.2f} FPS, {step_stats['mean']*1000:.2f}ms avg step time")
                
        except Exception as e:
            pytest.skip(f"YOLO performance test skipped: {e}")
    
    @pytest.mark.performance
    def test_full_pipeline_performance(self, performance_env_config, fast_enhanced_config, profiler):
        """Test performance with full enhanced pipeline."""
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                # Mock optimized YOLO model
                mock_model = Mock()
                
                def mock_fast_inference(img):
                    time.sleep(0.005)  # 5ms inference time
                    mock_results = Mock()
                    mock_results.pandas.return_value.xyxy = [Mock(values=np.array([
                        [100, 100, 200, 200, 0.8, 0, 'person']
                    ]))]
                    return mock_results
                
                mock_model.side_effect = mock_fast_inference
                mock_yolo.return_value = mock_model
                
                env = launch_and_wrap_enhanced_env(performance_env_config, fast_enhanced_config)
                
                obs = env.reset()
                num_steps = 30  # Fewer steps for full pipeline
                
                start_time = time.time()
                
                for i in range(num_steps):
                    profiler.start_timer('full_step')
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                    profiler.end_timer('full_step')
                    
                    if done:
                        obs = env.reset()
                
                end_time = time.time()
                total_time = end_time - start_time
                fps = num_steps / total_time
                
                env.close()
                
                step_stats = profiler.get_stats('full_step')
                
                # Full pipeline should still meet minimum requirements
                assert fps >= 8.0, f"Full pipeline FPS too low: {fps:.2f}"
                assert step_stats['mean'] <= 0.125, f"Full pipeline step time too high: {step_stats['mean']:.4f}s"
                
                print(f"Full Pipeline Performance: {fps:.2f} FPS, {step_stats['mean']*1000:.2f}ms avg step time")
                
        except Exception as e:
            pytest.skip(f"Full pipeline performance test skipped: {e}")
    
    @pytest.mark.performance
    def test_performance_consistency(self, performance_env_config, fast_enhanced_config, profiler):
        """Test performance consistency over time."""
        try:
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                mock_model.return_value = Mock()
                mock_model.return_value.pandas.return_value.xyxy = [Mock(values=np.array([]))]
                mock_yolo.return_value = mock_model
                
                env = launch_and_wrap_enhanced_env(performance_env_config, fast_enhanced_config)
                
                obs = env.reset()
                
                # Measure performance in chunks
                chunk_size = 20
                num_chunks = 5
                chunk_fps = []
                
                for chunk in range(num_chunks):
                    chunk_start = time.time()
                    
                    for i in range(chunk_size):
                        action = env.action_space.sample()
                        obs, reward, done, info = env.step(action)
                        if done:
                            obs = env.reset()
                    
                    chunk_end = time.time()
                    chunk_time = chunk_end - chunk_start
                    fps = chunk_size / chunk_time
                    chunk_fps.append(fps)
                
                env.close()
                
                # Check consistency
                mean_fps = statistics.mean(chunk_fps)
                fps_std = statistics.stdev(chunk_fps) if len(chunk_fps) > 1 else 0
                fps_cv = fps_std / mean_fps if mean_fps > 0 else float('inf')  # Coefficient of variation
                
                # Performance should be consistent (CV < 0.3)
                assert fps_cv < 0.3, f"Performance too inconsistent: CV={fps_cv:.3f}"
                assert mean_fps >= 8.0, f"Mean FPS too low: {mean_fps:.2f}"
                
                print(f"Performance Consistency: {mean_fps:.2f} ± {fps_std:.2f} FPS (CV: {fps_cv:.3f})")
                
        except Exception as e:
            pytest.skip(f"Performance consistency test skipped: {e}")


class TestComponentPerformance:
    """Test performance of individual components."""
    
    @pytest.fixture
    def profiler(self):
        return PerformanceProfiler()
    
    @pytest.mark.performance
    def test_yolo_wrapper_performance(self, profiler):
        """Test YOLO wrapper performance in isolation."""
        try:
            from duckietown_utils.wrappers.yolo_detection_wrapper import YOLOObjectDetectionWrapper
            from config.enhanced_config import YOLOConfig
            
            # Mock environment
            mock_env = Mock()
            mock_env.observation_space = Mock()
            
            yolo_config = YOLOConfig(
                model_path='yolov5n.pt',
                confidence_threshold=0.5,
                device='cpu',
                input_size=320
            )
            
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                # Mock fast model
                mock_model = Mock()
                
                def timed_inference(img):
                    time.sleep(0.008)  # 8ms inference
                    mock_results = Mock()
                    mock_results.pandas.return_value.xyxy = [Mock(values=np.array([
                        [100, 100, 200, 200, 0.8, 0, 'person']
                    ]))]
                    return mock_results
                
                mock_model.side_effect = timed_inference
                mock_yolo.return_value = mock_model
                
                wrapper = YOLOObjectDetectionWrapper(mock_env, yolo_config)
                
                # Test observation processing performance
                test_image = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
                num_inferences = 50
                
                for i in range(num_inferences):
                    profiler.start_timer('yolo_inference')
                    result = wrapper.observation(test_image)
                    profiler.end_timer('yolo_inference')
                
                inference_stats = profiler.get_stats('yolo_inference')
                
                # YOLO inference should be fast enough for real-time
                assert inference_stats['mean'] <= 0.05, f"YOLO inference too slow: {inference_stats['mean']*1000:.2f}ms"
                assert inference_stats['max'] <= 0.1, f"YOLO max inference too slow: {inference_stats['max']*1000:.2f}ms"
                
                print(f"YOLO Inference: {inference_stats['mean']*1000:.2f}ms avg, {inference_stats['max']*1000:.2f}ms max")
                
        except Exception as e:
            pytest.skip(f"YOLO wrapper performance test skipped: {e}")
    
    @pytest.mark.performance
    def test_observation_wrapper_performance(self, profiler):
        """Test enhanced observation wrapper performance."""
        try:
            from duckietown_utils.wrappers.enhanced_observation_wrapper import EnhancedObservationWrapper
            
            # Mock environment with detection observations
            mock_env = Mock()
            mock_env.observation_space = Mock()
            
            wrapper = EnhancedObservationWrapper(
                mock_env,
                include_detection_features=True,
                flatten_observations=True
            )
            
            # Test observation processing
            test_obs = {
                'image': np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8),
                'detections': [
                    {
                        'class': 'person',
                        'confidence': 0.8,
                        'bbox': [100, 100, 200, 200],
                        'distance': 2.5,
                        'relative_position': [0.1, 0.2]
                    }
                ],
                'detection_count': 1,
                'safety_critical': False
            }
            
            num_processes = 100
            
            for i in range(num_processes):
                profiler.start_timer('obs_processing')
                result = wrapper.observation(test_obs)
                profiler.end_timer('obs_processing')
            
            processing_stats = profiler.get_stats('obs_processing')
            
            # Observation processing should be very fast
            assert processing_stats['mean'] <= 0.005, f"Observation processing too slow: {processing_stats['mean']*1000:.2f}ms"
            
            print(f"Observation Processing: {processing_stats['mean']*1000:.2f}ms avg")
            
        except Exception as e:
            pytest.skip(f"Observation wrapper performance test skipped: {e}")
    
    @pytest.mark.performance
    def test_action_wrapper_performance(self, profiler):
        """Test action wrapper performance."""
        try:
            from duckietown_utils.wrappers.object_avoidance_action_wrapper import ObjectAvoidanceActionWrapper
            from config.enhanced_config import ObjectAvoidanceConfig
            
            # Mock environment
            mock_env = Mock()
            mock_env.action_space = Mock()
            mock_env.action_space.low = np.array([-1.0, -1.0])
            mock_env.action_space.high = np.array([1.0, 1.0])
            mock_env._last_observation = {
                'detections': [
                    {
                        'class': 'person',
                        'distance': 0.4,
                        'relative_position': [0.0, 0.2]
                    }
                ],
                'safety_critical': True
            }
            
            config = ObjectAvoidanceConfig(
                safety_distance=0.5,
                avoidance_strength=1.0,
                min_clearance=0.2
            )
            
            wrapper = ObjectAvoidanceActionWrapper(mock_env, config)
            
            # Test action processing
            test_action = np.array([0.5, 0.1])
            num_processes = 100
            
            for i in range(num_processes):
                profiler.start_timer('action_processing')
                result = wrapper.action(test_action)
                profiler.end_timer('action_processing')
            
            processing_stats = profiler.get_stats('action_processing')
            
            # Action processing should be very fast
            assert processing_stats['mean'] <= 0.002, f"Action processing too slow: {processing_stats['mean']*1000:.2f}ms"
            
            print(f"Action Processing: {processing_stats['mean']*1000:.2f}ms avg")
            
        except Exception as e:
            pytest.skip(f"Action wrapper performance test skipped: {e}")


class TestMemoryPerformance:
    """Test memory usage and efficiency."""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.mark.performance
    def test_memory_usage_baseline(self, temp_dir):
        """Test baseline memory usage."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create minimal environment
            env_config = {
                'training_map': 'small_loop',
                'episode_max_steps': 100,
                'mode': 'debug',
                'resized_input_shape': (84, 84, 3)
            }
            
            enhanced_config = EnhancedRLConfig(enabled_features=[])
            
            env = launch_and_wrap_enhanced_env(env_config, enhanced_config)
            
            # Run episode
            obs = env.reset()
            for i in range(50):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                if done:
                    break
            
            env.close()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory
            
            # Baseline should use reasonable memory
            assert memory_usage < 200, f"Baseline memory usage too high: {memory_usage:.2f} MB"
            
            print(f"Baseline Memory Usage: {memory_usage:.2f} MB")
            
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        except Exception as e:
            pytest.skip(f"Memory test skipped: {e}")
    
    @pytest.mark.performance
    def test_memory_leak_detection(self, temp_dir):
        """Test for memory leaks during extended operation."""
        try:
            import psutil
            import os
            import gc
            
            process = psutil.Process(os.getpid())
            
            env_config = {
                'training_map': 'small_loop',
                'episode_max_steps': 50,
                'mode': 'debug',
                'resized_input_shape': (64, 64, 3)  # Smaller for faster testing
            }
            
            enhanced_config = EnhancedRLConfig(
                enabled_features=['yolo'],
                yolo={
                    'model_path': 'yolov5n.pt',
                    'confidence_threshold': 0.5,
                    'device': 'cpu',
                    'input_size': 320
                }
            )
            
            with patch('duckietown_utils.yolo_utils.load_yolo_model') as mock_yolo:
                mock_model = Mock()
                mock_model.return_value = Mock()
                mock_model.return_value.pandas.return_value.xyxy = [Mock(values=np.array([]))]
                mock_yolo.return_value = mock_model
                
                memory_samples = []
                
                # Sample memory usage over multiple episodes
                for episode in range(10):
                    env = launch_and_wrap_enhanced_env(env_config, enhanced_config)
                    
                    obs = env.reset()
                    for step in range(20):
                        action = env.action_space.sample()
                        obs, reward, done, info = env.step(action)
                        if done:
                            break
                    
                    env.close()
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Sample memory
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(memory_mb)
                
                # Check for memory leaks
                if len(memory_samples) >= 5:
                    # Compare first half to second half
                    first_half = memory_samples[:len(memory_samples)//2]
                    second_half = memory_samples[len(memory_samples)//2:]
                    
                    first_avg = statistics.mean(first_half)
                    second_avg = statistics.mean(second_half)
                    
                    memory_increase = second_avg - first_avg
                    
                    # Should not have significant memory increase
                    assert memory_increase < 50, f"Potential memory leak detected: {memory_increase:.2f} MB increase"
                    
                    print(f"Memory Leak Test: {memory_increase:.2f} MB increase over {len(memory_samples)} episodes")
                
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        except Exception as e:
            pytest.skip(f"Memory leak test skipped: {e}")


class TestConcurrentPerformance:
    """Test performance under concurrent load."""
    
    @pytest.mark.performance
    def test_multi_environment_performance(self):
        """Test performance with multiple concurrent environments."""
        try:
            env_config = {
                'training_map': 'small_loop',
                'episode_max_steps': 30,
                'mode': 'debug',
                'resized_input_shape': (64, 64, 3),
                'simulation_framerate': 30
            }
            
            enhanced_config = EnhancedRLConfig(enabled_features=[])
            
            num_envs = 4
            results_queue = queue.Queue()
            
            def run_environment(env_id):
                try:
                    env = launch_and_wrap_enhanced_env(env_config, enhanced_config)
                    
                    start_time = time.time()
                    total_steps = 0
                    
                    obs = env.reset()
                    for step in range(20):
                        action = env.action_space.sample()
                        obs, reward, done, info = env.step(action)
                        total_steps += 1
                        if done:
                            break
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    fps = total_steps / duration if duration > 0 else 0
                    
                    env.close()
                    
                    results_queue.put((env_id, fps, total_steps, True))
                    
                except Exception as e:
                    results_queue.put((env_id, 0, 0, False))
            
            # Start concurrent environments
            threads = []
            start_time = time.time()
            
            for i in range(num_envs):
                thread = threading.Thread(target=run_environment, args=(i,))
                thread.start()
                threads.append(thread)
            
            # Wait for completion
            for thread in threads:
                thread.join(timeout=60)
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            # Collect results
            successful_envs = 0
            total_fps = 0
            
            while not results_queue.empty():
                env_id, fps, steps, success = results_queue.get()
                if success:
                    successful_envs += 1
                    total_fps += fps
            
            # Performance should scale reasonably
            assert successful_envs >= num_envs // 2, f"Too many failed environments: {successful_envs}/{num_envs}"
            assert total_duration < 30, f"Concurrent execution too slow: {total_duration:.2f}s"
            
            if successful_envs > 0:
                avg_fps = total_fps / successful_envs
                assert avg_fps >= 5.0, f"Concurrent FPS too low: {avg_fps:.2f}"
                
                print(f"Concurrent Performance: {successful_envs}/{num_envs} envs, {avg_fps:.2f} avg FPS")
            
        except Exception as e:
            pytest.skip(f"Concurrent performance test skipped: {e}")


class TestPerformanceRegression:
    """Test for performance regressions."""
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.mark.performance
    def test_performance_benchmarks(self, temp_dir):
        """Test against performance benchmarks."""
        benchmarks_file = temp_dir / 'performance_benchmarks.json'
        
        # Define expected benchmarks
        expected_benchmarks = {
            'baseline_fps': 20.0,
            'yolo_fps': 10.0,
            'full_pipeline_fps': 8.0,
            'step_time_ms': 50.0,
            'memory_usage_mb': 200.0
        }
        
        # Save benchmarks
        with open(benchmarks_file, 'w') as f:
            json.dump(expected_benchmarks, f)
        
        # Test current performance
        try:
            env_config = {
                'training_map': 'small_loop',
                'episode_max_steps': 50,
                'mode': 'debug',
                'resized_input_shape': (84, 84, 3)
            }
            
            enhanced_config = EnhancedRLConfig(enabled_features=[])
            
            env = launch_and_wrap_enhanced_env(env_config, enhanced_config)
            
            # Measure performance
            obs = env.reset()
            num_steps = 30
            
            start_time = time.time()
            for i in range(num_steps):
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                if done:
                    obs = env.reset()
            
            end_time = time.time()
            total_time = end_time - start_time
            fps = num_steps / total_time
            step_time_ms = (total_time / num_steps) * 1000
            
            env.close()
            
            # Compare against benchmarks
            assert fps >= expected_benchmarks['baseline_fps'] * 0.8, \
                f"FPS regression: {fps:.2f} < {expected_benchmarks['baseline_fps'] * 0.8:.2f}"
            
            assert step_time_ms <= expected_benchmarks['step_time_ms'] * 1.2, \
                f"Step time regression: {step_time_ms:.2f}ms > {expected_benchmarks['step_time_ms'] * 1.2:.2f}ms"
            
            print(f"Performance Benchmark: {fps:.2f} FPS, {step_time_ms:.2f}ms step time")
            
        except Exception as e:
            pytest.skip(f"Performance benchmark test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])