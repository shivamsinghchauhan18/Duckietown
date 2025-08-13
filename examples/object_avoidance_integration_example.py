"""
Object Avoidance Action Wrapper Integration Example.

This example demonstrates how to integrate the ObjectAvoidanceActionWrapper
with the Duckietown RL environment for object detection and avoidance.
"""

import numpy as np
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_detection_observation(
    objects: list = None,
    image_shape: tuple = (64, 64, 3)
) -> Dict[str, Any]:
    """
    Create a mock observation with detection data for testing.
    
    Args:
        objects: List of object dictionaries with position and distance info
        image_shape: Shape of the mock image
        
    Returns:
        Mock observation dictionary compatible with ObjectAvoidanceActionWrapper
    """
    if objects is None:
        objects = []
    
    # Create detection array (max 10 detections, 9 features each)
    detections = np.zeros((10, 9), dtype=np.float32)
    
    for i, obj in enumerate(objects[:10]):  # Limit to 10 objects
        # Format: [class_id, confidence, x1, y1, x2, y2, rel_x, rel_y, distance]
        detections[i] = [
            obj.get('class_id', 1),
            obj.get('confidence', 0.8),
            obj.get('bbox', [10, 10, 20, 20])[0],  # x1
            obj.get('bbox', [10, 10, 20, 20])[1],  # y1
            obj.get('bbox', [10, 10, 20, 20])[2],  # x2
            obj.get('bbox', [10, 10, 20, 20])[3],  # y2
            obj.get('relative_position', [0.0, 0.0])[0],  # rel_x
            obj.get('relative_position', [0.0, 0.0])[1],  # rel_y
            obj.get('distance', 1.0)  # distance
        ]
    
    return {
        'image': np.zeros(image_shape, dtype=np.uint8),
        'detections': detections,
        'detection_count': np.array([len(objects)], dtype=np.int32),
        'safety_critical': np.array([any(obj.get('distance', 1.0) < 0.5 for obj in objects)], dtype=np.int32),
        'inference_time': np.array([0.02], dtype=np.float32)  # 20ms inference time
    }


def simulate_avoidance_scenario():
    """
    Simulate a complete object avoidance scenario.
    
    This function demonstrates how the ObjectAvoidanceActionWrapper
    modifies actions in response to detected objects.
    """
    print("=" * 60)
    print("Object Avoidance Action Wrapper - Integration Example")
    print("=" * 60)
    
    try:
        # Import the wrapper (this will fail if gym is not available)
        from duckietown_utils.wrappers.object_avoidance_action_wrapper import ObjectAvoidanceActionWrapper
        
        # Create a mock environment
        class MockEnv:
            def __init__(self):
                self.action_space = type('ActionSpace', (), {'shape': (2,)})()
                self._last_observation = None
        
        mock_env = MockEnv()
        
        # Create wrapper with specific configuration
        wrapper = ObjectAvoidanceActionWrapper(
            mock_env,
            safety_distance=0.5,
            min_clearance=0.2,
            avoidance_strength=1.0,
            max_avoidance_action=0.6,
            smoothing_factor=0.7,
            emergency_brake_distance=0.15,
            enable_emergency_brake=True,
            debug_logging=True
        )
        
        print(f"✓ Wrapper initialized with configuration:")
        config = wrapper.get_configuration()
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        print("\\n" + "=" * 60)
        print("Scenario 1: No Objects Detected")
        print("=" * 60)
        
        # Scenario 1: No objects
        mock_env._last_observation = create_mock_detection_observation([])
        
        original_action = np.array([0.8, 0.8])
        modified_action = wrapper.action(original_action)
        
        print(f"Original action: {original_action}")
        print(f"Modified action: {modified_action}")
        print(f"Avoidance active: {wrapper.is_avoidance_active()}")
        print(f"Emergency brake: {wrapper.is_emergency_brake_active()}")
        
        print("\\n" + "=" * 60)
        print("Scenario 2: Object on Right Side")
        print("=" * 60)
        
        # Scenario 2: Object on right side within safety distance
        objects = [{
            'class_id': 1,
            'confidence': 0.9,
            'bbox': [30, 20, 50, 40],
            'relative_position': [0.3, 0.4],  # Right side, in front
            'distance': 0.35  # Within safety distance
        }]
        
        mock_env._last_observation = create_mock_detection_observation(objects)
        
        original_action = np.array([0.8, 0.8])
        modified_action = wrapper.action(original_action)
        
        print(f"Object detected at: position={objects[0]['relative_position']}, distance={objects[0]['distance']}m")
        print(f"Original action: {original_action}")
        print(f"Modified action: {modified_action}")
        print(f"Avoidance active: {wrapper.is_avoidance_active()}")
        print(f"Action change: Left wheel: {modified_action[0] - original_action[0]:+.3f}, Right wheel: {modified_action[1] - original_action[1]:+.3f}")
        
        if modified_action[0] > modified_action[1]:
            print("→ Robot steers LEFT to avoid right-side object")
        else:
            print("→ Robot steers RIGHT")
        
        print("\\n" + "=" * 60)
        print("Scenario 3: Object on Left Side")
        print("=" * 60)
        
        # Scenario 3: Object on left side
        objects = [{
            'class_id': 1,
            'confidence': 0.85,
            'bbox': [10, 20, 30, 40],
            'relative_position': [-0.25, 0.4],  # Left side, in front
            'distance': 0.3  # Close
        }]
        
        mock_env._last_observation = create_mock_detection_observation(objects)
        
        original_action = np.array([0.8, 0.8])
        modified_action = wrapper.action(original_action)
        
        print(f"Object detected at: position={objects[0]['relative_position']}, distance={objects[0]['distance']}m")
        print(f"Original action: {original_action}")
        print(f"Modified action: {modified_action}")
        print(f"Avoidance active: {wrapper.is_avoidance_active()}")
        print(f"Action change: Left wheel: {modified_action[0] - original_action[0]:+.3f}, Right wheel: {modified_action[1] - original_action[1]:+.3f}")
        
        if modified_action[1] > modified_action[0]:
            print("→ Robot steers RIGHT to avoid left-side object")
        else:
            print("→ Robot steers LEFT")
        
        print("\\n" + "=" * 60)
        print("Scenario 4: Emergency Brake Situation")
        print("=" * 60)
        
        # Scenario 4: Very close object requiring emergency brake
        objects = [{
            'class_id': 1,
            'confidence': 0.95,
            'bbox': [25, 25, 35, 35],
            'relative_position': [0.0, 0.2],  # Directly in front
            'distance': 0.1  # Very close - within emergency brake distance
        }]
        
        mock_env._last_observation = create_mock_detection_observation(objects)
        
        original_action = np.array([0.8, 0.8])
        modified_action = wrapper.action(original_action)
        
        print(f"Object detected at: position={objects[0]['relative_position']}, distance={objects[0]['distance']}m")
        print(f"Original action: {original_action}")
        print(f"Modified action: {modified_action}")
        print(f"Avoidance active: {wrapper.is_avoidance_active()}")
        print(f"Emergency brake: {wrapper.is_emergency_brake_active()}")
        
        if wrapper.is_emergency_brake_active():
            print("🚨 EMERGENCY BRAKE ACTIVATED - Robot stops immediately!")
        
        print("\\n" + "=" * 60)
        print("Scenario 5: Multiple Objects")
        print("=" * 60)
        
        # Scenario 5: Multiple objects with different priorities
        objects = [
            {
                'class_id': 1,
                'confidence': 0.8,
                'bbox': [20, 15, 35, 30],
                'relative_position': [0.2, 0.5],  # Right side
                'distance': 0.25  # Close
            },
            {
                'class_id': 1,
                'confidence': 0.7,
                'bbox': [40, 20, 55, 35],
                'relative_position': [-0.15, 0.6],  # Left side
                'distance': 0.4  # Farther
            }
        ]
        
        mock_env._last_observation = create_mock_detection_observation(objects)
        
        original_action = np.array([0.8, 0.8])
        modified_action = wrapper.action(original_action)
        
        print(f"Multiple objects detected:")
        for i, obj in enumerate(objects):
            print(f"  Object {i+1}: position={obj['relative_position']}, distance={obj['distance']}m")
        
        print(f"Original action: {original_action}")
        print(f"Modified action: {modified_action}")
        print(f"Avoidance active: {wrapper.is_avoidance_active()}")
        
        # Determine primary avoidance direction
        if modified_action[0] > modified_action[1]:
            print("→ Robot primarily steers LEFT (avoiding closer right object)")
        elif modified_action[1] > modified_action[0]:
            print("→ Robot primarily steers RIGHT (avoiding closer left object)")
        else:
            print("→ Robot maintains straight path (balanced forces)")
        
        print("\\n" + "=" * 60)
        print("Performance Statistics")
        print("=" * 60)
        
        # Display performance statistics
        stats = wrapper.get_avoidance_stats()
        print(f"Total steps: {stats['total_steps']}")
        print(f"Avoidance activations: {stats['avoidance_activations']}")
        print(f"Emergency brakes: {stats['emergency_brakes']}")
        print(f"Avoidance rate: {stats['avoidance_rate']:.2%}")
        print(f"Emergency brake rate: {stats['emergency_brake_rate']:.2%}")
        print(f"Max avoidance force: {stats['max_avoidance_force']:.3f}")
        
        print("\\n✓ Object Avoidance Integration Example completed successfully!")
        
    except ImportError as e:
        print(f"✗ Cannot run full integration example: {e}")
        print("This is expected if gym is not installed in the environment.")
        print("\\nThe ObjectAvoidanceActionWrapper has been implemented with:")
        print("- Potential field-based avoidance algorithm")
        print("- Configurable safety parameters")
        print("- Smooth action modifications")
        print("- Priority-based multi-object avoidance")
        print("- Emergency braking capability")
        print("- Comprehensive statistics tracking")
        
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def demonstrate_configuration_options():
    """Demonstrate different configuration options for the wrapper."""
    print("\\n" + "=" * 60)
    print("Configuration Options")
    print("=" * 60)
    
    configurations = [
        {
            'name': 'Conservative (High Safety)',
            'params': {
                'safety_distance': 0.8,
                'min_clearance': 0.3,
                'avoidance_strength': 1.5,
                'emergency_brake_distance': 0.2
            }
        },
        {
            'name': 'Aggressive (Low Safety)',
            'params': {
                'safety_distance': 0.3,
                'min_clearance': 0.1,
                'avoidance_strength': 0.8,
                'emergency_brake_distance': 0.05
            }
        },
        {
            'name': 'Smooth (High Smoothing)',
            'params': {
                'safety_distance': 0.5,
                'smoothing_factor': 0.9,
                'max_avoidance_action': 0.3
            }
        },
        {
            'name': 'Responsive (Low Smoothing)',
            'params': {
                'safety_distance': 0.5,
                'smoothing_factor': 0.3,
                'max_avoidance_action': 0.8
            }
        }
    ]
    
    for config in configurations:
        print(f"\\n{config['name']}:")
        for param, value in config['params'].items():
            print(f"  {param}: {value}")


if __name__ == "__main__":
    simulate_avoidance_scenario()
    demonstrate_configuration_options()
    
    print("\\n" + "=" * 60)
    print("Integration Notes")
    print("=" * 60)
    print("""
To use ObjectAvoidanceActionWrapper in your Duckietown RL training:

1. Wrap your environment with detection capability:
   env = YOLOObjectDetectionWrapper(env, model_path="yolov5s.pt")
   
2. Add the avoidance wrapper:
   env = ObjectAvoidanceActionWrapper(env, safety_distance=0.5)
   
3. Train your RL agent normally - the wrapper will automatically
   modify actions to avoid detected objects while maintaining
   the original action space interface.

4. Monitor avoidance statistics during training:
   stats = env.get_avoidance_stats()
   
5. Adjust configuration parameters based on performance:
   env.update_configuration(avoidance_strength=1.2)

The wrapper integrates seamlessly with existing PPO training
and maintains compatibility with all other Duckietown wrappers.
    """)