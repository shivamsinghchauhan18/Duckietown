#!/usr/bin/env python3
"""
DTS-compatible RL Inference Node
Combines inference and control in a single DTS-standard node.
"""

import rospy
import numpy as np
import cv2
from std_msgs.msg import Float32MultiArray, Bool, String
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from cv_bridge import CvBridge
import torch
import json
import time
import logging
from threading import Lock
import traceback

# DTS-compatible imports
import rospkg
from duckietown_utils import get_duckiebot_name


class DuckiebotRLInferenceNode:
    """
    DTS-compatible RL inference node that handles both inference and control.
    
    This node follows Duckietown Software Stack conventions:
    - Uses standard DT message types
    - Follows DT naming conventions
    - Integrates with DT infrastructure
    - Provides DT-standard diagnostics
    """
    
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('duckiebot_rl_inference_node', anonymous=False)
        
        # Get robot name using DT utilities
        self.robot_name = get_duckiebot_name()
        
        # Setup logging
        self.logger = rospy
        self.logger.loginfo(f"Initializing RL Inference Node for {self.robot_name}")
        
        # Load parameters
        self._load_parameters()
        
        # Initialize state
        self.current_image = None
        self.model_loaded = False
        self.inference_active = True
        self.emergency_stop = False
        self.bridge = CvBridge()
        self.image_lock = Lock()
        self.command_lock = Lock()
        
        # Control state
        self.last_steering = 0.0
        self.last_throttle = 0.0
        self.last_command_time = time.time()
        
        # Performance tracking
        self.inference_times = []
        
        # Load RL model
        self._load_rl_model()
        
        # Setup ROS communication
        self._setup_publishers()
        self._setup_subscribers()
        
        # Start control timer
        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_frequency),
            self.control_loop
        )
        
        self.logger.loginfo("DTS RL Inference Node initialized successfully")
    
    def _load_parameters(self):
        """Load ROS parameters with DTS-compatible defaults."""
        self.model_path = rospy.get_param('~model_path', '/models/champion_model.pth')
        self.config_path = rospy.get_param('~config_path', '/config/enhanced_config.yml')
        self.inference_frequency = rospy.get_param('~inference_frequency', 10)
        self.control_frequency = rospy.get_param('~control_frequency', 20)
        self.max_speed = rospy.get_param('~max_speed', 0.5)
        self.wheel_baseline = rospy.get_param('~wheel_baseline', 0.1)
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.0318)
        self.command_timeout = rospy.get_param('~command_timeout', 1.0)
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 0.7)
        self.image_resize = rospy.get_param('~image_resize', [120, 160])
        self.enable_safety_checks = rospy.get_param('~enable_safety_checks', True)
    
    def _load_rl_model(self):
        """Load the trained RL model."""
        try:
            self.logger.loginfo(f"Loading RL model from: {self.model_path}")
            
            if self.model_path.endswith('.pth') or self.model_path.endswith('.pt'):
                # PyTorch model
                self.model = torch.load(self.model_path, map_location='cpu')
                self.model.eval()
                self.trainer = None
                self.model_loaded = True
                self.logger.loginfo("PyTorch model loaded successfully")
                
            elif 'checkpoint' in self.model_path:
                # Ray RLLib checkpoint
                from ray.rllib.agents.ppo import PPOTrainer
                self.trainer = PPOTrainer(config={
                    'env': 'DummyEnv',
                    'framework': 'torch',
                    'num_workers': 0,
                    'explore': False
                })
                self.trainer.restore(self.model_path)
                self.model = None
                self.model_loaded = True
                self.logger.loginfo("Ray RLLib model loaded successfully")
                
            else:
                self.logger.logerr(f"Unknown model format: {self.model_path}")
                self.model_loaded = False
                
        except Exception as e:
            self.logger.logerr(f"Failed to load RL model: {e}")
            self.model_loaded = False
    
    def _setup_publishers(self):
        """Setup ROS publishers following DT conventions."""
        # Car command (high-level control) - DT standard
        self.car_cmd_pub = rospy.Publisher(
            "car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )
        
        # Wheel commands (low-level control) - DT standard
        self.wheels_cmd_pub = rospy.Publisher(
            "wheels_driver_node/wheels_cmd",
            WheelsCmdStamped,
            queue_size=1
        )
        
        # Status and diagnostics - DT standard
        self.status_pub = rospy.Publisher(
            "rl_inference/status",
            String,
            queue_size=1
        )
        
        # Debug information
        self.debug_pub = rospy.Publisher(
            "rl_inference/debug",
            Float32MultiArray,
            queue_size=1
        )
    
    def _setup_subscribers(self):
        """Setup ROS subscribers following DT conventions."""
        # Camera feed - DT standard topic
        self.camera_sub = rospy.Subscriber(
            "camera_node/image/compressed",
            CompressedImage,
            self.camera_callback,
            queue_size=1
        )
        
        # Emergency stop - DT standard
        self.estop_sub = rospy.Subscriber(
            "emergency_stop",
            Bool,
            self.emergency_stop_callback,
            queue_size=1
        )
        
        # Manual enable/disable
        self.enable_sub = rospy.Subscriber(
            "rl_inference/enable",
            Bool,
            self.enable_callback,
            queue_size=1
        )
    
    def camera_callback(self, msg):
        """Process camera images for RL inference."""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                with self.image_lock:
                    self.current_image = cv_image
                    
        except Exception as e:
            self.logger.logerr(f"Error processing camera image: {e}")
    
    def emergency_stop_callback(self, msg):
        """Handle emergency stop signals."""
        self.emergency_stop = msg.data
        if self.emergency_stop:
            self.logger.logwarn("EMERGENCY STOP ACTIVATED!")
            self.stop_robot()
    
    def enable_callback(self, msg):
        """Enable/disable RL inference."""
        self.inference_active = msg.data
        self.logger.loginfo(f"RL Inference {'enabled' if self.inference_active else 'disabled'}")
    
    def preprocess_image(self, image):
        """Preprocess camera image for RL model (same as training)."""
        if image is None:
            return None
        
        try:
            # Resize to training size
            resized = cv2.resize(image, tuple(self.image_resize[::-1]))
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized = rgb_image.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            self.logger.logerr(f"Error preprocessing image: {e}")
            return None
    
    def run_inference(self, observation):
        """Run RL model inference."""
        if not self.model_loaded or observation is None:
            return [0.0, 0.0]
        
        try:
            start_time = time.time()
            
            if self.trainer is not None:
                # Ray RLLib inference
                action = self.trainer.compute_action(observation, explore=False)
            else:
                # PyTorch model inference
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(observation)
                    if len(obs_tensor.shape) == 3:
                        obs_tensor = obs_tensor.unsqueeze(0)  # Add batch dimension
                    action = self.model(obs_tensor).cpu().numpy().flatten()
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Keep only recent times
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            # Ensure correct format
            if isinstance(action, np.ndarray):
                action = action.tolist()
            
            if len(action) != 2:
                self.logger.logwarn(f"Invalid action shape: {action}")
                return [0.0, 0.0]
            
            # Apply safety checks
            if self.enable_safety_checks:
                action = self.apply_safety_checks(action)
            
            # Clip to valid range
            steering = np.clip(action[0], -1.0, 1.0)
            throttle = np.clip(action[1], -1.0, 1.0)
            
            return [float(steering), float(throttle)]
            
        except Exception as e:
            self.logger.logerr(f"Error during inference: {e}")
            return [0.0, 0.0]
    
    def apply_safety_checks(self, action):
        """Apply safety limits to RL actions."""
        steering, throttle = action[0], action[1]
        
        # Limit maximum throttle
        max_throttle = 0.7
        throttle = np.clip(throttle, -max_throttle, max_throttle)
        
        # Limit steering rate of change
        if hasattr(self, 'last_steering'):
            max_steering_change = 0.3
            steering_change = steering - self.last_steering
            if abs(steering_change) > max_steering_change:
                steering = self.last_steering + np.sign(steering_change) * max_steering_change
        
        return [steering, throttle]
    
    def control_loop(self, event):
        """Main control loop - runs inference and publishes commands."""
        try:
            # Check for emergency stop
            if self.emergency_stop:
                self.stop_robot()
                return
            
            # Check if inference is active
            if not self.inference_active or not self.model_loaded:
                return
            
            # Get current image
            with self.image_lock:
                current_image = self.current_image.copy() if self.current_image is not None else None
            
            if current_image is None:
                return
            
            # Preprocess image
            observation = self.preprocess_image(current_image)
            if observation is None:
                return
            
            # Run inference
            action = self.run_inference(observation)
            
            # Update command state
            with self.command_lock:
                # Apply smoothing
                self.last_steering = (
                    self.smoothing_factor * self.last_steering +
                    (1 - self.smoothing_factor) * action[0]
                )
                self.last_throttle = (
                    self.smoothing_factor * self.last_throttle +
                    (1 - self.smoothing_factor) * action[1]
                )
                self.last_command_time = time.time()
            
            # Execute robot command
            self.execute_robot_command(self.last_steering, self.last_throttle)
            
            # Publish status
            self.publish_status()
            
        except Exception as e:
            self.logger.logerr(f"Error in control loop: {e}")
            self.stop_robot()
    
    def execute_robot_command(self, steering, throttle):
        """Convert RL commands to robot control."""
        # Convert to physical units
        linear_velocity = throttle * self.max_speed
        angular_velocity = steering * 2.0
        
        # Publish car command (DT standard)
        car_msg = Twist2DStamped()
        car_msg.header.stamp = rospy.Time.now()
        car_msg.v = linear_velocity
        car_msg.omega = angular_velocity
        self.car_cmd_pub.publish(car_msg)
        
        # Publish debug info
        debug_msg = Float32MultiArray()
        debug_msg.data = [steering, throttle, linear_velocity, angular_velocity]
        self.debug_pub.publish(debug_msg)
    
    def stop_robot(self):
        """Stop the robot immediately."""
        car_msg = Twist2DStamped()
        car_msg.header.stamp = rospy.Time.now()
        car_msg.v = 0.0
        car_msg.omega = 0.0
        self.car_cmd_pub.publish(car_msg)
        
        with self.command_lock:
            self.last_steering = 0.0
            self.last_throttle = 0.0
    
    def publish_status(self):
        """Publish node status."""
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0.0
        
        status = {
            'timestamp': time.time(),
            'model_loaded': self.model_loaded,
            'inference_active': self.inference_active,
            'emergency_stop': self.emergency_stop,
            'avg_inference_time': avg_inference_time,
            'last_command_age': time.time() - self.last_command_time,
            'steering': self.last_steering,
            'throttle': self.last_throttle
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status)
        self.status_pub.publish(status_msg)
    
    def on_shutdown(self):
        """Clean shutdown."""
        self.logger.loginfo("Shutting down RL Inference Node")
        self.stop_robot()
        rospy.sleep(0.5)


def main():
    """Main function."""
    try:
        node = DuckiebotRLInferenceNode()
        rospy.on_shutdown(node.on_shutdown)
        rospy.loginfo("DTS RL Inference Node is running...")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("RL Inference Node interrupted")
    except Exception as e:
        rospy.logerr(f"Error in RL Inference Node: {e}")
        raise


if __name__ == '__main__':
    main()