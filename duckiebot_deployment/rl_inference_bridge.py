#!/usr/bin/env python3
"""
RL Inference Bridge - Connects trained models to Duckiebot
Runs inference on trained RL models and publishes commands to the control node.
"""

import rospy
import numpy as np
import cv2
from std_msgs.msg import Float32MultiArray, Bool, String
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import torch
import json
import time
import logging
from pathlib import Path
from threading import Lock
import traceback

# Import your training utilities
import sys
sys.path.append('/code/duckietown-rl')  # Adjust path as needed
from duckietown_utils.env import launch_and_wrap_enhanced_env
from config.enhanced_config import load_enhanced_config
from ray.rllib.agents.ppo import PPOTrainer


class RLInferenceBridge:
    """
    Bridge between trained RL models and Duckiebot hardware.
    
    This node:
    1. Loads trained RL models
    2. Processes camera images 
    3. Runs inference to get actions
    4. Publishes commands to control node
    """
    
    def __init__(self):
        rospy.init_node('rl_inference_bridge', anonymous=False)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Node parameters
        self.robot_name = rospy.get_param('~robot_name', 'duckiebot')
        self.model_path = rospy.get_param('~model_path', '/models/champion_model')
        self.config_path = rospy.get_param('~config_path', '/config/enhanced_config.yml')
        self.inference_frequency = rospy.get_param('~inference_frequency', 10)  # Hz
        self.image_resize = rospy.get_param('~image_resize', [120, 160])  # [height, width]
        
        # Safety parameters
        self.max_inference_time = rospy.get_param('~max_inference_time', 0.1)  # seconds
        self.enable_safety_checks = rospy.get_param('~enable_safety_checks', True)
        
        # State
        self.current_image = None
        self.model_loaded = False
        self.inference_active = True
        self.bridge = CvBridge()
        self.image_lock = Lock()
        
        # Performance tracking
        self.inference_times = []
        self.last_inference_time = 0
        
        # Initialize model
        self.trainer = None
        self.enhanced_config = None
        self._load_model()
        
        # Setup publishers and subscribers
        self._setup_communication()
        
        # Start inference loop
        if self.model_loaded:
            self.inference_timer = rospy.Timer(
                rospy.Duration(1.0 / self.inference_frequency),
                self.inference_loop
            )
            self.logger.info("RL Inference Bridge initialized and running")
        else:
            self.logger.error("Failed to load model - inference disabled")
    
    def _load_model(self):
        """Load the trained RL model."""
        try:
            self.logger.info(f"Loading model from: {self.model_path}")
            
            # Load enhanced configuration
            if Path(self.config_path).exists():
                self.enhanced_config = load_enhanced_config(self.config_path)
                self.logger.info("Enhanced configuration loaded")
            else:
                self.logger.warning(f"Config file not found: {self.config_path}")
            
            # Load the trained model
            if Path(self.model_path).exists():
                # Method 1: Load Ray RLLib model
                try:
                    self.trainer = PPOTrainer(config={
                        'env': 'DummyEnv',  # We'll handle observations manually
                        'framework': 'torch',
                        'num_workers': 0,
                        'explore': False
                    })
                    self.trainer.restore(self.model_path)
                    self.model_loaded = True
                    self.logger.info("Ray RLLib model loaded successfully")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load as RLLib model: {e}")
                    
                    # Method 2: Load PyTorch model directly
                    try:
                        self.model = torch.load(self.model_path, map_location='cpu')
                        self.model.eval()
                        self.model_loaded = True
                        self.logger.info("PyTorch model loaded successfully")
                    except Exception as e2:
                        self.logger.error(f"Failed to load PyTorch model: {e2}")
            else:
                self.logger.error(f"Model file not found: {self.model_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            traceback.print_exc()
    
    def _setup_communication(self):
        """Setup ROS publishers and subscribers."""
        # Publishers
        self.command_pub = rospy.Publisher(
            f'/{self.robot_name}/rl_commands',
            Float32MultiArray,
            queue_size=1
        )
        
        self.status_pub = rospy.Publisher(
            f'/{self.robot_name}/inference_bridge/status',
            String,
            queue_size=1
        )
        
        # Subscribers
        self.camera_sub = rospy.Subscriber(
            f'/{self.robot_name}/camera_node/image/compressed',
            CompressedImage,
            self.camera_callback,
            queue_size=1
        )
        
        self.enable_sub = rospy.Subscriber(
            f'/{self.robot_name}/inference_bridge/enable',
            Bool,
            self.enable_callback,
            queue_size=1
        )
    
    def camera_callback(self, msg):
        """Process incoming camera images."""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                with self.image_lock:
                    self.current_image = cv_image
            
        except Exception as e:
            self.logger.error(f"Error processing camera image: {e}")
    
    def enable_callback(self, msg):
        """Enable/disable inference."""
        self.inference_active = msg.data
        self.logger.info(f"Inference {'enabled' if self.inference_active else 'disabled'}")
    
    def preprocess_image(self, image):
        """
        Preprocess camera image for RL model input.
        
        This should match the preprocessing used during training.
        """
        if image is None:
            return None
        
        try:
            # Resize to match training input size
            resized = cv2.resize(image, tuple(self.image_resize[::-1]))  # OpenCV uses (width, height)
            
            # Convert BGR to RGB (if needed)
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized = rgb_image.astype(np.float32) / 255.0
            
            # Add batch dimension if needed
            if len(normalized.shape) == 3:
                normalized = np.expand_dims(normalized, axis=0)
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            return None
    
    def run_inference(self, observation):
        """
        Run inference on the trained model.
        
        Args:
            observation: Preprocessed image observation
            
        Returns:
            action: [steering, throttle] in range [-1, 1]
        """
        if not self.model_loaded or observation is None:
            return [0.0, 0.0]  # Safe default
        
        try:
            start_time = time.time()
            
            if self.trainer is not None:
                # Ray RLLib inference
                action = self.trainer.compute_action(observation, explore=False)
            else:
                # PyTorch model inference
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(observation)
                    action = self.model(obs_tensor).cpu().numpy()
            
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Keep only recent inference times for performance tracking
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
            
            # Safety checks
            if self.enable_safety_checks:
                action = self.apply_safety_checks(action)
            
            # Ensure action is in correct format
            if isinstance(action, np.ndarray):
                action = action.tolist()
            
            # Ensure we have exactly 2 values
            if len(action) != 2:
                self.logger.warning(f"Invalid action shape: {action}")
                return [0.0, 0.0]
            
            # Clip to valid range
            steering = np.clip(action[0], -1.0, 1.0)
            throttle = np.clip(action[1], -1.0, 1.0)
            
            self.logger.debug(f"Inference: {inference_time:.4f}s, Action: [{steering:.3f}, {throttle:.3f}]")
            
            return [float(steering), float(throttle)]
            
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            traceback.print_exc()
            return [0.0, 0.0]  # Safe default
    
    def apply_safety_checks(self, action):
        """Apply safety checks to model output."""
        steering, throttle = action[0], action[1]
        
        # Limit maximum throttle for safety
        max_throttle = 0.7  # Reduce from full speed
        throttle = np.clip(throttle, -max_throttle, max_throttle)
        
        # Limit steering rate of change
        if hasattr(self, 'last_steering'):
            max_steering_change = 0.3  # Maximum change per step
            steering_change = steering - self.last_steering
            if abs(steering_change) > max_steering_change:
                steering = self.last_steering + np.sign(steering_change) * max_steering_change
        
        self.last_steering = steering
        
        return [steering, throttle]
    
    def inference_loop(self, event):
        """Main inference loop."""
        if not self.inference_active or not self.model_loaded:
            return
        
        try:
            # Get current image
            with self.image_lock:
                current_image = self.current_image.copy() if self.current_image is not None else None
            
            if current_image is None:
                self.logger.debug("No camera image available")
                return
            
            # Preprocess image
            observation = self.preprocess_image(current_image)
            
            if observation is None:
                self.logger.debug("Failed to preprocess image")
                return
            
            # Run inference
            action = self.run_inference(observation)
            
            # Publish command
            cmd_msg = Float32MultiArray()
            cmd_msg.data = action
            self.command_pub.publish(cmd_msg)
            
            # Update timing
            self.last_inference_time = time.time()
            
            # Publish status
            self.publish_status()
            
        except Exception as e:
            self.logger.error(f"Error in inference loop: {e}")
            traceback.print_exc()
    
    def publish_status(self):
        """Publish inference bridge status."""
        avg_inference_time = np.mean(self.inference_times) if self.inference_times else 0.0
        
        status = {
            'timestamp': time.time(),
            'model_loaded': self.model_loaded,
            'inference_active': self.inference_active,
            'avg_inference_time': avg_inference_time,
            'inference_frequency': len(self.inference_times) / max(1, len(self.inference_times) * avg_inference_time),
            'last_inference_age': time.time() - self.last_inference_time,
            'model_path': self.model_path
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status)
        self.status_pub.publish(status_msg)
    
    def shutdown(self):
        """Clean shutdown."""
        self.logger.info("Shutting down RL Inference Bridge")
        self.inference_active = False
        
        # Publish stop command
        cmd_msg = Float32MultiArray()
        cmd_msg.data = [0.0, 0.0]
        self.command_pub.publish(cmd_msg)


def main():
    """Main function."""
    try:
        bridge = RLInferenceBridge()
        
        # Register shutdown handler
        rospy.on_shutdown(bridge.shutdown)
        
        rospy.loginfo("RL Inference Bridge is running...")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("RL Inference Bridge interrupted")
    except Exception as e:
        rospy.logerr(f"Error in RL Inference Bridge: {e}")
        raise


if __name__ == '__main__':
    main()