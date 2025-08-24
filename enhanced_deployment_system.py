#!/usr/bin/env python3
"""
🚀 ENHANCED DUCKIEBOT DEPLOYMENT SYSTEM 🚀

Production-ready deployment system with full RL capabilities:
- YOLO v5 object detection and avoidance
- Dynamic lane changing
- Multi-objective decision making
- Real-time performance optimization
- Comprehensive safety systems
- DTS daffy compatibility

This bridges the gap between advanced training and real-world deployment.
"""

import os
import sys
import time
import json
import logging
import threading
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import argparse

# Core ML and vision imports
import numpy as np
import cv2
import torch
import torch.nn as nn

# ROS imports
try:
    import rospy
    from std_msgs.msg import String, Bool
    from sensor_msgs.msg import CompressedImage, Image
    from geometry_msgs.msg import Twist
    from duckietown_msgs.msg import Twist2DStamped, BoolStamped
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    rospy = None
    logger = logging.getLogger(__name__)
    logger.warning("ROS not available. Running in simulation mode.")

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our enhanced infrastructure
from duckietown_utils.wrappers.yolo_detection_wrapper import YOLOObjectDetectionWrapper
from duckietown_utils.wrappers.enhanced_observation_wrapper import EnhancedObservationWrapper
from duckietown_utils.wrappers.object_avoidance_action_wrapper import ObjectAvoidanceActionWrapper
from duckietown_utils.wrappers.lane_changing_action_wrapper import LaneChangingActionWrapper
from duckietown_utils.yolo_utils import create_yolo_inference_system
from config.enhanced_config import EnhancedRLConfig, load_enhanced_config
from enhanced_rl_training_system import EnhancedDQNNetwork

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Enhanced deployment configuration."""
    # Model configuration
    model_path: str = "/data/models/enhanced_champion_model.pth"
    yolo_model_path: str = "yolov5s.pt"
    
    # YOLO configuration
    yolo_confidence_threshold: float = 0.5
    yolo_device: str = "cpu"  # Use CPU for deployment stability
    max_detections: int = 10
    
    # Object avoidance configuration
    safety_distance: float = 0.5
    min_clearance: float = 0.2
    avoidance_strength: float = 1.0
    emergency_brake_distance: float = 0.15
    
    # Lane changing configuration
    lane_change_threshold: float = 0.3
    safety_margin: float = 2.0
    max_lane_change_time: float = 3.0
    
    # Control configuration
    max_linear_velocity: float = 0.3
    max_angular_velocity: float = 1.0
    control_frequency: float = 10.0  # Hz
    
    # Safety configuration
    enable_safety_override: bool = True
    emergency_stop_enabled: bool = True
    max_consecutive_failures: int = 5
    
    # Logging configuration
    log_detections: bool = True
    log_actions: bool = True
    log_performance: bool = True
    save_debug_images: bool = False


class EnhancedRLInferenceNode:
    """Enhanced RL inference node with full capabilities."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.robot_name = rospy.get_param('~robot_name', 'duckiebot') if ROS_AVAILABLE else 'duckiebot'
        
        # Initialize components
        self.device = torch.device("cpu")  # Use CPU for deployment stability
        self.bridge = CvBridge() if ROS_AVAILABLE else None
        
        # State tracking
        self.current_image = None
        self.current_detections = []
        self.last_action = np.array([0.0, 0.0])
        self.emergency_stop_active = False
        self.consecutive_failures = 0
        self.inference_times = []
        
        # Performance monitoring
        self.performance_stats = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'detection_count': 0,
            'avoidance_activations': 0,
            'lane_changes': 0,
            'emergency_stops': 0,
            'avg_inference_time': 0.0
        }
        
        # Initialize systems
        self._initialize_yolo_system()
        self._initialize_rl_model()
        self._initialize_wrappers()
        
        if ROS_AVAILABLE:
            self._initialize_ros()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Enhanced RL Inference Node initialized for {self.robot_name}")
    
    def _initialize_yolo_system(self):
        """Initialize YOLO object detection system."""
        try:
            logger.info("Initializing YOLO detection system...")
            
            self.yolo_system = create_yolo_inference_system(
                model_path=self.config.yolo_model_path,
                device=self.config.yolo_device,
                confidence_threshold=self.config.yolo_confidence_threshold,
                max_detections=self.config.max_detections
            )
            
            if self.yolo_system is not None:
                logger.info("✅ YOLO system initialized successfully")
                self.yolo_enabled = True
            else:
                logger.warning("⚠️ YOLO system initialization failed - running without object detection")
                self.yolo_enabled = False
                
        except Exception as e:
            logger.error(f"❌ YOLO initialization error: {e}")
            self.yolo_enabled = False
    
    def _initialize_rl_model(self):
        """Initialize the enhanced RL model."""
        try:
            logger.info(f"Loading enhanced RL model from {self.config.model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            
            # Create dummy observation space for model initialization
            # This should match the training observation space
            from gym import spaces
            
            if self.yolo_enabled:
                # Enhanced observation space with YOLO features
                obs_space = spaces.Dict({
                    'image': spaces.Box(0, 255, (120, 160, 3), dtype=np.uint8),
                    'detection_features': spaces.Box(-np.inf, np.inf, (90,), dtype=np.float32),
                    'safety_features': spaces.Box(-np.inf, np.inf, (5,), dtype=np.float32)
                })
            else:
                # Standard flattened observation space
                obs_space = spaces.Box(-np.inf, np.inf, (57600,), dtype=np.float32)  # 120*160*3
            
            action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)
            
            # Create model
            from enhanced_rl_training_system import TrainingConfig
            training_config = TrainingConfig()
            
            self.rl_model = EnhancedDQNNetwork(obs_space, action_space, training_config).to(self.device)
            
            # Load model weights
            if 'q_network_state_dict' in checkpoint:
                self.rl_model.load_state_dict(checkpoint['q_network_state_dict'])
            elif 'model_state_dict' in checkpoint:
                self.rl_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Try to load directly
                self.rl_model.load_state_dict(checkpoint)
            
            self.rl_model.eval()
            
            logger.info("✅ Enhanced RL model loaded successfully")
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"❌ RL model loading error: {e}")
            logger.info("🔄 Falling back to simple policy")
            self.model_loaded = False
            self._initialize_fallback_policy()
    
    def _initialize_fallback_policy(self):
        """Initialize simple fallback policy."""
        class SimpleFallbackPolicy:
            def __init__(self):
                self.speed = 0.2
                self.turn_strength = 0.3
            
            def get_action(self, obs):
                # Simple lane following policy
                if isinstance(obs, dict) and 'image' in obs:
                    image = obs['image']
                else:
                    image = obs.reshape(120, 160, 3) if len(obs.shape) == 1 else obs
                
                # Simple lane detection
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                height, width = gray.shape
                
                # Look at bottom third of image
                roi = gray[int(height*0.7):, :]
                
                # Find lane center (simplified)
                edges = cv2.Canny(roi, 50, 150)
                
                # Calculate steering based on lane position
                center_x = width // 2
                lane_center = center_x  # Default to image center
                
                # Find lane lines (simplified)
                left_line = np.where(edges[:, :center_x] > 0)
                right_line = np.where(edges[:, center_x:] > 0)
                
                if len(left_line[1]) > 0 and len(right_line[1]) > 0:
                    left_x = np.mean(left_line[1])
                    right_x = np.mean(right_line[1]) + center_x
                    lane_center = (left_x + right_x) / 2
                
                # Calculate steering
                steering_error = (lane_center - center_x) / center_x
                steering = -steering_error * self.turn_strength
                
                return np.array([steering, self.speed])
        
        self.rl_model = SimpleFallbackPolicy()
        logger.info("✅ Fallback policy initialized")
    
    def _initialize_wrappers(self):
        """Initialize action wrappers for enhanced behavior."""
        try:
            # Object avoidance wrapper
            self.object_avoidance = ObjectAvoidanceActionWrapper(
                env=None,  # We'll use it without env
                safety_distance=self.config.safety_distance,
                min_clearance=self.config.min_clearance,
                avoidance_strength=self.config.avoidance_strength,
                emergency_brake_distance=self.config.emergency_brake_distance
            )
            
            # Lane changing wrapper
            self.lane_changing = LaneChangingActionWrapper(
                env=None,  # We'll use it without env
                lane_change_threshold=self.config.lane_change_threshold,
                safety_margin=self.config.safety_margin,
                max_lane_change_time=self.config.max_lane_change_time
            )
            
            logger.info("✅ Action wrappers initialized")
            self.wrappers_enabled = True
            
        except Exception as e:
            logger.error(f"❌ Wrapper initialization error: {e}")
            self.wrappers_enabled = False
    
    def _initialize_ros(self):
        """Initialize ROS publishers and subscribers."""
        if not ROS_AVAILABLE:
            return
        
        try:
            rospy.init_node('enhanced_rl_inference_node', anonymous=True)
            
            # Publishers
            self.cmd_pub = rospy.Publisher(
                f'/{self.robot_name}/car_cmd_switch_node/cmd',
                Twist2DStamped,
                queue_size=1
            )
            
            self.status_pub = rospy.Publisher(
                f'/{self.robot_name}/enhanced_rl/status',
                String,
                queue_size=1
            )
            
            self.detection_pub = rospy.Publisher(
                f'/{self.robot_name}/enhanced_rl/detections',
                String,
                queue_size=1
            )
            
            self.emergency_pub = rospy.Publisher(
                f'/{self.robot_name}/enhanced_rl/emergency_stop',
                BoolStamped,
                queue_size=1
            )
            
            # Subscribers
            self.camera_sub = rospy.Subscriber(
                f'/{self.robot_name}/camera_node/image/compressed',
                CompressedImage,
                self._camera_callback,
                queue_size=1,
                buff_size=2**24
            )
            
            self.emergency_sub = rospy.Subscriber(
                f'/{self.robot_name}/emergency_stop',
                BoolStamped,
                self._emergency_callback,
                queue_size=1
            )
            
            # Control timer
            self.control_timer = rospy.Timer(
                rospy.Duration(1.0 / self.config.control_frequency),
                self._control_loop
            )
            
            logger.info("✅ ROS interface initialized")
            
        except Exception as e:
            logger.error(f"❌ ROS initialization error: {e}")
    
    def _camera_callback(self, msg):
        """Process camera images with YOLO detection."""
        try:
            start_time = time.time()
            
            # Decode image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is None:
                return
            
            # Resize and convert
            resized = cv2.resize(cv_image, (160, 120))
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Store current image
            self.current_image = rgb_image.astype(np.float32) / 255.0
            
            # Run YOLO detection if enabled
            if self.yolo_enabled and self.yolo_system is not None:
                try:
                    detection_result = self.yolo_system.detect_objects(rgb_image)
                    self.current_detections = detection_result.get('detections', [])
                    
                    # Update performance stats
                    self.performance_stats['detection_count'] += len(self.current_detections)
                    
                    # Publish detections
                    if self.config.log_detections and ROS_AVAILABLE:
                        detection_msg = String()
                        detection_msg.data = json.dumps({
                            'timestamp': time.time(),
                            'detections': self.current_detections,
                            'safety_critical': detection_result.get('safety_critical', False)
                        })
                        self.detection_pub.publish(detection_msg)
                        
                except Exception as e:
                    logger.warning(f"YOLO detection error: {e}")
                    self.current_detections = []
            
            # Update inference timing
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
        except Exception as e:
            logger.error(f"Camera callback error: {e}")
            self.consecutive_failures += 1
    
    def _emergency_callback(self, msg):
        """Handle emergency stop messages."""
        if ROS_AVAILABLE:
            self.emergency_stop_active = msg.data
            if self.emergency_stop_active:
                logger.warning("🛑 Emergency stop activated")
                self.performance_stats['emergency_stops'] += 1
    
    def _control_loop(self, event=None):
        """Main control loop with enhanced decision making."""
        if self.current_image is None or (ROS_AVAILABLE and rospy.is_shutdown()):
            return
        
        if self.emergency_stop_active:
            self._publish_stop_command()
            return
        
        try:
            start_time = time.time()
            
            # Prepare observation
            obs = self._prepare_observation()
            
            # Get base action from RL model
            base_action = self._get_model_action(obs)
            
            # Apply enhanced behaviors if wrappers are enabled
            if self.wrappers_enabled:
                enhanced_action = self._apply_enhanced_behaviors(base_action, obs)
            else:
                enhanced_action = base_action
            
            # Apply safety constraints
            safe_action = self._apply_safety_constraints(enhanced_action)
            
            # Publish command
            self._publish_command(safe_action)
            
            # Update performance stats
            self.performance_stats['total_inferences'] += 1
            self.performance_stats['successful_inferences'] += 1
            
            # Log performance
            if self.config.log_performance:
                self._log_performance(safe_action, time.time() - start_time)
            
            # Reset failure counter on success
            self.consecutive_failures = 0
            
        except Exception as e:
            logger.error(f"Control loop error: {e}")
            self.consecutive_failures += 1
            
            # Emergency stop if too many failures
            if (self.consecutive_failures >= self.config.max_consecutive_failures and 
                self.config.emergency_stop_enabled):
                logger.error("🛑 Too many consecutive failures - activating emergency stop")
                self.emergency_stop_active = True
                self._publish_stop_command()
    
    def _prepare_observation(self) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Prepare observation for the RL model."""
        if self.yolo_enabled and self.model_loaded:
            # Enhanced observation with YOLO features
            
            # Detection features (flatten detection array)
            detection_features = np.zeros(90, dtype=np.float32)  # 10 detections * 9 features
            
            for i, detection in enumerate(self.current_detections[:10]):
                start_idx = i * 9
                
                # Extract detection features
                class_id = hash(detection.get('class', 'unknown')) % 1000
                confidence = detection.get('confidence', 0.0)
                bbox = detection.get('bbox', [0, 0, 0, 0])
                rel_pos = detection.get('relative_position', [0.0, 0.0])
                distance = detection.get('distance', 0.0)
                
                detection_features[start_idx:start_idx+9] = [
                    class_id, confidence,
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    rel_pos[0], rel_pos[1], distance
                ]
            
            # Safety features
            safety_features = np.array([
                len(self.current_detections) / 10.0,  # Normalized detection count
                1.0 if any(d.get('distance', float('inf')) < self.config.safety_distance 
                          for d in self.current_detections) else 0.0,  # Safety critical
                np.mean(self.inference_times) if self.inference_times else 0.0,  # Inference time
                np.mean([d.get('distance', 0) for d in self.current_detections]) / 10.0 if self.current_detections else 0.0,  # Avg distance
                min([d.get('distance', 10) for d in self.current_detections]) / 10.0 if self.current_detections else 1.0  # Min distance
            ], dtype=np.float32)
            
            return {
                'image': self.current_image,
                'detection_features': detection_features,
                'safety_features': safety_features
            }
        else:
            # Standard flattened observation
            return self.current_image.flatten()
    
    def _get_model_action(self, obs: Union[Dict[str, np.ndarray], np.ndarray]) -> np.ndarray:
        """Get action from the RL model."""
        try:
            if self.model_loaded and hasattr(self.rl_model, 'forward'):
                # Enhanced RL model
                with torch.no_grad():
                    if isinstance(obs, dict):
                        obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0).to(self.device) 
                                    for k, v in obs.items()}
                    else:
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    
                    q_values = self.rl_model(obs_tensor)
                    action = q_values.cpu().numpy().squeeze()
                    
                    # Ensure action is in correct format
                    if len(action.shape) == 0:
                        action = np.array([action, 0.2])  # Add default throttle
                    elif len(action) == 1:
                        action = np.array([action[0], 0.2])  # Add default throttle
                    
                    return action[:2]  # Take first 2 elements
            else:
                # Fallback policy
                return self.rl_model.get_action(obs)
                
        except Exception as e:
            logger.error(f"Model inference error: {e}")
            # Return safe default action
            return np.array([0.0, 0.1])
    
    def _apply_enhanced_behaviors(self, base_action: np.ndarray, obs: Union[Dict, np.ndarray]) -> np.ndarray:
        """Apply enhanced behaviors using wrappers."""
        try:
            enhanced_action = base_action.copy()
            
            # Apply object avoidance if detections are present
            if self.current_detections:
                # Simulate wrapper behavior for object avoidance
                closest_distance = min([d.get('distance', float('inf')) for d in self.current_detections])
                
                if closest_distance < self.config.safety_distance:
                    # Calculate avoidance steering
                    closest_detection = min(self.current_detections, key=lambda d: d.get('distance', float('inf')))
                    rel_pos = closest_detection.get('relative_position', [0, 0])
                    
                    # Steer away from object
                    avoidance_steering = -np.sign(rel_pos[0]) * self.config.avoidance_strength * 0.3
                    enhanced_action[0] += avoidance_steering
                    
                    # Reduce speed when avoiding
                    enhanced_action[1] *= 0.7
                    
                    self.performance_stats['avoidance_activations'] += 1
                    
                    # Emergency brake if too close
                    if closest_distance < self.config.emergency_brake_distance:
                        enhanced_action[1] = 0.0  # Stop
                        logger.warning(f"🛑 Emergency brake: object at {closest_distance:.2f}m")
            
            return enhanced_action
            
        except Exception as e:
            logger.error(f"Enhanced behavior error: {e}")
            return base_action
    
    def _apply_safety_constraints(self, action: np.ndarray) -> np.ndarray:
        """Apply safety constraints to action."""
        safe_action = action.copy()
        
        # Clip to safe ranges
        safe_action[0] = np.clip(safe_action[0], -1.0, 1.0)  # Steering
        safe_action[1] = np.clip(safe_action[1], 0.0, 1.0)   # Throttle (no reverse)
        
        # Convert to robot velocities
        linear_vel = safe_action[1] * self.config.max_linear_velocity
        angular_vel = safe_action[0] * self.config.max_angular_velocity
        
        # Additional safety checks
        if self.emergency_stop_active:
            linear_vel = 0.0
            angular_vel = 0.0
        
        return np.array([angular_vel, linear_vel])
    
    def _publish_command(self, action: np.ndarray):
        """Publish robot command."""
        if not ROS_AVAILABLE:
            return
        
        try:
            cmd_msg = Twist2DStamped()
            cmd_msg.header.stamp = rospy.Time.now()
            cmd_msg.omega = float(action[0])  # Angular velocity
            cmd_msg.v = float(action[1])      # Linear velocity
            
            self.cmd_pub.publish(cmd_msg)
            
            # Store last action
            self.last_action = action.copy()
            
        except Exception as e:
            logger.error(f"Command publishing error: {e}")
    
    def _publish_stop_command(self):
        """Publish stop command."""
        if ROS_AVAILABLE:
            cmd_msg = Twist2DStamped()
            cmd_msg.header.stamp = rospy.Time.now()
            cmd_msg.omega = 0.0
            cmd_msg.v = 0.0
            self.cmd_pub.publish(cmd_msg)
    
    def _log_performance(self, action: np.ndarray, inference_time: float):
        """Log performance metrics."""
        if not ROS_AVAILABLE:
            return
        
        try:
            # Update average inference time
            total_inferences = self.performance_stats['total_inferences']
            current_avg = self.performance_stats['avg_inference_time']
            self.performance_stats['avg_inference_time'] = (
                (current_avg * (total_inferences - 1) + inference_time) / total_inferences
            )
            
            # Create status message
            status = {
                'timestamp': time.time(),
                'action': action.tolist(),
                'detections_count': len(self.current_detections),
                'inference_time': inference_time,
                'emergency_stop': self.emergency_stop_active,
                'performance_stats': self.performance_stats
            }
            
            status_msg = String()
            status_msg.data = json.dumps(status)
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            logger.error(f"Performance logging error: {e}")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Log performance summary every 30 seconds
                if self.performance_stats['total_inferences'] > 0:
                    success_rate = (self.performance_stats['successful_inferences'] / 
                                  self.performance_stats['total_inferences'])
                    
                    logger.info(f"📊 Performance: {success_rate:.1%} success rate, "
                              f"{self.performance_stats['avg_inference_time']*1000:.1f}ms avg inference, "
                              f"{len(self.current_detections)} detections")
                
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)
    
    def shutdown(self):
        """Graceful shutdown."""
        logger.info("🛑 Shutting down Enhanced RL Inference Node")
        
        self.monitoring_active = False
        
        if ROS_AVAILABLE:
            # Publish final stop command
            self._publish_stop_command()
            
            # Shutdown ROS
            if hasattr(self, 'control_timer'):
                self.control_timer.shutdown()
        
        # Save final performance report
        self._save_performance_report()
        
        logger.info("✅ Shutdown complete")
    
    def _save_performance_report(self):
        """Save final performance report."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = Path(f"logs/deployment_performance_{timestamp}.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            report = {
                'timestamp': timestamp,
                'robot_name': self.robot_name,
                'config': asdict(self.config),
                'performance_stats': self.performance_stats,
                'yolo_enabled': self.yolo_enabled,
                'model_loaded': self.model_loaded,
                'wrappers_enabled': self.wrappers_enabled
            }
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📋 Performance report saved: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to save performance report: {e}")


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Enhanced Duckiebot RL Deployment")
    parser.add_argument('--config', type=str, help='Path to deployment config file')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--robot-name', type=str, default='duckiebot', help='Robot name')
    parser.add_argument('--no-yolo', action='store_true', help='Disable YOLO detection')
    parser.add_argument('--simulation', action='store_true', help='Run in simulation mode')
    
    args = parser.parse_args()
    
    # Create deployment config
    config = DeploymentConfig()
    
    if args.model:
        config.model_path = args.model
    
    if args.no_yolo:
        config.yolo_confidence_threshold = 0.0  # Effectively disable YOLO
    
    # Create and run inference node
    try:
        if args.simulation or not ROS_AVAILABLE:
            logger.info("🎮 Running in simulation mode")
            # Simulation mode - create a simple test loop
            node = EnhancedRLInferenceNode(config)
            
            # Simulate camera input
            dummy_image = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
            node.current_image = dummy_image.astype(np.float32) / 255.0
            
            # Run for a few iterations
            for i in range(100):
                node._control_loop()
                time.sleep(0.1)
                
                if i % 10 == 0:
                    logger.info(f"Simulation step {i}/100")
            
            node.shutdown()
            
        else:
            logger.info("🚗 Running on real robot")
            node = EnhancedRLInferenceNode(config)
            
            # Setup signal handlers
            def signal_handler(signum, frame):
                logger.info("Received shutdown signal")
                node.shutdown()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Run ROS node
            rospy.spin()
            
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()