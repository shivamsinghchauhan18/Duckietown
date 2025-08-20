#!/usr/bin/env python3
"""
Duckiebot Control Node - Real Hardware Interface
Receives commands from trained RL models and converts them to actual robot control.
"""

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray, Bool, String
from sensor_msgs.msg import CompressedImage, Joy
from geometry_msgs.msg import Twist
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
import cv2
from cv_bridge import CvBridge
import json
import time
from threading import Lock
import logging

class DuckiebotControlNode:
    """
    ROS node that interfaces between RL model commands and Duckiebot actuators.
    
    Subscribes to:
    - /rl_commands: RL model output [steering, throttle]
    - /camera/image/compressed: Camera feed for observation
    - /emergency_stop: Emergency stop signal
    
    Publishes to:
    - /wheels_driver_node/wheels_cmd: Wheel commands
    - /car_cmd_switch_node/cmd: High-level car commands
    - /status: Node status and diagnostics
    """
    
    def __init__(self):
        rospy.init_node('duckiebot_control_node', anonymous=False)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Node parameters
        self.robot_name = rospy.get_param('~robot_name', 'duckiebot')
        self.max_speed = rospy.get_param('~max_speed', 0.5)  # m/s
        self.wheel_baseline = rospy.get_param('~wheel_baseline', 0.1)  # meters
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.0318)  # meters
        self.control_frequency = rospy.get_param('~control_frequency', 20)  # Hz
        
        # Safety parameters
        self.emergency_stop = False
        self.command_timeout = rospy.get_param('~command_timeout', 1.0)  # seconds
        self.last_command_time = time.time()
        
        # Command smoothing
        self.smoothing_factor = rospy.get_param('~smoothing_factor', 0.7)
        self.last_steering = 0.0
        self.last_throttle = 0.0
        
        # State tracking
        self.current_observation = None
        self.bridge = CvBridge()
        self.command_lock = Lock()
        
        # Initialize publishers
        self._setup_publishers()
        
        # Initialize subscribers
        self._setup_subscribers()
        
        # Start control loop
        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_frequency), 
            self.control_loop
        )
        
        self.logger.info(f"Duckiebot Control Node initialized for {self.robot_name}")
        self.logger.info(f"Max speed: {self.max_speed} m/s, Control freq: {self.control_frequency} Hz")
    
    def _setup_publishers(self):
        """Setup ROS publishers for robot control."""
        # Wheel commands (low-level control)
        self.wheels_cmd_pub = rospy.Publisher(
            f'/{self.robot_name}/wheels_driver_node/wheels_cmd',
            WheelsCmdStamped,
            queue_size=1
        )
        
        # Car commands (high-level control)
        self.car_cmd_pub = rospy.Publisher(
            f'/{self.robot_name}/car_cmd_switch_node/cmd',
            Twist2DStamped,
            queue_size=1
        )
        
        # Status and diagnostics
        self.status_pub = rospy.Publisher(
            f'/{self.robot_name}/control_node/status',
            String,
            queue_size=1
        )
        
        # Debug information
        self.debug_pub = rospy.Publisher(
            f'/{self.robot_name}/control_node/debug',
            Float32MultiArray,
            queue_size=1
        )
    
    def _setup_subscribers(self):
        """Setup ROS subscribers for commands and sensors."""
        # RL model commands
        self.rl_cmd_sub = rospy.Subscriber(
            f'/{self.robot_name}/rl_commands',
            Float32MultiArray,
            self.rl_command_callback,
            queue_size=1
        )
        
        # Camera feed
        self.camera_sub = rospy.Subscriber(
            f'/{self.robot_name}/camera_node/image/compressed',
            CompressedImage,
            self.camera_callback,
            queue_size=1
        )
        
        # Emergency stop
        self.estop_sub = rospy.Subscriber(
            f'/{self.robot_name}/emergency_stop',
            Bool,
            self.emergency_stop_callback,
            queue_size=1
        )
        
        # Joystick override (for manual control)
        self.joy_sub = rospy.Subscriber(
            f'/{self.robot_name}/joy',
            Joy,
            self.joystick_callback,
            queue_size=1
        )
    
    def rl_command_callback(self, msg):
        """
        Process RL model commands and convert to robot actions.
        
        Expected format: [steering, throttle] where both are in [-1, 1]
        """
        if len(msg.data) != 2:
            self.logger.error(f"Invalid RL command format. Expected 2 values, got {len(msg.data)}")
            return
        
        with self.command_lock:
            raw_steering, raw_throttle = msg.data
            
            # Validate command ranges
            raw_steering = np.clip(raw_steering, -1.0, 1.0)
            raw_throttle = np.clip(raw_throttle, -1.0, 1.0)
            
            # Apply command smoothing
            self.last_steering = (
                self.smoothing_factor * self.last_steering + 
                (1 - self.smoothing_factor) * raw_steering
            )
            self.last_throttle = (
                self.smoothing_factor * self.last_throttle + 
                (1 - self.smoothing_factor) * raw_throttle
            )
            
            self.last_command_time = time.time()
            
            self.logger.debug(f"RL Command: steering={raw_steering:.3f}, throttle={raw_throttle:.3f}")
            self.logger.debug(f"Smoothed: steering={self.last_steering:.3f}, throttle={self.last_throttle:.3f}")
    
    def camera_callback(self, msg):
        """Process camera images for observation."""
        try:
            # Convert compressed image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Store for potential use by RL model
            self.current_observation = cv_image
            
        except Exception as e:
            self.logger.error(f"Error processing camera image: {e}")
    
    def emergency_stop_callback(self, msg):
        """Handle emergency stop signals."""
        self.emergency_stop = msg.data
        if self.emergency_stop:
            self.logger.warning("EMERGENCY STOP ACTIVATED!")
            self.stop_robot()
    
    def joystick_callback(self, msg):
        """Handle joystick override for manual control."""
        # If joystick is active, it overrides RL commands
        if len(msg.axes) >= 2 and len(msg.buttons) >= 1:
            # Check if manual override button is pressed
            if msg.buttons[0]:  # Button 0 for manual override
                with self.command_lock:
                    self.last_steering = msg.axes[0]  # Left stick horizontal
                    self.last_throttle = msg.axes[1]  # Left stick vertical
                    self.last_command_time = time.time()
                    self.logger.info("Manual joystick override active")
    
    def control_loop(self, event):
        """Main control loop - converts RL commands to robot actions."""
        try:
            # Check for emergency stop
            if self.emergency_stop:
                self.stop_robot()
                return
            
            # Check for command timeout
            if time.time() - self.last_command_time > self.command_timeout:
                self.logger.warning("Command timeout - stopping robot")
                self.stop_robot()
                return
            
            with self.command_lock:
                steering = self.last_steering
                throttle = self.last_throttle
            
            # Convert RL commands to robot commands
            self.execute_robot_command(steering, throttle)
            
            # Publish status
            self.publish_status()
            
        except Exception as e:
            self.logger.error(f"Error in control loop: {e}")
            self.stop_robot()
    
    def execute_robot_command(self, steering, throttle):
        """
        Convert normalized RL commands to actual robot control.
        
        Args:
            steering: Normalized steering command [-1, 1]
            throttle: Normalized throttle command [-1, 1]
        """
        # Convert to physical units
        linear_velocity = throttle * self.max_speed  # m/s
        angular_velocity = steering * 2.0  # rad/s (max ~115 degrees/s)
        
        # Method 1: High-level car command (recommended)
        self.publish_car_command(linear_velocity, angular_velocity)
        
        # Method 2: Low-level wheel commands (for precise control)
        # self.publish_wheel_commands(linear_velocity, angular_velocity)
        
        # Publish debug information
        debug_msg = Float32MultiArray()
        debug_msg.data = [
            steering, throttle, linear_velocity, angular_velocity,
            time.time() - self.last_command_time
        ]
        self.debug_pub.publish(debug_msg)
    
    def publish_car_command(self, linear_vel, angular_vel):
        """Publish high-level car command."""
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = linear_vel
        msg.omega = angular_vel
        
        self.car_cmd_pub.publish(msg)
    
    def publish_wheel_commands(self, linear_vel, angular_vel):
        """Publish low-level wheel commands (alternative method)."""
        # Convert to wheel velocities using differential drive kinematics
        # v_left = (2*v - omega*L) / (2*R)
        # v_right = (2*v + omega*L) / (2*R)
        
        v_left = (linear_vel - angular_vel * self.wheel_baseline / 2) / self.wheel_radius
        v_right = (linear_vel + angular_vel * self.wheel_baseline / 2) / self.wheel_radius
        
        msg = WheelsCmdStamped()
        msg.header.stamp = rospy.Time.now()
        msg.vel_left = v_left
        msg.vel_right = v_right
        
        self.wheels_cmd_pub.publish(msg)
    
    def stop_robot(self):
        """Immediately stop the robot."""
        # Publish zero commands
        self.publish_car_command(0.0, 0.0)
        self.publish_wheel_commands(0.0, 0.0)
        
        with self.command_lock:
            self.last_steering = 0.0
            self.last_throttle = 0.0
    
    def publish_status(self):
        """Publish node status and diagnostics."""
        status = {
            'timestamp': time.time(),
            'emergency_stop': self.emergency_stop,
            'last_command_age': time.time() - self.last_command_time,
            'steering': self.last_steering,
            'throttle': self.last_throttle,
            'node_healthy': True
        }
        
        status_msg = String()
        status_msg.data = json.dumps(status)
        self.status_pub.publish(status_msg)
    
    def shutdown(self):
        """Clean shutdown of the node."""
        self.logger.info("Shutting down Duckiebot Control Node")
        self.stop_robot()
        rospy.sleep(0.5)  # Give time for stop commands to be sent


def main():
    """Main function to run the Duckiebot Control Node."""
    try:
        control_node = DuckiebotControlNode()
        
        # Register shutdown handler
        rospy.on_shutdown(control_node.shutdown)
        
        rospy.loginfo("Duckiebot Control Node is running...")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Duckiebot Control Node interrupted")
    except Exception as e:
        rospy.logerr(f"Error in Duckiebot Control Node: {e}")
        raise


if __name__ == '__main__':
    main()