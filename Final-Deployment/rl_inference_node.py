#!/usr/bin/env python3
"""
Simple RL Inference Node for Duckiebot
"""

import rospy
import numpy as np
import cv2
import torch
import json
import time
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from cv_bridge import CvBridge


class SimpleRLNode:
    def __init__(self):
        rospy.init_node('rl_inference_node')
        
        # Get robot name
        self.robot_name = rospy.get_param('~robot_name', 'duckiebot')
        
        # Load model
        self.model = self.load_model('/data/models/champion_model.pth')
        
        # ROS setup
        self.bridge = CvBridge()
        self.current_image = None
        
        # Publishers
        self.cmd_pub = rospy.Publisher(
            f'/{self.robot_name}/car_cmd_switch_node/cmd',
            Twist2DStamped,
            queue_size=1
        )
        
        self.status_pub = rospy.Publisher(
            f'/{self.robot_name}/rl_inference/status',
            String,
            queue_size=1
        )
        
        # Subscribers
        self.camera_sub = rospy.Subscriber(
            f'/{self.robot_name}/camera_node/image/compressed',
            CompressedImage,
            self.camera_callback
        )
        
        # Control timer
        self.control_timer = rospy.Timer(
            rospy.Duration(0.1),  # 10 Hz
            self.control_loop
        )
        
        rospy.loginfo(f"RL Node started for {self.robot_name}")
    
    def load_model(self, model_path):
        """Load the champion model"""
        try:
            rospy.loginfo(f"Loading model: {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Create model (simplified version)
            model = SimpleDuckiebotModel()
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            
            model.eval()
            rospy.loginfo("Model loaded successfully")
            return model
            
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            return None
    
    def camera_callback(self, msg):
        """Process camera images"""
        try:
            # Decode image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if cv_image is not None:
                # Resize to model input size
                resized = cv2.resize(cv_image, (160, 120))
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                # Normalize
                self.current_image = rgb_image.astype(np.float32) / 255.0
                
        except Exception as e:
            rospy.logerr(f"Camera callback error: {e}")
    
    def control_loop(self, event):
        """Main control loop"""
        if self.current_image is None or self.model is None:
            return
        
        try:
            # Run inference
            with torch.no_grad():
                # Convert to tensor (HWC -> CHW)
                obs_tensor = torch.FloatTensor(self.current_image).permute(2, 0, 1).unsqueeze(0)
                
                # Get action
                action = self.model.get_action(obs_tensor)
                
                if len(action) >= 2:
                    steering = float(np.clip(action[0], -1.0, 1.0))
                    throttle = float(np.clip(action[1], -1.0, 1.0))
                    
                    # Convert to robot commands
                    linear_vel = throttle * 0.3  # Max 0.3 m/s
                    angular_vel = steering * 1.0  # Max 1.0 rad/s
                    
                    # Publish command
                    cmd_msg = Twist2DStamped()
                    cmd_msg.header.stamp = rospy.Time.now()
                    cmd_msg.v = linear_vel
                    cmd_msg.omega = angular_vel
                    self.cmd_pub.publish(cmd_msg)
                    
                    # Publish status
                    status = {
                        'timestamp': time.time(),
                        'steering': steering,
                        'throttle': throttle,
                        'linear_vel': linear_vel,
                        'angular_vel': angular_vel
                    }
                    
                    status_msg = String()
                    status_msg.data = json.dumps(status)
                    self.status_pub.publish(status_msg)
        
        except Exception as e:
            rospy.logerr(f"Control loop error: {e}")


class SimpleDuckiebotModel(torch.nn.Module):
    """Simplified model class for deployment"""
    
    def __init__(self):
        super().__init__()
        
        # Simple CNN
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 8, 4, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 5))
        )
        
        # FC layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1280, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
        )
        
        # Policy head
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.Tanh()
        )
    
    def forward(self, x):
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        hidden = self.fc(features)
        action = self.policy(hidden)
        return action
    
    def get_action(self, obs):
        output = self.forward(obs)
        return output.squeeze().cpu().numpy()


if __name__ == '__main__':
    try:
        node = SimpleRLNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass