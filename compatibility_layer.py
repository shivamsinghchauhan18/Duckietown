#!/usr/bin/env python3
"""
🔧 COMPATIBILITY LAYER 🔧

Handles missing dependencies and provides fallback implementations
for the enhanced RL system to work without full Duckietown installation.
"""

import sys
import warnings
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))


def setup_compatibility():
    """Setup compatibility layer for missing dependencies."""
    
    # Handle gym_duckietown import
    try:
        import gym_duckietown
    except ImportError:
        print("⚠️ gym_duckietown not available - using mock implementation")
        _create_mock_gym_duckietown()
    
    # Handle ultralytics import
    try:
        import ultralytics
    except ImportError:
        print("⚠️ ultralytics not available - using mock implementation")
        _create_mock_ultralytics()
    
    # Handle ROS imports
    try:
        import rospy
    except ImportError:
        print("⚠️ ROS not available - using mock implementation")
        _create_mock_ros()


def _create_mock_gym_duckietown():
    """Create mock gym_duckietown module."""
    import sys
    from types import ModuleType
    
    # Create mock module
    mock_module = ModuleType('gym_duckietown')
    
    # Mock simulator
    class MockSimulator:
        def __init__(self, **kwargs):
            self.observation_space = None
            self.action_space = None
        
        def reset(self):
            import numpy as np
            return np.zeros((120, 160, 3), dtype=np.uint8)
        
        def step(self, action):
            import numpy as np
            obs = np.zeros((120, 160, 3), dtype=np.uint8)
            reward = 0.0
            done = False
            info = {}
            return obs, reward, done, info
    
    # Mock constants
    mock_module.Simulator = MockSimulator
    mock_module.DEFAULT_ROBOT_SPEED = 0.3
    mock_module.DEFAULT_CAMERA_WIDTH = 640
    mock_module.DEFAULT_CAMERA_HEIGHT = 480
    
    # Mock simulator submodule
    simulator_module = ModuleType('gym_duckietown.simulator')
    simulator_module.Simulator = MockSimulator
    simulator_module.DEFAULT_ROBOT_SPEED = 0.3
    simulator_module.DEFAULT_CAMERA_WIDTH = 640
    simulator_module.DEFAULT_CAMERA_HEIGHT = 480
    simulator_module.CAMERA_FOV_Y = 60.0
    
    # Mock exceptions
    class NotInLane(Exception):
        pass
    
    simulator_module.NotInLane = NotInLane
    
    # Add to sys.modules
    sys.modules['gym_duckietown'] = mock_module
    sys.modules['gym_duckietown.simulator'] = simulator_module


def _create_mock_ultralytics():
    """Create mock ultralytics module."""
    import sys
    from types import ModuleType
    import numpy as np
    
    # Create mock module
    mock_module = ModuleType('ultralytics')
    
    class MockYOLO:
        def __init__(self, model_path='yolov5s.pt'):
            self.model_path = model_path
        
        def __call__(self, image):
            # Return mock detection results
            return MockResults()
        
        def predict(self, image, **kwargs):
            return [MockResults()]
    
    class MockResults:
        def __init__(self):
            self.boxes = MockBoxes()
        
        @property
        def names(self):
            return {0: 'person', 1: 'bicycle', 2: 'car'}
    
    class MockBoxes:
        def __init__(self):
            # Mock some detections
            self.xyxy = np.array([[100, 50, 150, 100]])  # [x1, y1, x2, y2]
            self.conf = np.array([0.8])
            self.cls = np.array([0])  # person class
        
        def cpu(self):
            return self
        
        def numpy(self):
            return self
    
    mock_module.YOLO = MockYOLO
    
    # Add to sys.modules
    sys.modules['ultralytics'] = mock_module


def _create_mock_ros():
    """Create mock ROS modules."""
    import sys
    from types import ModuleType
    
    # Create mock rospy
    rospy_module = ModuleType('rospy')
    
    def mock_init_node(name, **kwargs):
        print(f"Mock ROS node initialized: {name}")
    
    def mock_loginfo(msg):
        print(f"ROS INFO: {msg}")
    
    def mock_logerr(msg):
        print(f"ROS ERROR: {msg}")
    
    def mock_logwarn(msg):
        print(f"ROS WARN: {msg}")
    
    class MockTime:
        @staticmethod
        def now():
            import time
            return time.time()
    
    class MockDuration:
        def __init__(self, secs):
            self.secs = secs
    
    class MockTimer:
        def __init__(self, duration, callback):
            self.duration = duration
            self.callback = callback
        
        def shutdown(self):
            pass
    
    class MockPublisher:
        def __init__(self, topic, msg_type, **kwargs):
            self.topic = topic
            self.msg_type = msg_type
        
        def publish(self, msg):
            pass
    
    class MockSubscriber:
        def __init__(self, topic, msg_type, callback, **kwargs):
            self.topic = topic
            self.msg_type = msg_type
            self.callback = callback
    
    def mock_get_param(param, default=None):
        return default
    
    def mock_is_shutdown():
        return False
    
    def mock_spin():
        import time
        time.sleep(1)
    
    # Add functions to module
    rospy_module.init_node = mock_init_node
    rospy_module.loginfo = mock_loginfo
    rospy_module.logerr = mock_logerr
    rospy_module.logwarn = mock_logwarn
    rospy_module.Time = MockTime
    rospy_module.Duration = MockDuration
    rospy_module.Timer = MockTimer
    rospy_module.Publisher = MockPublisher
    rospy_module.Subscriber = MockSubscriber
    rospy_module.get_param = mock_get_param
    rospy_module.is_shutdown = mock_is_shutdown
    rospy_module.spin = mock_spin
    
    # Mock message types
    std_msgs_module = ModuleType('std_msgs')
    std_msgs_msg_module = ModuleType('std_msgs.msg')
    
    class MockString:
        def __init__(self):
            self.data = ""
    
    class MockBool:
        def __init__(self):
            self.data = False
    
    std_msgs_msg_module.String = MockString
    std_msgs_msg_module.Bool = MockBool
    
    # Mock sensor messages
    sensor_msgs_module = ModuleType('sensor_msgs')
    sensor_msgs_msg_module = ModuleType('sensor_msgs.msg')
    
    class MockCompressedImage:
        def __init__(self):
            self.data = b""
            self.format = "jpeg"
    
    class MockImage:
        def __init__(self):
            self.data = b""
            self.width = 640
            self.height = 480
    
    sensor_msgs_msg_module.CompressedImage = MockCompressedImage
    sensor_msgs_msg_module.Image = MockImage
    
    # Mock geometry messages
    geometry_msgs_module = ModuleType('geometry_msgs')
    geometry_msgs_msg_module = ModuleType('geometry_msgs.msg')
    
    class MockTwist:
        def __init__(self):
            self.linear = MockVector3()
            self.angular = MockVector3()
    
    class MockVector3:
        def __init__(self):
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
    
    geometry_msgs_msg_module.Twist = MockTwist
    
    # Mock duckietown messages
    duckietown_msgs_module = ModuleType('duckietown_msgs')
    duckietown_msgs_msg_module = ModuleType('duckietown_msgs.msg')
    
    class MockTwist2DStamped:
        def __init__(self):
            self.header = MockHeader()
            self.v = 0.0
            self.omega = 0.0
    
    class MockBoolStamped:
        def __init__(self):
            self.header = MockHeader()
            self.data = False
    
    class MockHeader:
        def __init__(self):
            self.stamp = MockTime.now()
    
    duckietown_msgs_msg_module.Twist2DStamped = MockTwist2DStamped
    duckietown_msgs_msg_module.BoolStamped = MockBoolStamped
    
    # Mock cv_bridge
    cv_bridge_module = ModuleType('cv_bridge')
    
    class MockCvBridge:
        def imgmsg_to_cv2(self, msg, encoding="bgr8"):
            import numpy as np
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        def cv2_to_imgmsg(self, cv_image, encoding="bgr8"):
            return MockImage()
    
    cv_bridge_module.CvBridge = MockCvBridge
    
    # Add all modules to sys.modules
    sys.modules['rospy'] = rospy_module
    sys.modules['std_msgs'] = std_msgs_module
    sys.modules['std_msgs.msg'] = std_msgs_msg_module
    sys.modules['sensor_msgs'] = sensor_msgs_module
    sys.modules['sensor_msgs.msg'] = sensor_msgs_msg_module
    sys.modules['geometry_msgs'] = geometry_msgs_module
    sys.modules['geometry_msgs.msg'] = geometry_msgs_msg_module
    sys.modules['duckietown_msgs'] = duckietown_msgs_module
    sys.modules['duckietown_msgs.msg'] = duckietown_msgs_msg_module
    sys.modules['cv_bridge'] = cv_bridge_module


if __name__ == "__main__":
    setup_compatibility()
    print("✅ Compatibility layer setup complete")