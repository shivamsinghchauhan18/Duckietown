# 🤖 DTS-Compatible Duckiebot RL Deployment

This version uses the official **Duckietown Software Stack (DTS)** tools and conventions for deployment.

## 🏗️ DTS vs Custom Deployment

| Feature | Custom Deployment | DTS Deployment |
|---------|------------------|----------------|
| **Build System** | Custom Dockerfile | `dts devel build` |
| **Deployment** | Custom scripts | `dts devel run` |
| **Robot Access** | SSH + Docker | DTS tools |
| **ROS Integration** | Manual setup | DTS standard |
| **Compatibility** | Generic | Duckietown optimized |

## 🚀 Quick Start with DTS

### Prerequisites
```bash
# Install DTS (Duckietown Shell)
pip install duckietown-shell

# Initialize DTS
dts --set-version daffy
dts update

# Check robot connectivity
ping YOUR_ROBOT_NAME.local
```

### Deploy with DTS
```bash
# Build and deploy in one command
python duckiebot_deployment_dts/dts_deploy.py \
    --robot-name YOUR_ROBOT_NAME \
    --model-path models/champion_model.pth \
    --config-path config/enhanced_config.yml
```

## 🔧 DTS Workflow Explained

### 1. **DTS Build Process**
```bash
# What happens when you run dts_deploy.py:
dts devel build --arch arm64v8 --push
# → Builds Docker image using official DT base images
# → Pushes to DockerHub for robot access
# → Uses DT-standard directory structure
```

### 2. **DTS Deployment Process**
```bash
# DTS handles the deployment:
dts devel run \
    --robot YOUR_ROBOT_NAME \
    --mount /data/models:/models \
    --mount /data/config:/config \
    --name rl-inference-YOUR_ROBOT_NAME
# → Automatically connects to robot
# → Sets up proper ROS environment
# → Mounts data directories
# → Starts container with DT conventions
```

### 3. **DTS Monitoring**
```bash
# Check deployment status
python duckiebot_deployment_dts/dts_deploy.py \
    --robot-name YOUR_ROBOT_NAME \
    --health-check

# Monitor ROS topics using DTS
dts start_gui_tools YOUR_ROBOT_NAME -- rostopic list
dts start_gui_tools YOUR_ROBOT_NAME -- rostopic echo /YOUR_ROBOT_NAME/rl_inference/status
```

## 📁 DTS File Structure

```
duckiebot_deployment_dts/
├── Dockerfile                    # DTS-compatible Dockerfile
├── package.xml                   # ROS package metadata
├── CMakeLists.txt               # ROS build configuration
├── launch/
│   └── rl_inference.launch      # ROS launch file
├── src/
│   └── duckiebot_rl_inference_node.py  # Combined inference+control node
├── config/
│   └── default.yaml             # Default parameters
├── dts_deploy.py                # DTS deployment script
└── README_DTS.md               # This file
```

## 🎯 Key DTS Features

### **1. Official DT Integration**
- Uses `duckietown/dt-ros-commons` base image
- Follows DT naming conventions
- Integrates with DT infrastructure
- Compatible with DT monitoring tools

### **2. Simplified Node Structure**
```python
# Single node handles both inference and control
class DuckiebotRLInferenceNode:
    def __init__(self):
        self.robot_name = get_duckiebot_name()  # DT utility
        # ... setup inference and control
    
    def control_loop(self, event):
        # Run inference
        action = self.run_inference(observation)
        # Execute command
        self.execute_robot_command(action[0], action[1])
```

### **3. DT-Standard Topics**
```bash
# Input topics (DT standard)
/ROBOT_NAME/camera_node/image/compressed     # Camera feed
/ROBOT_NAME/emergency_stop                   # Emergency stop

# Output topics (DT standard)  
/ROBOT_NAME/car_cmd_switch_node/cmd          # Car commands
/ROBOT_NAME/wheels_driver_node/wheels_cmd    # Wheel commands
/ROBOT_NAME/rl_inference/status              # Status info
```

## 🛠️ Advanced DTS Usage

### **Build Only**
```bash
# Just build the image
python duckiebot_deployment_dts/dts_deploy.py \
    --robot-name YOUR_ROBOT_NAME \
    --build-only
```

### **Health Check Only**
```bash
# Check if deployment is working
python duckiebot_deployment_dts/dts_deploy.py \
    --robot-name YOUR_ROBOT_NAME \
    --health-check
```

### **Manual DTS Commands**
```bash
# Build manually
dts devel build --arch arm64v8

# Run manually
dts devel run --robot YOUR_ROBOT_NAME

# Access robot shell
dts start_gui_tools YOUR_ROBOT_NAME

# Monitor logs
dts start_gui_tools YOUR_ROBOT_NAME -- docker logs rl-inference-YOUR_ROBOT_NAME
```

## 🔍 Troubleshooting DTS

### **Common Issues**

1. **DTS Not Found**
   ```bash
   pip install duckietown-shell
   dts --set-version daffy
   ```

2. **Robot Not Reachable**
   ```bash
   ping YOUR_ROBOT_NAME.local
   # If fails, check robot WiFi and network
   ```

3. **Build Fails**
   ```bash
   # Check DTS version
   dts --version
   
   # Update DTS
   dts update
   ```

4. **Deployment Fails**
   ```bash
   # Check robot Docker
   dts start_gui_tools YOUR_ROBOT_NAME -- docker ps
   
   # Check available space
   dts start_gui_tools YOUR_ROBOT_NAME -- df -h
   ```

### **Debug Mode**
```bash
# Enable verbose logging
export DTS_DEBUG=1

# Check container logs
dts start_gui_tools YOUR_ROBOT_NAME -- docker logs rl-inference-YOUR_ROBOT_NAME -f
```

## 🎮 DTS vs Custom Comparison

### **When to Use DTS:**
- ✅ You want official DT compatibility
- ✅ You're using standard DT hardware
- ✅ You want simplified deployment
- ✅ You need DT ecosystem integration

### **When to Use Custom:**
- ✅ You need maximum flexibility
- ✅ You're using non-standard hardware
- ✅ You want custom Docker configurations
- ✅ You need advanced deployment features

## 🚀 Next Steps with DTS

1. **Test DTS deployment** with your trained models
2. **Customize launch parameters** in `rl_inference.launch`
3. **Add custom DT nodes** for additional functionality
4. **Integrate with DT fleet management** for multiple robots

The DTS version provides official Duckietown compatibility while maintaining the same RL bridging functionality!