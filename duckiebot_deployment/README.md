# 🤖 Duckiebot Deployment System

This directory contains everything needed to deploy your trained RL models to real Duckiebot hardware.

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Your Computer     │    │    Duckiebot        │    │   ROS Network       │
│                     │    │                     │    │                     │
│ • Trained Models    │───►│ • RL Inference      │───►│ • Control Commands  │
│ • Deployment Script │    │ • Control Node      │    │ • Camera Feed       │
│ • Configuration     │    │ • Safety Systems    │    │ • Status Messages   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## 📁 Files Overview

### Core Components

1. **`rl_inference_bridge.py`** - Runs your trained model and generates actions
2. **`duckiebot_control_node.py`** - Converts RL actions to robot commands
3. **`deploy_to_duckiebot.py`** - Automated deployment script
4. **`Dockerfile.duckiebot`** - Docker container for robot deployment

### Data Flow

```
Camera Image → RL Model → [steering, throttle] → Robot Actuators
     ↑              ↑            ↑                    ↑
  640x480        Inference    [-1,1] range      Wheel Commands
   RGB           Bridge        Actions           & Car Commands
```

## 🚀 Quick Start

### 1. Run Tests First
```bash
# Make sure your system works
python tests/run_comprehensive_tests.py --quick
```

### 2. Deploy to Robot
```bash
# Deploy your trained model
python duckiebot_deployment/deploy_to_duckiebot.py \
    --robot-name YOUR_ROBOT_NAME \
    --robot-ip YOUR_ROBOT_IP \
    --model-path models/your_champion_model.pth \
    --config-path config/enhanced_config.yml
```

### 3. Monitor Status
```bash
# Check if everything is working
python duckiebot_deployment/deploy_to_duckiebot.py \
    --robot-name YOUR_ROBOT_NAME \
    --robot-ip YOUR_ROBOT_IP \
    --health-check
```

## 🔧 Detailed Setup

### Prerequisites

1. **Duckiebot Setup**
   ```bash
   # On your Duckiebot, make sure these are running:
   docker ps  # Should show dt-core containers
   rostopic list  # Should show camera and wheel topics
   ```

2. **Network Access**
   ```bash
   # Test SSH access
   ssh duckie@YOUR_ROBOT_IP
   
   # Test Docker access
   ssh duckie@YOUR_ROBOT_IP 'docker ps'
   ```

3. **Model Ready**
   ```bash
   # Make sure you have a trained model
   ls models/  # Should contain your .pth or checkpoint files
   ```

### Step-by-Step Deployment

#### Step 1: Validate Your Model
```bash
python duckiebot_deployment/deploy_to_duckiebot.py \
    --robot-name mybot \
    --robot-ip 192.168.1.100 \
    --model-path models/champion_model.pth \
    --config-path config/enhanced_config.yml \
    --build-docker
```

#### Step 2: Deploy to Robot
```bash
# Full deployment (this does everything)
python duckiebot_deployment/deploy_to_duckiebot.py \
    --robot-name mybot \
    --robot-ip 192.168.1.100 \
    --model-path models/champion_model.pth \
    --config-path config/enhanced_config.yml
```

#### Step 3: Verify Deployment
```bash
# Check system health
python duckiebot_deployment/deploy_to_duckiebot.py \
    --robot-name mybot \
    --robot-ip 192.168.1.100 \
    --health-check
```

## 🎮 Manual Control & Testing

### ROS Topics for Monitoring

```bash
# On the robot, monitor these topics:
rostopic echo /mybot/rl_commands              # RL model outputs
rostopic echo /mybot/wheels_driver_node/wheels_cmd  # Wheel commands
rostopic echo /mybot/camera_node/image/compressed   # Camera feed
rostopic echo /mybot/control_node/status      # Control node status
```

### Emergency Stop
```bash
# Send emergency stop
rostopic pub /mybot/emergency_stop std_msgs/Bool "data: true"

# Resume operation
rostopic pub /mybot/emergency_stop std_msgs/Bool "data: false"
```

### Manual Override
```bash
# Disable RL inference
rostopic pub /mybot/inference_bridge/enable std_msgs/Bool "data: false"

# Send manual commands
rostopic pub /mybot/rl_commands std_msgs/Float32MultiArray "data: [0.5, 0.3]"  # [steering, throttle]
```

## 🛡️ Safety Features

### Built-in Safety Systems

1. **Command Timeout** - Robot stops if no commands received for 1 second
2. **Emergency Stop** - Immediate stop via ROS topic
3. **Speed Limiting** - Maximum throttle limited to 70% for safety
4. **Steering Rate Limiting** - Prevents sudden steering changes
5. **Joystick Override** - Manual control always takes priority

### Safety Checklist

- [ ] Emergency stop button accessible
- [ ] Robot in safe testing area
- [ ] Human supervisor present
- [ ] Speed limits configured
- [ ] Rollback plan ready

## 🔍 Troubleshooting

### Common Issues

#### 1. Model Won't Load
```bash
# Check model file
ls -la models/your_model.pth

# Test model loading locally
python -c "import torch; torch.load('models/your_model.pth')"
```

#### 2. Robot Not Responding
```bash
# Check container status
ssh duckie@ROBOT_IP 'docker ps'

# Check ROS topics
ssh duckie@ROBOT_IP 'rostopic list'

# Restart deployment
python duckiebot_deployment/deploy_to_duckiebot.py --robot-name mybot --robot-ip ROBOT_IP --rollback
```

#### 3. Poor Performance
```bash
# Check inference timing
rostopic echo /mybot/inference_bridge/status

# Monitor system resources
ssh duckie@ROBOT_IP 'htop'
```

### Debug Mode

Enable detailed logging:
```bash
# Set debug environment variable
export ROS_LOG_LEVEL=DEBUG

# Or modify the deployment script to enable debug logging
```

## 📊 Performance Monitoring

### Key Metrics to Watch

1. **Inference Time** - Should be < 50ms
2. **Command Frequency** - Should be ~10Hz
3. **Memory Usage** - Should be < 70% of available RAM
4. **CPU Usage** - Should be < 60% average

### Monitoring Commands

```bash
# Real-time status
watch -n 1 'rostopic echo /mybot/inference_bridge/status | head -20'

# Performance logging
rostopic echo /mybot/control_node/debug > performance.log
```

## 🔄 Model Updates

### Hot-Swapping Models

```bash
# Deploy new model without stopping robot
python duckiebot_deployment/deploy_to_duckiebot.py \
    --robot-name mybot \
    --robot-ip 192.168.1.100 \
    --model-path models/new_champion_model.pth \
    --config-path config/enhanced_config.yml
```

### A/B Testing

```bash
# Deploy model A
python duckiebot_deployment/deploy_to_duckiebot.py \
    --robot-name mybot-a \
    --robot-ip 192.168.1.100 \
    --model-path models/model_a.pth \
    --config-path config/config_a.yml

# Deploy model B  
python duckiebot_deployment/deploy_to_duckiebot.py \
    --robot-name mybot-b \
    --robot-ip 192.168.1.101 \
    --model-path models/model_b.pth \
    --config-path config/config_b.yml
```

## 🎯 Next Steps

1. **Test in Simulation First** - Always validate in gym before deploying
2. **Start Slow** - Begin with low speeds and simple scenarios
3. **Monitor Continuously** - Watch performance metrics and robot behavior
4. **Iterate Quickly** - Use the deployment system for rapid testing cycles
5. **Scale Gradually** - Add more complex behaviors once basics work

## 📞 Support

If you encounter issues:

1. Check the logs: `docker logs rl-inference-mybot`
2. Verify ROS connectivity: `rostopic list`
3. Test model locally first
4. Use rollback if needed: `--rollback`
5. Monitor system resources on the robot

The deployment system is designed to be safe and recoverable - don't hesitate to use the rollback feature if something goes wrong!