# 🚀 BULLETPROOF DUCKIEBOT DEPLOYMENT

This directory contains everything needed for a **bulletproof deployment** of your champion RL model to a Duckiebot.

## 📁 Files Overview

### Core Files
- `champion_model.pth` - Your trained RL model (1.9MB PyTorch file)
- `enhanced_config.yml` - Configuration file
- `rl_inference_node.py` - Simple ROS node for inference
- `Dockerfile` - Docker container definition

### Deployment Scripts
- `deploy.sh` - **Main deployment script** (Docker-based)
- `deploy_manual.sh` - Manual deployment (fallback if Docker fails)
- `verify_deployment.sh` - Check if deployment is working
- `emergency_stop.sh` - Emergency stop the robot

### Support Files
- `launch_rl.sh` - Container startup script
- `README.md` - This file

## 🎯 Quick Start

### Option 1: Docker Deployment (Recommended)

```bash
# 1. Edit deploy.sh and set your Docker Hub username
vim deploy.sh  # Change "your-dockerhub-username" to your actual username

# 2. Deploy to your robot
./deploy.sh YOUR_ROBOT_NAME your-dockerhub-username

# 3. Verify deployment
./verify_deployment.sh YOUR_ROBOT_NAME
```

### Option 2: Manual Deployment (Fallback)

```bash
# If Docker deployment fails, use manual deployment
./deploy_manual.sh YOUR_ROBOT_NAME

# Then SSH into robot and start manually
ssh duckie@YOUR_ROBOT_NAME.local
/code/start_rl.sh
```

## 🔍 Verification

After deployment, verify everything is working:

```bash
# Check deployment status
./verify_deployment.sh YOUR_ROBOT_NAME

# Monitor robot behavior
ssh duckie@YOUR_ROBOT_NAME.local
rostopic echo /YOUR_ROBOT_NAME/rl_inference/status
```

## 🚨 Emergency Stop

If something goes wrong:

```bash
# Emergency stop
./emergency_stop.sh YOUR_ROBOT_NAME

# Or manually
ssh duckie@YOUR_ROBOT_NAME.local 'docker stop rl-inference'
```

## 📊 Expected Behavior

When working correctly, you should see:

1. **Container Running**: `docker ps` shows `rl-inference` container
2. **ROS Topics Active**: 
   - `/YOUR_ROBOT_NAME/rl_inference/status` - Model status
   - `/YOUR_ROBOT_NAME/car_cmd_switch_node/cmd` - Robot commands
3. **Robot Movement**: Conservative lane-following behavior
4. **Status Messages**: JSON status every 0.1 seconds

## 🛠️ Troubleshooting

### Common Issues

**1. Robot not reachable**
```bash
ping YOUR_ROBOT_NAME.local
# If fails: Check robot WiFi, power, network
```

**2. Container won't start**
```bash
ssh duckie@YOUR_ROBOT_NAME.local 'docker logs rl-inference'
# Check logs for errors
```

**3. No camera feed**
```bash
ssh duckie@YOUR_ROBOT_NAME.local 'rostopic list | grep camera'
# Should show camera topics
```

**4. Model loading fails**
```bash
# Check model file
ssh duckie@YOUR_ROBOT_NAME.local 'ls -la /data/models/'
# Should show champion_model.pth (~2MB)
```

### Debug Commands

```bash
# Check container status
ssh duckie@YOUR_ROBOT_NAME.local 'docker ps'

# View container logs
ssh duckie@YOUR_ROBOT_NAME.local 'docker logs rl-inference -f'

# Check ROS topics
ssh duckie@YOUR_ROBOT_NAME.local 'rostopic list'

# Monitor inference status
ssh duckie@YOUR_ROBOT_NAME.local 'rostopic echo /YOUR_ROBOT_NAME/rl_inference/status'

# Check system resources
ssh duckie@YOUR_ROBOT_NAME.local 'htop'
```

## ⚙️ Configuration

### Model Behavior
- **Max Speed**: 0.3 m/s (conservative)
- **Inference Rate**: 10 Hz
- **Input Size**: 160x120 RGB images
- **Output**: [steering, throttle] in [-1, 1] range

### Safety Features
- Speed limiting (max 0.3 m/s)
- Command timeout (stops if no commands)
- Emergency stop capability
- Conservative steering limits

## 🔧 Customization

### Adjust Speed
Edit `rl_inference_node.py`, line ~85:
```python
linear_vel = throttle * 0.5  # Change from 0.3 to 0.5 for higher speed
```

### Change Inference Rate
Edit `rl_inference_node.py`, line ~50:
```python
rospy.Duration(0.05),  # Change from 0.1 to 0.05 for 20 Hz
```

### Modify Docker Image
Edit `Dockerfile` to add dependencies or change base image.

## 📈 Performance Expectations

### Normal Operation
- **Inference Time**: ~10-50ms per frame
- **Memory Usage**: ~200-500MB
- **CPU Usage**: ~20-40%
- **Behavior**: Conservative lane following

### Success Indicators
- ✅ Container runs without crashes
- ✅ Status messages published at 10 Hz
- ✅ Robot follows lane centerline
- ✅ Smooth, conservative movements
- ✅ No collisions or erratic behavior

## 🎯 Next Steps

1. **Test in Safe Area**: Always test in controlled environment first
2. **Monitor Performance**: Watch for consistent behavior
3. **Adjust Parameters**: Tune speed/sensitivity as needed
4. **Collect Data**: Log performance for analysis
5. **Iterate**: Improve model based on real-world performance

## 📞 Support

If you encounter issues:

1. Run `./verify_deployment.sh YOUR_ROBOT_NAME`
2. Check container logs: `docker logs rl-inference`
3. Verify ROS topics are active
4. Use emergency stop if needed
5. Try manual deployment as fallback

## ⚠️ Safety Reminders

- **Always test in safe, open areas**
- **Have emergency stop ready**
- **Monitor robot behavior closely**
- **Start with low speeds**
- **Keep human supervisor present**

---

**🎉 Your champion model is ready for deployment!**

The system is designed to be safe, reliable, and easy to debug. Follow the steps above and your Duckiebot will be running your RL model in no time!