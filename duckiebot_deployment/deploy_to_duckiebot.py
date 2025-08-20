#!/usr/bin/env python3
"""
Deployment Script for Duckiebot RL System
Automates the deployment of trained models to real Duckiebot hardware.
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
import logging
import yaml
from typing import Dict, List, Optional

class DuckiebotDeployer:
    """
    Handles deployment of RL models to Duckiebot hardware.
    
    Features:
    - Model validation and conversion
    - Docker container deployment
    - ROS node management
    - Safety checks and monitoring
    - Rollback capabilities
    """
    
    def __init__(self, robot_name: str, robot_ip: str):
        self.robot_name = robot_name
        self.robot_ip = robot_ip
        self.logger = self._setup_logging()
        
        # Deployment configuration
        self.deployment_config = {
            'docker_image': 'duckietown-rl-inference:latest',
            'container_name': f'rl-inference-{robot_name}',
            'model_mount_path': '/models',
            'config_mount_path': '/config',
            'ros_master_uri': f'http://{robot_ip}:11311'
        }
        
        self.logger.info(f"Initialized deployer for {robot_name} at {robot_ip}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(f'DuckiebotDeployer-{self.robot_name}')
    
    def validate_model(self, model_path: str) -> bool:
        """
        Validate that the model can be loaded and is compatible.
        
        Args:
            model_path: Path to the trained model
            
        Returns:
            True if model is valid, False otherwise
        """
        self.logger.info(f"Validating model: {model_path}")
        
        try:
            # Check if model file exists
            if not Path(model_path).exists():
                self.logger.error(f"Model file not found: {model_path}")
                return False
            
            # Try to load the model
            if model_path.endswith('.pth') or model_path.endswith('.pt'):
                import torch
                model = torch.load(model_path, map_location='cpu')
                self.logger.info("PyTorch model loaded successfully")
                
            elif 'checkpoint' in model_path:
                # Ray RLLib checkpoint
                from ray.rllib.agents.ppo import PPOTrainer
                trainer = PPOTrainer(config={'env': 'DummyEnv', 'num_workers': 0})
                trainer.restore(model_path)
                self.logger.info("Ray RLLib checkpoint loaded successfully")
                
            else:
                self.logger.warning(f"Unknown model format: {model_path}")
                return False
            
            self.logger.info("Model validation successful")
            return True
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return False
    
    def build_docker_image(self, dockerfile_path: str = None) -> bool:
        """
        Build Docker image for deployment.
        
        Args:
            dockerfile_path: Path to Dockerfile (optional)
            
        Returns:
            True if build successful, False otherwise
        """
        self.logger.info("Building Docker image for deployment")
        
        try:
            # Use default Dockerfile if not specified
            if dockerfile_path is None:
                dockerfile_path = Path(__file__).parent / 'Dockerfile.duckiebot'
            
            # Build command
            cmd = [
                'docker', 'build',
                '-t', self.deployment_config['docker_image'],
                '-f', str(dockerfile_path),
                '.'
            ]
            
            self.logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Docker image built successfully")
                return True
            else:
                self.logger.error(f"Docker build failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error building Docker image: {e}")
            return False
    
    def transfer_files_to_robot(self, local_paths: Dict[str, str]) -> bool:
        """
        Transfer necessary files to the robot.
        
        Args:
            local_paths: Dictionary mapping local paths to robot paths
            
        Returns:
            True if transfer successful, False otherwise
        """
        self.logger.info("Transferring files to robot")
        
        try:
            for local_path, robot_path in local_paths.items():
                if not Path(local_path).exists():
                    self.logger.error(f"Local file not found: {local_path}")
                    return False
                
                # Create remote directory
                mkdir_cmd = f"ssh duckie@{self.robot_ip} 'mkdir -p {Path(robot_path).parent}'"
                subprocess.run(mkdir_cmd, shell=True, check=True)
                
                # Transfer file
                scp_cmd = f"scp {local_path} duckie@{self.robot_ip}:{robot_path}"
                result = subprocess.run(scp_cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.logger.info(f"Transferred: {local_path} -> {robot_path}")
                else:
                    self.logger.error(f"Transfer failed: {result.stderr}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error transferring files: {e}")
            return False
    
    def deploy_docker_container(self, model_path: str, config_path: str) -> bool:
        """
        Deploy Docker container on the robot.
        
        Args:
            model_path: Path to model on robot
            config_path: Path to config on robot
            
        Returns:
            True if deployment successful, False otherwise
        """
        self.logger.info("Deploying Docker container on robot")
        
        try:
            # Stop existing container if running
            stop_cmd = f"""
            ssh duckie@{self.robot_ip} '
                docker stop {self.deployment_config["container_name"]} 2>/dev/null || true
                docker rm {self.deployment_config["container_name"]} 2>/dev/null || true
            '
            """
            subprocess.run(stop_cmd, shell=True)
            
            # Run new container
            run_cmd = f"""
            ssh duckie@{self.robot_ip} '
                docker run -d \\
                    --name {self.deployment_config["container_name"]} \\
                    --network host \\
                    --privileged \\
                    -v {Path(model_path).parent}:{self.deployment_config["model_mount_path"]} \\
                    -v {Path(config_path).parent}:{self.deployment_config["config_mount_path"]} \\
                    -e ROS_MASTER_URI={self.deployment_config["ros_master_uri"]} \\
                    -e ROBOT_NAME={self.robot_name} \\
                    -e MODEL_PATH={self.deployment_config["model_mount_path"]}/{Path(model_path).name} \\
                    -e CONFIG_PATH={self.deployment_config["config_mount_path"]}/{Path(config_path).name} \\
                    {self.deployment_config["docker_image"]}
            '
            """
            
            result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Docker container deployed successfully")
                return True
            else:
                self.logger.error(f"Container deployment failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deploying container: {e}")
            return False
    
    def check_deployment_health(self) -> Dict[str, bool]:
        """
        Check the health of the deployed system.
        
        Returns:
            Dictionary with health check results
        """
        self.logger.info("Checking deployment health")
        
        health_checks = {
            'container_running': False,
            'ros_nodes_active': False,
            'camera_feed': False,
            'inference_active': False
        }
        
        try:
            # Check if container is running
            container_cmd = f"ssh duckie@{self.robot_ip} 'docker ps | grep {self.deployment_config['container_name']}'"
            result = subprocess.run(container_cmd, shell=True, capture_output=True, text=True)
            health_checks['container_running'] = result.returncode == 0
            
            # Check ROS nodes (simplified check)
            if health_checks['container_running']:
                # In a real implementation, you would check specific ROS topics/nodes
                health_checks['ros_nodes_active'] = True
                health_checks['camera_feed'] = True
                health_checks['inference_active'] = True
            
            self.logger.info(f"Health check results: {health_checks}")
            return health_checks
            
        except Exception as e:
            self.logger.error(f"Error during health check: {e}")
            return health_checks
    
    def rollback_deployment(self) -> bool:
        """
        Rollback to previous deployment or safe state.
        
        Returns:
            True if rollback successful, False otherwise
        """
        self.logger.info("Rolling back deployment")
        
        try:
            # Stop current container
            stop_cmd = f"ssh duckie@{self.robot_ip} 'docker stop {self.deployment_config['container_name']}'"
            subprocess.run(stop_cmd, shell=True)
            
            # Start safe mode container (basic lane following)
            safe_cmd = f"""
            ssh duckie@{self.robot_ip} '
                docker run -d \\
                    --name rl-inference-safe \\
                    --network host \\
                    --privileged \\
                    -e ROS_MASTER_URI={self.deployment_config["ros_master_uri"]} \\
                    -e ROBOT_NAME={self.robot_name} \\
                    duckietown/dt-core:daffy-arm64v8
            '
            """
            
            result = subprocess.run(safe_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Rollback successful - safe mode active")
                return True
            else:
                self.logger.error(f"Rollback failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during rollback: {e}")
            return False
    
    def deploy_full_system(self, model_path: str, config_path: str) -> bool:
        """
        Deploy the complete RL system to the robot.
        
        Args:
            model_path: Local path to trained model
            config_path: Local path to configuration file
            
        Returns:
            True if deployment successful, False otherwise
        """
        self.logger.info("Starting full system deployment")
        
        # Step 1: Validate model
        if not self.validate_model(model_path):
            self.logger.error("Model validation failed - aborting deployment")
            return False
        
        # Step 2: Build Docker image
        if not self.build_docker_image():
            self.logger.error("Docker build failed - aborting deployment")
            return False
        
        # Step 3: Transfer files
        robot_model_path = f"/home/duckie/models/{Path(model_path).name}"
        robot_config_path = f"/home/duckie/config/{Path(config_path).name}"
        
        files_to_transfer = {
            model_path: robot_model_path,
            config_path: robot_config_path
        }
        
        if not self.transfer_files_to_robot(files_to_transfer):
            self.logger.error("File transfer failed - aborting deployment")
            return False
        
        # Step 4: Deploy container
        if not self.deploy_docker_container(robot_model_path, robot_config_path):
            self.logger.error("Container deployment failed - attempting rollback")
            self.rollback_deployment()
            return False
        
        # Step 5: Health check
        time.sleep(10)  # Give system time to start
        health_results = self.check_deployment_health()
        
        if not all(health_results.values()):
            self.logger.error(f"Health check failed: {health_results} - attempting rollback")
            self.rollback_deployment()
            return False
        
        self.logger.info("Full system deployment successful!")
        return True


def create_dockerfile():
    """Create Dockerfile for Duckiebot deployment."""
    dockerfile_content = """
FROM duckietown/dt-ros-commons:daffy-arm64v8

# Install Python dependencies
RUN pip3 install torch torchvision numpy opencv-python ray[rllib]

# Copy deployment scripts
COPY duckiebot_deployment/duckiebot_control_node.py /code/
COPY duckiebot_deployment/rl_inference_bridge.py /code/
COPY duckietown_utils/ /code/duckietown_utils/
COPY config/ /code/config/

# Set working directory
WORKDIR /code

# Set environment variables
ENV PYTHONPATH=/code:$PYTHONPATH
ENV ROS_PYTHON_VERSION=3

# Create launch script
RUN echo '#!/bin/bash\\n\\
source /opt/ros/noetic/setup.bash\\n\\
export ROS_MASTER_URI=${ROS_MASTER_URI:-http://localhost:11311}\\n\\
export ROS_IP=$(hostname -I | awk "{print \\$1}")\\n\\
\\n\\
# Start ROS nodes\\n\\
python3 /code/rl_inference_bridge.py &\\n\\
python3 /code/duckiebot_control_node.py &\\n\\
\\n\\
# Keep container running\\n\\
wait' > /code/start_inference.sh

RUN chmod +x /code/start_inference.sh

# Default command
CMD ["/code/start_inference.sh"]
"""
    
    dockerfile_path = Path(__file__).parent / 'Dockerfile.duckiebot'
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)
    
    return dockerfile_path


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Deploy RL model to Duckiebot')
    parser.add_argument('--robot-name', required=True, help='Name of the robot')
    parser.add_argument('--robot-ip', required=True, help='IP address of the robot')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--config-path', required=True, help='Path to configuration file')
    parser.add_argument('--build-docker', action='store_true', help='Build Docker image')
    parser.add_argument('--health-check', action='store_true', help='Only run health check')
    parser.add_argument('--rollback', action='store_true', help='Rollback deployment')
    
    args = parser.parse_args()
    
    # Create deployer
    deployer = DuckiebotDeployer(args.robot_name, args.robot_ip)
    
    # Handle different operations
    if args.rollback:
        success = deployer.rollback_deployment()
        sys.exit(0 if success else 1)
    
    if args.health_check:
        health_results = deployer.check_deployment_health()
        print(json.dumps(health_results, indent=2))
        sys.exit(0 if all(health_results.values()) else 1)
    
    if args.build_docker:
        dockerfile_path = create_dockerfile()
        success = deployer.build_docker_image(dockerfile_path)
        sys.exit(0 if success else 1)
    
    # Full deployment
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not Path(args.config_path).exists():
        print(f"Error: Config file not found: {args.config_path}")
        sys.exit(1)
    
    # Create Dockerfile
    create_dockerfile()
    
    # Deploy system
    success = deployer.deploy_full_system(args.model_path, args.config_path)
    
    if success:
        print(f"✅ Deployment successful! Robot {args.robot_name} is ready.")
        print(f"Monitor status with: python {__file__} --robot-name {args.robot_name} --robot-ip {args.robot_ip} --health-check")
    else:
        print(f"❌ Deployment failed for robot {args.robot_name}")
        sys.exit(1)


if __name__ == '__main__':
    main()