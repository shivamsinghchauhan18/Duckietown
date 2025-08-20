#!/usr/bin/env python3
"""
DTS-compatible deployment script for Duckiebot RL system.
Uses official Duckietown Software Stack tools and conventions.
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


class DTSRLDeployer:
    """
    DTS-compatible deployer for RL models on Duckiebot.
    
    Uses official DTS tools:
    - dts devel build
    - dts devel run
    - dts start_gui_tools
    """
    
    def __init__(self, robot_name: str):
        self.robot_name = robot_name
        self.logger = self._setup_logging()
        
        # DTS configuration
        self.package_name = "duckiebot-rl-inference"
        self.image_name = f"duckietown/{self.package_name}:latest-arm64v8"
        
        self.logger.info(f"Initialized DTS deployer for {robot_name}")
    
    def _setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(f'DTSDeployer-{self.robot_name}')
    
    def validate_dts_environment(self) -> bool:
        """Validate that DTS is properly installed and configured."""
        try:
            # Check if dts command is available
            result = subprocess.run(['dts', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error("DTS not found. Please install dts: pip install duckietown-shell")
                return False
            
            self.logger.info(f"DTS version: {result.stdout.strip()}")
            
            # Check if robot is reachable
            ping_result = subprocess.run(['ping', '-c', '1', f'{self.robot_name}.local'], 
                                       capture_output=True, text=True)
            if ping_result.returncode != 0:
                self.logger.warning(f"Cannot ping {self.robot_name}.local - robot may not be reachable")
            else:
                self.logger.info(f"Robot {self.robot_name} is reachable")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating DTS environment: {e}")
            return False
    
    def build_dts_image(self) -> bool:
        """Build Docker image using DTS."""
        try:
            self.logger.info("Building DTS-compatible Docker image...")
            
            # Use dts devel build command
            cmd = [
                'dts', 'devel', 'build',
                '--arch', 'arm64v8',
                '--push'  # Push to DockerHub for robot access
            ]
            
            self.logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=Path(__file__).parent)
            
            if result.returncode == 0:
                self.logger.info("DTS image built and pushed successfully")
                return True
            else:
                self.logger.error("DTS image build failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error building DTS image: {e}")
            return False
    
    def transfer_models_to_robot(self, model_path: str, config_path: str) -> bool:
        """Transfer model and config files to robot using DTS."""
        try:
            self.logger.info("Transferring files to robot...")
            
            # Create directories on robot
            mkdir_cmd = [
                'dts', 'start_gui_tools', self.robot_name,
                '--', 'mkdir', '-p', '/data/models', '/data/config'
            ]
            subprocess.run(mkdir_cmd)
            
            # Transfer model file
            if Path(model_path).exists():
                scp_model_cmd = [
                    'scp', model_path, 
                    f'duckie@{self.robot_name}.local:/data/models/'
                ]
                result = subprocess.run(scp_model_cmd)
                if result.returncode != 0:
                    self.logger.error("Failed to transfer model file")
                    return False
                self.logger.info(f"Model transferred: {model_path}")
            
            # Transfer config file
            if Path(config_path).exists():
                scp_config_cmd = [
                    'scp', config_path,
                    f'duckie@{self.robot_name}.local:/data/config/'
                ]
                result = subprocess.run(scp_config_cmd)
                if result.returncode != 0:
                    self.logger.error("Failed to transfer config file")
                    return False
                self.logger.info(f"Config transferred: {config_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error transferring files: {e}")
            return False
    
    def deploy_with_dts(self, model_path: str, config_path: str) -> bool:
        """Deploy using DTS run command."""
        try:
            self.logger.info("Deploying with DTS...")
            
            # Stop any existing containers
            self.stop_existing_deployment()
            
            # Run container using dts
            cmd = [
                'dts', 'devel', 'run',
                '--robot', self.robot_name,
                '--mount', f'/data/models:/models',
                '--mount', f'/data/config:/config',
                '--env', f'ROBOT_NAME={self.robot_name}',
                '--env', f'MODEL_PATH=/models/{Path(model_path).name}',
                '--env', f'CONFIG_PATH=/config/{Path(config_path).name}',
                '--name', f'rl-inference-{self.robot_name}',
                '--detach'
            ]
            
            self.logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            
            if result.returncode == 0:
                self.logger.info("DTS deployment successful")
                return True
            else:
                self.logger.error("DTS deployment failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deploying with DTS: {e}")
            return False
    
    def stop_existing_deployment(self):
        """Stop any existing RL inference containers."""
        try:
            # Use dts to stop containers on robot
            stop_cmd = [
                'dts', 'start_gui_tools', self.robot_name,
                '--', 'docker', 'stop', f'rl-inference-{self.robot_name}'
            ]
            subprocess.run(stop_cmd, capture_output=True)
            
            remove_cmd = [
                'dts', 'start_gui_tools', self.robot_name,
                '--', 'docker', 'rm', f'rl-inference-{self.robot_name}'
            ]
            subprocess.run(remove_cmd, capture_output=True)
            
        except Exception as e:
            self.logger.debug(f"Error stopping existing deployment: {e}")
    
    def check_deployment_health_dts(self) -> dict:
        """Check deployment health using DTS tools."""
        health_status = {
            'container_running': False,
            'ros_master_active': False,
            'camera_feed': False,
            'inference_active': False
        }
        
        try:
            # Check if container is running
            ps_cmd = [
                'dts', 'start_gui_tools', self.robot_name,
                '--', 'docker', 'ps', '--filter', f'name=rl-inference-{self.robot_name}'
            ]
            result = subprocess.run(ps_cmd, capture_output=True, text=True)
            health_status['container_running'] = f'rl-inference-{self.robot_name}' in result.stdout
            
            if health_status['container_running']:
                # Check ROS topics
                rostopic_cmd = [
                    'dts', 'start_gui_tools', self.robot_name,
                    '--', 'rostopic', 'list'
                ]
                result = subprocess.run(rostopic_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    topics = result.stdout
                    health_status['ros_master_active'] = '/rosout' in topics
                    health_status['camera_feed'] = 'camera_node/image' in topics
                    health_status['inference_active'] = 'rl_inference' in topics
            
            self.logger.info(f"Health check: {health_status}")
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error checking health: {e}")
            return health_status
    
    def deploy_full_system_dts(self, model_path: str, config_path: str) -> bool:
        """Deploy complete system using DTS."""
        self.logger.info("Starting DTS deployment...")
        
        # Step 1: Validate DTS environment
        if not self.validate_dts_environment():
            return False
        
        # Step 2: Build DTS image
        if not self.build_dts_image():
            return False
        
        # Step 3: Transfer files
        if not self.transfer_models_to_robot(model_path, config_path):
            return False
        
        # Step 4: Deploy with DTS
        if not self.deploy_with_dts(model_path, config_path):
            return False
        
        # Step 5: Health check
        time.sleep(15)  # Give system time to start
        health_results = self.check_deployment_health_dts()
        
        if not health_results['container_running']:
            self.logger.error("Deployment health check failed")
            return False
        
        self.logger.info("DTS deployment completed successfully!")
        return True


def create_dts_metadata():
    """Create DTS-compatible metadata files."""
    
    # Create package.xml for ROS package
    package_xml = """<?xml version="1.0"?>
<package format="2">
  <name>duckiebot_rl_inference</name>
  <version>1.0.0</version>
  <description>RL model inference for Duckiebot using DTS</description>
  
  <maintainer email="your.email@example.com">Your Name</maintainer>
  <license>MIT</license>
  
  <buildtool_depend>catkin</buildtool_depend>
  
  <depend>rospy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>duckietown_msgs</depend>
  <depend>cv_bridge</depend>
  
  <export>
  </export>
</package>"""
    
    # Create CMakeLists.txt
    cmake_lists = """cmake_minimum_required(VERSION 3.0.2)
project(duckiebot_rl_inference)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  duckietown_msgs
)

catkin_package()

# Install Python scripts
catkin_install_python(PROGRAMS
  src/duckiebot_rl_inference_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)"""
    
    # Write files
    Path('duckiebot_deployment_dts/package.xml').write_text(package_xml)
    Path('duckiebot_deployment_dts/CMakeLists.txt').write_text(cmake_lists)


def main():
    """Main DTS deployment function."""
    parser = argparse.ArgumentParser(description='Deploy RL model using DTS')
    parser.add_argument('--robot-name', required=True, help='Name of the robot')
    parser.add_argument('--model-path', required=True, help='Path to trained model')
    parser.add_argument('--config-path', required=True, help='Path to configuration file')
    parser.add_argument('--build-only', action='store_true', help='Only build the image')
    parser.add_argument('--health-check', action='store_true', help='Only run health check')
    
    args = parser.parse_args()
    
    # Create DTS metadata files
    create_dts_metadata()
    
    # Create deployer
    deployer = DTSRLDeployer(args.robot_name)
    
    if args.health_check:
        health_results = deployer.check_deployment_health_dts()
        print(json.dumps(health_results, indent=2))
        sys.exit(0 if health_results['container_running'] else 1)
    
    if args.build_only:
        success = deployer.build_dts_image()
        sys.exit(0 if success else 1)
    
    # Full deployment
    success = deployer.deploy_full_system_dts(args.model_path, args.config_path)
    
    if success:
        print(f"✅ DTS deployment successful! Robot {args.robot_name} is ready.")
        print(f"Monitor with: dts start_gui_tools {args.robot_name} -- rostopic echo /{args.robot_name}/rl_inference/status")
    else:
        print(f"❌ DTS deployment failed for robot {args.robot_name}")
        sys.exit(1)


if __name__ == '__main__':
    main()