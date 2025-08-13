#!/usr/bin/env python3
"""
Example script demonstrating YOLO v5 integration with Duckietown RL environment.
This script shows how to use the YOLO utilities for object detection.
"""

import os
import sys
import numpy as np
import cv2

# Add parent directory to path to import duckietown_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from duckietown_utils.yolo_utils import create_yolo_inference_system


def main():
    """
    Demonstrate YOLO integration with sample image processing.
    """
    print("YOLO v5 Integration Example")
    print("=" * 40)
    
    # Configuration
    model_path = "yolov5s.pt"  # This would be downloaded automatically by ultralytics
    confidence_threshold = 0.5
    device = 'auto'
    
    print(f"Model path: {model_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Device: {device}")
    print()
    
    # Create YOLO inference system
    print("Creating YOLO inference system...")
    yolo_system = create_yolo_inference_system(
        model_path=model_path,
        device=device,
        confidence_threshold=confidence_threshold,
        max_detections=10
    )
    
    if yolo_system is None:
        print("❌ Failed to create YOLO inference system")
        print("This is expected if ultralytics is not installed.")
        print("To install: pip install ultralytics")
        return
    
    print("✅ YOLO inference system created successfully")
    print()
    
    # Create a sample image (simulating Duckietown camera feed)
    print("Creating sample image...")
    sample_image = create_sample_duckietown_image()
    
    # Perform object detection
    print("Running object detection...")
    results = yolo_system.detect_objects(sample_image)
    
    # Display results
    print("Detection Results:")
    print("-" * 20)
    print(f"Number of detections: {results['detection_count']}")
    print(f"Inference time: {results['inference_time']:.3f}s")
    print(f"Safety critical: {results['safety_critical']}")
    print()
    
    if results['detections']:
        print("Detected objects:")
        for i, detection in enumerate(results['detections']):
            print(f"  {i+1}. {detection['class']} "
                  f"(confidence: {detection['confidence']:.2f}, "
                  f"distance: {detection['distance']:.2f}m)")
    else:
        print("No objects detected")
    
    # Performance statistics
    print()
    print("Performance Statistics:")
    print("-" * 20)
    stats = yolo_system.get_performance_stats()
    print(f"Total inferences: {stats['total_inferences']}")
    print(f"Average inference time: {stats['avg_inference_time']:.3f}s")
    
    print()
    print("Example completed successfully! 🎉")


def create_sample_duckietown_image():
    """
    Create a sample image that simulates a Duckietown camera feed.
    In a real scenario, this would come from the simulator or real robot.
    """
    # Create a simple road-like image
    height, width = 480, 640
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add road (gray)
    image[height//2:, :] = [128, 128, 128]
    
    # Add yellow lane lines
    cv2.line(image, (width//4, height//2), (width//4, height), (0, 255, 255), 5)
    cv2.line(image, (3*width//4, height//2), (3*width//4, height), (0, 255, 255), 5)
    
    # Add some simple shapes to simulate objects
    # Red rectangle (could be a duckie)
    cv2.rectangle(image, (300, 350), (350, 400), (0, 0, 255), -1)
    
    # Blue circle (could be another object)
    cv2.circle(image, (500, 300), 30, (255, 0, 0), -1)
    
    return image


if __name__ == "__main__":
    main()