# YOLO v5 Integration for Duckietown RL

This document describes the YOLO v5 integration infrastructure for the Enhanced Duckietown RL system.

## Overview

The YOLO integration provides real-time object detection capabilities for the Duckietown autonomous driving environment. It includes robust error handling, performance optimization, and seamless integration with the existing RL training pipeline.

## Components

### YOLOModelLoader

Handles loading and managing YOLO v5 models with comprehensive error handling.

**Features:**
- Automatic device detection (CPU/CUDA)
- Model validation and testing
- Error recovery and reloading
- Performance monitoring

**Usage:**
```python
from duckietown_utils.yolo_utils import YOLOModelLoader

loader = YOLOModelLoader(
    model_path="yolov5s.pt",
    device="auto",
    confidence_threshold=0.5
)

if loader.load_model():
    print("Model loaded successfully!")
    model = loader.get_model()
```

### YOLOInferenceWrapper

High-performance inference wrapper with optimization and safety features.

**Features:**
- Real-time object detection
- Safety-critical object identification
- Performance metrics tracking
- Input validation and error handling

**Usage:**
```python
from duckietown_utils.yolo_utils import YOLOInferenceWrapper

wrapper = YOLOInferenceWrapper(
    model_loader=loader,
    max_detections=10
)

# Detect objects in image
results = wrapper.detect_objects(image)
print(f"Found {results['detection_count']} objects")
```

### Factory Function

Convenient factory function for creating complete YOLO systems.

**Usage:**
```python
from duckietown_utils.yolo_utils import create_yolo_inference_system

yolo_system = create_yolo_inference_system(
    model_path="yolov5s.pt",
    device="auto",
    confidence_threshold=0.5,
    max_detections=10
)

if yolo_system:
    results = yolo_system.detect_objects(image)
```

## Installation

### Dependencies

Add the following to your `requirements.txt`:

```
torch>=1.7.0
torchvision>=0.8.0
ultralytics>=8.0.0
Pillow>=8.0.0
PyYAML>=5.3.1
```

### Install Dependencies

```bash
pip install torch torchvision ultralytics Pillow PyYAML
```

## Detection Output Format

The detection system returns a structured dictionary:

```python
{
    'detections': [
        {
            'class': 'person',           # Object class name
            'confidence': 0.85,          # Detection confidence (0-1)
            'bbox': [100, 100, 200, 200], # Bounding box [x1, y1, x2, y2]
            'center': [150.0, 150.0],    # Center coordinates
            'relative_position': [0.1, 0.2], # Position relative to image center
            'distance': 1.5,             # Estimated distance in meters
            'area': 10000.0              # Bounding box area in pixels
        }
    ],
    'detection_count': 1,                # Number of detections
    'inference_time': 0.045,             # Processing time in seconds
    'frame_shape': (480, 640, 3),       # Input image shape
    'safety_critical': False             # Any objects within safety distance
}
```

## Performance Considerations

### Real-time Processing

The system is optimized for real-time processing:
- Target: >= 10 FPS for RL training
- Typical inference time: 20-50ms on GPU
- Automatic device selection (CUDA/CPU)

### Memory Usage

- GPU memory: < 2GB for YOLO inference
- CPU fallback available for systems without GPU
- Configurable batch processing for optimization

### Error Handling

The system includes comprehensive error handling:
- Graceful degradation when YOLO is unavailable
- Automatic fallback to CPU if GPU fails
- Model reloading for error recovery
- Input validation and sanitization

## Integration with Duckietown RL

### Environment Wrappers

The YOLO system integrates with gym wrappers:

```python
# This will be implemented in subsequent tasks
from duckietown_utils.wrappers import YOLOObjectDetectionWrapper

env = YOLOObjectDetectionWrapper(
    env=base_env,
    model_path="yolov5s.pt",
    confidence_threshold=0.5
)
```

### Safety Features

- **Safety Distance Monitoring**: Identifies objects within configurable safety distance
- **Collision Risk Assessment**: Evaluates potential collision risks based on object position
- **Emergency Response**: Provides safety-critical flags for immediate action

## Testing

Run the comprehensive test suite:

```bash
python -m unittest tests.test_yolo_utils -v
```

### Test Coverage

- Model loading and error handling
- Inference processing and validation
- Performance metrics and statistics
- Safety-critical object detection
- Error recovery and graceful degradation

## Example Usage

See `examples/yolo_integration_example.py` for a complete demonstration:

```bash
python examples/yolo_integration_example.py
```

## Troubleshooting

### Common Issues

1. **ultralytics not installed**: Install with `pip install ultralytics`
2. **CUDA out of memory**: Reduce image resolution or use CPU device
3. **Model file not found**: Ensure model path is correct and file exists
4. **Slow inference**: Check GPU availability and model optimization

### Performance Optimization

1. **Use GPU**: Ensure CUDA is available and properly configured
2. **Model Selection**: Use smaller models (yolov5s) for faster inference
3. **Image Resolution**: Reduce input image size if needed
4. **Batch Processing**: Process multiple images together when possible

## Future Enhancements

- Support for custom trained models
- Advanced distance estimation algorithms
- Multi-object tracking capabilities
- Integration with semantic segmentation
- Real-time performance monitoring dashboard