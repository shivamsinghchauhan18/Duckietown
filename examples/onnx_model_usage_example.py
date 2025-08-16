#!/usr/bin/env python3
"""
ONNX Model Usage Example for Enhanced Duckietown RL

This example demonstrates how to use the exported ONNX model for inference
in production environments or cross-platform deployment.
"""

import numpy as np
import json
from pathlib import Path
import time


def load_onnx_model(model_path: str):
    """
    Load ONNX model for inference.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        ONNX inference session
        
    Note:
        Requires onnxruntime: pip install onnxruntime
    """
    try:
        import onnxruntime as ort
        
        # Create inference session
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # Use CPU by default
        )
        
        print(f"✅ ONNX model loaded successfully: {model_path}")
        print(f"   Providers: {session.get_providers()}")
        
        # Print input/output info
        print(f"   Inputs:")
        for input_meta in session.get_inputs():
            print(f"     - {input_meta.name}: {input_meta.shape} ({input_meta.type})")
        
        print(f"   Outputs:")
        for output_meta in session.get_outputs():
            print(f"     - {output_meta.name}: {output_meta.shape} ({output_meta.type})")
        
        return session
        
    except ImportError:
        print("❌ onnxruntime not available. Install with: pip install onnxruntime")
        return None
    except Exception as e:
        print(f"❌ Failed to load ONNX model: {e}")
        return None


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for model inference.
    
    Args:
        image: RGB image array of shape (H, W, 3)
        
    Returns:
        Preprocessed image array of shape (1, 3, H, W)
    """
    # Ensure image is in correct format
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    # Normalize to [0, 1] if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    # Resize to model input size (64x64)
    # Note: In practice, you'd use cv2.resize or similar
    if image.shape[:2] != (64, 64):
        print(f"⚠️  Image shape {image.shape[:2]} != (64, 64). Resize needed.")
    
    # Convert from HWC to CHW format
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = np.transpose(image, (2, 0, 1))
    
    # Add batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    return image


def postprocess_outputs(actions: np.ndarray, values: np.ndarray) -> dict:
    """
    Postprocess model outputs.
    
    Args:
        actions: Action predictions of shape (1, 2)
        values: Value predictions of shape (1, 1)
        
    Returns:
        Dictionary with processed outputs
    """
    # Extract values from batch dimension
    actions = actions[0]  # Shape: (2,)
    values = values[0, 0]  # Shape: scalar
    
    # Actions are [throttle, steering] in range [-1, 1]
    throttle = float(actions[0])
    steering = float(actions[1])
    state_value = float(values)
    
    return {
        'throttle': throttle,
        'steering': steering,
        'state_value': state_value,
        'raw_actions': actions.tolist(),
        'confidence': abs(state_value)  # Use absolute value as confidence
    }


def run_inference_example():
    """Run inference example with mock data."""
    print("🚀 ONNX Model Inference Example")
    print("=" * 50)
    
    # Model path
    model_path = "models/legendary_fusion_champions/LEGENDARY_CHAMPION_ULTIMATE.onnx"
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("   Run: python create_mock_onnx_model.py")
        return False
    
    # Load model
    session = load_onnx_model(model_path)
    if session is None:
        print("⚠️  Cannot run inference without onnxruntime")
        print("   Install with: pip install onnxruntime")
        return False
    
    # Create mock input image (RGB, 64x64)
    print(f"\n📸 Creating mock input image...")
    mock_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    print(f"   Mock image shape: {mock_image.shape}")
    
    # Preprocess image
    input_image = preprocess_image(mock_image)
    print(f"   Preprocessed shape: {input_image.shape}")
    
    # Run inference
    print(f"\n🧠 Running inference...")
    start_time = time.time()
    
    try:
        outputs = session.run(None, {'image': input_image})
        inference_time = time.time() - start_time
        
        actions, values = outputs
        print(f"   Inference time: {inference_time*1000:.2f} ms")
        print(f"   Raw outputs:")
        print(f"     Actions: {actions}")
        print(f"     Values: {values}")
        
        # Postprocess outputs
        result = postprocess_outputs(actions, values)
        print(f"\n📊 Processed Results:")
        print(f"   Throttle: {result['throttle']:.3f}")
        print(f"   Steering: {result['steering']:.3f}")
        print(f"   State Value: {result['state_value']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False


def benchmark_inference():
    """Benchmark inference performance."""
    print(f"\n⚡ Performance Benchmark")
    print("-" * 30)
    
    model_path = "models/legendary_fusion_champions/LEGENDARY_CHAMPION_ULTIMATE.onnx"
    session = load_onnx_model(model_path)
    
    if session is None:
        print("⚠️  Cannot benchmark without onnxruntime")
        return
    
    # Create batch of test images
    batch_size = 10
    test_images = np.random.rand(batch_size, 3, 64, 64).astype(np.float32)
    
    # Warmup
    for _ in range(5):
        session.run(None, {'image': test_images[:1]})
    
    # Benchmark
    times = []
    for i in range(batch_size):
        start_time = time.time()
        session.run(None, {'image': test_images[i:i+1]})
        times.append(time.time() - start_time)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    fps = 1000 / avg_time
    
    print(f"   Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   Throughput: {fps:.1f} FPS")
    print(f"   Min time: {min(times)*1000:.2f} ms")
    print(f"   Max time: {max(times)*1000:.2f} ms")


def deployment_example():
    """Example of deployment considerations."""
    print(f"\n🚀 Deployment Example")
    print("-" * 25)
    
    model_path = "models/legendary_fusion_champions/LEGENDARY_CHAMPION_ULTIMATE.onnx"
    
    # Load model metadata
    json_path = "models/legendary_fusion_champions/LEGENDARY_CHAMPION_ULTIMATE_20250816_012453.json"
    
    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        deployment_specs = config.get('deployment_specifications', {})
        
        print(f"📋 Deployment Specifications:")
        print(f"   Target Platforms: {deployment_specs.get('target_platforms', [])}")
        
        real_time_req = deployment_specs.get('real_time_requirements', {})
        print(f"   Real-time Requirements:")
        print(f"     Max inference time: {real_time_req.get('max_inference_time_ms', 'N/A')} ms")
        print(f"     Target FPS: {real_time_req.get('target_fps', 'N/A')}")
        print(f"     Memory limit: {real_time_req.get('memory_limit_mb', 'N/A')} MB")
        print(f"     Max CPU utilization: {real_time_req.get('cpu_utilization_max', 'N/A')}")
        
        safety_features = deployment_specs.get('safety_features', {})
        print(f"   Safety Features:")
        for feature, enabled in safety_features.items():
            status = "✅" if enabled else "❌"
            print(f"     {status} {feature.replace('_', ' ').title()}")
        
        print(f"\n💡 Deployment Tips:")
        print(f"   1. Use GPU acceleration for better performance")
        print(f"   2. Implement safety monitoring and fallback systems")
        print(f"   3. Test thoroughly on target hardware")
        print(f"   4. Monitor inference times in production")
        print(f"   5. Implement model versioning and rollback")
        
    except Exception as e:
        print(f"❌ Could not load deployment specifications: {e}")


def main():
    """Main function to run all examples."""
    print("Enhanced Duckietown RL - ONNX Model Usage Examples")
    print("=" * 60)
    
    # Run inference example
    success = run_inference_example()
    
    if success:
        # Run benchmark
        benchmark_inference()
        
        # Show deployment info
        deployment_example()
    
    print(f"\n" + "=" * 60)
    print("Examples completed!")
    
    if not success:
        print("\n💡 To run full examples:")
        print("   1. Install onnxruntime: pip install onnxruntime")
        print("   2. Create ONNX model: python create_mock_onnx_model.py")
        print("   3. Run this script again")


if __name__ == "__main__":
    main()