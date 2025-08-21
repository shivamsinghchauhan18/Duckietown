#!/usr/bin/env python3
"""
Test the working champion model to ensure it can be used for deployment.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add the project root to path so we can import the model class
sys.path.append('.')

def test_model_loading_and_inference():
    """Test loading and using the champion model."""
    print("🧪 Testing champion model loading and inference...")
    
    # Test 1: Load state dict version (recommended for deployment)
    print("\n1️⃣ Testing state dict loading...")
    try:
        checkpoint = torch.load('champion_model.pth', map_location='cpu', weights_only=False)
        print(f"✅ Loaded checkpoint with keys: {list(checkpoint.keys())}")
        print(f"   Model class: {checkpoint.get('model_class', 'Unknown')}")
        print(f"   Input shape: {checkpoint.get('input_shape', 'Unknown')}")
        
        # Import the model class
        from create_working_champion_model import DuckiebotRLModel
        
        # Recreate model
        model = DuckiebotRLModel(
            input_shape=checkpoint['input_shape'],
            hidden_dim=checkpoint['hidden_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Model recreated and loaded successfully")
        
    except Exception as e:
        print(f"❌ State dict loading failed: {e}")
        return False
    
    # Test 2: Inference with different input formats
    print("\n2️⃣ Testing inference with different input formats...")
    
    try:
        # Test with HWC format (camera-like input)
        obs_hwc = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        action_hwc = model.get_action(obs_hwc)
        print(f"✅ HWC input (120,160,3): Action = {action_hwc}")
        
        # Test with CHW format (tensor-like input)
        obs_chw = np.random.rand(3, 120, 160).astype(np.float32)
        action_chw = model.get_action(obs_chw)
        print(f"✅ CHW input (3,120,160): Action = {action_chw}")
        
        # Test with normalized input
        obs_norm = np.random.rand(120, 160, 3).astype(np.float32)
        action_norm = model.get_action(obs_norm)
        print(f"✅ Normalized input: Action = {action_norm}")
        
        # Verify output format
        assert len(action_hwc) == 2, f"Expected 2 actions, got {len(action_hwc)}"
        assert -1 <= action_hwc[0] <= 1, f"Steering out of range: {action_hwc[0]}"
        assert -1 <= action_hwc[1] <= 1, f"Throttle out of range: {action_hwc[1]}"
        
        print("✅ All inference tests passed")
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False
    
    # Test 3: Complete model loading (alternative method)
    print("\n3️⃣ Testing complete model loading...")
    
    try:
        complete_model_path = Path("models/champion_model_complete.pth")
        if complete_model_path.exists():
            complete_model = torch.load(complete_model_path, map_location='cpu', weights_only=False)
            complete_model.eval()
            
            # Test inference
            test_obs = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
            action_complete = complete_model.get_action(test_obs)
            print(f"✅ Complete model inference: Action = {action_complete}")
        else:
            print("⚠️ Complete model file not found (optional)")
            
    except Exception as e:
        print(f"⚠️ Complete model loading failed (expected): {e}")
    
    # Test 4: Deployment simulation
    print("\n4️⃣ Testing deployment simulation...")
    
    try:
        # Simulate camera feed processing
        for i in range(5):
            # Simulate camera image (like from Duckiebot)
            camera_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Resize to model input size (like in deployment)
            import cv2
            resized = cv2.resize(camera_image, (160, 120))
            
            # Convert to RGB and normalize
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb_image.astype(np.float32) / 255.0
            
            # Get action
            action = model.get_action(normalized)
            
            print(f"   Frame {i+1}: Steering={action[0]:.3f}, Throttle={action[1]:.3f}")
        
        print("✅ Deployment simulation successful")
        
    except Exception as e:
        print(f"❌ Deployment simulation failed: {e}")
        return False
    
    return True


def test_model_compatibility():
    """Test compatibility with deployment scripts."""
    print("\n🔧 Testing deployment script compatibility...")
    
    try:
        # Test the loading pattern used in deployment scripts
        model_path = "champion_model.pth"
        
        # Method 1: PyTorch direct loading (used in deployment)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            print("✅ Compatible with state dict loading pattern")
        else:
            print("❌ Missing model_state_dict key")
            return False
        
        # Method 2: Check metadata
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
            print(f"✅ Model metadata available: {metadata.get('model_type', 'Unknown')}")
        else:
            print("⚠️ No metadata found")
        
        # Method 3: Test with deployment-like loading
        from create_working_champion_model import DuckiebotRLModel
        
        model = DuckiebotRLModel(
            input_shape=checkpoint['input_shape'],
            hidden_dim=checkpoint['hidden_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Test compute_action interface (like Ray RLLib)
        obs = np.random.rand(120, 160, 3).astype(np.float32)
        
        # This is the interface used in deployment
        with torch.no_grad():
            if hasattr(model, 'get_action'):
                action = model.get_action(obs)
            else:
                # Fallback to forward pass
                obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0)
                output = model(obs_tensor)
                action = output['action'].squeeze().numpy()
        
        print(f"✅ Deployment interface test: Action = {action}")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment compatibility test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Testing Champion Model for Deployment")
    print("=" * 50)
    
    # Check if model file exists
    if not Path("champion_model.pth").exists():
        print("❌ champion_model.pth not found!")
        print("   Run: python create_working_champion_model.py")
        return False
    
    # Run tests
    test1_passed = test_model_loading_and_inference()
    test2_passed = test_model_compatibility()
    
    print("\n" + "=" * 50)
    
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Champion model is ready for deployment")
        print("\n📋 Deployment checklist:")
        print("   ✅ Model file exists and is valid PyTorch format")
        print("   ✅ Model can be loaded with torch.load()")
        print("   ✅ Model supports get_action() interface")
        print("   ✅ Model handles different input formats")
        print("   ✅ Model outputs are in correct range [-1, 1]")
        print("   ✅ Model is compatible with deployment scripts")
        
        print("\n🚀 Ready to deploy with:")
        print("   python duckiebot_deployment/deploy_to_duckiebot.py \\")
        print("       --robot-name YOUR_ROBOT \\")
        print("       --robot-ip YOUR_IP \\")
        print("       --model-path champion_model.pth \\")
        print("       --config-path config/enhanced_config.yml")
        
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        print("   Check the error messages above and fix issues")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)