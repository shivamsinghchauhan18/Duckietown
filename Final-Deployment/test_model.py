#!/usr/bin/env python3
"""
Test script to verify the champion model works locally
"""

import torch
import numpy as np
import cv2
import time

def test_model_loading():
    """Test if the model can be loaded"""
    print("🧪 Testing model loading...")
    
    try:
        # Load the model
        model_path = "champion_model.pth"
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"   Keys: {list(checkpoint.keys())}")
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                total_params = sum(p.numel() for p in state_dict.values())
                print(f"   Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_model_inference():
    """Test model inference with dummy data"""
    print("\n🧪 Testing model inference...")
    
    try:
        # Create simplified model class
        class SimpleDuckiebotModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 32, 8, 4, 2),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(32, 64, 4, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 64, 3, 1, 1),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((4, 5))
                )
                self.fc = torch.nn.Sequential(
                    torch.nn.Linear(1280, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 256),
                    torch.nn.ReLU(),
                )
                self.policy = torch.nn.Sequential(
                    torch.nn.Linear(256, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 2),
                    torch.nn.Tanh()
                )
            
            def forward(self, x):
                features = self.cnn(x)
                features = features.view(features.size(0), -1)
                hidden = self.fc(features)
                action = self.policy(hidden)
                return action
            
            def get_action(self, obs):
                output = self.forward(obs)
                return output.squeeze().cpu().numpy()
        
        # Load model
        checkpoint = torch.load("champion_model.pth", map_location='cpu', weights_only=False)
        model = SimpleDuckiebotModel()
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        
        # Create dummy input (like camera image)
        dummy_image = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
        
        # Preprocess (same as in deployment)
        normalized = dummy_image.astype(np.float32) / 255.0
        tensor_input = torch.FloatTensor(normalized).permute(2, 0, 1).unsqueeze(0)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            action = model.get_action(tensor_input)
        inference_time = time.time() - start_time
        
        print(f"✅ Inference successful")
        print(f"   Input shape: {tensor_input.shape}")
        print(f"   Output action: [{action[0]:.3f}, {action[1]:.3f}]")
        print(f"   Inference time: {inference_time*1000:.1f}ms")
        print(f"   Expected range: [-1, 1] for both steering and throttle")
        
        # Validate output
        if len(action) == 2 and -1 <= action[0] <= 1 and -1 <= action[1] <= 1:
            print(f"✅ Output format is correct")
            return True
        else:
            print(f"❌ Output format is incorrect")
            return False
            
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_processing():
    """Test image processing pipeline"""
    print("\n🧪 Testing image processing...")
    
    try:
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"   Original image shape: {test_image.shape}")
        
        # Resize (same as in deployment)
        resized = cv2.resize(test_image, (160, 120))
        print(f"   Resized shape: {resized.shape}")
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        print(f"   RGB shape: {rgb_image.shape}")
        
        # Normalize
        normalized = rgb_image.astype(np.float32) / 255.0
        print(f"   Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        
        # Convert to tensor
        tensor = torch.FloatTensor(normalized).permute(2, 0, 1).unsqueeze(0)
        print(f"   Tensor shape: {tensor.shape}")
        
        print(f"✅ Image processing pipeline works")
        return True
        
    except Exception as e:
        print(f"❌ Image processing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 CHAMPION MODEL TESTING")
    print("=" * 40)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Image Processing", test_image_processing),
        ("Model Inference", test_model_inference),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n📊 TEST RESULTS")
    print("=" * 40)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your model is ready for deployment!")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please fix the issues before deployment.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)