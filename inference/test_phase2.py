#!/usr/bin/env python3
"""
Test Script for Phase 2: Inference System
Verifies all components are working correctly
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
import cv2

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from inference_engine import InferenceEngine
from post_processing import PostProcessor
from inference import BuildingSegmentationInference
from training.model_architectures import get_model_configs

def test_inference_engine():
    """Test inference engine functionality"""
    print("Testing Inference Engine...")
    
    try:
        # Create a dummy model for testing
        config = get_model_configs()['unet_basic']
        
        # Create a temporary model file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_path = temp_path / "test_model.pth"
            
            # Create a simple model and save it
            from training.model_architectures import create_model
            model = create_model(config)
            torch.save(model.state_dict(), model_path)
            
            # Test inference engine
            engine = InferenceEngine(str(model_path), config)
            
            # Test model info
            info = engine.get_model_info()
            print(f"  [OK] Model: {info['model_type']}")
            print(f"  [OK] Parameters: {info['total_parameters']:,}")
            
            # Test preprocessing
            test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            tensor = engine.preprocess_image(test_image, (512, 512))
            print(f"  [OK] Preprocessing: {test_image.shape} -> {tensor.shape}")
            
            # Test prediction
            results = engine.predict(test_image, threshold=0.5, target_size=(512, 512))
            print(f"  [OK] Prediction: {list(results.keys())}")
            
            # Test batch prediction
            batch_results = engine.predict_batch([test_image, test_image])
            print(f"  [OK] Batch prediction: {len(batch_results)} results")
            
            # Test saving results
            saved_paths = engine.save_results(results, str(temp_path), "test")
            print(f"  [OK] Results saved: {list(saved_paths.keys())}")
            
            return True
            
    except Exception as e:
        print(f"  [FAILED] Inference engine test failed: {e}")
        return False

def test_post_processing():
    """Test post-processing functionality"""
    print("Testing Post-processing...")
    
    try:
        post_processor = PostProcessor()
        
        # Create test mask
        test_mask = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        test_mask = (test_mask > 127).astype(np.uint8) * 255
        
        # Test morphological cleaning
        cleaned = post_processor.morphological_cleaning(test_mask)
        print(f"  [OK] Morphological cleaning: {cleaned.shape}")
        
        # Test small object removal
        cleaned = post_processor.remove_small_objects(test_mask, min_area=50)
        print(f"  [OK] Small object removal: {cleaned.shape}")
        
        # Test boundary smoothing
        smoothed = post_processor.smooth_boundaries(test_mask)
        print(f"  [OK] Boundary smoothing: {smoothed.shape}")
        
        # Test hole filling
        filled = post_processor.fill_holes(test_mask)
        print(f"  [OK] Hole filling: {filled.shape}")
        
        # Test contour extraction
        contours = post_processor.create_contours(test_mask)
        print(f"  [OK] Contour extraction: {len(contours)} contours")
        
        # Test area metrics
        metrics = post_processor.calculate_area_metrics(test_mask)
        print(f"  [OK] Area metrics: {list(metrics.keys())}")
        
        # Test full pipeline
        processed = post_processor.apply_full_pipeline(test_mask)
        print(f"  [OK] Full pipeline: {processed.shape}")
        
        # Test overlay creation
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        overlay = post_processor.create_enhanced_overlay(test_image, processed)
        print(f"  [OK] Enhanced overlay: {overlay.shape}")
        
        return True
        
    except Exception as e:
        print(f"  [FAILED] Post-processing test failed: {e}")
        return False

def test_inference_pipeline():
    """Test complete inference pipeline"""
    print("Testing Inference Pipeline...")
    
    try:
        # Create a dummy model for testing
        config = get_model_configs()['unet_basic']
        
        # Create a temporary model file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_path = temp_path / "test_model.pth"
            
            # Create a simple model and save it
            from training.model_architectures import create_model
            model = create_model(config)
            torch.save(model.state_dict(), model_path)
            
            # Create test image
            test_image_path = temp_path / "test_image.jpg"
            test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(test_image_path), test_image)
            
            # Test inference pipeline
            pipeline = BuildingSegmentationInference(str(model_path), config)
            
            # Test single image processing
            results = pipeline.process_single_image(
                str(test_image_path),
                str(temp_path / "output"),
                threshold=0.5,
                target_size=(512, 512),
                apply_post_processing=True
            )
            print(f"  [OK] Single image processing: {list(results.keys())}")
            
            # Test batch processing
            batch_results = pipeline.process_batch(
                str(temp_path),
                str(temp_path / "batch_output"),
                threshold=0.5,
                target_size=(512, 512),
                apply_post_processing=True
            )
            print(f"  [OK] Batch processing: {len(batch_results)} results")
            
            # Test summary report
            summary_path = pipeline.create_summary_report(batch_results, str(temp_path))
            print(f"  [OK] Summary report: {summary_path}")
            
            return True
            
    except Exception as e:
        print(f"  [FAILED] Inference pipeline test failed: {e}")
        return False

def test_command_line_interface():
    """Test command-line interface"""
    print("Testing Command-line Interface...")
    
    try:
        # Test help output
        import subprocess
        result = subprocess.run([
            sys.executable, "inference/inference.py", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  [OK] Help command works")
        else:
            print("  [FAILED] Help command failed")
            return False
        
        # Test inference engine help
        result = subprocess.run([
            sys.executable, "inference/inference_engine.py", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  [OK] Inference engine help works")
        else:
            print("  [FAILED] Inference engine help failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  [FAILED] Command-line interface test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("PHASE 2 TESTING - INFERENCE SYSTEM COMPONENTS")
    print("=" * 60)
    
    tests = [
        ("Inference Engine", test_inference_engine),
        ("Post-processing", test_post_processing),
        ("Inference Pipeline", test_inference_pipeline),
        ("Command-line Interface", test_command_line_interface)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"  [FAILED] {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED! Phase 2 is ready for use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
