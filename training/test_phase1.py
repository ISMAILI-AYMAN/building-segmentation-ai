#!/usr/bin/env python3
"""
Test Script for Phase 1: Model Training
Verifies all components are working correctly
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.data_loader import DataPreparation
from training.model_architectures import create_model, get_model_configs
from training.training_utils import (
    CombinedLoss, SegmentationMetrics, EarlyStopping, 
    LearningRateScheduler, TrainingLogger, save_checkpoint,
    load_checkpoint, get_device, count_parameters
)

def test_data_loader():
    """Test data loader functionality"""
    print("Testing Data Loader...")
    
    # Create temporary test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test images and masks
        test_dir = temp_path / "test_data"
        test_dir.mkdir()
        
        # Create dummy images and masks
        for i in range(5):
            # Create dummy image (random data)
            dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(test_dir / f"test_{i:03d}.JPG"), dummy_image)
            
            # Create dummy mask
            dummy_mask = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            cv2.imwrite(str(test_dir / f"test_{i:03d}_mask.png"), dummy_mask)
        
        # Test data preparation with a simpler output directory
        data_prep = DataPreparation(str(test_dir), "test_output")
        
        try:
            # Test finding data pairs
            pairs = data_prep.find_data_pairs()
            print(f"  [OK] Found {len(pairs)} image-mask pairs")
            
            # Test validation
            validation = data_prep.validate_data()
            print(f"  [OK] Validation: {validation['valid_pairs']} valid pairs")
            
            # Test train/test split
            splits = data_prep.create_train_test_split(test_size=0.2, val_size=0.2)
            print(f"  [OK] Train/Val/Test split: {len(splits['train']['image_paths'])}/{len(splits['val']['image_paths'])}/{len(splits['test']['image_paths'])}")
            
            # Test K-fold splits
            fold_splits = data_prep.create_kfold_splits(n_splits=3)
            print(f"  [OK] K-fold splits: {len(fold_splits)} folds")
            
            return True
            
        except Exception as e:
            print(f"  [FAILED] Data loader test failed: {e}")
            return False

def test_model_architectures():
    """Test model architectures"""
    print("Testing Model Architectures...")
    
    try:
        configs = get_model_configs()
        
        for config_name, config in configs.items():
            print(f"  Testing {config_name}...")
            
            # Create model
            model = create_model(config)
            
            # Test forward pass
            x = torch.randn(1, 3, 512, 512)
            with torch.no_grad():
                output = model(x)
            
            # Get model info
            info = model.get_model_info()
            param_counts = count_parameters(model)
            
            print(f"    [OK] Model: {info['model_type']}")
            print(f"    [OK] Input: {x.shape} -> Output: {output.shape}")
            print(f"    [OK] Parameters: {param_counts['total']:,}")
        
        return True
        
    except Exception as e:
        print(f"  [FAILED] Model architectures test failed: {e}")
        return False

def test_training_utils():
    """Test training utilities"""
    print("Testing Training Utilities...")
    
    try:
        # Test loss functions
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        combined_loss = CombinedLoss()
        loss = combined_loss(pred, target)
        print(f"  [OK] Combined loss: {loss.item():.4f}")
        
        # Test metrics
        metrics = SegmentationMetrics()
        metrics.update(pred, target)
        results = metrics.compute()
        print(f"  [OK] Metrics computed: {list(results.keys())}")
        
        # Test device detection
        device = get_device()
        print(f"  [OK] Device: {device}")
        
        # Test early stopping
        early_stopping = EarlyStopping(patience=3)
        model = torch.nn.Linear(10, 1)
        
        # Simulate training
        for epoch in range(5):
            val_loss = 1.0 - epoch * 0.1  # Decreasing loss
            should_stop = early_stopping(val_loss, model)
            if should_stop:
                print(f"  [OK] Early stopping triggered at epoch {epoch}")
                break
        
        return True
        
    except Exception as e:
        print(f"  [FAILED] Training utilities test failed: {e}")
        return False

def test_training_integration():
    """Test training integration"""
    print("Testing Training Integration...")
    
    try:
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, 1)
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create loss function
        criterion = CombinedLoss()
        
        # Test training step
        x = torch.randn(1, 3, 64, 64)
        y = torch.randint(0, 2, (1, 1, 64, 64)).float()
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        print(f"  [OK] Training step completed, loss: {loss.item():.4f}")
        
        # Test checkpoint saving/loading
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pth"
            
            # Save checkpoint
            save_checkpoint(model, optimizer, 1, loss.item(), {'test': 0.5}, str(checkpoint_path))
            print(f"  [OK] Checkpoint saved to {checkpoint_path}")
            
            # Load checkpoint
            new_model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 1, 1)
            )
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-4)
            
            epoch, loss_val, metrics = load_checkpoint(new_model, new_optimizer, str(checkpoint_path))
            print(f"  [OK] Checkpoint loaded: epoch {epoch}, loss {loss_val:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  [FAILED] Training integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("PHASE 1 TESTING - MODEL TRAINING COMPONENTS")
    print("=" * 60)
    
    tests = [
        ("Data Loader", test_data_loader),
        ("Model Architectures", test_model_architectures),
        ("Training Utilities", test_training_utils),
        ("Training Integration", test_training_integration)
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
        print("\n[SUCCESS] ALL TESTS PASSED! Phase 1 is ready for use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
