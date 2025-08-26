#!/usr/bin/env python3
"""
Data Loader for Building Segmentation Training
Handles data preparation, train/test split, and K-fold validation
"""
import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import argparse
from datetime import datetime
import logging
import json
from sklearn.model_selection import train_test_split, KFold
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for non-interactive use
import matplotlib
matplotlib.use('Agg')

class BuildingSegmentationDataset(Dataset):
    """Custom dataset for building segmentation"""
    
    def __init__(self, image_paths: List[str], mask_paths: List[str], 
                 transform: Optional[A.Compose] = None, mode: str = 'train'):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Read image and mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float() / 255.0
        
        return {
            'image': image,
            'mask': mask,
            'image_path': image_path,
            'mask_path': mask_path
        }

class DataPreparation:
    """Data preparation and management for building segmentation training"""
    
    def __init__(self, data_dir: str, output_dir: str = "training_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # If we can't create the directory, use a fallback
            self.output_dir = Path("training_data")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Data storage
        self.image_paths = []
        self.mask_paths = []
        self.data_info = {}
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "data_preparation.log"
        
        # Create a custom logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Create file handler
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            # If file logging fails, continue with console only
            pass
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def find_data_pairs(self, image_pattern: str = "*.JPG", mask_pattern: str = "*_mask.png"):
        """Find matching image-mask pairs"""
        self.logger.info(f"Searching for data pairs in {self.data_dir}")
        
        # Find all image files
        image_files = list(self.data_dir.rglob(image_pattern))
        image_files.extend(list(self.data_dir.rglob(image_pattern.lower())))
        
        self.logger.info(f"Found {len(image_files)} image files")
        
        # Find corresponding mask files
        valid_pairs = []
        for img_path in image_files:
            # Try to find corresponding mask
            img_name = img_path.stem
            mask_path = img_path.parent / f"{img_name}_mask.png"
            
            if mask_path.exists():
                valid_pairs.append((str(img_path), str(mask_path)))
            else:
                self.logger.warning(f"No mask found for {img_path}")
        
        self.logger.info(f"Found {len(valid_pairs)} valid image-mask pairs")
        
        # Store paths
        self.image_paths = [pair[0] for pair in valid_pairs]
        self.mask_paths = [pair[1] for pair in valid_pairs]
        
        return valid_pairs
    
    def validate_data(self) -> Dict:
        """Validate data quality and consistency"""
        self.logger.info("Validating data quality...")
        
        validation_results = {
            'total_pairs': len(self.image_paths),
            'valid_pairs': 0,
            'invalid_pairs': 0,
            'errors': [],
            'image_sizes': [],
            'mask_sizes': []
        }
        
        for i, (img_path, mask_path) in enumerate(zip(self.image_paths, self.mask_paths)):
            try:
                # Load and validate image
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError(f"Could not load image: {img_path}")
                
                # Load and validate mask
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Could not load mask: {mask_path}")
                
                # Check dimensions
                if image.shape[:2] != mask.shape[:2]:
                    raise ValueError(f"Size mismatch: image {image.shape[:2]} vs mask {mask.shape[:2]}")
                
                # Store sizes
                validation_results['image_sizes'].append(image.shape[:2])
                validation_results['mask_sizes'].append(mask.shape[:2])
                validation_results['valid_pairs'] += 1
                
            except Exception as e:
                validation_results['errors'].append(f"Pair {i}: {str(e)}")
                validation_results['invalid_pairs'] += 1
        
        # Calculate statistics
        if validation_results['image_sizes']:
            sizes = np.array(validation_results['image_sizes'])
            validation_results['avg_width'] = float(np.mean(sizes[:, 1]))
            validation_results['avg_height'] = float(np.mean(sizes[:, 0]))
            validation_results['size_std'] = float(np.std(sizes))
        
        self.logger.info(f"Validation complete: {validation_results['valid_pairs']} valid pairs")
        return validation_results
    
    def create_train_test_split(self, test_size: float = 0.15, val_size: float = 0.15, 
                               random_state: int = 42) -> Dict:
        """Create train/validation/test split"""
        self.logger.info(f"Creating train/validation/test split (test={test_size}, val={val_size})")
        
        # First split: train+val vs test
        train_val_indices, test_indices = train_test_split(
            range(len(self.image_paths)), 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Second split: train vs val
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size/(1-test_size),  # Adjust for the remaining data
            random_state=random_state
        )
        
        # Create splits
        splits = {
            'train': {
                'image_paths': [self.image_paths[i] for i in train_indices],
                'mask_paths': [self.mask_paths[i] for i in train_indices],
                'indices': train_indices
            },
            'val': {
                'image_paths': [self.image_paths[i] for i in val_indices],
                'mask_paths': [self.mask_paths[i] for i in val_indices],
                'indices': val_indices
            },
            'test': {
                'image_paths': [self.image_paths[i] for i in test_indices],
                'mask_paths': [self.mask_paths[i] for i in test_indices],
                'indices': test_indices
            }
        }
        
        # Log split information
        for split_name, split_data in splits.items():
            self.logger.info(f"{split_name.capitalize()}: {len(split_data['image_paths'])} samples")
        
        return splits
    
    def create_kfold_splits(self, n_splits: int = 5, random_state: int = 42) -> List[Dict]:
        """Create K-fold cross-validation splits"""
        self.logger.info(f"Creating {n_splits}-fold cross-validation splits")
        
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_splits = []
        
        for fold, (train_indices, val_indices) in enumerate(kfold.split(self.image_paths)):
            fold_data = {
                'fold': fold + 1,
                'train': {
                    'image_paths': [self.image_paths[i] for i in train_indices],
                    'mask_paths': [self.mask_paths[i] for i in train_indices],
                    'indices': train_indices.tolist()
                },
                'val': {
                    'image_paths': [self.image_paths[i] for i in val_indices],
                    'mask_paths': [self.mask_paths[i] for i in val_indices],
                    'indices': val_indices.tolist()
                }
            }
            fold_splits.append(fold_data)
            
            self.logger.info(f"Fold {fold + 1}: Train={len(fold_data['train']['image_paths'])}, "
                           f"Val={len(fold_data['val']['image_paths'])}")
        
        return fold_splits
    
    def get_transforms(self, image_size: Tuple[int, int] = (512, 512)) -> Dict:
        """Get data augmentation transforms"""
        train_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        val_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        return {
            'train': train_transform,
            'val': val_transform,
            'test': val_transform
        }
    
    def create_data_loaders(self, splits: Dict, batch_size: int = 8, 
                           image_size: Tuple[int, int] = (512, 512)) -> Dict:
        """Create PyTorch data loaders"""
        self.logger.info(f"Creating data loaders with batch_size={batch_size}, image_size={image_size}")
        
        transforms = self.get_transforms(image_size)
        data_loaders = {}
        
        for split_name, split_data in splits.items():
            if split_name in ['train', 'val', 'test']:
                dataset = BuildingSegmentationDataset(
                    image_paths=split_data['image_paths'],
                    mask_paths=split_data['mask_paths'],
                    transform=transforms[split_name],
                    mode=split_name
                )
                
                shuffle = (split_name == 'train')
                data_loaders[split_name] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=4,
                    pin_memory=True
                )
                
                self.logger.info(f"Created {split_name} loader: {len(dataset)} samples")
        
        return data_loaders
    
    def save_data_info(self, splits: Dict, validation_results: Dict, 
                      fold_splits: Optional[List[Dict]] = None):
        """Save data information to JSON"""
        data_info = {
            'timestamp': datetime.now().isoformat(),
            'data_dir': str(self.data_dir),
            'output_dir': str(self.output_dir),
            'validation_results': validation_results,
            'splits': {
                split_name: {
                    'count': len(split_data['image_paths']),
                    'indices': split_data['indices']
                }
                for split_name, split_data in splits.items()
            }
        }
        
        if fold_splits:
            data_info['kfold_splits'] = [
                {
                    'fold': fold_data['fold'],
                    'train_count': len(fold_data['train']['image_paths']),
                    'val_count': len(fold_data['val']['image_paths'])
                }
                for fold_data in fold_splits
            ]
        
        # Save to JSON
        info_path = self.output_dir / "data_info.json"
        with open(info_path, 'w') as f:
            json.dump(data_info, f, indent=2)
        
        self.logger.info(f"Data information saved to {info_path}")
    
    def visualize_samples(self, splits: Dict, num_samples: int = 4):
        """Visualize sample images and masks"""
        self.logger.info("Creating sample visualizations...")
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        
        for i in range(num_samples):
            # Get sample from each split
            for j, (split_name, split_data) in enumerate(splits.items()):
                if i < len(split_data['image_paths']):
                    # Load image and mask
                    image = cv2.imread(split_data['image_paths'][i])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mask = cv2.imread(split_data['mask_paths'][i], cv2.IMREAD_GRAYSCALE)
                    
                    # Plot
                    axes[i, j].imshow(image)
                    axes[i, j].set_title(f'{split_name.capitalize()} - Image')
                    axes[i, j].axis('off')
                    
                    # Overlay mask
                    overlay = image.copy()
                    overlay[mask > 0] = [255, 0, 0]  # Red overlay for buildings
                    axes[i, j+1].imshow(overlay)
                    axes[i, j+1].set_title(f'{split_name.capitalize()} - Overlay')
                    axes[i, j+1].axis('off')
                    
                    # Mask only
                    axes[i, j+2].imshow(mask, cmap='gray')
                    axes[i, j+2].set_title(f'{split_name.capitalize()} - Mask')
                    axes[i, j+2].axis('off')
        
        plt.tight_layout()
        viz_path = self.output_dir / "sample_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Sample visualization saved to {viz_path}")
    
    def prepare_data(self, test_size: float = 0.15, val_size: float = 0.15,
                    n_folds: int = 5, batch_size: int = 8, 
                    image_size: Tuple[int, int] = (512, 512)) -> Dict:
        """Complete data preparation pipeline"""
        self.logger.info("Starting complete data preparation pipeline...")
        
        # Step 1: Find data pairs
        self.find_data_pairs()
        
        # Step 2: Validate data
        validation_results = self.validate_data()
        
        # Step 3: Create train/test split
        splits = self.create_train_test_split(test_size, val_size)
        
        # Step 4: Create K-fold splits
        fold_splits = self.create_kfold_splits(n_folds)
        
        # Step 5: Create data loaders
        data_loaders = self.create_data_loaders(splits, batch_size, image_size)
        
        # Step 6: Save data information
        self.save_data_info(splits, validation_results, fold_splits)
        
        # Step 7: Visualize samples
        self.visualize_samples(splits)
        
        self.logger.info("Data preparation completed successfully!")
        
        return {
            'splits': splits,
            'fold_splits': fold_splits,
            'data_loaders': data_loaders,
            'validation_results': validation_results
        }

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Data Preparation for Building Segmentation")
    parser.add_argument("data_dir", help="Directory containing images and masks")
    parser.add_argument("--output-dir", default="training_data", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test set size")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation set size")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of K-fold splits")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512], help="Image size (width height)")
    
    args = parser.parse_args()
    
    # Create data preparation instance
    data_prep = DataPreparation(args.data_dir, args.output_dir)
    
    try:
        # Run complete data preparation
        results = data_prep.prepare_data(
            test_size=args.test_size,
            val_size=args.val_size,
            n_folds=args.n_folds,
            batch_size=args.batch_size,
            image_size=tuple(args.image_size)
        )
        
        print("\n" + "="*60)
        print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total pairs: {results['validation_results']['total_pairs']}")
        print(f"Valid pairs: {results['validation_results']['valid_pairs']}")
        print(f"Train samples: {len(results['splits']['train']['image_paths'])}")
        print(f"Val samples: {len(results['splits']['val']['image_paths'])}")
        print(f"Test samples: {len(results['splits']['test']['image_paths'])}")
        print(f"K-fold splits: {len(results['fold_splits'])}")
        print(f"Output directory: {args.output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during data preparation: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
