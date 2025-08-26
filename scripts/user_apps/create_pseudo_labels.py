#!/usr/bin/env python3
"""
Pseudo-Label Generation for Building Segmentation
Uses our enhanced pipeline to generate high-quality masks from unlabeled aerial images
Then prepares them for training a deep learning model
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime
import json
import shutil
from typing import List, Dict, Tuple, Optional
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
import torch

# Add scripts to path
sys.path.append(str(Path(__file__).parent / 'scripts'))
from enhanced_pipeline import EnhancedBuildingSegmentationPipeline

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pseudo_label_generation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def process_single_image(args):
    """Process a single image to generate pseudo-label"""
    image_path, output_dir, pipeline, target_size, quality_level = args
    
    try:
        # Load and preprocess image
        image = cv2.imread(str(image_path))
        if image is None:
            return None, f"Failed to load image: {image_path}"
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image_resized = cv2.resize(image, target_size)
        
        # Save temporary image for pipeline processing
        temp_image_path = f"temp_{image_path.stem}.png"
        cv2.imwrite(temp_image_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
        
        # Create temporary output directory
        temp_output_dir = f"temp_output_{image_path.stem}"
        os.makedirs(temp_output_dir, exist_ok=True)
        
        try:
            # Process with enhanced pipeline
            result = pipeline.process_single_image(temp_image_path, temp_output_dir)
            
            # Extract mask from saved results
            if result and 'mask_path' in result:
                mask_path = result['mask_path']
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                else:
                    mask = None
            else:
                mask = None
            
            if mask is not None:
                # Ensure mask is binary and correct size
                if len(mask.shape) == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                
                # Resize mask if needed
                if mask.shape[:2] != target_size[::-1]:
                    mask = cv2.resize(mask, target_size)
                
                # Binarize mask
                mask = (mask > 127).astype(np.uint8) * 255
                
                # Save results
                image_name = f"{image_path.stem}.png"
                mask_name = f"{image_path.stem}_mask.png"
                
                image_save_path = output_dir / 'images' / image_name
                mask_save_path = output_dir / 'masks' / mask_name
                
                cv2.imwrite(str(image_save_path), cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(mask_save_path), mask)
                
                return {
                    'image': image_name,
                    'mask': mask_name,
                    'original': str(image_path),
                    'mask_area_ratio': np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
                }, None
            else:
                return None, f"No mask generated for {image_path}"
                
        finally:
            # Clean up temporary files
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            if os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir, ignore_errors=True)
                
    except Exception as e:
        return None, f"Error processing {image_path}: {str(e)}"

def create_pseudo_label_dataset(
    input_dir: str,
    output_dir: str,
    num_samples: int = 100,
    target_size: Tuple[int, int] = (512, 512),
    quality_level: str = 'balanced',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_workers: int = 4,
    min_mask_area_ratio: float = 0.01,  # Minimum building area ratio
    max_mask_area_ratio: float = 0.8,   # Maximum building area ratio
    seed: int = 42
):
    """
    Create pseudo-label dataset using enhanced pipeline
    
    Args:
        input_dir: Directory containing aerial images
        output_dir: Output directory for pseudo-label dataset
        num_samples: Number of samples to process
        target_size: Target image size (width, height)
        quality_level: Quality level for pipeline ('fast', 'balanced', 'high')
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        max_workers: Maximum number of parallel workers
        min_mask_area_ratio: Minimum building area ratio to keep
        max_mask_area_ratio: Maximum building area ratio to keep
        seed: Random seed for reproducibility
    """
    logger = setup_logging()
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Initialize enhanced pipeline with GPU optimization
    logger.info("Initializing enhanced pipeline for pseudo-label generation...")
    pipeline = EnhancedBuildingSegmentationPipeline(
        device=device,
        enable_gpu_optimization=True,
        enable_multiscale=True,
        enable_quality_enhancement=True,
        quality_level=quality_level
    )
    
    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    test_dir = output_path / 'test'
    
    for split_dir in [train_dir, val_dir, test_dir]:
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Get input images
    input_path = Path(input_dir)
    image_files = list(input_path.rglob('*.JPG'))
    image_files.extend(list(input_path.rglob('*.jpg')))
    image_files.extend(list(input_path.rglob('*.png')))
    
    if not image_files:
        raise ValueError(f"No images found in {input_dir}")
    
    # Randomly sample images
    if len(image_files) > num_samples:
        image_files = random.sample(image_files, num_samples)
    
    logger.info(f"Processing {len(image_files)} images for pseudo-label generation...")
    
    # Process images in parallel with progress bar
    successful_results = []
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare arguments for each image
        args_list = [
            (image_file, train_dir, pipeline, target_size, quality_level)
            for image_file in image_files
        ]
        
        # Submit all tasks
        future_to_image = {
            executor.submit(process_single_image, args): args[0]
            for args in args_list
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(image_files), desc="Generating pseudo-labels", unit="image") as pbar:
            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    result, error = future.result()
                    if result is not None:
                        # Check mask area ratio
                        mask_area_ratio = result['mask_area_ratio']
                        if min_mask_area_ratio <= mask_area_ratio <= max_mask_area_ratio:
                            successful_results.append(result)
                            pbar.set_postfix({
                                'success': len(successful_results),
                                'failed': failed_count,
                                'area_ratio': f"{mask_area_ratio:.3f}"
                            })
                            logger.info(f"[OK] Generated pseudo-label for {image_path.name} (area ratio: {mask_area_ratio:.3f})")
                        else:
                            pbar.set_postfix({
                                'success': len(successful_results),
                                'failed': failed_count,
                                'skipped': 'area_ratio'
                            })
                            logger.info(f"[SKIP] Skipped {image_path.name} (area ratio: {mask_area_ratio:.3f} outside range)")
                    else:
                        failed_count += 1
                        pbar.set_postfix({
                            'success': len(successful_results),
                            'failed': failed_count,
                            'error': 'mask_gen'
                        })
                        logger.error(f"[FAILED] Failed to process {image_path.name}: {error}")
                except Exception as e:
                    failed_count += 1
                    pbar.set_postfix({
                        'success': len(successful_results),
                        'failed': failed_count,
                        'error': 'exception'
                    })
                    logger.error(f"[FAILED] Exception processing {image_path.name}: {str(e)}")
                
                pbar.update(1)
    
    logger.info(f"Pseudo-label generation completed:")
    logger.info(f"  - Successful: {len(successful_results)}")
    logger.info(f"  - Failed: {failed_count}")
    
    if not successful_results:
        raise ValueError("No successful pseudo-labels generated!")
    
    # Split data
    random.shuffle(successful_results)
    train_size = int(len(successful_results) * train_ratio)
    val_size = int(len(successful_results) * val_ratio)
    
    train_data = successful_results[:train_size]
    val_data = successful_results[train_size:train_size + val_size]
    test_data = successful_results[train_size + val_size:]
    
    logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Move files to appropriate splits
    splits = [
        ('train', train_data, train_dir),
        ('val', val_data, val_dir),
        ('test', test_data, test_dir)
    ]
    
    dataset_info = {
        'creation_date': datetime.now().isoformat(),
        'total_samples': len(successful_results),
        'splits': {},
        'processing_config': {
            'target_size': target_size,
            'quality_level': quality_level,
            'min_mask_area_ratio': min_mask_area_ratio,
            'max_mask_area_ratio': max_mask_area_ratio,
            'seed': seed
        },
        'statistics': {
            'average_mask_area_ratio': np.mean([r['mask_area_ratio'] for r in successful_results]),
            'mask_area_ratio_std': np.std([r['mask_area_ratio'] for r in successful_results])
        }
    }
    
    for split_name, data, split_dir in splits:
        logger.info(f"Organizing {split_name} split ({len(data)} samples)...")
        
        split_info = {
            'count': len(data),
            'files': []
        }
        
        # Organize files with progress bar
        with tqdm(total=len(data), desc=f"Organizing {split_name} split", unit="file") as pbar:
            for item in data:
                # Move files from train_dir to appropriate split directory
                src_image = train_dir / 'images' / item['image']
                src_mask = train_dir / 'masks' / item['mask']
                
                dst_image = split_dir / 'images' / item['image']
                dst_mask = split_dir / 'masks' / item['mask']
                
                if src_image.exists() and src_mask.exists():
                    shutil.move(str(src_image), str(dst_image))
                    shutil.move(str(src_mask), str(dst_mask))
                    
                    split_info['files'].append({
                        'image': item['image'],
                        'mask': item['mask'],
                        'original': item['original'],
                        'mask_area_ratio': item['mask_area_ratio']
                    })
                
                pbar.update(1)
        
        dataset_info['splits'][split_name] = split_info
    
    # Save dataset info
    with open(output_path / 'pseudo_label_dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    logger.info(f"Pseudo-label dataset created successfully in {output_dir}")
    logger.info(f"Dataset statistics:")
    logger.info(f"  - Average mask area ratio: {dataset_info['statistics']['average_mask_area_ratio']:.3f}")
    logger.info(f"  - Mask area ratio std: {dataset_info['statistics']['mask_area_ratio_std']:.3f}")
    
    return dataset_info

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create pseudo-label dataset from unlabeled aerial images')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing aerial images')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for pseudo-label dataset')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to process')
    parser.add_argument('--target-size', type=int, nargs=2, default=[512, 512],
                        help='Target image size (width height)')
    parser.add_argument('--quality-level', type=str, default='balanced',
                        choices=['fast', 'balanced', 'high'],
                        help='Quality level for pipeline')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--max-workers', type=int, default=4,
                        help='Maximum number of parallel workers')
    parser.add_argument('--min-mask-area-ratio', type=float, default=0.01,
                        help='Minimum building area ratio to keep')
    parser.add_argument('--max-mask-area-ratio', type=float, default=0.8,
                        help='Maximum building area ratio to keep')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    create_pseudo_label_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        target_size=tuple(args.target_size),
        quality_level=args.quality_level,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_workers=args.max_workers,
        min_mask_area_ratio=args.min_mask_area_ratio,
        max_mask_area_ratio=args.max_mask_area_ratio,
        seed=args.seed
    )

if __name__ == '__main__':
    main()
