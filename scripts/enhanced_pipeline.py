#!/usr/bin/env python3
"""
Enhanced Building Segmentation Pipeline
Comprehensive integration of all advanced modules

This module integrates:
- GPU optimization and memory management
- Multi-scale processing
- Advanced quality enhancement
- Comprehensive evaluation
- Performance monitoring
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import time
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# Set matplotlib backend for thread safety
import matplotlib
matplotlib.use('Agg')

# Import our custom modules
from .gpu_optimizer import GPUOptimizer
from .multiscale_processor import MultiScaleProcessor
from .quality_enhancer import QualityEnhancer
from .evaluation import BuildingSegmentationEvaluator

class EnhancedBuildingSegmentationPipeline:
    """
    Enhanced building segmentation pipeline with all advanced features
    """
    
    def __init__(self, device='auto', quality_level='balanced', 
                 enable_gpu_optimization=True, enable_multiscale=True,
                 enable_quality_enhancement=True, log_dir="logs"):
        """Initialize enhanced pipeline"""
        self.log_dir = log_dir
        self._setup_logging()
        
        # Initialize components
        self.gpu_optimizer = GPUOptimizer(device, log_dir) if enable_gpu_optimization else None
        self.multiscale_processor = MultiScaleProcessor(log_dir) if enable_multiscale else None
        self.quality_enhancer = QualityEnhancer(log_dir) if enable_quality_enhancement else None
        self.evaluator = BuildingSegmentationEvaluator(log_dir)
        
        # Configuration
        self.quality_level = quality_level
        self.enable_gpu_optimization = enable_gpu_optimization
        self.enable_multiscale = enable_multiscale
        self.enable_quality_enhancement = enable_quality_enhancement
        
        # Performance tracking
        self.processing_stats = {}
        
        self.logger.info("Enhanced Building Segmentation Pipeline initialized")
        self.logger.info(f"GPU Optimization: {enable_gpu_optimization}")
        self.logger.info(f"Multi-scale Processing: {enable_multiscale}")
        self.logger.info(f"Quality Enhancement: {enable_quality_enhancement}")
        self.logger.info(f"Quality Level: {quality_level}")
        
    def _setup_logging(self):
        """Setup logging for enhanced pipeline"""
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"enhanced_pipeline_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image with GPU optimization and adaptive resolution
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        self.logger.info("Starting image preprocessing")
        
        # GPU-optimized preprocessing
        if self.gpu_optimizer:
            # Monitor memory usage
            memory_info = self.gpu_optimizer.get_memory_usage()
            self.logger.info(f"Initial memory - System: {memory_info['system_percent']:.1f}%, "
                           f"GPU: {memory_info.get('gpu_percent', 0):.1f}%")
        
        processed_image = image.copy()
        
        # Multi-scale preprocessing
        if self.multiscale_processor:
            # Adaptive scale selection
            optimal_scale = self.multiscale_processor.adaptive_scale_selection(processed_image)
            if optimal_scale != 1.0:
                height, width = processed_image.shape[:2]
                new_height = int(height * optimal_scale)
                new_width = int(width * optimal_scale)
                processed_image = cv2.resize(processed_image, (new_width, new_height), 
                                           interpolation=cv2.INTER_LANCZOS4)
        
        self.logger.info(f"Preprocessing completed: {processed_image.shape}")
        return processed_image
    
    def segment_buildings(self, image: np.ndarray) -> np.ndarray:
        """
        Perform building segmentation with multi-scale processing
        
        Args:
            image: Preprocessed image
            
        Returns:
            Building segmentation mask
        """
        self.logger.info("Starting building segmentation")
        
        def basic_segmentation(img):
            """Basic segmentation function for multi-scale processing"""
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            return binary
        
        # Multi-scale segmentation
        if self.multiscale_processor:
            self.logger.info("Applying multi-scale segmentation")
            segmentation_results = self.multiscale_processor.process_at_multiple_scales(
                image, basic_segmentation
            )
            
            # Fuse results
            mask = self.multiscale_processor.fuse_multiscale_results(segmentation_results)
        else:
            # Single-scale segmentation
            mask = basic_segmentation(image)
        
        self.logger.info(f"Segmentation completed: {np.sum(mask > 0)} pixels detected")
        return mask
    
    def enhance_quality(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply quality enhancement to segmentation results
        
        Args:
            image: Original image
            mask: Segmentation mask
            
        Returns:
            Quality-enhanced mask
        """
        if not self.quality_enhancer:
            return mask
        
        self.logger.info("Starting quality enhancement")
        
        # Apply quality enhancement pipeline
        enhanced_mask = self.quality_enhancer.enhance_segmentation_quality(
            image, mask, self.quality_level
        )
        
        self.logger.info(f"Quality enhancement completed: "
                        f"{np.sum(enhanced_mask > 0)} pixels retained")
        
        return enhanced_mask
    
    def process_single_image(self, image_path: str, output_dir: str) -> Dict:
        """
        Process a single image with full enhanced pipeline
        
        Args:
            image_path: Path to input image
            output_dir: Output directory
            
        Returns:
            Processing results dictionary
        """
        self.logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            self.logger.error(f"Could not load image: {image_path}")
            return {}
        
        start_time = time.time()
        
        # Step 1: Preprocessing
        preprocess_start = time.time()
        processed_image = self.preprocess_image(image)
        preprocess_time = time.time() - preprocess_start
        
        # Step 2: Segmentation
        segmentation_start = time.time()
        mask = self.segment_buildings(processed_image)
        segmentation_time = time.time() - segmentation_start
        
        # Step 3: Quality Enhancement
        enhancement_start = time.time()
        # Resize mask back to original image size for quality enhancement
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                            interpolation=cv2.INTER_LINEAR)
        enhanced_mask = self.enhance_quality(image, mask)
        enhancement_time = time.time() - enhancement_start
        
        # Step 4: Post-processing
        postprocess_start = time.time()
        final_mask = self.postprocess_mask(enhanced_mask)
        postprocess_time = time.time() - postprocess_start
        
        total_time = time.time() - start_time
        
        # Save results
        results = self.save_results(image, final_mask, image_path, output_dir)
        
        # Update processing stats
        self.processing_stats = {
            'image_path': image_path,
            'image_size': image.shape,
            'processing_time': total_time,
            'preprocess_time': preprocess_time,
            'segmentation_time': segmentation_time,
            'enhancement_time': enhancement_time,
            'postprocess_time': postprocess_time,
            'num_buildings': results.get('num_buildings', 0),
            'total_area': results.get('total_area', 0)
        }
        
        self.logger.info(f"Processing completed in {total_time:.2f}s")
        return results
    
    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply final post-processing to mask
        
        Args:
            mask: Input mask
            
        Returns:
            Post-processed mask
        """
        # Final morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def save_results(self, image: np.ndarray, mask: np.ndarray, 
                    image_path: str, output_dir: str) -> Dict:
        """
        Save processing results
        
        Args:
            image: Original image
            mask: Final mask
            image_path: Input image path
            output_dir: Output directory
            
        Returns:
            Results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filenames
        base_name = Path(image_path).stem
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
        summary_path = os.path.join(output_dir, f"{base_name}_summary.json")
        
        # Save mask
        cv2.imwrite(mask_path, mask)
        
        # Create overlay
        overlay = image.copy()
        overlay[mask > 0] = [0, 255, 0]  # Green overlay
        cv2.imwrite(overlay_path, overlay)
        
        # Calculate statistics
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_buildings = len(contours)
        total_area = int(np.sum(mask > 0))  # Convert to Python int
        
        # Save summary
        summary = {
            'image_path': image_path,
            'mask_path': mask_path,
            'overlay_path': overlay_path,
            'num_buildings': num_buildings,
            'total_area_pixels': total_area,
            'processing_stats': self.processing_stats
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     max_workers: int = 4) -> Dict:
        """
        Process a batch of images
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            max_workers: Maximum number of workers
            
        Returns:
            Batch processing results
        """
        self.logger.info(f"Starting batch processing: {input_dir} -> {output_dir}")
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = [f for f in Path(input_dir).rglob('*')
                      if f.suffix.lower() in image_extensions]
        
        self.logger.info(f"Found {len(image_paths)} images to process")
        
        # Process images
        results = []
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for image_path in image_paths:
                    future = executor.submit(self.process_single_image, str(image_path), output_dir)
                    futures.append(future)
                
                for future in futures:
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error processing image: {e}")
        else:
            for image_path in image_paths:
                try:
                    result = self.process_single_image(str(image_path), output_dir)
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {e}")
        
        # Generate batch summary
        batch_summary = self.generate_batch_summary(results, output_dir)
        
        self.logger.info(f"Batch processing completed: {len(results)} images processed")
        return batch_summary
    
    def generate_batch_summary(self, results: List[Dict], output_dir: str) -> Dict:
        """
        Generate comprehensive batch processing summary
        
        Args:
            results: List of processing results
            output_dir: Output directory
            
        Returns:
            Batch summary dictionary
        """
        if not results:
            return {}
        
        # Calculate statistics
        total_images = len(results)
        total_buildings = sum(r.get('num_buildings', 0) for r in results)
        total_area = sum(r.get('total_area_pixels', 0) for r in results)
        total_time = sum(r.get('processing_stats', {}).get('processing_time', 0) for r in results)
        
        avg_buildings_per_image = total_buildings / total_images if total_images > 0 else 0
        avg_area_per_image = total_area / total_images if total_images > 0 else 0
        avg_time_per_image = total_time / total_images if total_images > 0 else 0
        
        # Create summary
        summary = {
            'total_images': total_images,
            'total_buildings': total_buildings,
            'total_area_pixels': total_area,
            'total_processing_time': total_time,
            'avg_buildings_per_image': avg_buildings_per_image,
            'avg_area_per_image': avg_area_per_image,
            'avg_time_per_image': avg_time_per_image,
            'processing_date': datetime.now().isoformat(),
            'pipeline_config': {
                'quality_level': self.quality_level,
                'enable_gpu_optimization': self.enable_gpu_optimization,
                'enable_multiscale': self.enable_multiscale,
                'enable_quality_enhancement': self.enable_quality_enhancement
            }
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate visualization
        self.generate_batch_visualization(results, output_dir)
        
        return summary
    
    def generate_batch_visualization(self, results: List[Dict], output_dir: str):
        """
        Generate batch processing visualizations
        
        Args:
            results: List of processing results
            output_dir: Output directory
        """
        if not results:
            return
        
        # Extract data for plotting
        buildings_per_image = [r.get('num_buildings', 0) for r in results]
        areas_per_image = [r.get('total_area_pixels', 0) for r in results]
        times_per_image = [r.get('processing_stats', {}).get('processing_time', 0) for r in results]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Enhanced Pipeline Batch Processing Results', fontsize=16)
        
        # Buildings per image distribution
        axes[0, 0].hist(buildings_per_image, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Buildings per Image Distribution')
        axes[0, 0].set_xlabel('Number of Buildings')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Area per image distribution
        axes[0, 1].hist(areas_per_image, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Area per Image Distribution')
        axes[0, 1].set_xlabel('Area (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Processing time per image
        axes[1, 0].hist(times_per_image, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Processing Time per Image')
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Buildings vs Area scatter plot
        axes[1, 1].scatter(areas_per_image, buildings_per_image, alpha=0.6)
        axes[1, 1].set_title('Buildings vs Area')
        axes[1, 1].set_xlabel('Area (pixels)')
        axes[1, 1].set_ylabel('Number of Buildings')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "batch_processing_analysis.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Enhanced Building Segmentation Pipeline")
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("--output", "-o", default="enhanced_results", help="Output directory")
    parser.add_argument("--quality", "-q", choices=['light', 'balanced', 'aggressive'], 
                       default='balanced', help="Quality level")
    parser.add_argument("--device", "-d", default='auto', help="Device (auto, cuda, cpu)")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Number of workers")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU optimization")
    parser.add_argument("--no-multiscale", action="store_true", help="Disable multi-scale processing")
    parser.add_argument("--no-quality", action="store_true", help="Disable quality enhancement")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EnhancedBuildingSegmentationPipeline(
        device=args.device,
        quality_level=args.quality,
        enable_gpu_optimization=not args.no_gpu,
        enable_multiscale=not args.no_multiscale,
        enable_quality_enhancement=not args.no_quality
    )
    
    # Process input
    if os.path.isfile(args.input):
        # Single image
        result = pipeline.process_single_image(args.input, args.output)
        print(f"✅ Single image processing completed!")
        print(f"📊 Results: {result.get('num_buildings', 0)} buildings detected")
    else:
        # Batch processing
        summary = pipeline.process_batch(args.input, args.output, args.workers)
        print(f"✅ Batch processing completed!")
        print(f"📊 Summary:")
        print(f"  Images processed: {summary.get('total_images', 0)}")
        print(f"  Total buildings: {summary.get('total_buildings', 0)}")
        print(f"  Average buildings per image: {summary.get('avg_buildings_per_image', 0):.1f}")
        print(f"  Total processing time: {summary.get('total_processing_time', 0):.1f}s")

if __name__ == "__main__":
    main()
