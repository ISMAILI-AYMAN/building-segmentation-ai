#!/usr/bin/env python3
"""
Main Inference Script for Building Segmentation
Combines inference engine and post-processing for complete pipeline
"""
import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from datetime import datetime
import logging
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from inference_engine import InferenceEngine
from post_processing import PostProcessor
from training.model_architectures import get_model_configs

class BuildingSegmentationInference:
    """Complete inference pipeline for building segmentation"""
    
    def __init__(self, model_path: str, model_config: Dict, device: Optional[str] = None):
        self.model_path = model_path
        self.model_config = model_config
        
        # Initialize components
        self.engine = InferenceEngine(model_path, model_config, device)
        self.post_processor = PostProcessor()
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("Building segmentation inference pipeline initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def process_single_image(self, image_path: str, output_dir: str,
                           threshold: float = 0.5,
                           target_size: Tuple[int, int] = (512, 512),
                           apply_post_processing: bool = True,
                           post_processing_config: Optional[Dict] = None) -> Dict:
        """Process a single image through the complete pipeline"""
        self.logger.info(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Run inference
        results = self.engine.predict(image, threshold, target_size)
        
        # Apply post-processing if requested
        if apply_post_processing:
            self.logger.info("Applying post-processing...")
            
            # Use default config if none provided
            if post_processing_config is None:
                post_processing_config = {
                    'morphological_kernel': 3,
                    'min_area': 100,
                    'smooth_kernel': 3,
                    'fill_holes': True
                }
            
            # Apply post-processing pipeline
            processed_mask = self.post_processor.apply_full_pipeline(
                results['binary_mask'],
                **post_processing_config
            )
            
            # Calculate metrics
            metrics = self.post_processor.calculate_area_metrics(processed_mask)
            
            # Save results with post-processing
            filename = Path(image_path).stem
            saved_paths = self.post_processor.save_processed_results(
                results['binary_mask'],
                processed_mask,
                image,
                output_dir,
                filename,
                metrics
            )
            
            # Add processed results to output
            results['processed_mask'] = processed_mask
            results['metrics'] = metrics
            results['saved_paths'] = saved_paths
            
        else:
            # Save basic results
            filename = Path(image_path).stem
            saved_paths = self.engine.save_results(results, output_dir, filename)
            results['saved_paths'] = saved_paths
        
        self.logger.info(f"Processing completed for {image_path}")
        return results
    
    def process_batch(self, image_dir: str, output_dir: str,
                     threshold: float = 0.5,
                     target_size: Tuple[int, int] = (512, 512),
                     apply_post_processing: bool = True,
                     post_processing_config: Optional[Dict] = None) -> List[Dict]:
        """Process a batch of images"""
        self.logger.info(f"Processing batch from directory: {image_dir}")
        
        # Find image files
        image_path = Path(image_dir)
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(image_path.glob(ext))
        
        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        results = []
        for i, image_file in enumerate(image_files):
            self.logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                result = self.process_single_image(
                    str(image_file),
                    output_dir,
                    threshold,
                    target_size,
                    apply_post_processing,
                    post_processing_config
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing {image_file}: {e}")
                continue
        
        self.logger.info(f"Batch processing completed. {len(results)} images processed successfully.")
        return results
    
    def create_summary_report(self, results: List[Dict], output_dir: str) -> str:
        """Create a summary report of batch processing"""
        if not results:
            return "No results to summarize"
        
        # Calculate overall metrics
        total_images = len(results)
        total_building_area = 0
        total_area = 0
        all_metrics = []
        
        for result in results:
            if 'metrics' in result:
                metrics = result['metrics']
                all_metrics.append(metrics)
                total_building_area += metrics['building_area']
                total_area += metrics['total_area']
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_images_processed': total_images,
            'successful_processing': len(all_metrics),
            'overall_metrics': {
                'total_building_area': total_building_area,
                'total_area': total_area,
                'overall_building_percentage': (total_building_area / total_area * 100) if total_area > 0 else 0
            },
            'average_metrics': {}
        }
        
        if all_metrics:
            # Calculate averages
            avg_metrics = {}
            for key in all_metrics[0].keys():
                if isinstance(all_metrics[0][key], (int, float)):
                    avg_metrics[key] = np.mean([m[key] for m in all_metrics])
            
            summary['average_metrics'] = avg_metrics
        
        # Save summary
        summary_path = Path(output_dir) / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary report saved to {summary_path}")
        return str(summary_path)
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return self.engine.get_model_info()

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Building Segmentation Inference Pipeline")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--model-config", help="Model configuration (JSON file or preset)")
    parser.add_argument("--input-image", help="Path to input image")
    parser.add_argument("--input-dir", help="Directory containing input images")
    parser.add_argument("--output-dir", default="inference_results", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold")
    parser.add_argument("--target-size", type=int, nargs=2, default=[512, 512], help="Target image size")
    parser.add_argument("--device", help="Device to use (cpu/cuda)")
    parser.add_argument("--no-post-processing", action='store_true', help="Skip post-processing")
    parser.add_argument("--morphological-kernel", type=int, default=3, help="Morphological kernel size")
    parser.add_argument("--min-area", type=int, default=100, help="Minimum object area")
    parser.add_argument("--smooth-kernel", type=int, default=3, help="Smoothing kernel size")
    parser.add_argument("--no-fill-holes", action='store_true', help="Skip hole filling")
    
    args = parser.parse_args()
    
    # Load model configuration
    if args.model_config and os.path.exists(args.model_config):
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)
    else:
        # Use default configuration
        model_config = get_model_configs()['resnet34_unet']
    
    # Create post-processing configuration
    post_processing_config = {
        'morphological_kernel': args.morphological_kernel,
        'min_area': args.min_area,
        'smooth_kernel': args.smooth_kernel,
        'fill_holes': not args.no_fill_holes
    }
    
    # Create inference pipeline
    pipeline = BuildingSegmentationInference(args.model_path, model_config, args.device)
    
    # Print model info
    model_info = pipeline.get_model_info()
    print(f"Model: {model_info['model_type']}")
    print(f"Parameters: {model_info['total_parameters']:,}")
    print(f"Device: {model_info['device']}")
    
    # Process input
    if args.input_image:
        # Single image
        try:
            results = pipeline.process_single_image(
                args.input_image,
                args.output_dir,
                args.threshold,
                tuple(args.target_size),
                not args.no_post_processing,
                post_processing_config
            )
            print(f"Processing completed for {args.input_image}")
            
            if 'metrics' in results:
                metrics = results['metrics']
                print(f"Building area: {metrics['building_area']:.2f}")
                print(f"Building percentage: {metrics['building_percentage']:.2f}%")
                print(f"Number of buildings: {metrics['num_buildings']}")
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return 1
        
    elif args.input_dir:
        # Directory of images
        try:
            results = pipeline.process_batch(
                args.input_dir,
                args.output_dir,
                args.threshold,
                tuple(args.target_size),
                not args.no_post_processing,
                post_processing_config
            )
            
            # Create summary report
            summary_path = pipeline.create_summary_report(results, args.output_dir)
            print(f"Batch processing completed. Summary saved to {summary_path}")
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            return 1
    
    else:
        print("Please provide either --input-image or --input-dir")
        return 1
    
    print("Inference pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
