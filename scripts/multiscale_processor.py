#!/usr/bin/env python3
"""
Multi-Scale Processing Module
Advanced multi-scale building segmentation with pyramid processing

This module provides:
- Image pyramid processing
- Adaptive resolution selection
- Scale-aware segmentation
- Multi-scale fusion
- Resolution optimization
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import time
from scipy import ndimage
from skimage import transform
import matplotlib.pyplot as plt

# Set matplotlib backend for thread safety
import matplotlib
matplotlib.use('Agg')

class MultiScaleProcessor:
    """
    Multi-scale processing for building segmentation
    """
    
    def __init__(self, log_dir="logs"):
        """Initialize multi-scale processor"""
        self.log_dir = log_dir
        self._setup_logging()
        self.scale_levels = [0.5, 1.0, 2.0]  # Default scale levels
        self.fusion_weights = [0.2, 0.6, 0.2]  # Default fusion weights
        
    def _setup_logging(self):
        """Setup logging for multi-scale processing"""
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"multiscale_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def create_image_pyramid(self, image: np.ndarray, 
                           scale_levels: List[float] = None) -> List[Tuple[np.ndarray, float]]:
        """
        Create image pyramid at different scales
        
        Args:
            image: Input image
            scale_levels: List of scale factors
            
        Returns:
            List of (scaled_image, scale_factor) tuples
        """
        if scale_levels is None:
            scale_levels = self.scale_levels
        
        pyramid = []
        height, width = image.shape[:2]
        
        for scale in scale_levels:
            if scale == 1.0:
                scaled_image = image.copy()
            else:
                new_height = int(height * scale)
                new_width = int(width * scale)
                scaled_image = cv2.resize(image, (new_width, new_height), 
                                        interpolation=cv2.INTER_LANCZOS4)
            
            pyramid.append((scaled_image, scale))
            self.logger.info(f"Created pyramid level: {scale}x ({new_width}x{new_height})")
        
        return pyramid
    
    def adaptive_scale_selection(self, image: np.ndarray, 
                               target_size: Tuple[int, int] = (1024, 1024)) -> float:
        """
        Automatically select optimal scale based on image characteristics
        
        Args:
            image: Input image
            target_size: Target processing size
            
        Returns:
            Optimal scale factor
        """
        height, width = image.shape[:2]
        target_height, target_width = target_size
        
        # Calculate scale factors
        scale_h = target_height / height
        scale_w = target_width / width
        scale = min(scale_h, scale_w)
        
        # Ensure scale is within reasonable bounds
        scale = max(0.25, min(scale, 4.0))
        
        # Adjust based on image complexity
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Adjust scale based on edge density
        if edge_density > 0.1:  # High complexity
            scale *= 0.8
        elif edge_density < 0.02:  # Low complexity
            scale *= 1.2
        
        self.logger.info(f"Adaptive scale selection: {scale:.2f}x "
                        f"(edge density: {edge_density:.3f})")
        
        return scale
    
    def process_at_multiple_scales(self, image: np.ndarray, 
                                 processor_func, 
                                 scale_levels: List[float] = None) -> List[np.ndarray]:
        """
        Process image at multiple scales using provided processor function
        
        Args:
            image: Input image
            processor_func: Function to process each scale
            scale_levels: List of scale factors
            
        Returns:
            List of processed results at each scale
        """
        if scale_levels is None:
            scale_levels = self.scale_levels
        
        pyramid = self.create_image_pyramid(image, scale_levels)
        results = []
        
        for scaled_image, scale in pyramid:
            self.logger.info(f"Processing at scale {scale}x...")
            
            # Process at current scale
            result = processor_func(scaled_image)
            
            # Resize result back to original size
            if scale != 1.0:
                original_height, original_width = image.shape[:2]
                result = cv2.resize(result, (original_width, original_height), 
                                  interpolation=cv2.INTER_LINEAR)
            
            results.append(result)
        
        return results
    
    def fuse_multiscale_results(self, results: List[np.ndarray], 
                              weights: List[float] = None) -> np.ndarray:
        """
        Fuse results from multiple scales using weighted combination
        
        Args:
            results: List of results from different scales
            weights: Weights for each scale (must sum to 1)
            
        Returns:
            Fused result
        """
        if weights is None:
            weights = self.fusion_weights
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Ensure we have the right number of weights
        if len(weights) != len(results):
            self.logger.warning(f"Weight count ({len(weights)}) doesn't match "
                              f"result count ({len(results)}), using equal weights")
            weights = np.ones(len(results)) / len(results)
        
        # Weighted combination
        fused = np.zeros_like(results[0], dtype=np.float32)
        for result, weight in zip(results, weights):
            fused += result.astype(np.float32) * weight
        
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        
        self.logger.info(f"Fused {len(results)} scale results with weights: {weights}")
        return fused
    
    def scale_aware_segmentation(self, image: np.ndarray, 
                               processor_func,
                               confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Perform scale-aware segmentation with confidence-based scale selection
        
        Args:
            image: Input image
            processor_func: Function to process each scale
            confidence_threshold: Threshold for confidence-based selection
            
        Returns:
            Scale-aware segmentation result
        """
        # Process at multiple scales
        results = self.process_at_multiple_scales(image, processor_func)
        
        # Calculate confidence for each scale
        confidences = []
        for result in results:
            # Simple confidence based on edge strength
            edges = cv2.Canny(result, 50, 150)
            confidence = np.sum(edges > 0) / (result.shape[0] * result.shape[1])
            confidences.append(confidence)
        
        # Select best scale based on confidence
        best_scale_idx = np.argmax(confidences)
        best_result = results[best_scale_idx]
        best_confidence = confidences[best_scale_idx]
        
        self.logger.info(f"Scale-aware selection: scale {self.scale_levels[best_scale_idx]}x "
                        f"with confidence {best_confidence:.3f}")
        
        # If confidence is low, use fusion
        if best_confidence < confidence_threshold:
            self.logger.info("Low confidence, using multi-scale fusion")
            return self.fuse_multiscale_results(results)
        
        return best_result
    
    def adaptive_resolution_processing(self, image: np.ndarray,
                                    min_size: int = 512,
                                    max_size: int = 2048) -> np.ndarray:
        """
        Process image at adaptive resolution based on content
        
        Args:
            image: Input image
            min_size: Minimum processing size
            max_size: Maximum processing size
            
        Returns:
            Processed result
        """
        height, width = image.shape[:2]
        
        # Calculate optimal processing size
        if height > width:
            target_size = min(max_size, max(min_size, height))
            scale = target_size / height
        else:
            target_size = min(max_size, max(min_size, width))
            scale = target_size / width
        
        # Initialize variables
        new_height = height
        new_width = width
        
        # Resize for processing
        if scale != 1.0:
            new_height = int(height * scale)
            new_width = int(width * scale)
            resized = cv2.resize(image, (new_width, new_height), 
                               interpolation=cv2.INTER_LANCZOS4)
        else:
            resized = image.copy()
        
        self.logger.info(f"Adaptive resolution: {height}x{width} -> {new_height}x{new_width} "
                        f"(scale: {scale:.2f})")
        
        return resized
    
    def pyramid_attention_fusion(self, results: List[np.ndarray]) -> np.ndarray:
        """
        Advanced fusion using pyramid attention mechanism
        
        Args:
            results: List of results from different scales
            
        Returns:
            Attention-weighted fused result
        """
        if len(results) < 2:
            return results[0] if results else np.zeros((100, 100), dtype=np.uint8)
        
        # Create attention weights based on local contrast
        attention_weights = []
        for result in results:
            # Calculate local contrast as attention
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            attention = np.abs(laplacian)
            attention = cv2.GaussianBlur(attention, (15, 15), 0)
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
            attention_weights.append(attention)
        
        # Normalize attention weights
        attention_sum = np.zeros_like(attention_weights[0])
        for weight in attention_weights:
            attention_sum += weight
        
        attention_sum[attention_sum == 0] = 1  # Avoid division by zero
        
        # Apply attention-weighted fusion
        fused = np.zeros_like(results[0], dtype=np.float32)
        for result, attention in zip(results, attention_weights):
            normalized_attention = attention / attention_sum
            # Handle both 2D and 3D results
            if len(result.shape) == 3:
                fused += result.astype(np.float32) * normalized_attention[:, :, np.newaxis]
            else:
                fused += result.astype(np.float32) * normalized_attention
        
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        
        self.logger.info(f"Applied pyramid attention fusion to {len(results)} scales")
        return fused
    
    def generate_scale_analysis(self, image: np.ndarray, 
                              output_dir: str = "multiscale_analysis") -> Dict:
        """
        Generate comprehensive analysis of multi-scale processing
        
        Args:
            image: Input image
            output_dir: Output directory for analysis
            
        Returns:
            Analysis results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create pyramid
        pyramid = self.create_image_pyramid(image)
        
        # Analyze each scale
        analysis = {
            'scales': [],
            'sizes': [],
            'edge_densities': [],
            'processing_times': []
        }
        
        fig, axes = plt.subplots(2, len(pyramid), figsize=(4*len(pyramid), 8))
        fig.suptitle('Multi-Scale Analysis', fontsize=16)
        
        for i, (scaled_image, scale) in enumerate(pyramid):
            # Analyze current scale
            start_time = time.time()
            
            # Calculate edge density
            gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (scaled_image.shape[0] * scaled_image.shape[1])
            
            processing_time = time.time() - start_time
            
            # Store analysis
            analysis['scales'].append(scale)
            analysis['sizes'].append(scaled_image.shape[:2])
            analysis['edge_densities'].append(edge_density)
            analysis['processing_times'].append(processing_time)
            
            # Visualize
            if len(pyramid) == 1:
                ax1, ax2 = axes[0], axes[1]
            else:
                ax1, ax2 = axes[0, i], axes[1, i]
            
            ax1.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB))
            ax1.set_title(f'Scale {scale}x\n{scaled_image.shape[1]}x{scaled_image.shape[0]}')
            ax1.axis('off')
            
            ax2.imshow(edges, cmap='gray')
            ax2.set_title(f'Edge Density: {edge_density:.3f}')
            ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "scale_analysis.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save analysis report
        report_path = os.path.join(output_dir, "scale_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write("MULTI-SCALE ANALYSIS REPORT\n")
            f.write("=" * 30 + "\n\n")
            
            for i, scale in enumerate(analysis['scales']):
                f.write(f"Scale {scale}x:\n")
                f.write(f"  Size: {analysis['sizes'][i][1]}x{analysis['sizes'][i][0]}\n")
                f.write(f"  Edge Density: {analysis['edge_densities'][i]:.3f}\n")
                f.write(f"  Processing Time: {analysis['processing_times'][i]:.3f}s\n\n")
            
            # Recommendations
            best_scale_idx = np.argmax(analysis['edge_densities'])
            f.write(f"RECOMMENDATIONS:\n")
            f.write(f"  Best scale for detail: {analysis['scales'][best_scale_idx]}x\n")
            f.write(f"  Fastest processing: {analysis['scales'][np.argmin(analysis['processing_times'])]}x\n")
        
        self.logger.info(f"Scale analysis saved to: {output_dir}")
        return analysis

def main():
    """Test multi-scale processing functionality"""
    print("🚀 Multi-Scale Processing Module Test")
    print("=" * 45)
    
    # Initialize processor
    processor = MultiScaleProcessor()
    
    # Create test image
    print("\n📸 Creating test image...")
    test_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    
    # Add some structure to make it more realistic
    cv2.rectangle(test_image, (200, 200), (400, 400), (255, 255, 255), -1)
    cv2.rectangle(test_image, (600, 600), (800, 800), (128, 128, 128), -1)
    
    # Test pyramid creation
    print("\n🏗️ Testing image pyramid creation...")
    pyramid = processor.create_image_pyramid(test_image)
    print(f"Created pyramid with {len(pyramid)} levels")
    
    # Test adaptive scale selection
    print("\n⚙️ Testing adaptive scale selection...")
    optimal_scale = processor.adaptive_scale_selection(test_image)
    print(f"Optimal scale: {optimal_scale:.2f}x")
    
    # Test multi-scale processing
    print("\n🔄 Testing multi-scale processing...")
    def dummy_processor(image):
        # Dummy processor that returns edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 50, 150)
    
    results = processor.process_at_multiple_scales(test_image, dummy_processor)
    print(f"Processed at {len(results)} scales")
    
    # Test fusion
    print("\n🔗 Testing result fusion...")
    fused = processor.fuse_multiscale_results(results)
    print(f"Fused result shape: {fused.shape}")
    
    # Test scale-aware segmentation
    print("\n🎯 Testing scale-aware segmentation...")
    scale_aware_result = processor.scale_aware_segmentation(test_image, dummy_processor)
    print(f"Scale-aware result shape: {scale_aware_result.shape}")
    
    # Test adaptive resolution
    print("\n📏 Testing adaptive resolution...")
    adaptive_result = processor.adaptive_resolution_processing(test_image)
    print(f"Adaptive resolution result shape: {adaptive_result.shape}")
    
    # Test pyramid attention fusion
    print("\n🧠 Testing pyramid attention fusion...")
    attention_fused = processor.pyramid_attention_fusion(results)
    print(f"Attention fused result shape: {attention_fused.shape}")
    
    # Generate analysis
    print("\n📊 Generating scale analysis...")
    analysis = processor.generate_scale_analysis(test_image)
    
    print("\n✅ Multi-scale processing tests completed!")
    print(f"📁 Analysis saved to: multiscale_analysis/")

if __name__ == "__main__":
    main()
