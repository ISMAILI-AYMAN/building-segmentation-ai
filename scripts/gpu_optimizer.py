#!/usr/bin/env python3
"""
GPU Optimization and Memory Management Module
Advanced GPU acceleration with memory monitoring and tile-based processing

This module provides:
- GPU memory monitoring and management
- Automatic batch size adjustment
- Tile-based processing for large images
- Memory-efficient polygon processing
- Performance profiling and optimization
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import time
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# Set matplotlib backend for thread safety
import matplotlib
matplotlib.use('Agg')

class GPUOptimizer:
    """
    GPU optimization and memory management for building segmentation
    """
    
    def __init__(self, device='auto', log_dir="logs"):
        """Initialize GPU optimizer"""
        self.log_dir = log_dir
        self._setup_logging()
        self.device = self._setup_device(device)
        self.memory_history = []
        self.performance_metrics = {}
        
    def _setup_logging(self):
        """Setup logging for GPU optimization"""
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"gpu_optimization_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup and validate GPU device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                self.logger.info(f"Auto-detected GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                self.logger.info("No GPU available, using CPU")
        
        if device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'
        
        torch_device = torch.device(device)
        self.logger.info(f"Using device: {torch_device}")
        
        if torch_device.type == 'cuda':
            self._log_gpu_info()
        
        return torch_device
    
    def _log_gpu_info(self):
        """Log detailed GPU information"""
        if torch.cuda.is_available():
            self.logger.info(f"GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"PyTorch Version: {torch.__version__}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage (GPU and system)"""
        memory_info = {}
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info['system_total_gb'] = system_memory.total / 1e9
        memory_info['system_used_gb'] = system_memory.used / 1e9
        memory_info['system_available_gb'] = system_memory.available / 1e9
        memory_info['system_percent'] = system_memory.percent
        
        # GPU memory
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.memory_stats()
            memory_info['gpu_allocated_gb'] = gpu_memory['allocated_bytes.all.current'] / 1e9
            memory_info['gpu_reserved_gb'] = gpu_memory['reserved_bytes.all.current'] / 1e9
            memory_info['gpu_free_gb'] = (torch.cuda.get_device_properties(0).total_memory - 
                                        gpu_memory['reserved_bytes.all.current']) / 1e9
            memory_info['gpu_percent'] = (gpu_memory['reserved_bytes.all.current'] / 
                                        torch.cuda.get_device_properties(0).total_memory) * 100
        
        return memory_info
    
    def monitor_memory(self, interval: float = 1.0, duration: float = 60.0):
        """Monitor memory usage over time"""
        self.logger.info(f"Starting memory monitoring for {duration}s...")
        
        start_time = time.time()
        while time.time() - start_time < duration:
            memory_info = self.get_memory_usage()
            memory_info['timestamp'] = time.time()
            self.memory_history.append(memory_info)
            
            self.logger.info(f"Memory - System: {memory_info['system_percent']:.1f}%, "
                           f"GPU: {memory_info.get('gpu_percent', 0):.1f}%")
            
            time.sleep(interval)
        
        self.logger.info("Memory monitoring completed")
    
    def optimize_batch_size(self, image_size: Tuple[int, int], 
                          target_memory_gb: float = 2.0) -> int:
        """
        Automatically determine optimal batch size based on image size and memory
        
        Args:
            image_size: (height, width) of input images
            target_memory_gb: Target GPU memory usage in GB
            
        Returns:
            Optimal batch size
        """
        if self.device.type != 'cuda':
            return 4  # Default for CPU
        
        # Estimate memory per image (rough approximation)
        height, width = image_size
        channels = 3
        
        # Memory for input image
        input_memory = height * width * channels * 4  # 4 bytes per float32
        
        # Memory for intermediate tensors (rough estimate)
        intermediate_memory = input_memory * 2  # 2x for intermediate processing
        
        # Memory for model weights and gradients
        model_memory = 500 * 1024 * 1024  # 500MB estimate
        
        # Available GPU memory
        available_memory = torch.cuda.get_device_properties(0).total_memory
        target_memory = target_memory_gb * 1e9
        
        # Calculate optimal batch size
        memory_per_image = input_memory + intermediate_memory
        optimal_batch_size = max(1, int((target_memory - model_memory) / memory_per_image))
        
        # Ensure reasonable limits
        optimal_batch_size = min(optimal_batch_size, 16)
        optimal_batch_size = max(optimal_batch_size, 1)
        
        self.logger.info(f"Optimized batch size: {optimal_batch_size} "
                        f"(image: {height}x{width}, target: {target_memory_gb}GB)")
        
        return optimal_batch_size
    
    def tile_image(self, image: np.ndarray, tile_size: int = 512, 
                  overlap: int = 64) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Split large image into overlapping tiles
        
        Args:
            image: Input image
            tile_size: Size of each tile
            overlap: Overlap between tiles
            
        Returns:
            List of (tile, position) tuples
        """
        height, width = image.shape[:2]
        tiles = []
        
        # Calculate tile positions
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                # Extract tile
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)
                
                tile = image[y:y_end, x:x_end]
                
                # Pad if necessary
                if tile.shape[:2] != (tile_size, tile_size):
                    padded_tile = np.zeros((tile_size, tile_size, tile.shape[2]), dtype=tile.dtype)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile
                
                tiles.append((tile, (x, y)))
        
        self.logger.info(f"Split image {height}x{width} into {len(tiles)} tiles")
        return tiles
    
    def merge_tiles(self, tiles: List[Tuple[np.ndarray, Tuple[int, int]]], 
                   original_size: Tuple[int, int], overlap: int = 64) -> np.ndarray:
        """
        Merge processed tiles back into full image
        
        Args:
            tiles: List of (processed_tile, position) tuples
            original_size: Original image size (height, width)
            overlap: Overlap between tiles
            
        Returns:
            Merged image
        """
        height, width = original_size
        merged = np.zeros((height, width), dtype=np.uint8)
        weights = np.zeros((height, width), dtype=np.float32)
        
        tile_size = tiles[0][0].shape[0]
        
        for tile, (x, y) in tiles:
            # Create weight mask for smooth blending
            weight_mask = np.ones((tile_size, tile_size), dtype=np.float32)
            
            # Apply feathering at edges
            if overlap > 0:
                feather_size = overlap // 2
                for i in range(feather_size):
                    weight = (i + 1) / feather_size
                    weight_mask[i, :] *= weight
                    weight_mask[-i-1, :] *= weight
                    weight_mask[:, i] *= weight
                    weight_mask[:, -i-1] *= weight
            
            # Apply tile to merged image
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)
            
            tile_height = y_end - y
            tile_width = x_end - x
            
            merged[y:y_end, x:x_end] += (tile[:tile_height, :tile_width] * 
                                       weight_mask[:tile_height, :tile_width]).astype(np.uint8)
            weights[y:y_end, x:x_end] += weight_mask[:tile_height, :tile_width]
        
        # Normalize by weights
        weights[weights == 0] = 1  # Avoid division by zero
        merged = (merged / weights).astype(np.uint8)
        
        self.logger.info(f"Merged {len(tiles)} tiles into {height}x{width} image")
        return merged
    
    def optimize_polygon_processing(self, polygons: List, max_points: int = 1000) -> List:
        """
        Optimize polygon processing for memory efficiency
        
        Args:
            polygons: List of polygon objects
            max_points: Maximum points per polygon
            
        Returns:
            Optimized polygon list
        """
        optimized_polygons = []
        
        for polygon in polygons:
            # Simplify polygon if too complex
            if len(polygon.exterior.coords) > max_points:
                tolerance = polygon.length / max_points
                simplified = polygon.simplify(tolerance, preserve_topology=True)
                optimized_polygons.append(simplified)
            else:
                optimized_polygons.append(polygon)
        
        self.logger.info(f"Optimized {len(polygons)} polygons "
                        f"(reduced complexity for {len(polygons) - len(optimized_polygons)} polygons)")
        return optimized_polygons
    
    def profile_performance(self, func, *args, **kwargs) -> Dict[str, float]:
        """
        Profile function performance with memory tracking
        
        Args:
            func: Function to profile
            *args, **kwargs: Function arguments
            
        Returns:
            Performance metrics
        """
        # Record initial memory
        initial_memory = self.get_memory_usage()
        
        # Record start time
        start_time = time.time()
        
        # Run function
        result = func(*args, **kwargs)
        
        # Record end time
        end_time = time.time()
        
        # Record final memory
        final_memory = self.get_memory_usage()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_delta = (final_memory.get('gpu_allocated_gb', 0) - 
                       initial_memory.get('gpu_allocated_gb', 0))
        
        metrics = {
            'execution_time_seconds': execution_time,
            'memory_delta_gb': memory_delta,
            'peak_memory_gb': max(initial_memory.get('gpu_allocated_gb', 0),
                                final_memory.get('gpu_allocated_gb', 0))
        }
        
        self.logger.info(f"Performance: {execution_time:.2f}s, "
                        f"Memory delta: {memory_delta:+.2f}GB")
        
        return metrics
    
    def generate_performance_report(self, output_dir: str = "gpu_performance"):
        """Generate comprehensive performance report"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.memory_history:
            self.logger.warning("No memory history available for report")
            return
        
        # Convert to DataFrame for analysis
        import pandas as pd
        
        df = pd.DataFrame(self.memory_history)
        
        # Generate plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GPU Performance Analysis', fontsize=16)
        
        # System memory over time
        if 'system_percent' in df.columns:
            axes[0, 0].plot(df['timestamp'], df['system_percent'])
            axes[0, 0].set_title('System Memory Usage')
            axes[0, 0].set_ylabel('Memory Usage (%)')
            axes[0, 0].grid(True)
        
        # GPU memory over time
        if 'gpu_percent' in df.columns:
            axes[0, 1].plot(df['timestamp'], df['gpu_percent'])
            axes[0, 1].set_title('GPU Memory Usage')
            axes[0, 1].set_ylabel('Memory Usage (%)')
            axes[0, 1].grid(True)
        
        # Memory comparison
        if 'system_used_gb' in df.columns and 'gpu_allocated_gb' in df.columns:
            axes[1, 0].plot(df['timestamp'], df['system_used_gb'], label='System')
            axes[1, 0].plot(df['timestamp'], df['gpu_allocated_gb'], label='GPU')
            axes[1, 0].set_title('Memory Usage Comparison')
            axes[1, 0].set_ylabel('Memory (GB)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Performance metrics summary
        if self.performance_metrics:
            metrics_df = pd.DataFrame([self.performance_metrics])
            axes[1, 1].bar(metrics_df.columns, metrics_df.iloc[0])
            axes[1, 1].set_title('Performance Metrics')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_analysis.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed report
        report_path = os.path.join(output_dir, "performance_report.txt")
        with open(report_path, 'w') as f:
            f.write("GPU PERFORMANCE REPORT\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Monitoring Duration: {len(self.memory_history)} samples\n\n")
            
            if 'system_percent' in df.columns:
                f.write(f"System Memory:\n")
                f.write(f"  Average: {df['system_percent'].mean():.1f}%\n")
                f.write(f"  Peak: {df['system_percent'].max():.1f}%\n\n")
            
            if 'gpu_percent' in df.columns:
                f.write(f"GPU Memory:\n")
                f.write(f"  Average: {df['gpu_percent'].mean():.1f}%\n")
                f.write(f"  Peak: {df['gpu_percent'].max():.1f}%\n\n")
            
            if self.performance_metrics:
                f.write("Performance Metrics:\n")
                for metric, value in self.performance_metrics.items():
                    f.write(f"  {metric}: {value:.3f}\n")
        
        self.logger.info(f"Performance report saved to: {output_dir}")

def main():
    """Test GPU optimization functionality"""
    print("🚀 GPU Optimization Module Test")
    print("=" * 40)
    
    # Initialize optimizer
    optimizer = GPUOptimizer(device='auto')
    
    # Test memory monitoring
    print("\n📊 Testing memory monitoring...")
    optimizer.monitor_memory(duration=5.0)  # Monitor for 5 seconds
    
    # Test batch size optimization
    print("\n⚙️ Testing batch size optimization...")
    optimal_batch = optimizer.optimize_batch_size((1024, 1024), target_memory_gb=2.0)
    print(f"Optimal batch size: {optimal_batch}")
    
    # Test tile processing
    print("\n🧩 Testing tile processing...")
    test_image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    tiles = optimizer.tile_image(test_image, tile_size=512, overlap=64)
    print(f"Created {len(tiles)} tiles")
    
    # Test polygon optimization
    print("\n🔷 Testing polygon optimization...")
    from shapely.geometry import Polygon
    test_polygons = [Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]) for _ in range(10)]
    optimized = optimizer.optimize_polygon_processing(test_polygons, max_points=50)
    print(f"Optimized {len(test_polygons)} polygons")
    
    # Generate performance report
    print("\n📈 Generating performance report...")
    optimizer.generate_performance_report()
    
    print("\n✅ GPU optimization tests completed!")

if __name__ == "__main__":
    main()
