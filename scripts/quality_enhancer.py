#!/usr/bin/env python3
"""
Advanced Quality Enhancement Module
Sophisticated filtering and enhancement for building segmentation

This module provides:
- Advanced shadow removal and correction
- Vegetation and water body detection
- Building shape validation and refinement
- Texture-based filtering
- Quality assessment and improvement
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
from scipy import ndimage
from skimage import filters, feature, morphology, segmentation
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Set matplotlib backend for thread safety
import matplotlib
matplotlib.use('Agg')

class QualityEnhancer:
    """
    Advanced quality enhancement for building segmentation
    """
    
    def __init__(self, log_dir="logs"):
        """Initialize quality enhancer"""
        self.log_dir = log_dir
        self._setup_logging()
        self.enhancement_stats = {}
        
    def _setup_logging(self):
        """Setup logging for quality enhancement"""
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"quality_enhancement_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def detect_and_remove_shadows(self, image: np.ndarray, 
                                shadow_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and remove shadows from image
        
        Args:
            image: Input image
            shadow_threshold: Threshold for shadow detection
            
        Returns:
            Tuple of (shadow_mask, corrected_image)
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate local mean and standard deviation
        kernel_size = 15
        local_mean = cv2.blur(l_channel.astype(np.float32), (kernel_size, kernel_size))
        local_std = cv2.blur((l_channel.astype(np.float32) - local_mean) ** 2, (kernel_size, kernel_size)) ** 0.5
        
        # Detect shadows based on low lightness and low local contrast
        shadow_mask = np.zeros_like(l_channel, dtype=np.uint8)
        
        # Low lightness condition
        low_lightness = l_channel < (np.mean(l_channel) - shadow_threshold * np.std(l_channel))
        
        # Low local contrast condition
        low_contrast = local_std < (np.mean(local_std) - shadow_threshold * np.std(local_std))
        
        # Combine conditions
        shadow_mask[low_lightness & low_contrast] = 255
        
        # Morphological operations to clean up shadow mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        # Correct shadows by adjusting lightness
        corrected_image = image.copy()
        shadow_correction_factor = 1.5
        
        for i in range(3):  # Apply to all channels
            corrected_image[:, :, i] = np.clip(
                corrected_image[:, :, i].astype(np.float32) * 
                (1 + shadow_mask.astype(np.float32) / 255 * (shadow_correction_factor - 1)),
                0, 255
            ).astype(np.uint8)
        
        shadow_area = np.sum(shadow_mask > 0) / (shadow_mask.shape[0] * shadow_mask.shape[1])
        self.logger.info(f"Shadow detection: {shadow_area:.3f} of image area")
        
        return shadow_mask, corrected_image
    
    def detect_vegetation(self, image: np.ndarray, 
                         ndvi_threshold: float = 0.1) -> np.ndarray:
        """
        Detect vegetation using NDVI-like approach
        
        Args:
            image: Input image
            ndvi_threshold: Threshold for vegetation detection
            
        Returns:
            Vegetation mask
        """
        # Convert to different color spaces for vegetation detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Extract channels
        h, s, v = cv2.split(hsv)
        l, a, b = cv2.split(lab)
        
        # Create vegetation mask using multiple criteria
        vegetation_mask = np.zeros_like(h, dtype=np.uint8)
        
        # 1. Green channel dominance (BGR format)
        green = image[:, :, 1]
        blue = image[:, :, 0]
        red = image[:, :, 2]
        
        green_dominance = (green > red) & (green > blue)
        
        # 2. Saturation threshold
        high_saturation = s > np.mean(s) + 0.5 * np.std(s)
        
        # 3. Hue in green range
        green_hue = (h > 35) & (h < 85)
        
        # 4. Local texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        texture_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        high_texture = texture_variance > np.mean(texture_variance)
        
        # Combine criteria
        vegetation_mask[green_dominance & high_saturation & green_hue] = 255
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel)
        
        vegetation_area = np.sum(vegetation_mask > 0) / (vegetation_mask.shape[0] * vegetation_mask.shape[1])
        self.logger.info(f"Vegetation detection: {vegetation_area:.3f} of image area")
        
        return vegetation_mask
    
    def detect_water_bodies(self, image: np.ndarray, 
                          water_threshold: float = 0.2) -> np.ndarray:
        """
        Detect water bodies using color and texture analysis
        
        Args:
            image: Input image
            water_threshold: Threshold for water detection
            
        Returns:
            Water mask
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        h, s, v = cv2.split(hsv)
        l, a, b = cv2.split(lab)
        
        # Create water mask using multiple criteria
        water_mask = np.zeros_like(h, dtype=np.uint8)
        
        # 1. Low saturation (water is typically desaturated)
        low_saturation = s < np.mean(s) - 0.5 * np.std(s)
        
        # 2. Blue channel dominance
        blue = image[:, :, 0]
        green = image[:, :, 1]
        red = image[:, :, 2]
        blue_dominance = (blue > green) & (blue > red)
        
        # 3. Low texture (water is typically smooth)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 5), np.float32) / 25
        blurred = cv2.filter2D(gray, -1, kernel)
        texture_diff = np.abs(gray.astype(np.float32) - blurred)
        low_texture = texture_diff < np.mean(texture_diff) + 0.5 * np.std(texture_diff)
        
        # 4. Hue in blue range
        blue_hue = (h > 100) & (h < 130)
        
        # Combine criteria
        water_mask[low_saturation & blue_dominance & low_texture & blue_hue] = 255
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        
        water_area = np.sum(water_mask > 0) / (water_mask.shape[0] * water_mask.shape[1])
        self.logger.info(f"Water detection: {water_area:.3f} of image area")
        
        return water_mask
    
    def validate_building_shapes(self, mask: np.ndarray, 
                               min_area: int = 100,
                               min_aspect_ratio: float = 0.1,
                               max_aspect_ratio: float = 10.0) -> np.ndarray:
        """
        Validate and filter building shapes based on geometric properties
        
        Args:
            mask: Binary building mask
            min_area: Minimum building area
            min_aspect_ratio: Minimum aspect ratio
            max_aspect_ratio: Maximum aspect ratio
            
        Returns:
            Validated building mask
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create output mask
        validated_mask = np.zeros_like(mask)
        
        valid_buildings = 0
        total_buildings = len(contours)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < min_area:
                continue
            
            # Calculate bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / (min(w, h) + 1e-8)
            
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue
            
            # Calculate compactness
            perimeter = cv2.arcLength(contour, True)
            compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-8)
            
            # Filter based on compactness (buildings should be reasonably compact)
            if compactness < 0.1:  # Too irregular
                continue
            
            # Fill the contour
            cv2.fillPoly(validated_mask, [contour], 255)
            valid_buildings += 1
        
        self.logger.info(f"Building validation: {valid_buildings}/{total_buildings} buildings passed")
        
        return validated_mask
    
    def apply_texture_filtering(self, image: np.ndarray, mask: np.ndarray,
                              texture_threshold: float = 0.5) -> np.ndarray:
        """
        Apply texture-based filtering to improve building detection
        
        Args:
            image: Input image
            mask: Binary building mask
            texture_threshold: Threshold for texture filtering
            
        Returns:
            Texture-filtered mask
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate local texture measures
        # 1. Local variance
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # 2. Local entropy
        def calculate_entropy(image, window_size=5):
            entropy = np.zeros_like(image, dtype=np.float32)
            half_window = window_size // 2
            
            for i in range(half_window, image.shape[0] - half_window):
                for j in range(half_window, image.shape[1] - half_window):
                    window = image[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
                    hist, _ = np.histogram(window, bins=256, range=(0, 256))
                    hist = hist[hist > 0]
                    entropy[i, j] = -np.sum(hist * np.log2(hist / np.sum(hist)))
            
            return entropy
        
        local_entropy = calculate_entropy(gray)
        
        # Normalize texture measures
        local_variance_norm = (local_variance - local_variance.min()) / (local_variance.max() - local_variance.min() + 1e-8)
        local_entropy_norm = (local_entropy - local_entropy.min()) / (local_entropy.max() - local_entropy.min() + 1e-8)
        
        # Combine texture measures
        texture_measure = (local_variance_norm + local_entropy_norm) / 2
        
        # Apply texture filtering
        texture_filtered = np.zeros_like(mask)
        texture_filtered[(mask > 0) & (texture_measure > texture_threshold)] = 255
        
        filtered_area = np.sum(texture_filtered > 0) / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0
        self.logger.info(f"Texture filtering: {filtered_area:.3f} of buildings retained")
        
        return texture_filtered
    
    def cluster_based_refinement(self, mask: np.ndarray, 
                               eps: float = 50.0,
                               min_samples: int = 5) -> np.ndarray:
        """
        Apply clustering-based refinement to group nearby buildings
        
        Args:
            mask: Binary building mask
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            
        Returns:
            Clustering-refined mask
        """
        # Find building centroids
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 2:
            return mask
        
        # Extract centroids
        centroids = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append([cx, cy])
        
        if len(centroids) < 2:
            return mask
        
        # Apply DBSCAN clustering
        centroids = np.array(centroids)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
        
        # Create refined mask
        refined_mask = np.zeros_like(mask)
        
        # Group buildings by cluster
        unique_labels = set(clustering.labels_)
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            # Find buildings in this cluster
            cluster_indices = np.where(clustering.labels_ == label)[0]
            
            # Combine buildings in cluster
            for idx in cluster_indices:
                cv2.fillPoly(refined_mask, [contours[idx]], 255)
        
        # Apply morphological operations to smooth cluster boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        
        clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
        self.logger.info(f"Clustering refinement: {clusters_found} clusters found from {len(contours)} buildings")
        
        return refined_mask
    
    def enhance_segmentation_quality(self, image: np.ndarray, mask: np.ndarray,
                                   enhancement_level: str = 'balanced') -> np.ndarray:
        """
        Apply comprehensive quality enhancement pipeline
        
        Args:
            image: Input image
            mask: Binary building mask
            enhancement_level: 'light', 'balanced', or 'aggressive'
            
        Returns:
            Quality-enhanced mask
        """
        self.logger.info(f"Starting quality enhancement (level: {enhancement_level})")
        
        # Configure parameters based on enhancement level
        if enhancement_level == 'light':
            shadow_threshold = 0.4
            texture_threshold = 0.3
            min_area = 50
            eps = 30.0
        elif enhancement_level == 'aggressive':
            shadow_threshold = 0.2
            texture_threshold = 0.7
            min_area = 200
            eps = 100.0
        else:  # balanced
            shadow_threshold = 0.3
            texture_threshold = 0.5
            min_area = 100
            eps = 50.0
        
        enhanced_mask = mask.copy()
        
        # Step 1: Shadow removal
        self.logger.info("Step 1: Shadow removal")
        shadow_mask, corrected_image = self.detect_and_remove_shadows(image, shadow_threshold)
        
        # Step 2: Vegetation detection and removal
        self.logger.info("Step 2: Vegetation detection")
        vegetation_mask = self.detect_vegetation(corrected_image)
        enhanced_mask[vegetation_mask > 0] = 0
        
        # Step 3: Water body detection and removal
        self.logger.info("Step 3: Water body detection")
        water_mask = self.detect_water_bodies(corrected_image)
        enhanced_mask[water_mask > 0] = 0
        
        # Step 4: Building shape validation
        self.logger.info("Step 4: Building shape validation")
        enhanced_mask = self.validate_building_shapes(enhanced_mask, min_area)
        
        # Step 5: Texture filtering
        self.logger.info("Step 5: Texture filtering")
        enhanced_mask = self.apply_texture_filtering(corrected_image, enhanced_mask, texture_threshold)
        
        # Step 6: Clustering-based refinement
        self.logger.info("Step 6: Clustering refinement")
        enhanced_mask = self.cluster_based_refinement(enhanced_mask, eps)
        
        # Calculate improvement statistics
        original_area = np.sum(mask > 0)
        enhanced_area = np.sum(enhanced_mask > 0)
        improvement_ratio = enhanced_area / original_area if original_area > 0 else 0
        
        self.enhancement_stats = {
            'original_area': original_area,
            'enhanced_area': enhanced_area,
            'improvement_ratio': improvement_ratio,
            'shadow_area': np.sum(shadow_mask > 0),
            'vegetation_area': np.sum(vegetation_mask > 0),
            'water_area': np.sum(water_mask > 0)
        }
        
        self.logger.info(f"Quality enhancement completed: {improvement_ratio:.3f} area ratio")
        
        return enhanced_mask
    
    def generate_quality_report(self, image: np.ndarray, original_mask: np.ndarray, 
                              enhanced_mask: np.ndarray,
                              output_dir: str = "quality_enhancement") -> Dict:
        """
        Generate comprehensive quality enhancement report
        
        Args:
            image: Input image
            original_mask: Original building mask
            enhanced_mask: Enhanced building mask
            output_dir: Output directory
            
        Returns:
            Quality report dictionary
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Quality Enhancement Analysis', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Original mask
        axes[0, 1].imshow(original_mask, cmap='gray')
        axes[0, 1].set_title(f'Original Mask\n({np.sum(original_mask > 0)} pixels)')
        axes[0, 1].axis('off')
        
        # Enhanced mask
        axes[0, 2].imshow(enhanced_mask, cmap='gray')
        axes[0, 2].set_title(f'Enhanced Mask\n({np.sum(enhanced_mask > 0)} pixels)')
        axes[0, 2].axis('off')
        
        # Overlay comparison
        overlay_original = image.copy()
        overlay_original[original_mask > 0] = [0, 255, 0]  # Green for original
        axes[1, 0].imshow(cv2.cvtColor(overlay_original, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Original Overlay')
        axes[1, 0].axis('off')
        
        overlay_enhanced = image.copy()
        overlay_enhanced[enhanced_mask > 0] = [0, 0, 255]  # Red for enhanced
        axes[1, 1].imshow(cv2.cvtColor(overlay_enhanced, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Enhanced Overlay')
        axes[1, 1].axis('off')
        
        # Difference
        difference = np.zeros_like(image)
        difference[original_mask > 0] = [0, 255, 0]  # Green for removed
        difference[enhanced_mask > 0] = [0, 0, 255]  # Red for added
        axes[1, 2].imshow(cv2.cvtColor(difference, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Difference (Green=Removed, Red=Added)')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "quality_enhancement_analysis.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed report
        report_path = os.path.join(output_dir, "quality_enhancement_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("QUALITY ENHANCEMENT REPORT\n")
            f.write("=" * 30 + "\n\n")
            
            f.write(f"Enhancement Statistics:\n")
            f.write(f"  Original area: {self.enhancement_stats['original_area']:,} pixels\n")
            f.write(f"  Enhanced area: {self.enhancement_stats['enhanced_area']:,} pixels\n")
            f.write(f"  Improvement ratio: {self.enhancement_stats['improvement_ratio']:.3f}\n")
            f.write(f"  Shadow area detected: {self.enhancement_stats['shadow_area']:,} pixels\n")
            f.write(f"  Vegetation area detected: {self.enhancement_stats['vegetation_area']:,} pixels\n")
            f.write(f"  Water area detected: {self.enhancement_stats['water_area']:,} pixels\n\n")
            
            f.write("Quality Assessment:\n")
            if self.enhancement_stats['improvement_ratio'] > 0.8:
                f.write("  ✅ Good quality preservation\n")
            elif self.enhancement_stats['improvement_ratio'] > 0.5:
                f.write("  ⚠️ Moderate quality reduction\n")
            else:
                f.write("  ❌ Significant quality reduction\n")
        
        self.logger.info(f"Quality report saved to: {output_dir}")
        return self.enhancement_stats

def main():
    """Test quality enhancement functionality"""
    print("🚀 Quality Enhancement Module Test")
    print("=" * 40)
    
    # Initialize enhancer
    enhancer = QualityEnhancer()
    
    # Create test image
    print("\n📸 Creating test image...")
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Add some realistic features
    # Buildings (rectangular shapes)
    cv2.rectangle(test_image, (100, 100), (200, 200), (128, 128, 128), -1)
    cv2.rectangle(test_image, (300, 150), (400, 250), (100, 100, 100), -1)
    
    # Vegetation (green areas)
    cv2.rectangle(test_image, (50, 300), (150, 400), (0, 255, 0), -1)
    
    # Water (blue areas)
    cv2.rectangle(test_image, (350, 350), (450, 450), (255, 0, 0), -1)
    
    # Shadows (dark areas)
    cv2.rectangle(test_image, (200, 50), (250, 100), (50, 50, 50), -1)
    
    # Create initial mask
    print("\n🎭 Creating initial mask...")
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    initial_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    
    # Test individual enhancement steps
    print("\n🔍 Testing shadow detection...")
    shadow_mask, corrected_image = enhancer.detect_and_remove_shadows(test_image)
    print(f"Shadow mask created: {np.sum(shadow_mask > 0)} pixels")
    
    print("\n🌿 Testing vegetation detection...")
    vegetation_mask = enhancer.detect_vegetation(test_image)
    print(f"Vegetation mask created: {np.sum(vegetation_mask > 0)} pixels")
    
    print("\n💧 Testing water detection...")
    water_mask = enhancer.detect_water_bodies(test_image)
    print(f"Water mask created: {np.sum(water_mask > 0)} pixels")
    
    print("\n🏗️ Testing building validation...")
    validated_mask = enhancer.validate_building_shapes(initial_mask)
    print(f"Validated mask: {np.sum(validated_mask > 0)} pixels")
    
    print("\n🧠 Testing texture filtering...")
    texture_filtered = enhancer.apply_texture_filtering(test_image, initial_mask)
    print(f"Texture filtered: {np.sum(texture_filtered > 0)} pixels")
    
    print("\n🔗 Testing clustering refinement...")
    clustered_mask = enhancer.cluster_based_refinement(initial_mask)
    print(f"Clustered mask: {np.sum(clustered_mask > 0)} pixels")
    
    # Test full enhancement pipeline
    print("\n🚀 Testing full enhancement pipeline...")
    enhanced_mask = enhancer.enhance_segmentation_quality(test_image, initial_mask, 'balanced')
    print(f"Enhanced mask: {np.sum(enhanced_mask > 0)} pixels")
    
    # Generate quality report
    print("\n📊 Generating quality report...")
    stats = enhancer.generate_quality_report(test_image, initial_mask, enhanced_mask)
    
    print("\n✅ Quality enhancement tests completed!")
    print(f"📁 Report saved to: quality_enhancement/")
    print(f"📈 Enhancement ratio: {stats['improvement_ratio']:.3f}")

if __name__ == "__main__":
    main()
