#!/usr/bin/env python3
"""
Post-processing Utilities for Building Segmentation
Enhances and cleans up segmentation results
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

class PostProcessor:
    """Post-processing utilities for segmentation results"""
    
    def __init__(self):
        pass
    
    def morphological_cleaning(self, mask: np.ndarray, 
                             kernel_size: int = 3, 
                             operation: str = 'both') -> np.ndarray:
        """Clean mask using morphological operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if operation == 'opening':
            # Remove small noise
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == 'closing':
            # Fill small holes
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        elif operation == 'both':
            # Both opening and closing
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        else:
            cleaned = mask
        
        return cleaned
    
    def remove_small_objects(self, mask: np.ndarray, min_area: int = 100) -> np.ndarray:
        """Remove small objects from mask"""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Create new mask
        cleaned_mask = np.zeros_like(mask)
        
        # Keep only components with area >= min_area
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_mask[labels == i] = 255
        
        return cleaned_mask
    
    def smooth_boundaries(self, mask: np.ndarray, blur_kernel: int = 3) -> np.ndarray:
        """Smooth the boundaries of the mask"""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(mask.astype(np.float32), (blur_kernel, blur_kernel), 0)
        
        # Threshold back to binary
        smoothed = (blurred > 127).astype(np.uint8) * 255
        
        return smoothed
    
    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fill holes in the mask"""
        # Create a copy of the mask
        filled = mask.copy()
        
        # Create a mask for flood filling
        h, w = mask.shape
        flood_mask = np.zeros((h+2, w+2), dtype=np.uint8)
        
        # Flood fill from the corners
        cv2.floodFill(filled, flood_mask, (0, 0), 255)
        
        # Invert to get holes
        holes = cv2.bitwise_not(filled)
        
        # Combine original mask with filled holes
        result = cv2.bitwise_or(mask, holes)
        
        return result
    
    def enhance_contrast(self, mask: np.ndarray, alpha: float = 1.5, beta: float = 0) -> np.ndarray:
        """Enhance contrast of the mask"""
        enhanced = cv2.convertScaleAbs(mask, alpha=alpha, beta=beta)
        return enhanced
    
    def create_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Extract contours from mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def draw_contours(self, image: np.ndarray, contours: List[np.ndarray], 
                     color: Tuple[int, int, int] = (0, 255, 0), 
                     thickness: int = 2) -> np.ndarray:
        """Draw contours on image"""
        result = image.copy()
        cv2.drawContours(result, contours, -1, color, thickness)
        return result
    
    def calculate_area_metrics(self, mask: np.ndarray, pixel_area: float = 1.0) -> Dict:
        """Calculate area-based metrics"""
        # Count pixels
        total_pixels = mask.shape[0] * mask.shape[1]
        building_pixels = np.sum(mask > 0)
        
        # Calculate areas
        total_area = total_pixels * pixel_area
        building_area = building_pixels * pixel_area
        building_percentage = (building_area / total_area) * 100
        
        # Find contours for additional metrics
        contours = self.create_contours(mask)
        num_buildings = len(contours)
        
        # Calculate average building size
        if num_buildings > 0:
            building_areas = [cv2.contourArea(contour) * pixel_area for contour in contours]
            avg_building_area = np.mean(building_areas)
            max_building_area = np.max(building_areas)
            min_building_area = np.min(building_areas)
        else:
            avg_building_area = max_building_area = min_building_area = 0
        
        return {
            'total_area': total_area,
            'building_area': building_area,
            'building_percentage': building_percentage,
            'num_buildings': num_buildings,
            'avg_building_area': avg_building_area,
            'max_building_area': max_building_area,
            'min_building_area': min_building_area
        }
    
    def create_enhanced_overlay(self, image: np.ndarray, mask: np.ndarray,
                              alpha: float = 0.6, 
                              color: Tuple[int, int, int] = (0, 255, 0),
                              show_contours: bool = True,
                              contour_color: Tuple[int, int, int] = (255, 0, 0),
                              contour_thickness: int = 2) -> np.ndarray:
        """Create enhanced overlay with contours"""
        # Create basic overlay
        overlay = self._create_basic_overlay(image, mask, alpha, color)
        
        # Add contours if requested
        if show_contours:
            contours = self.create_contours(mask)
            overlay = self.draw_contours(overlay, contours, contour_color, contour_thickness)
        
        return overlay
    
    def _create_basic_overlay(self, image: np.ndarray, mask: np.ndarray,
                            alpha: float = 0.6, 
                            color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Create basic overlay"""
        # Ensure image is in BGR format
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.dtype == np.uint8 else image
        else:
            image_bgr = image
        
        # Create colored mask
        colored_mask = np.zeros_like(image_bgr)
        colored_mask[mask > 0] = color
        
        # Create overlay
        overlay = cv2.addWeighted(image_bgr, 1 - alpha, colored_mask, alpha, 0)
        
        return overlay
    
    def apply_full_pipeline(self, mask: np.ndarray, 
                          morphological_kernel: int = 3,
                          min_area: int = 100,
                          smooth_kernel: int = 3,
                          fill_holes: bool = True) -> np.ndarray:
        """Apply full post-processing pipeline"""
        # Step 1: Morphological cleaning
        cleaned = self.morphological_cleaning(mask, morphological_kernel, 'both')
        
        # Step 2: Remove small objects
        cleaned = self.remove_small_objects(cleaned, min_area)
        
        # Step 3: Fill holes
        if fill_holes:
            cleaned = self.fill_holes(cleaned)
        
        # Step 4: Smooth boundaries
        cleaned = self.smooth_boundaries(cleaned, smooth_kernel)
        
        return cleaned
    
    def create_comparison_visualization(self, image: np.ndarray, 
                                      original_mask: np.ndarray,
                                      processed_mask: np.ndarray,
                                      save_path: Optional[str] = None) -> np.ndarray:
        """Create comparison visualization"""
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Original mask
        axes[0, 1].imshow(original_mask, cmap='gray')
        axes[0, 1].set_title('Original Mask')
        axes[0, 1].axis('off')
        
        # Original overlay
        original_overlay = self._create_basic_overlay(image, original_mask)
        axes[0, 2].imshow(cv2.cvtColor(original_overlay, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Original Overlay')
        axes[0, 2].axis('off')
        
        # Processed mask
        axes[1, 0].imshow(processed_mask, cmap='gray')
        axes[1, 0].set_title('Processed Mask')
        axes[1, 0].axis('off')
        
        # Processed overlay
        processed_overlay = self._create_basic_overlay(image, processed_mask)
        axes[1, 1].imshow(cv2.cvtColor(processed_overlay, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Processed Overlay')
        axes[1, 1].axis('off')
        
        # Enhanced overlay with contours
        enhanced_overlay = self.create_enhanced_overlay(image, processed_mask)
        axes[1, 2].imshow(cv2.cvtColor(enhanced_overlay, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Enhanced Overlay')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return enhanced_overlay
        else:
            plt.show()
            return enhanced_overlay
    
    def save_processed_results(self, original_mask: np.ndarray,
                             processed_mask: np.ndarray,
                             image: np.ndarray,
                             output_dir: str,
                             filename: str,
                             metrics: Optional[Dict] = None) -> Dict:
        """Save processed results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save processed mask
        processed_mask_path = output_path / f"{filename}_processed_mask.png"
        cv2.imwrite(str(processed_mask_path), processed_mask)
        
        # Create enhanced overlay
        enhanced_overlay = self.create_enhanced_overlay(image, processed_mask)
        overlay_path = output_path / f"{filename}_enhanced_overlay.png"
        cv2.imwrite(str(overlay_path), enhanced_overlay)
        
        # Create comparison visualization
        comparison_path = output_path / f"{filename}_comparison.png"
        self.create_comparison_visualization(image, original_mask, processed_mask, str(comparison_path))
        
        # Save metrics
        if metrics is None:
            metrics = self.calculate_area_metrics(processed_mask)
        
        metrics_path = output_path / f"{filename}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return {
            'processed_mask_path': str(processed_mask_path),
            'overlay_path': str(overlay_path),
            'comparison_path': str(comparison_path),
            'metrics_path': str(metrics_path),
            'metrics': metrics
        }

def main():
    """Test post-processing functions"""
    # Create a simple test
    print("Post-processing utilities ready for use!")
    print("Available functions:")
    print("- morphological_cleaning()")
    print("- remove_small_objects()")
    print("- smooth_boundaries()")
    print("- fill_holes()")
    print("- calculate_area_metrics()")
    print("- apply_full_pipeline()")
    print("- create_enhanced_overlay()")

if __name__ == "__main__":
    main()
