#!/usr/bin/env python3
"""
Image Data Analysis Module
Analyzes image data of full dataset and individual datasets
Generates histograms, statistics, and visualizations
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import argparse
from datetime import datetime
import logging
from collections import defaultdict
import json
from tqdm import tqdm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for non-interactive use
import matplotlib
matplotlib.use('Agg')

class ImageDataAnalyzer:
    """Comprehensive image data analysis for building segmentation datasets"""
    
    def __init__(self, output_dir: str = "analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Analysis results storage
        self.dataset_stats = {}
        self.individual_stats = {}
        self.histogram_data = {}
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / "analysis.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def analyze_single_image(self, image_path: str) -> Dict:
        """Analyze a single image and return comprehensive statistics"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            # Convert to different color spaces
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Basic statistics
            stats = {
                'filename': Path(image_path).name,
                'path': image_path,
                'dimensions': image.shape,
                'width': image.shape[1],
                'height': image.shape[0],
                'aspect_ratio': image.shape[1] / image.shape[0],
                'total_pixels': image.shape[0] * image.shape[1],
                'file_size_mb': os.path.getsize(image_path) / (1024 * 1024),
            }
            
            # Color statistics for each channel
            for i, channel_name in enumerate(['blue', 'green', 'red']):
                channel = image[:, :, i]
                stats[f'{channel_name}_mean'] = float(np.mean(channel))
                stats[f'{channel_name}_std'] = float(np.std(channel))
                stats[f'{channel_name}_min'] = float(np.min(channel))
                stats[f'{channel_name}_max'] = float(np.max(channel))
                stats[f'{channel_name}_median'] = float(np.median(channel))
                
            # Grayscale statistics
            stats['gray_mean'] = float(np.mean(gray))
            stats['gray_std'] = float(np.std(gray))
            stats['gray_min'] = float(np.min(gray))
            stats['gray_max'] = float(np.max(gray))
            stats['gray_median'] = float(np.median(gray))
            
            # HSV statistics
            for i, channel_name in enumerate(['hue', 'saturation', 'value']):
                channel = hsv[:, :, i]
                stats[f'{channel_name}_mean'] = float(np.mean(channel))
                stats[f'{channel_name}_std'] = float(np.std(channel))
                
            # LAB statistics
            for i, channel_name in enumerate(['l', 'a', 'b']):
                channel = lab[:, :, i]
                stats[f'{channel_name}_mean'] = float(np.mean(channel))
                stats[f'{channel_name}_std'] = float(np.std(channel))
                
            # Texture analysis
            stats.update(self._analyze_texture(gray))
            
            # Edge analysis
            stats.update(self._analyze_edges(gray))
            
            # Brightness and contrast analysis
            stats.update(self._analyze_brightness_contrast(gray))
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing {image_path}: {str(e)}")
            return {'filename': Path(image_path).name, 'error': str(e)}
    
    def _analyze_texture(self, gray_image: np.ndarray) -> Dict:
        """Analyze texture characteristics of the image"""
        # Local Binary Pattern approximation
        from skimage.feature import local_binary_pattern
        
        # Calculate LBP
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
        
        # LBP histogram
        n_bins = n_points + 2
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
        
        # Texture statistics
        texture_stats = {
            'lbp_uniformity': float(np.sum(lbp_hist ** 2)),
            'lbp_entropy': float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))),
            'lbp_contrast': float(np.std(lbp)),
        }
        
        return texture_stats
    
    def _analyze_edges(self, gray_image: np.ndarray) -> Dict:
        """Analyze edge characteristics of the image"""
        # Sobel edge detection
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Canny edge detection
        canny_edges = cv2.Canny(gray_image, 50, 150)
        
        edge_stats = {
            'sobel_mean': float(np.mean(sobel_magnitude)),
            'sobel_std': float(np.std(sobel_magnitude)),
            'sobel_max': float(np.max(sobel_magnitude)),
            'canny_edge_density': float(np.sum(canny_edges > 0) / canny_edges.size),
            'edge_complexity': float(np.sum(sobel_magnitude > np.mean(sobel_magnitude) + np.std(sobel_magnitude)) / sobel_magnitude.size),
        }
        
        return edge_stats
    
    def _analyze_brightness_contrast(self, gray_image: np.ndarray) -> Dict:
        """Analyze brightness and contrast characteristics"""
        # Brightness analysis
        brightness = np.mean(gray_image)
        
        # Contrast analysis using standard deviation
        contrast = np.std(gray_image)
        
        # Histogram analysis
        hist, _ = np.histogram(gray_image, bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()
        
        # Calculate percentiles
        percentiles = np.percentile(gray_image, [10, 25, 50, 75, 90])
        
        brightness_contrast_stats = {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'brightness_percentile_10': float(percentiles[0]),
            'brightness_percentile_25': float(percentiles[1]),
            'brightness_percentile_50': float(percentiles[2]),
            'brightness_percentile_75': float(percentiles[3]),
            'brightness_percentile_90': float(percentiles[4]),
            'histogram_entropy': float(-np.sum(hist * np.log2(hist + 1e-10))),
            'histogram_uniformity': float(np.sum(hist ** 2)),
        }
        
        return brightness_contrast_stats
    
    def analyze_dataset(self, dataset_path: str, file_patterns: List[str] = None) -> Dict:
        """Analyze entire dataset and generate comprehensive statistics"""
        if file_patterns is None:
            file_patterns = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
            
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
            
        self.logger.info(f"Starting analysis of dataset: {dataset_path}")
        
        # Find all image files
        image_files = []
        for pattern in file_patterns:
            image_files.extend(dataset_path.rglob(pattern))
            image_files.extend(dataset_path.rglob(pattern.upper()))
            
        if not image_files:
            raise ValueError(f"No image files found in {dataset_path}")
            
        self.logger.info(f"Found {len(image_files)} image files")
        
        # Analyze each image
        individual_stats = []
        for image_file in tqdm(image_files, desc="Analyzing images"):
            stats = self.analyze_single_image(str(image_file))
            individual_stats.append(stats)
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(individual_stats)
        
        # Remove rows with errors
        error_rows = df[df['error'].notna()]
        if not error_rows.empty:
            self.logger.warning(f"Found {len(error_rows)} images with errors")
            df = df[df['error'].isna()].drop('error', axis=1)
            
        # Calculate dataset-level statistics
        dataset_stats = self._calculate_dataset_statistics(df)
        
        # Store results
        self.dataset_stats = dataset_stats
        self.individual_stats = df.to_dict('records')
        
        self.logger.info(f"Analysis completed. Processed {len(df)} images successfully.")
        
        return {
            'dataset_stats': dataset_stats,
            'individual_stats': self.individual_stats,
            'total_images': len(df),
            'failed_images': len(error_rows)
        }
    
    def _calculate_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive dataset-level statistics"""
        stats = {
            'total_images': len(df),
            'analysis_timestamp': datetime.now().isoformat(),
        }
        
        # Basic statistics for numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            stats[f'{col}_mean'] = float(df[col].mean())
            stats[f'{col}_std'] = float(df[col].std())
            stats[f'{col}_min'] = float(df[col].min())
            stats[f'{col}_max'] = float(df[col].max())
            stats[f'{col}_median'] = float(df[col].median())
            
        # Dimension analysis
        if 'width' in df.columns and 'height' in df.columns:
            stats['avg_width'] = float(df['width'].mean())
            stats['avg_height'] = float(df['height'].mean())
            stats['avg_aspect_ratio'] = float(df['aspect_ratio'].mean())
            stats['resolution_variation'] = float(df['width'].std() / df['width'].mean())
            
        # File size analysis
        if 'file_size_mb' in df.columns:
            stats['avg_file_size_mb'] = float(df['file_size_mb'].mean())
            stats['total_size_gb'] = float(df['file_size_mb'].sum() / 1024)
            
        # Color distribution analysis
        color_channels = ['blue', 'green', 'red', 'gray']
        for channel in color_channels:
            mean_col = f'{channel}_mean'
            if mean_col in df.columns:
                stats[f'{channel}_distribution_mean'] = float(df[mean_col].mean())
                stats[f'{channel}_distribution_std'] = float(df[mean_col].std())
                
        return stats
    
    def generate_histograms(self, save_individual: bool = True) -> Dict:
        """Generate comprehensive histograms for the dataset"""
        if not self.individual_stats:
            raise ValueError("No analysis data available. Run analyze_dataset() first.")
            
        df = pd.DataFrame(self.individual_stats)
        
        # Create histogram directory
        hist_dir = self.output_dir / "histograms"
        hist_dir.mkdir(exist_ok=True)
        
        histogram_paths = {}
        
        # 1. Color channel histograms
        color_channels = ['blue', 'green', 'red', 'gray']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, channel in enumerate(color_channels):
            mean_col = f'{channel}_mean'
            if mean_col in df.columns:
                axes[i].hist(df[mean_col], bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{channel.capitalize()} Channel Distribution')
                axes[i].set_xlabel('Mean Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
                
        plt.tight_layout()
        color_hist_path = hist_dir / "color_channel_distributions.png"
        plt.savefig(color_hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        histogram_paths['color_channels'] = str(color_hist_path)
        
        # 2. Image dimensions histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if 'width' in df.columns:
            ax1.hist(df['width'], bins=30, alpha=0.7, edgecolor='black', color='blue')
            ax1.set_title('Image Width Distribution')
            ax1.set_xlabel('Width (pixels)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
            
        if 'height' in df.columns:
            ax2.hist(df['height'], bins=30, alpha=0.7, edgecolor='black', color='red')
            ax2.set_title('Image Height Distribution')
            ax2.set_xlabel('Height (pixels)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
        plt.tight_layout()
        dim_hist_path = hist_dir / "image_dimensions.png"
        plt.savefig(dim_hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        histogram_paths['dimensions'] = str(dim_hist_path)
        
        # 3. File size histogram
        if 'file_size_mb' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df['file_size_mb'], bins=30, alpha=0.7, edgecolor='black', color='green')
            plt.title('File Size Distribution')
            plt.xlabel('File Size (MB)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            size_hist_path = hist_dir / "file_sizes.png"
            plt.savefig(size_hist_path, dpi=300, bbox_inches='tight')
            plt.close()
            histogram_paths['file_sizes'] = str(size_hist_path)
            
        # 4. Texture and edge analysis histograms
        texture_metrics = ['lbp_uniformity', 'lbp_entropy', 'lbp_contrast', 
                          'sobel_mean', 'canny_edge_density', 'edge_complexity']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(texture_metrics):
            if metric in df.columns:
                axes[i].hist(df[metric], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
                
        plt.tight_layout()
        texture_hist_path = hist_dir / "texture_analysis.png"
        plt.savefig(texture_hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        histogram_paths['texture_analysis'] = str(texture_hist_path)
        
        # 5. Brightness and contrast histograms
        brightness_metrics = ['brightness', 'contrast', 'histogram_entropy', 'histogram_uniformity']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(brightness_metrics):
            if metric in df.columns:
                axes[i].hist(df[metric], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
                
        plt.tight_layout()
        brightness_hist_path = hist_dir / "brightness_contrast.png"
        plt.savefig(brightness_hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        histogram_paths['brightness_contrast'] = str(brightness_hist_path)
        
        # 6. Correlation matrix heatmap
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            correlation_matrix = df[numerical_cols].corr()
            
            plt.figure(figsize=(20, 16))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8})
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            corr_heatmap_path = hist_dir / "correlation_matrix.png"
            plt.savefig(corr_heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            histogram_paths['correlation_matrix'] = str(corr_heatmap_path)
            
        # Save individual image histograms if requested
        if save_individual:
            individual_hist_dir = hist_dir / "individual_images"
            individual_hist_dir.mkdir(exist_ok=True)
            
            # Sample a few images for individual histograms
            sample_size = min(10, len(df))
            sample_indices = np.random.choice(len(df), sample_size, replace=False)
            
            for idx in sample_indices:
                img_stats = df.iloc[idx]
                filename = img_stats['filename']
                
                # Create individual image histogram
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Color channel means
                color_means = [img_stats.get(f'{ch}_mean', 0) for ch in ['blue', 'green', 'red']]
                axes[0, 0].bar(['Blue', 'Green', 'Red'], color_means, color=['blue', 'green', 'red'])
                axes[0, 0].set_title('Color Channel Means')
                axes[0, 0].set_ylabel('Mean Value')
                
                # Grayscale histogram
                if 'gray_mean' in img_stats:
                    axes[0, 1].axvline(img_stats['gray_mean'], color='red', linestyle='--', label='Mean')
                    axes[0, 1].axvline(img_stats['gray_median'], color='green', linestyle='--', label='Median')
                    axes[0, 1].set_title('Grayscale Statistics')
                    axes[0, 1].legend()
                    
                # Texture metrics
                texture_vals = [img_stats.get(metric, 0) for metric in ['lbp_uniformity', 'lbp_entropy', 'lbp_contrast']]
                axes[1, 0].bar(['Uniformity', 'Entropy', 'Contrast'], texture_vals)
                axes[1, 0].set_title('Texture Analysis')
                
                # Edge metrics
                edge_vals = [img_stats.get(metric, 0) for metric in ['sobel_mean', 'canny_edge_density', 'edge_complexity']]
                axes[1, 1].bar(['Sobel Mean', 'Edge Density', 'Edge Complexity'], edge_vals)
                axes[1, 1].set_title('Edge Analysis')
                
                plt.suptitle(f'Analysis: {filename}')
                plt.tight_layout()
                
                individual_path = individual_hist_dir / f"{filename.replace('.', '_')}_analysis.png"
                plt.savefig(individual_path, dpi=300, bbox_inches='tight')
                plt.close()
                
        self.logger.info(f"Generated {len(histogram_paths)} histogram sets")
        return histogram_paths
    
    def generate_analysis_report(self) -> str:
        """Generate a comprehensive analysis report"""
        if not self.dataset_stats:
            raise ValueError("No analysis data available. Run analyze_dataset() first.")
            
        report_path = self.output_dir / "dataset_analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DATASET ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images Analyzed: {self.dataset_stats.get('total_images', 0)}\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 40 + "\n")
            if 'avg_width' in self.dataset_stats:
                f.write(f"Average Image Dimensions: {self.dataset_stats['avg_width']:.0f} x {self.dataset_stats['avg_height']:.0f} pixels\n")
                f.write(f"Average Aspect Ratio: {self.dataset_stats['avg_aspect_ratio']:.3f}\n")
                f.write(f"Resolution Variation: {self.dataset_stats['resolution_variation']:.3f}\n")
                
            if 'avg_file_size_mb' in self.dataset_stats:
                f.write(f"Average File Size: {self.dataset_stats['avg_file_size_mb']:.2f} MB\n")
                f.write(f"Total Dataset Size: {self.dataset_stats['total_size_gb']:.2f} GB\n")
                
            f.write("\n")
            
            # Color analysis
            f.write("COLOR ANALYSIS\n")
            f.write("-" * 40 + "\n")
            color_channels = ['blue', 'green', 'red', 'gray']
            for channel in color_channels:
                mean_key = f'{channel}_distribution_mean'
                std_key = f'{channel}_distribution_std'
                if mean_key in self.dataset_stats:
                    f.write(f"{channel.capitalize()} Channel: Mean={self.dataset_stats[mean_key]:.2f}, Std={self.dataset_stats[std_key]:.2f}\n")
                    
            f.write("\n")
            
            # Quality metrics
            f.write("QUALITY METRICS\n")
            f.write("-" * 40 + "\n")
            quality_metrics = ['brightness', 'contrast', 'lbp_uniformity', 'lbp_entropy', 'sobel_mean']
            for metric in quality_metrics:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                if mean_key in self.dataset_stats:
                    f.write(f"{metric.replace('_', ' ').title()}: Mean={self.dataset_stats[mean_key]:.3f}, Std={self.dataset_stats[std_key]:.3f}\n")
                    
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            
            # Brightness recommendations
            if 'brightness_mean' in self.dataset_stats:
                brightness = self.dataset_stats['brightness_mean']
                if brightness < 100:
                    f.write("• Dataset appears to be relatively dark. Consider brightness enhancement.\n")
                elif brightness > 150:
                    f.write("• Dataset appears to be relatively bright. Consider contrast enhancement.\n")
                else:
                    f.write("• Dataset has good brightness levels for building segmentation.\n")
                    
            # Contrast recommendations
            if 'contrast_mean' in self.dataset_stats:
                contrast = self.dataset_stats['contrast_mean']
                if contrast < 30:
                    f.write("• Dataset has low contrast. Consider contrast enhancement techniques.\n")
                elif contrast > 60:
                    f.write("• Dataset has high contrast, which is good for building detection.\n")
                    
            # Resolution recommendations
            if 'resolution_variation' in self.dataset_stats:
                variation = self.dataset_stats['resolution_variation']
                if variation > 0.3:
                    f.write("• High resolution variation detected. Consider multi-scale processing.\n")
                else:
                    f.write("• Consistent resolution across dataset. Standard processing should work well.\n")
                    
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
            
        self.logger.info(f"Analysis report saved to: {report_path}")
        return str(report_path)
    
    def save_analysis_data(self) -> Dict:
        """Save analysis data to various formats"""
        if not self.dataset_stats:
            raise ValueError("No analysis data available. Run analyze_dataset() first.")
            
        # Save as JSON
        json_path = self.output_dir / "analysis_data.json"
        analysis_data = {
            'dataset_stats': self.dataset_stats,
            'individual_stats': self.individual_stats,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            
        # Save as CSV
        csv_path = self.output_dir / "individual_image_stats.csv"
        df = pd.DataFrame(self.individual_stats)
        df.to_csv(csv_path, index=False)
        
        # Save dataset summary as CSV
        summary_path = self.output_dir / "dataset_summary.csv"
        summary_df = pd.DataFrame([self.dataset_stats])
        summary_df.to_csv(summary_path, index=False)
        
        self.logger.info(f"Analysis data saved to: {json_path}, {csv_path}, {summary_path}")
        
        return {
            'json': str(json_path),
            'individual_csv': str(csv_path),
            'summary_csv': str(summary_path)
        }

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Image Data Analysis for Building Segmentation")
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument("--output", default="analysis_results", help="Output directory for results")
    parser.add_argument("--patterns", nargs="+", default=['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff'],
                       help="File patterns to search for")
    parser.add_argument("--no-individual-histograms", action="store_true", 
                       help="Skip generating individual image histograms")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ImageDataAnalyzer(args.output)
    
    try:
        # Analyze dataset
        print(f"Analyzing dataset: {args.dataset_path}")
        results = analyzer.analyze_dataset(args.dataset_path, args.patterns)
        
        # Generate histograms
        print("Generating histograms...")
        histogram_paths = analyzer.generate_histograms(save_individual=not args.no_individual_histograms)
        
        # Generate report
        print("Generating analysis report...")
        report_path = analyzer.generate_analysis_report()
        
        # Save data
        print("Saving analysis data...")
        data_paths = analyzer.save_analysis_data()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total images processed: {results['total_images']}")
        print(f"Failed images: {results['failed_images']}")
        print(f"Output directory: {args.output}")
        print(f"Report: {report_path}")
        print(f"Histograms: {len(histogram_paths)} sets generated")
        print("="*60)
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
