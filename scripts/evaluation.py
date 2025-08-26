#!/usr/bin/env python3
"""
Building Segmentation Evaluation Framework
Comprehensive evaluation tools for measuring segmentation performance

This module provides:
- Multiple evaluation metrics (IoU, Precision, Recall, F1, Boundary Accuracy)
- Ground truth comparison tools
- Performance visualization
- Statistical analysis
- Report generation
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import argparse
from datetime import datetime
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial.distance import directed_hausdorff
import seaborn as sns
from tqdm import tqdm

# Set matplotlib backend for thread safety
import matplotlib
matplotlib.use('Agg')

class BuildingSegmentationEvaluator:
    """
    Comprehensive evaluator for building segmentation results
    """
    
    def __init__(self, log_dir="logs"):
        """Initialize the evaluator"""
        self.log_dir = log_dir
        self._setup_logging()
        self.results = {}
        
    def _setup_logging(self):
        """Setup logging for evaluation"""
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"evaluation_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_iou(self, ground_truth: np.ndarray, prediction: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU)
        
        Args:
            ground_truth: Binary ground truth mask
            prediction: Binary prediction mask
            
        Returns:
            IoU score
        """
        intersection = np.logical_and(ground_truth, prediction)
        union = np.logical_or(ground_truth, prediction)
        
        if np.sum(union) == 0:
            return 1.0 if np.sum(ground_truth) == 0 and np.sum(prediction) == 0 else 0.0
        
        return np.sum(intersection) / np.sum(union)
    
    def calculate_precision_recall_f1(self, ground_truth: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
        """
        Calculate Precision, Recall, and F1 Score
        
        Args:
            ground_truth: Binary ground truth mask
            prediction: Binary prediction mask
            
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        # Flatten arrays for sklearn metrics
        gt_flat = ground_truth.flatten()
        pred_flat = prediction.flatten()
        
        precision = precision_score(gt_flat, pred_flat, zero_division=0)
        recall = recall_score(gt_flat, pred_flat, zero_division=0)
        f1 = f1_score(gt_flat, pred_flat, zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def calculate_boundary_accuracy(self, ground_truth: np.ndarray, prediction: np.ndarray, 
                                  tolerance: int = 3) -> Dict[str, float]:
        """
        Calculate boundary accuracy using Hausdorff distance
        
        Args:
            ground_truth: Binary ground truth mask
            prediction: Binary prediction mask
            tolerance: Pixel tolerance for boundary matching
            
        Returns:
            Dictionary with boundary accuracy metrics
        """
        # Extract boundaries
        gt_boundary = self._extract_boundary(ground_truth)
        pred_boundary = self._extract_boundary(prediction)
        
        if len(gt_boundary) == 0 or len(pred_boundary) == 0:
            return {
                'hausdorff_distance': float('inf'),
                'boundary_accuracy': 0.0,
                'boundary_precision': 0.0,
                'boundary_recall': 0.0
            }
        
        # Calculate Hausdorff distance
        hausdorff_dist = max(
            directed_hausdorff(gt_boundary, pred_boundary)[0],
            directed_hausdorff(pred_boundary, gt_boundary)[0]
        )
        
        # Calculate boundary accuracy within tolerance
        gt_points = set(map(tuple, gt_boundary))
        pred_points = set(map(tuple, pred_boundary))
        
        # Find points within tolerance
        within_tolerance = 0
        total_gt_points = len(gt_points)
        
        for gt_point in gt_points:
            for pred_point in pred_points:
                if abs(gt_point[0] - pred_point[0]) <= tolerance and \
                   abs(gt_point[1] - pred_point[1]) <= tolerance:
                    within_tolerance += 1
                    break
        
        boundary_accuracy = within_tolerance / total_gt_points if total_gt_points > 0 else 0.0
        
        # Boundary precision and recall
        boundary_precision = within_tolerance / len(pred_points) if len(pred_points) > 0 else 0.0
        boundary_recall = boundary_accuracy
        
        return {
            'hausdorff_distance': hausdorff_dist,
            'boundary_accuracy': boundary_accuracy,
            'boundary_precision': boundary_precision,
            'boundary_recall': boundary_recall
        }
    
    def _extract_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Extract boundary points from binary mask"""
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = mask - eroded
        return np.column_stack(np.where(boundary > 0))
    
    def calculate_shape_metrics(self, ground_truth: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
        """
        Calculate shape-based metrics
        
        Args:
            ground_truth: Binary ground truth mask
            prediction: Binary prediction mask
            
        Returns:
            Dictionary with shape metrics
        """
        # Find contours
        gt_contours, _ = cv2.findContours(ground_truth.astype(np.uint8), 
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pred_contours, _ = cv2.findContours(prediction.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate areas
        gt_area = sum(cv2.contourArea(contour) for contour in gt_contours)
        pred_area = sum(cv2.contourArea(contour) for contour in pred_contours)
        
        # Area ratio
        area_ratio = pred_area / gt_area if gt_area > 0 else 0.0
        
        # Perimeter ratio
        gt_perimeter = sum(cv2.arcLength(contour, True) for contour in gt_contours)
        pred_perimeter = sum(cv2.arcLength(contour, True) for contour in pred_contours)
        perimeter_ratio = pred_perimeter / gt_perimeter if gt_perimeter > 0 else 0.0
        
        # Compactness ratio
        gt_compactness = (4 * np.pi * gt_area) / (gt_perimeter ** 2) if gt_perimeter > 0 else 0.0
        pred_compactness = (4 * np.pi * pred_area) / (pred_perimeter ** 2) if pred_perimeter > 0 else 0.0
        compactness_ratio = pred_compactness / gt_compactness if gt_compactness > 0 else 0.0
        
        return {
            'area_ratio': area_ratio,
            'perimeter_ratio': perimeter_ratio,
            'compactness_ratio': compactness_ratio,
            'gt_area': gt_area,
            'pred_area': pred_area,
            'gt_perimeter': gt_perimeter,
            'pred_perimeter': pred_perimeter
        }
    
    def evaluate_single_image(self, ground_truth_path: str, prediction_path: str, 
                            image_name: str = None) -> Dict[str, float]:
        """
        Evaluate a single image
        
        Args:
            ground_truth_path: Path to ground truth mask
            prediction_path: Path to prediction mask
            image_name: Name of the image for logging
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Load masks
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        prediction = cv2.imread(prediction_path, cv2.IMREAD_GRAYSCALE)
        
        if ground_truth is None or prediction is None:
            self.logger.error(f"Could not load masks for {image_name}")
            return {}
        
        # Ensure binary masks
        ground_truth = (ground_truth > 127).astype(np.uint8)
        prediction = (prediction > 127).astype(np.uint8)
        
        # Ensure same size
        if ground_truth.shape != prediction.shape:
            prediction = cv2.resize(prediction, (ground_truth.shape[1], ground_truth.shape[0]))
        
        # Calculate metrics
        iou = self.calculate_iou(ground_truth, prediction)
        prf_metrics = self.calculate_precision_recall_f1(ground_truth, prediction)
        boundary_metrics = self.calculate_boundary_accuracy(ground_truth, prediction)
        shape_metrics = self.calculate_shape_metrics(ground_truth, prediction)
        
        # Combine all metrics
        results = {
            'image_name': image_name or Path(ground_truth_path).stem,
            'iou': iou,
            **prf_metrics,
            **boundary_metrics,
            **shape_metrics
        }
        
        self.logger.info(f"Evaluated {image_name}: IoU={iou:.3f}, F1={prf_metrics['f1_score']:.3f}")
        
        return results
    
    def evaluate_batch(self, ground_truth_dir: str, prediction_dir: str, 
                      output_dir: str = "evaluation_results") -> Dict[str, Union[Dict, pd.DataFrame]]:
        """
        Evaluate a batch of images
        
        Args:
            ground_truth_dir: Directory containing ground truth masks
            prediction_dir: Directory containing prediction masks
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary with batch evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all mask files
        gt_files = list(Path(ground_truth_dir).glob("*.png"))
        pred_files = list(Path(prediction_dir).glob("*_mask.png"))
        
        self.logger.info(f"Found {len(gt_files)} ground truth files and {len(pred_files)} prediction files")
        
        # Match files
        results = []
        for gt_file in tqdm(gt_files, desc="Evaluating images"):
            # Find corresponding prediction file
            pred_file = None
            for p_file in pred_files:
                if p_file.stem.replace("_mask", "") == gt_file.stem:
                    pred_file = p_file
                    break
            
            if pred_file is None:
                self.logger.warning(f"No prediction found for {gt_file.name}")
                continue
            
            # Evaluate single image
            result = self.evaluate_single_image(str(gt_file), str(pred_file), gt_file.stem)
            if result:
                results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(df)
        
        # Save results
        self._save_evaluation_results(df, summary_stats, output_dir)
        
        # Generate visualizations
        self._generate_evaluation_plots(df, summary_stats, output_dir)
        
        return {
            'detailed_results': df,
            'summary_statistics': summary_stats,
            'output_directory': output_dir
        }
    
    def _calculate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate summary statistics for all metrics"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        summary = {}
        
        for col in numeric_columns:
            if col in ['gt_area', 'pred_area', 'gt_perimeter', 'pred_perimeter']:
                continue  # Skip raw values
            
            summary[f'{col}_mean'] = df[col].mean()
            summary[f'{col}_std'] = df[col].std()
            summary[f'{col}_median'] = df[col].median()
            summary[f'{col}_min'] = df[col].min()
            summary[f'{col}_max'] = df[col].max()
        
        return summary
    
    def _save_evaluation_results(self, df: pd.DataFrame, summary_stats: Dict, output_dir: str):
        """Save evaluation results to files"""
        # Save detailed results
        df.to_csv(os.path.join(output_dir, "detailed_results.csv"), index=False)
        
        # Save summary statistics
        with open(os.path.join(output_dir, "summary_statistics.json"), 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Save summary report
        self._generate_summary_report(df, summary_stats, output_dir)
    
    def _generate_summary_report(self, df: pd.DataFrame, summary_stats: Dict, output_dir: str):
        """Generate a human-readable summary report"""
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("BUILDING SEGMENTATION EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images Evaluated: {len(df)}\n\n")
            
            f.write("KEY METRICS SUMMARY:\n")
            f.write("-" * 20 + "\n")
            f.write(f"IoU (Intersection over Union):\n")
            f.write(f"  Mean: {summary_stats.get('iou_mean', 0):.3f} ± {summary_stats.get('iou_std', 0):.3f}\n")
            f.write(f"  Median: {summary_stats.get('iou_median', 0):.3f}\n")
            f.write(f"  Range: [{summary_stats.get('iou_min', 0):.3f}, {summary_stats.get('iou_max', 0):.3f}]\n\n")
            
            f.write(f"F1 Score:\n")
            f.write(f"  Mean: {summary_stats.get('f1_score_mean', 0):.3f} ± {summary_stats.get('f1_score_std', 0):.3f}\n")
            f.write(f"  Median: {summary_stats.get('f1_score_median', 0):.3f}\n\n")
            
            f.write(f"Precision:\n")
            f.write(f"  Mean: {summary_stats.get('precision_mean', 0):.3f} ± {summary_stats.get('precision_std', 0):.3f}\n")
            f.write(f"  Median: {summary_stats.get('precision_median', 0):.3f}\n\n")
            
            f.write(f"Recall:\n")
            f.write(f"  Mean: {summary_stats.get('recall_mean', 0):.3f} ± {summary_stats.get('recall_std', 0):.3f}\n")
            f.write(f"  Median: {summary_stats.get('recall_median', 0):.3f}\n\n")
            
            f.write(f"Boundary Accuracy:\n")
            f.write(f"  Mean: {summary_stats.get('boundary_accuracy_mean', 0):.3f} ± {summary_stats.get('boundary_accuracy_std', 0):.3f}\n")
            f.write(f"  Median: {summary_stats.get('boundary_accuracy_median', 0):.3f}\n\n")
            
            f.write("PERFORMANCE ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            
            # Performance categories
            excellent = len(df[df['iou'] >= 0.8])
            good = len(df[(df['iou'] >= 0.6) & (df['iou'] < 0.8)])
            fair = len(df[(df['iou'] >= 0.4) & (df['iou'] < 0.6)])
            poor = len(df[df['iou'] < 0.4])
            
            f.write(f"Performance Distribution:\n")
            f.write(f"  Excellent (IoU ≥ 0.8): {excellent} images ({excellent/len(df)*100:.1f}%)\n")
            f.write(f"  Good (0.6 ≤ IoU < 0.8): {good} images ({good/len(df)*100:.1f}%)\n")
            f.write(f"  Fair (0.4 ≤ IoU < 0.6): {fair} images ({fair/len(df)*100:.1f}%)\n")
            f.write(f"  Poor (IoU < 0.4): {poor} images ({poor/len(df)*100:.1f}%)\n\n")
            
            # Top and bottom performers
            top_5 = df.nlargest(5, 'iou')[['image_name', 'iou', 'f1_score']]
            bottom_5 = df.nsmallest(5, 'iou')[['image_name', 'iou', 'f1_score']]
            
            f.write("TOP 5 PERFORMERS:\n")
            for _, row in top_5.iterrows():
                f.write(f"  {row['image_name']}: IoU={row['iou']:.3f}, F1={row['f1_score']:.3f}\n")
            
            f.write("\nBOTTOM 5 PERFORMERS:\n")
            for _, row in bottom_5.iterrows():
                f.write(f"  {row['image_name']}: IoU={row['iou']:.3f}, F1={row['f1_score']:.3f}\n")
    
    def _generate_evaluation_plots(self, df: pd.DataFrame, summary_stats: Dict, output_dir: str):
        """Generate evaluation plots and visualizations"""
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Metrics distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Building Segmentation Evaluation Metrics Distribution', fontsize=16)
        
        metrics = ['iou', 'precision', 'recall', 'f1_score', 'boundary_accuracy', 'hausdorff_distance']
        titles = ['IoU', 'Precision', 'Recall', 'F1 Score', 'Boundary Accuracy', 'Hausdorff Distance']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = i // 3, i % 3
            axes[row, col].hist(df[metric], bins=20, alpha=0.7, edgecolor='black')
            axes[row, col].axvline(df[metric].mean(), color='red', linestyle='--', 
                                 label=f'Mean: {df[metric].mean():.3f}')
            axes[row, col].axvline(df[metric].median(), color='green', linestyle='--', 
                                 label=f'Median: {df[metric].median():.3f}')
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel(metric)
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation matrix
        numeric_cols = ['iou', 'precision', 'recall', 'f1_score', 'boundary_accuracy', 
                       'hausdorff_distance', 'area_ratio', 'perimeter_ratio']
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Metrics Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. IoU vs F1 Score scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(df['iou'], df['f1_score'], alpha=0.6)
        plt.xlabel('IoU')
        plt.ylabel('F1 Score')
        plt.title('IoU vs F1 Score')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['iou'], df['f1_score'], 1)
        p = np.poly1d(z)
        plt.plot(df['iou'], p(df['iou']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "iou_vs_f1.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Performance ranking
        plt.figure(figsize=(12, 8))
        df_sorted = df.sort_values('iou', ascending=True)
        plt.barh(range(len(df_sorted)), df_sorted['iou'])
        plt.xlabel('IoU Score')
        plt.ylabel('Image Rank')
        plt.title('Performance Ranking by IoU Score')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "performance_ranking.png"), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Building Segmentation Evaluation Framework")
    parser.add_argument("--ground-truth", "-gt", required=True, help="Ground truth directory or file")
    parser.add_argument("--predictions", "-pred", required=True, help="Predictions directory or file")
    parser.add_argument("--output", "-o", default="evaluation_results", help="Output directory")
    parser.add_argument("--single", action="store_true", help="Evaluate single image pair")
    
    args = parser.parse_args()
    
    evaluator = BuildingSegmentationEvaluator()
    
    if args.single:
        # Single image evaluation
        result = evaluator.evaluate_single_image(args.ground_truth, args.predictions)
        print(f"Single Image Evaluation Results:")
        for metric, value in result.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
    else:
        # Batch evaluation
        results = evaluator.evaluate_batch(args.ground_truth, args.predictions, args.output)
        print(f"✅ Batch evaluation completed!")
        print(f"📊 Results saved to: {args.output}")
        print(f"📈 Summary statistics:")
        summary = results['summary_statistics']
        print(f"  IoU: {summary.get('iou_mean', 0):.3f} ± {summary.get('iou_std', 0):.3f}")
        print(f"  F1 Score: {summary.get('f1_score_mean', 0):.3f} ± {summary.get('f1_score_std', 0):.3f}")

if __name__ == "__main__":
    main()
