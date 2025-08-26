#!/usr/bin/env python3
"""
Inference Engine for Building Segmentation
Loads trained models and performs predictions on new images
"""
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import argparse
from datetime import datetime
import logging
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.model_architectures import create_model, get_model_configs
from training.training_utils import get_device

class InferenceEngine:
    """Main inference engine for building segmentation"""
    
    def __init__(self, model_path: str, model_config: Dict, device: Optional[str] = None):
        self.model_path = Path(model_path)
        self.model_config = model_config
        self.device = get_device() if device is None else torch.device(device)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model
        self.model = None
        self._load_model()
        
        self.logger.info(f"Inference engine initialized on {self.device}")
        self.logger.info(f"Model loaded from: {self.model_path}")
    
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
    
    def _load_model(self):
        """Load the trained model"""
        self.logger.info("Loading model...")
        
        try:
            # Create model architecture
            self.model = create_model(self.model_config)
            self.model = self.model.to(self.device)
            
            # Load trained weights
            if self.model_path.exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    # Full checkpoint format
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    # Just model weights
                    self.model.load_state_dict(checkpoint)
                    self.logger.info("Loaded model weights")
                
                self.model.eval()
                self.logger.info("Model loaded successfully")
            else:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
        """Preprocess image for inference"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image_resized = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_normalized = (image_normalized - mean) / std
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).unsqueeze(0).float()
        
        return image_tensor.to(self.device)
    
    def predict(self, image: np.ndarray, threshold: float = 0.01, 
                target_size: Tuple[int, int] = (512, 512)) -> Dict:
        """Perform prediction on a single image"""
        self.logger.info("Running inference...")
        
        # Store original size
        original_size = image.shape[:2]
        
        # Preprocess image
        input_tensor = self.preprocess_image(image, target_size)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # Apply sigmoid if needed
            if output.max() > 1:
                output = torch.sigmoid(output)
            
            # Convert to numpy
            prediction = output.cpu().numpy()[0, 0]  # Remove batch and channel dimensions
            
            # Threshold to get binary mask
            binary_mask = (prediction > threshold).astype(np.uint8) * 255
            
            # Resize back to original size
            binary_mask_resized = cv2.resize(binary_mask, (original_size[1], original_size[0]))
            
            # Create overlay
            overlay = self._create_overlay(image, binary_mask_resized)
        
        return {
            'prediction': prediction,
            'binary_mask': binary_mask_resized,
            'overlay': overlay,
            'confidence': prediction.max(),
            'original_size': original_size,
            'target_size': target_size
        }
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray, 
                       alpha: float = 0.6, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Create overlay of prediction on original image"""
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
    
    def predict_batch(self, images: List[np.ndarray], threshold: float = 0.01,
                     target_size: Tuple[int, int] = (512, 512)) -> List[Dict]:
        """Perform prediction on a batch of images"""
        self.logger.info(f"Running batch inference on {len(images)} images...")
        
        results = []
        for i, image in enumerate(images):
            self.logger.info(f"Processing image {i+1}/{len(images)}")
            result = self.predict(image, threshold, target_size)
            results.append(result)
        
        return results
    
    def save_results(self, results: Dict, output_dir: str, filename: str):
        """Save inference results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save binary mask
        mask_path = output_path / f"{filename}_mask.png"
        cv2.imwrite(str(mask_path), results['binary_mask'])
        
        # Save overlay
        overlay_path = output_path / f"{filename}_overlay.png"
        cv2.imwrite(str(overlay_path), results['overlay'])
        
        # Save prediction as numpy array
        pred_path = output_path / f"{filename}_prediction.npy"
        np.save(str(pred_path), results['prediction'])
        
        # Save metadata
        metadata = {
            'filename': filename,
            'confidence': float(results['confidence']),
            'original_size': results['original_size'],
            'target_size': results['target_size'],
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = output_path / f"{filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
        return {
            'mask_path': str(mask_path),
            'overlay_path': str(overlay_path),
            'prediction_path': str(pred_path),
            'metadata_path': str(metadata_path)
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {'error': 'Model not loaded'}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': type(self.model.model).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'model_path': str(self.model_path)
        }

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Building Segmentation Inference")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--model-config", help="Model configuration (JSON file or preset)")
    parser.add_argument("--input-image", help="Path to input image")
    parser.add_argument("--input-dir", help="Directory containing input images")
    parser.add_argument("--output-dir", default="inference_results", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.01, help="Prediction threshold")
    parser.add_argument("--target-size", type=int, nargs=2, default=[512, 512], help="Target image size")
    parser.add_argument("--device", help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Load model configuration
    if args.model_config and os.path.exists(args.model_config):
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)
    else:
        # Use default configuration - use UNet since that's what we trained
        model_config = get_model_configs()['unet_basic']
    
    # Create inference engine
    engine = InferenceEngine(args.model_path, model_config, args.device)
    
    # Print model info
    model_info = engine.get_model_info()
    print(f"Model: {model_info['model_type']}")
    print(f"Parameters: {model_info['total_parameters']:,}")
    print(f"Device: {model_info['device']}")
    
    # Process input
    if args.input_image:
        # Single image
        image = cv2.imread(args.input_image)
        if image is None:
            print(f"Error: Could not load image {args.input_image}")
            return 1
        
        results = engine.predict(image, args.threshold, tuple(args.target_size))
        engine.save_results(results, args.output_dir, Path(args.input_image).stem)
        
    elif args.input_dir:
        # Directory of images
        input_path = Path(args.input_dir)
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")) + list(input_path.glob("*.png"))
        
        if not image_files:
            print(f"No image files found in {args.input_dir}")
            return 1
        
        for image_file in image_files:
            print(f"Processing {image_file.name}...")
            image = cv2.imread(str(image_file))
            if image is not None:
                results = engine.predict(image, args.threshold, tuple(args.target_size))
                engine.save_results(results, args.output_dir, image_file.stem)
            else:
                print(f"Error: Could not load {image_file}")
    
    else:
        print("Please provide either --input-image or --input-dir")
        return 1
    
    print("Inference completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
