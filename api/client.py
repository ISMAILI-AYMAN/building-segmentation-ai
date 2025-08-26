#!/usr/bin/env python3
"""
API Client for Building Segmentation API
Provides easy-to-use interface for API interactions
"""
import os
import sys
import requests
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BuildingSegmentationClient:
    """Client for Building Segmentation API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'BuildingSegmentationClient/1.0.0'
        })
        
        logger.info(f"API client initialized for {self.base_url}")
    
    def health_check(self) -> Dict:
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get model info: {e}")
            raise
    
    def inference_single(self, image_path: str, 
                        threshold: float = 0.01,
                        target_size: tuple = (512, 512),
                        apply_post_processing: bool = True,
                        morphological_kernel: int = 3,
                        min_area: int = 100,
                        smooth_kernel: int = 3,
                        fill_holes: bool = True) -> Dict:
        """Perform single image inference"""
        try:
            # Prepare file
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Prepare form data
            files = {
                'file': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')
            }
            
            data = {
                'threshold': str(threshold),
                'target_size': f"{target_size[0]},{target_size[1]}",
                'apply_post_processing': str(apply_post_processing).lower(),
                'morphological_kernel': str(morphological_kernel),
                'min_area': str(min_area),
                'smooth_kernel': str(smooth_kernel),
                'fill_holes': str(fill_holes).lower()
            }
            
            # Make request
            response = self.session.post(f"{self.base_url}/inference/single", 
                                       files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Close file
            files['file'][1].close()
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Single inference failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in single inference: {e}")
            raise
    
    def inference_batch(self, image_paths: List[str],
                       threshold: float = 0.01,
                       target_size: tuple = (512, 512),
                       apply_post_processing: bool = True,
                       morphological_kernel: int = 3,
                       min_area: int = 100,
                       smooth_kernel: int = 3,
                       fill_holes: bool = True) -> Dict:
        """Perform batch inference"""
        try:
            # Validate files
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Prepare files
            files = []
            for image_path in image_paths:
                files.append(('files', (os.path.basename(image_path), 
                                      open(image_path, 'rb'), 'image/jpeg')))
            
            data = {
                'threshold': str(threshold),
                'target_size': f"{target_size[0]},{target_size[1]}",
                'apply_post_processing': str(apply_post_processing).lower(),
                'morphological_kernel': str(morphological_kernel),
                'min_area': str(min_area),
                'smooth_kernel': str(smooth_kernel),
                'fill_holes': str(fill_holes).lower()
            }
            
            # Make request
            response = self.session.post(f"{self.base_url}/inference/batch", 
                                       files=files, data=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Close files
            for _, file_tuple in files:
                file_tuple[1].close()
            
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Batch inference failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in batch inference: {e}")
            raise
    
    def get_results(self, request_id: str) -> Dict:
        """Get inference results"""
        try:
            response = self.session.get(f"{self.base_url}/results/{request_id}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get results: {e}")
            raise
    
    def download_results(self, request_id: str, output_path: str) -> str:
        """Download inference results"""
        try:
            response = self.session.get(f"{self.base_url}/results/{request_id}/download")
            response.raise_for_status()
            
            # Save file
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Results downloaded to {output_path}")
            return output_path
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download results: {e}")
            raise
    
    def delete_results(self, request_id: str) -> Dict:
        """Delete inference results"""
        try:
            response = self.session.delete(f"{self.base_url}/results/{request_id}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to delete results: {e}")
            raise
    
    def wait_for_results(self, request_id: str, timeout: int = 300, 
                        poll_interval: int = 5) -> Dict:
        """Wait for inference results to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                results = self.get_results(request_id)
                logger.info(f"Results ready for {request_id}")
                return results
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    # Results not ready yet
                    logger.info(f"Waiting for results {request_id}...")
                    time.sleep(poll_interval)
                    continue
                else:
                    raise
            except Exception as e:
                logger.error(f"Error waiting for results: {e}")
                raise
        
        raise TimeoutError(f"Timeout waiting for results {request_id}")
    
    def process_directory(self, input_dir: str, output_dir: str,
                         threshold: float = 0.01,
                         target_size: tuple = (512, 512),
                         apply_post_processing: bool = True,
                         **kwargs) -> Dict:
        """Process all images in a directory"""
        try:
            # Find image files
            input_path = Path(input_dir)
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(input_path.glob(ext))
            
            if not image_files:
                raise ValueError(f"No image files found in {input_dir}")
            
            logger.info(f"Found {len(image_files)} images to process")
            
            # Process images
            results = []
            for i, image_file in enumerate(image_files):
                logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
                
                try:
                    result = self.inference_single(
                        str(image_file),
                        threshold,
                        target_size,
                        apply_post_processing,
                        **kwargs
                    )
                    
                    if result['status'] == 'success':
                        results.append(result)
                        
                        # Download results if needed
                        if output_dir:
                            request_id = result['request_id']
                            output_path = Path(output_dir) / f"{request_id}_results.zip"
                            self.download_results(request_id, str(output_path))
                    
                except Exception as e:
                    logger.error(f"Error processing {image_file}: {e}")
                    continue
            
            return {
                'total_images': len(image_files),
                'successful_processing': len(results),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error processing directory: {e}")
            raise

def main():
    """Example usage of the API client"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Building Segmentation API Client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--action", choices=['health', 'info', 'single', 'batch', 'directory'], 
                       required=True, help="Action to perform")
    parser.add_argument("--input", help="Input image or directory")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.01, help="Prediction threshold")
    parser.add_argument("--target-size", type=int, nargs=2, default=[512, 512], help="Target image size")
    
    args = parser.parse_args()
    
    # Create client
    client = BuildingSegmentationClient(args.url)
    
    try:
        if args.action == 'health':
            # Health check
            health = client.health_check()
            print(f"API Health: {health}")
            
        elif args.action == 'info':
            # Get model info
            info = client.get_model_info()
            print(f"Model Info: {json.dumps(info, indent=2)}")
            
        elif args.action == 'single':
            # Single image inference
            if not args.input:
                print("Error: --input required for single inference")
                return 1
            
            result = client.inference_single(
                args.input,
                args.threshold,
                tuple(args.target_size)
            )
            print(f"Inference Result: {json.dumps(result, indent=2)}")
            
        elif args.action == 'batch':
            # Batch inference
            if not args.input:
                print("Error: --input required for batch inference")
                return 1
            
            # Assume input is a directory
            result = client.process_directory(
                args.input,
                args.output,
                args.threshold,
                tuple(args.target_size)
            )
            print(f"Batch Result: {json.dumps(result, indent=2)}")
            
        elif args.action == 'directory':
            # Directory processing
            if not args.input:
                print("Error: --input required for directory processing")
                return 1
            
            result = client.process_directory(
                args.input,
                args.output,
                args.threshold,
                tuple(args.target_size)
            )
            print(f"Directory Result: {json.dumps(result, indent=2)}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
