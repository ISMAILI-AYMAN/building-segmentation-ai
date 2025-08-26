#!/usr/bin/env python3
"""
FastAPI Application for Building Segmentation API
Provides RESTful endpoints for model inference
"""
import os
import sys
import time
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Union
import tempfile
import shutil
from datetime import datetime
import logging
import json
import uuid

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from inference.inference_engine import InferenceEngine
from inference.post_processing import PostProcessor
from training.model_architectures import get_model_configs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class InferenceRequest(BaseModel):
    threshold: float = Field(default=0.01, ge=0.0, le=1.0, description="Prediction threshold")
    target_size: List[int] = Field(default=[512, 512], description="Target image size")
    apply_post_processing: bool = Field(default=True, description="Apply post-processing")
    morphological_kernel: int = Field(default=3, description="Morphological kernel size")
    min_area: int = Field(default=100, description="Minimum object area")
    smooth_kernel: int = Field(default=3, description="Smoothing kernel size")
    fill_holes: bool = Field(default=True, description="Fill holes in masks")

class InferenceResponse(BaseModel):
    request_id: str
    status: str
    message: str
    results: Optional[Dict] = None
    error: Optional[str] = None
    timestamp: str

class ModelInfo(BaseModel):
    model_type: str
    total_parameters: int
    device: str
    model_path: str

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    device: str

class BuildingSegmentationAPI:
    """FastAPI application for building segmentation"""
    
    def __init__(self, model_path: str, model_config: Dict, device: Optional[str] = None):
        self.model_path = model_path
        self.model_config = model_config
        self.device = device
        
        # Initialize components
        self.engine = None
        self.post_processor = None
        self.app = None
        
        # Setup storage
        self.setup_storage()
        
        # Initialize model
        self.load_model()
        
        # Create FastAPI app
        self.create_app()
        
        logger.info("Building Segmentation API initialized")
    
    def setup_storage(self):
        """Setup storage directories"""
        self.upload_dir = Path("api/uploads")
        self.results_dir = Path("api/results")
        self.temp_dir = Path("api/temp")
        
        # Create directories
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Storage directories created")
    
    def load_model(self):
        """Load the inference model"""
        try:
            logger.info("Loading inference model...")
            self.engine = InferenceEngine(self.model_path, self.model_config, self.device)
            self.post_processor = PostProcessor()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def create_app(self):
        """Create FastAPI application"""
        self.app = FastAPI(
            title="Building Segmentation API",
            description="API for building segmentation using deep learning models",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="api/static"), name="static")
        self.app.mount("/api/results", StaticFiles(directory="api/results"), name="results")
        
        # Setup routes
        self.setup_routes()
        
        logger.info("FastAPI application created")
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_model=Dict)
        async def root():
            """Root endpoint"""
            return {
                "message": "Building Segmentation API",
                "version": "1.0.0",
                "docs": "/docs",
                "health": "/health"
            }
        
        @self.app.get("/health", response_model=HealthCheck)
        async def health_check():
            """Health check endpoint"""
            return HealthCheck(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                model_loaded=self.engine is not None,
                device=str(self.engine.device) if self.engine else "unknown"
            )
        
        @self.app.get("/model/info", response_model=ModelInfo)
        async def get_model_info():
            """Get model information"""
            if self.engine is None:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            info = self.engine.get_model_info()
            return ModelInfo(**info)
        
        @self.app.post("/inference/single", response_model=InferenceResponse)
        async def inference_single(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            threshold: float = Form(0.01),
            target_size: str = Form("512,512"),
            apply_post_processing: bool = Form(True),
            morphological_kernel: int = Form(3),
            min_area: int = Form(100),
            smooth_kernel: int = Form(3),
            fill_holes: bool = Form(True)
        ):
            """Single image inference endpoint"""
            request_id = str(uuid.uuid4())
            
            try:
                # Parse target size
                target_size_list = [int(x.strip()) for x in target_size.split(",")]
                if len(target_size_list) != 2:
                    raise ValueError("target_size must be in format 'width,height'")
                
                # Validate file
                if not file.content_type.startswith("image/"):
                    raise HTTPException(status_code=400, detail="File must be an image")
                
                # Save uploaded file
                file_path = self.upload_dir / f"{request_id}_{file.filename}"
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Process image
                results = await self.process_single_image(
                    str(file_path),
                    request_id,
                    threshold,
                    tuple(target_size_list),
                    apply_post_processing,
                    {
                        'morphological_kernel': morphological_kernel,
                        'min_area': min_area,
                        'smooth_kernel': smooth_kernel,
                        'fill_holes': fill_holes
                    }
                )
                
                # Cleanup uploaded file
                background_tasks.add_task(self.cleanup_file, file_path)
                
                return InferenceResponse(
                    request_id=request_id,
                    status="success",
                    message="Inference completed successfully",
                    results=results,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Error in single inference: {e}")
                return InferenceResponse(
                    request_id=request_id,
                    status="error",
                    message="Inference failed",
                    error=str(e),
                    timestamp=datetime.now().isoformat()
                )
        
        @self.app.post("/inference/batch", response_model=InferenceResponse)
        async def inference_batch(
            background_tasks: BackgroundTasks,
            files: List[UploadFile] = File(...),
            threshold: float = Form(0.01),
            target_size: str = Form("512,512"),
            apply_post_processing: bool = Form(True),
            morphological_kernel: int = Form(3),
            min_area: int = Form(100),
            smooth_kernel: int = Form(3),
            fill_holes: bool = Form(True)
        ):
            """Batch inference endpoint"""
            request_id = str(uuid.uuid4())
            
            try:
                # Parse target size
                target_size_list = [int(x.strip()) for x in target_size.split(",")]
                if len(target_size_list) != 2:
                    raise ValueError("target_size must be in format 'width,height'")
                
                # Validate files
                for file in files:
                    if not file.content_type.startswith("image/"):
                        raise HTTPException(status_code=400, detail="All files must be images")
                
                # Save uploaded files
                file_paths = []
                for file in files:
                    file_path = self.upload_dir / f"{request_id}_{file.filename}"
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    file_paths.append(file_path)
                
                # Process images
                results = await self.process_batch_images(
                    file_paths,
                    request_id,
                    threshold,
                    tuple(target_size_list),
                    apply_post_processing,
                    {
                        'morphological_kernel': morphological_kernel,
                        'min_area': min_area,
                        'smooth_kernel': smooth_kernel,
                        'fill_holes': fill_holes
                    }
                )
                
                # Cleanup uploaded files
                for file_path in file_paths:
                    background_tasks.add_task(self.cleanup_file, file_path)
                
                return InferenceResponse(
                    request_id=request_id,
                    status="success",
                    message=f"Batch inference completed for {len(files)} images",
                    results=results,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Error in batch inference: {e}")
                return InferenceResponse(
                    request_id=request_id,
                    status="error",
                    message="Batch inference failed",
                    error=str(e),
                    timestamp=datetime.now().isoformat()
                )
        
        @self.app.get("/results/{request_id}")
        async def get_results(request_id: str):
            """Get inference results"""
            results_dir = self.results_dir / request_id
            if not results_dir.exists():
                raise HTTPException(status_code=404, detail="Results not found")
            
            # Return results summary
            summary_file = results_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                return summary
            else:
                raise HTTPException(status_code=404, detail="Results summary not found")
        
        @self.app.get("/results/{request_id}/download")
        async def download_results(request_id: str):
            """Download inference results as zip file"""
            results_dir = self.results_dir / request_id
            if not results_dir.exists():
                raise HTTPException(status_code=404, detail="Results not found")
            
            # Create zip file
            import zipfile
            zip_path = self.temp_dir / f"{request_id}_results.zip"
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file_path in results_dir.rglob("*"):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(results_dir))
            
            return FileResponse(
                zip_path,
                media_type="application/zip",
                filename=f"{request_id}_results.zip"
            )
        
        @self.app.delete("/results/{request_id}")
        async def delete_results(request_id: str):
            """Delete inference results"""
            results_dir = self.results_dir / request_id
            if not results_dir.exists():
                raise HTTPException(status_code=404, detail="Results not found")
            
            # Delete results directory
            shutil.rmtree(results_dir)
            
            return {"message": f"Results for {request_id} deleted successfully"}
    
    async def process_single_image(self, image_path: str, request_id: str,
                                 threshold: float, target_size: tuple,
                                 apply_post_processing: bool,
                                 post_processing_config: Dict) -> Dict:
        """Process a single image"""
        try:
            start_time = time.time()
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Run inference
            results = self.engine.predict(image, threshold, target_size)
            
            # Create results directory
            results_dir = self.results_dir / request_id
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save original image
            filename = Path(image_path).stem
            original_path = results_dir / f"{filename}_original.png"
            cv2.imwrite(str(original_path), image)
            
            # Apply post-processing if requested
            if apply_post_processing:
                processed_mask = self.post_processor.apply_full_pipeline(
                    results['binary_mask'],
                    **post_processing_config
                )
                
                # Calculate metrics
                metrics = self.post_processor.calculate_area_metrics(processed_mask)
                
                # Save processed results
                saved_paths = self.post_processor.save_processed_results(
                    results['binary_mask'],
                    processed_mask,
                    image,
                    str(results_dir),
                    filename,
                    metrics
                )
                
                # Add original image path and processed results
                saved_paths['original_path'] = str(original_path)
                results['processed_mask'] = processed_mask.tolist()
                results['metrics'] = metrics
                results['saved_paths'] = saved_paths
            else:
                # Save basic results
                saved_paths = self.engine.save_results(results, str(results_dir), filename)
                saved_paths['original_path'] = str(original_path)
                results['saved_paths'] = saved_paths
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    # For large arrays, just store shape and type info
                    if value.size > 1000:  # Large arrays
                        serializable_results[key] = {
                            'shape': value.shape,
                            'dtype': str(value.dtype),
                            'min': float(value.min()),
                            'max': float(value.max()),
                            'mean': float(value.mean()),
                            'note': 'Large array - see saved files for full data'
                        }
                    else:
                        serializable_results[key] = value.tolist()
                elif isinstance(value, np.integer):
                    serializable_results[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable_results[key] = float(value)
                else:
                    serializable_results[key] = value
            
            # Add processing time and other metadata
            serializable_results['processing_time'] = round(processing_time, 3)
            serializable_results['model_type'] = self.engine.get_model_info()['model_type']
            serializable_results['threshold'] = threshold
            serializable_results['apply_post_processing'] = apply_post_processing
            
            # Save summary
            summary = {
                'request_id': request_id,
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'threshold': threshold,
                'target_size': target_size,
                'apply_post_processing': apply_post_processing,
                'results': serializable_results
            }
            
            with open(results_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            return serializable_results
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    async def process_batch_images(self, image_paths: List[Path], request_id: str,
                                 threshold: float, target_size: tuple,
                                 apply_post_processing: bool,
                                 post_processing_config: Dict) -> Dict:
        """Process a batch of images"""
        try:
            results = []
            
            for i, image_path in enumerate(image_paths):
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path.name}")
                
                try:
                    result = await self.process_single_image(
                        str(image_path),
                        f"{request_id}_{i}",
                        threshold,
                        target_size,
                        apply_post_processing,
                        post_processing_config
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    continue
            
            # Create batch summary
            batch_results_dir = self.results_dir / request_id
            batch_results_dir.mkdir(parents=True, exist_ok=True)
            
            summary = {
                'request_id': request_id,
                'timestamp': datetime.now().isoformat(),
                'total_images': len(image_paths),
                'successful_processing': len(results),
                'threshold': threshold,
                'target_size': target_size,
                'apply_post_processing': apply_post_processing,
                'results': results
            }
            
            with open(batch_results_dir / "batch_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
    
    async def cleanup_file(self, file_path: Path):
        """Cleanup temporary file"""
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {e}")

def main():
    """Main function to run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Building Segmentation API Server")
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--model-config", help="Model configuration (JSON file or preset)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--device", help="Device to use (cpu/cuda)")
    parser.add_argument("--reload", action='store_true', help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Load model configuration
    if args.model_config and os.path.exists(args.model_config):
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)
    else:
        # Use default configuration - use UNet since that's what we trained
        model_config = get_model_configs()['unet_basic']
    
    # Create API
    api = BuildingSegmentationAPI(args.model_path, model_config, args.device)
    
    # Run server
    uvicorn.run(
        api.app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
