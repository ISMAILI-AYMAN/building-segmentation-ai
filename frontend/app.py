#!/usr/bin/env python3
"""
Flask Web Application for Building Segmentation Frontend
Provides user-friendly web interface for model inference
"""
import os
import sys
import json
import requests
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
import zipfile
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BuildingSegmentationFrontend:
    """Flask web application for building segmentation"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip('/')
        self.app = None
        
        # Setup storage
        self.setup_storage()
        
        # Create Flask app
        self.create_app()
        
        logger.info("Building Segmentation Frontend initialized")
    
    def setup_storage(self):
        """Setup storage directories"""
        self.upload_dir = Path("frontend/uploads")
        self.results_dir = Path("frontend/results")
        self.temp_dir = Path("frontend/temp")
        
        # Create directories
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Frontend storage directories created")
    
    def create_app(self):
        """Create Flask application"""
        self.app = Flask(__name__)
        self.app.secret_key = 'building_segmentation_secret_key'
        
        # Enable CORS
        CORS(self.app)
        
        # Configure upload settings
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        self.app.config['UPLOAD_FOLDER'] = str(self.upload_dir)
        
        # Allowed file extensions
        self.app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
        
        # Setup routes
        self.setup_routes()
        
        logger.info("Flask application created")
    
    def allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in self.app.config['ALLOWED_EXTENSIONS']
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Home page"""
            return render_template('index.html')
        
        @self.app.route('/health')
        def health():
            """Health check endpoint"""
            try:
                response = requests.get(f"{self.api_url}/health", timeout=5)
                if response.status_code == 200:
                    return jsonify({
                        'status': 'healthy',
                        'api_status': 'connected',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'status': 'unhealthy',
                        'api_status': 'error',
                        'error': f"API returned status {response.status_code}",
                        'timestamp': datetime.now().isoformat()
                    })
            except requests.exceptions.RequestException as e:
                return jsonify({
                    'status': 'unhealthy',
                    'api_status': 'disconnected',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        @self.app.route('/model/info')
        def model_info():
            """Get model information"""
            try:
                response = requests.get(f"{self.api_url}/model/info", timeout=5)
                if response.status_code == 200:
                    return jsonify(response.json())
                else:
                    return jsonify({
                        'error': f"API returned status {response.status_code}"
                    }), 500
            except requests.exceptions.RequestException as e:
                return jsonify({
                    'error': f"Failed to connect to API: {str(e)}"
                }), 503
        
        @self.app.route('/upload', methods=['GET', 'POST'])
        def upload_file():
            """File upload page"""
            if request.method == 'POST':
                # Check if files were uploaded
                if 'files' not in request.files:
                    flash('No files selected')
                    return redirect(request.url)
                
                files = request.files.getlist('files')
                if not files or files[0].filename == '':
                    flash('No files selected')
                    return redirect(request.url)
                
                # Validate files
                valid_files = []
                for file in files:
                    if file and self.allowed_file(file.filename):
                        valid_files.append(file)
                    else:
                        flash(f'Invalid file type: {file.filename}')
                
                if not valid_files:
                    flash('No valid files found')
                    return redirect(request.url)
                
                # Get parameters
                threshold = float(request.form.get('threshold', 0.01))
                target_size = request.form.get('target_size', '512,512')
                apply_post_processing = request.form.get('apply_post_processing', 'true').lower() == 'true'
                morphological_kernel = int(request.form.get('morphological_kernel', 3))
                min_area = int(request.form.get('min_area', 100))
                smooth_kernel = int(request.form.get('smooth_kernel', 3))
                fill_holes = request.form.get('fill_holes', 'true').lower() == 'true'
                
                # Process files
                try:
                    if len(valid_files) == 1:
                        # Single file
                        result = self.process_single_file(
                            valid_files[0],
                            threshold,
                            target_size,
                            apply_post_processing,
                            morphological_kernel,
                            min_area,
                            smooth_kernel,
                            fill_holes
                        )
                        return render_template('result.html', result=result)
                    else:
                        # Multiple files
                        result = self.process_multiple_files(
                            valid_files,
                            threshold,
                            target_size,
                            apply_post_processing,
                            morphological_kernel,
                            min_area,
                            smooth_kernel,
                            fill_holes
                        )
                        return render_template('batch_result.html', result=result)
                
                except Exception as e:
                    flash(f'Error processing files: {str(e)}')
                    return redirect(request.url)
            
            return render_template('upload.html')
        
        @self.app.route('/api/inference', methods=['POST'])
        def api_inference():
            """API endpoint for inference"""
            try:
                # Get uploaded file
                if 'file' not in request.files:
                    return jsonify({'error': 'No file provided'}), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({'error': 'No file selected'}), 400
                
                if not self.allowed_file(file.filename):
                    return jsonify({'error': 'No file selected'}), 400
                
                # Get parameters
                threshold = float(request.form.get('threshold', 0.01))
                target_size = request.form.get('target_size', '512,512')
                apply_post_processing = request.form.get('apply_post_processing', 'true').lower() == 'true'
                morphological_kernel = int(request.form.get('morphological_kernel', 3))
                min_area = int(request.form.get('min_area', 100))
                smooth_kernel = int(request.form.get('smooth_kernel', 3))
                fill_holes = request.form.get('fill_holes', 'true').lower() == 'true'
                
                # Process file
                result = self.process_single_file(
                    file,
                    threshold,
                    target_size,
                    apply_post_processing,
                    morphological_kernel,
                    min_area,
                    smooth_kernel,
                    fill_holes
                )
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/results/<request_id>')
        def get_results(request_id: str):
            """Get inference results"""
            try:
                response = requests.get(f"{self.api_url}/results/{request_id}", timeout=5)
                if response.status_code == 200:
                    return jsonify(response.json())
                else:
                    return jsonify({'error': 'Results not found'}), 404
            except requests.exceptions.RequestException as e:
                return jsonify({'error': str(e)}), 503
        
        @self.app.route('/download/<request_id>')
        def download_results(request_id: str):
            """Download inference results"""
            try:
                response = requests.get(f"{self.api_url}/results/{request_id}/download", timeout=30)
                if response.status_code == 200:
                    # Save to temp file
                    temp_file = self.temp_dir / f"{request_id}_results.zip"
                    with open(temp_file, 'wb') as f:
                        f.write(response.content)
                    
                    return send_file(
                        temp_file,
                        as_attachment=True,
                        download_name=f"{request_id}_results.zip",
                        mimetype='application/zip'
                    )
                else:
                    flash('Results not found')
                    return redirect(url_for('index'))
            except requests.exceptions.RequestException as e:
                flash(f'Error downloading results: {str(e)}')
                return redirect(url_for('index'))
        
        @self.app.route('/gallery')
        def gallery():
            """Results gallery"""
            # Get list of recent results
            results = []
            try:
                # This would typically query a database
                # For now, we'll return an empty list
                pass
            except Exception as e:
                logger.error(f"Error loading gallery: {e}")
            
            return render_template('gallery.html', results=results)
        
        @self.app.route('/about')
        def about():
            """About page"""
            return render_template('about.html')
        
        @self.app.errorhandler(404)
        def not_found(error):
            """404 error handler"""
            return render_template('404.html'), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            """500 error handler"""
            return render_template('500.html'), 500
    
    def process_single_file(self, file, threshold: float, target_size: str,
                           apply_post_processing: bool, morphological_kernel: int,
                           min_area: int, smooth_kernel: int, fill_holes: bool) -> Dict:
        """Process a single file"""
        try:
            # Save file temporarily
            filename = secure_filename(file.filename)
            temp_path = self.upload_dir / filename
            file.save(temp_path)
            
            # Prepare form data for API
            files = {'file': (filename, open(temp_path, 'rb'), 'image/jpeg')}
            data = {
                'threshold': str(threshold),
                'target_size': target_size,
                'apply_post_processing': str(apply_post_processing).lower(),
                'morphological_kernel': str(morphological_kernel),
                'min_area': str(min_area),
                'smooth_kernel': str(smooth_kernel),
                'fill_holes': str(fill_holes).lower()
            }
            
            # Send to API
            response = requests.post(
                f"{self.api_url}/inference/single",
                files=files,
                data=data,
                timeout=60
            )
            
            # Close file
            files['file'][1].close()
            
            # Clean up temp file
            temp_path.unlink()
            
            if response.status_code == 200:
                result = response.json()
                result['filename'] = filename
                result['upload_time'] = datetime.now().isoformat()
                return result
            else:
                raise Exception(f"API returned status {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Error processing single file: {e}")
            raise
    
    def process_multiple_files(self, files, threshold: float, target_size: str,
                             apply_post_processing: bool, morphological_kernel: int,
                             min_area: int, smooth_kernel: int, fill_holes: bool) -> Dict:
        """Process multiple files"""
        try:
            # Save files temporarily
            file_paths = []
            for file in files:
                filename = secure_filename(file.filename)
                temp_path = self.upload_dir / filename
                file.save(temp_path)
                file_paths.append(temp_path)
            
            # Prepare form data for API
            files_data = []
            for file_path in file_paths:
                files_data.append(('files', (file_path.name, open(file_path, 'rb'), 'image/jpeg')))
            
            data = {
                'threshold': str(threshold),
                'target_size': target_size,
                'apply_post_processing': str(apply_post_processing).lower(),
                'morphological_kernel': str(morphological_kernel),
                'min_area': str(min_area),
                'smooth_kernel': str(smooth_kernel),
                'fill_holes': str(fill_holes).lower()
            }
            
            # Send to API
            response = requests.post(
                f"{self.api_url}/inference/batch",
                files=files_data,
                data=data,
                timeout=300  # 5 minutes for batch processing
            )
            
            # Close files
            for _, file_tuple in files_data:
                file_tuple[1].close()
            
            # Clean up temp files
            for file_path in file_paths:
                file_path.unlink()
            
            if response.status_code == 200:
                result = response.json()
                result['filenames'] = [f.name for f in files]
                result['upload_time'] = datetime.now().isoformat()
                return result
            else:
                raise Exception(f"API returned status {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Error processing multiple files: {e}")
            raise

def main():
    """Main function to run the Flask server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Building Segmentation Frontend Server")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API server URL")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create frontend
    frontend = BuildingSegmentationFrontend(args.api_url)
    
    # Run server
    frontend.app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
