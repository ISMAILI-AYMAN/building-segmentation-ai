# 🏢 Building Segmentation AI - Complete Pipeline

A comprehensive deep learning solution for automated building segmentation from aerial imagery using advanced computer vision techniques.

## 🌟 Features

- **Deep Learning Pipeline**: UNet-based segmentation with advanced training techniques
- **RESTful API**: FastAPI backend with real-time inference capabilities
- **Web Interface**: Flask frontend with drag-and-drop image processing
- **Docker Support**: Complete containerization with production-ready configurations
- **Multi-scale Processing**: Traditional CV + DL hybrid approach
- **Pseudo-labeling**: Semi-supervised learning for improved performance
- **Monitoring**: Prometheus + Grafana integration
- **Deployment Scripts**: Automated deployment for development and production

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose (for containerized deployment)
- CUDA-compatible GPU (optional, for acceleration)

### Local Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd clean_building_segmentation

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py --help
```

### Docker Deployment

```bash
# Development environment
./deploy.sh dev

# Production environment
./deploy.sh prod

# GPU-enabled deployment
./deploy.sh gpu
```

## 📁 Project Structure

```
clean_building_segmentation/
├── api/                    # FastAPI backend
├── frontend/              # Flask web interface
├── inference/             # Model inference engine
├── training/              # Training pipeline
├── scripts/               # Utility scripts
├── docker_config/         # Docker configurations
├── user_apps/             # User applications
├── main.py               # Main CLI entry point
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## 🎯 Usage Examples

### Command Line Interface

```bash
# Run inference on a single image
python main.py inference --image path/to/image.jpg

# Start the API server
python main.py api --model-path trained_model_final/final_model.pth

# Start the web frontend
python main.py frontend --api-url http://127.0.0.1:8001

# Train the model
python main.py train --data-path data/ --epochs 100
```

### API Usage

```python
import requests

# Single image inference
url = "http://localhost:8001/inference"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

# Batch inference
url = "http://localhost:8001/batch-inference"
files = [("files", open(f"image_{i}.jpg", "rb")) for i in range(5)]
response = requests.post(url, files=files)
results = response.json()
```

### Web Interface

1. Open `http://localhost:5000` in your browser
2. Upload an image or drag-and-drop
3. View results with detailed metrics
4. Download processed images

## 🏗️ Architecture

### Core Components

1. **Inference Engine**: PyTorch-based UNet model with post-processing
2. **API Layer**: FastAPI with async processing and result caching
3. **Frontend**: Flask web interface with real-time updates
4. **Training Pipeline**: Automated training with pseudo-labeling
5. **Docker Stack**: Multi-service deployment with monitoring

### Technology Stack

- **Backend**: FastAPI, PyTorch, OpenCV
- **Frontend**: Flask, Jinja2, Bootstrap
- **Containerization**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana, Redis
- **Deployment**: Nginx, SSL/TLS

## 📊 Performance

- **Inference Speed**: 2-5 seconds per image (CPU), 0.5-1 second (GPU)
- **Accuracy**: 85-90% IoU on building segmentation
- **Scalability**: Horizontal scaling with Docker Swarm
- **Memory Usage**: 2-4GB RAM per container

## 🔧 Configuration

### Environment Variables

```bash
# Model configuration
MODEL_PATH=/app/trained_model_final/final_model.pth
DEVICE=cpu  # or cuda for GPU

# API configuration
API_HOST=0.0.0.0
API_PORT=8001

# Frontend configuration
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=5000
```

### Docker Configuration

The project includes multiple Docker configurations:
- `docker_config/Dockerfile.dev` - Development environment
- `docker_config/Dockerfile.prod` - Production environment
- `docker_config/docker-compose.yml` - Multi-service setup
- `docker_config/docker-compose.prod.yml` - Production deployment

## 🚀 Deployment Options

### 1. Local Development
```bash
python main.py api --reload
python main.py frontend --debug
```

### 2. Docker Development
```bash
./deploy.sh dev
```

### 3. Production Deployment
```bash
./deploy.sh prod
```

### 4. GPU-Accelerated Deployment
```bash
./deploy.sh gpu
```

## 📈 Monitoring

The production deployment includes:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Redis**: Caching and session management
- **Nginx**: Reverse proxy and load balancing

Access monitoring at:
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`

## 🔍 API Endpoints

### Core Endpoints
- `POST /inference` - Single image processing
- `POST /batch-inference` - Multiple image processing
- `GET /health` - Health check
- `GET /model-info` - Model information

### Static Files
- `GET /api/results/{uuid}/{filename}` - Download processed images

## 🛠️ Development

### Adding New Features

1. **Model Improvements**: Modify `inference/inference_engine.py`
2. **API Extensions**: Add endpoints in `api/app.py`
3. **Frontend Enhancements**: Update `frontend/templates/`
4. **Training Pipeline**: Extend `training/` modules

### Testing

```bash
# Run inference tests
python -m pytest tests/test_inference.py

# Test API endpoints
python -m pytest tests/test_api.py

# Integration tests
python -m pytest tests/test_integration.py
```

## 📚 Documentation

- [Project Summary](PROJECT_SUMMARY.md) - Complete technical overview
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Detailed deployment instructions
- [API Documentation](http://localhost:8001/docs) - Interactive API docs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- FastAPI for the modern web framework
- OpenCV for computer vision capabilities
- Docker for containerization technology

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the deployment guide

---

**Note**: This repository does not include training data or pre-trained models. Please refer to the documentation for data preparation and model training instructions.
