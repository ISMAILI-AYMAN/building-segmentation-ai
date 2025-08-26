#!/bin/bash

# Building Segmentation Pipeline Deployment Script
# This script handles deployment for both development and production environments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Prerequisites check passed!"
}

# Function to check if model exists
check_model() {
    print_status "Checking if trained model exists..."
    
    if [ ! -f "trained_model_final/final_model.pth" ]; then
        print_warning "Trained model not found at trained_model_final/final_model.pth"
        print_warning "You may need to train a model first or download a pre-trained model."
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_error "Deployment cancelled."
            exit 1
        fi
    else
        print_success "Trained model found!"
    fi
}

# Function to create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p data/raw
    mkdir -p api/uploads
    mkdir -p api/results
    mkdir -p api/static
    
    print_success "Directories created!"
}

# Function to set proper permissions
set_permissions() {
    print_status "Setting proper permissions..."
    
    chmod -R 755 api/
    chmod -R 755 frontend/
    chmod -R 755 logs/
    
    print_success "Permissions set!"
}

# Function to deploy development environment
deploy_dev() {
    print_status "Deploying development environment..."
    
    cd docker_config
    
    # Stop existing containers
    print_status "Stopping existing containers..."
    docker-compose down --remove-orphans
    
    # Build and start containers
    print_status "Building and starting containers..."
    docker-compose up --build -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    print_status "Checking service health..."
    
    # Check API health
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_success "API service is healthy!"
    else
        print_warning "API service health check failed. Check logs with: docker-compose logs api"
    fi
    
    # Check frontend health
    if curl -f http://localhost:5000/health &> /dev/null; then
        print_success "Frontend service is healthy!"
    else
        print_warning "Frontend service health check failed. Check logs with: docker-compose logs frontend"
    fi
    
    cd ..
    
    print_success "Development environment deployed successfully!"
    print_status "Access points:"
    echo "  - Frontend: http://localhost:5000"
    echo "  - API: http://localhost:8000"
    echo "  - API Docs: http://localhost:8000/docs"
}

# Function to deploy production environment
deploy_prod() {
    print_status "Deploying production environment..."
    
    cd docker_config
    
    # Stop existing containers
    print_status "Stopping existing containers..."
    docker-compose -f docker-compose.prod.yml down --remove-orphans
    
    # Build and start containers
    print_status "Building and starting production containers..."
    docker-compose -f docker-compose.prod.yml up --build -d
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 60
    
    # Check service health
    print_status "Checking service health..."
    
    # Check API health
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_success "API service is healthy!"
    else
        print_warning "API service health check failed. Check logs with: docker-compose -f docker-compose.prod.yml logs api"
    fi
    
    # Check frontend health
    if curl -f http://localhost:5000/health &> /dev/null; then
        print_success "Frontend service is healthy!"
    else
        print_warning "Frontend service health check failed. Check logs with: docker-compose -f docker-compose.prod.yml logs frontend"
    fi
    
    # Check monitoring services
    if curl -f http://localhost:9090/-/healthy &> /dev/null; then
        print_success "Prometheus monitoring is healthy!"
    else
        print_warning "Prometheus health check failed."
    fi
    
    if curl -f http://localhost:3000/api/health &> /dev/null; then
        print_success "Grafana dashboard is healthy!"
    else
        print_warning "Grafana health check failed."
    fi
    
    cd ..
    
    print_success "Production environment deployed successfully!"
    print_status "Access points:"
    echo "  - Frontend: http://localhost:5000"
    echo "  - API: http://localhost:8000"
    echo "  - API Docs: http://localhost:8000/docs"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000 (admin/admin123)"
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."
    
    cd docker_config
    
    if [ "$1" = "prod" ]; then
        docker-compose -f docker-compose.prod.yml down
        print_success "Production services stopped!"
    else
        docker-compose down
        print_success "Development services stopped!"
    fi
    
    cd ..
}

# Function to show logs
show_logs() {
    print_status "Showing logs..."
    
    cd docker_config
    
    if [ "$1" = "prod" ]; then
        docker-compose -f docker-compose.prod.yml logs -f
    else
        docker-compose logs -f
    fi
    
    cd ..
}

# Function to show status
show_status() {
    print_status "Service status..."
    
    cd docker_config
    
    if [ "$1" = "prod" ]; then
        docker-compose -f docker-compose.prod.yml ps
    else
        docker-compose ps
    fi
    
    cd ..
}

# Function to clean up
cleanup() {
    print_status "Cleaning up..."
    
    cd docker_config
    
    if [ "$1" = "prod" ]; then
        docker-compose -f docker-compose.prod.yml down -v --remove-orphans
    else
        docker-compose down -v --remove-orphans
    fi
    
    cd ..
    
    print_success "Cleanup completed!"
}

# Function to show help
show_help() {
    echo "Building Segmentation Pipeline Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  dev           Deploy development environment"
    echo "  prod          Deploy production environment"
    echo "  stop [env]    Stop services (env: dev/prod, default: dev)"
    echo "  logs [env]    Show logs (env: dev/prod, default: dev)"
    echo "  status [env]  Show service status (env: dev/prod, default: dev)"
    echo "  cleanup [env] Clean up containers and volumes (env: dev/prod, default: dev)"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 dev                    # Deploy development environment"
    echo "  $0 prod                   # Deploy production environment"
    echo "  $0 stop prod              # Stop production services"
    echo "  $0 logs dev               # Show development logs"
    echo "  $0 status                 # Show development service status"
    echo "  $0 cleanup prod           # Clean up production environment"
}

# Main script logic
main() {
    case "$1" in
        "dev")
            check_prerequisites
            check_model
            create_directories
            set_permissions
            deploy_dev
            ;;
        "prod")
            check_prerequisites
            check_model
            create_directories
            set_permissions
            deploy_prod
            ;;
        "stop")
            stop_services "$2"
            ;;
        "logs")
            show_logs "$2"
            ;;
        "status")
            show_status "$2"
            ;;
        "cleanup")
            cleanup "$2"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
