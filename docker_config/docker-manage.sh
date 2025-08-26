#!/bin/bash

# Building Segmentation Pipeline - Docker Management Script
# This script provides easy management of the Docker environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROD_COMPOSE_FILE="docker-compose.prod.yml"
PROJECT_NAME="building-segmentation"

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
    exit 1
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
    fi
    print_success "Docker is running"
}

# Function to check if Docker Compose is available
check_compose() {
    if ! docker-compose --version > /dev/null 2>&1; then
        print_error "Docker Compose is not available. Please install Docker Compose and try again."
    fi
    print_success "Docker Compose is available"
}

# Function to build images
build_images() {
    print_status "Building Docker images..."
    docker-compose -f $COMPOSE_FILE build --no-cache
    print_success "Images built successfully"
}

# Function to start services
start_services() {
    print_status "Starting services..."
    docker-compose -f $COMPOSE_FILE up -d
    print_success "Services started successfully"
}

# Function to start production services
start_production() {
    print_status "Starting production services..."
    docker-compose -f $PROD_COMPOSE_FILE up -d
    print_success "Production services started successfully"
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."
    docker-compose -f $COMPOSE_FILE down
    print_success "Services stopped successfully"
}

# Function to stop production services
stop_production() {
    print_status "Stopping production services..."
    docker-compose -f $PROD_COMPOSE_FILE down
    print_success "Production services stopped successfully"
}

# Function to restart services
restart_services() {
    print_status "Restarting services..."
    docker-compose -f $COMPOSE_FILE restart
    print_success "Services restarted successfully"
}

# Function to view logs
view_logs() {
    local service=${1:-""}
    if [ -z "$service" ]; then
        print_status "Showing logs for all services..."
        docker-compose -f $COMPOSE_FILE logs -f
    else
        print_status "Showing logs for service: $service"
        docker-compose -f $COMPOSE_FILE logs -f $service
    fi
}

# Function to view production logs
view_production_logs() {
    local service=${1:-""}
    if [ -z "$service" ]; then
        print_status "Showing production logs for all services..."
        docker-compose -f $PROD_COMPOSE_FILE logs -f
    else
        print_status "Showing production logs for service: $service"
        docker-compose -f $PROD_COMPOSE_FILE logs -f $service
    fi
}

# Function to check service status
check_status() {
    print_status "Checking service status..."
    docker-compose -f $COMPOSE_FILE ps
}

# Function to check production status
check_production_status() {
    print_status "Checking production service status..."
    docker-compose -f $PROD_COMPOSE_FILE ps
}

# Function to clean up
cleanup() {
    print_warning "This will remove all containers, networks, and volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up Docker environment..."
        docker-compose -f $COMPOSE_FILE down -v --rmi all
        docker system prune -f
        print_success "Cleanup completed"
    else
        print_status "Cleanup cancelled"
    fi
}

# Function to clean up production
cleanup_production() {
    print_warning "This will remove all production containers, networks, and volumes. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_status "Cleaning up production Docker environment..."
        docker-compose -f $PROD_COMPOSE_FILE down -v --rmi all
        docker system prune -f
        print_success "Production cleanup completed"
    else
        print_status "Production cleanup cancelled"
    fi
}

# Function to scale services
scale_services() {
    local service=${1:-"api"}
    local replicas=${2:-"2"}
    print_status "Scaling $service to $replicas replicas..."
    docker-compose -f $COMPOSE_FILE up -d --scale $service=$replicas
    print_success "Service scaled successfully"
}

# Function to show system info
system_info() {
    print_status "Docker system information:"
    echo "Docker version: $(docker --version)"
    echo "Docker Compose version: $(docker-compose --version)"
    echo "Docker info:"
    docker info --format 'table {{.ServerVersion}}\t{{.OperatingSystem}}\t{{.KernelVersion}}'
}

# Function to show usage
show_usage() {
    echo "Building Segmentation Pipeline - Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build              Build Docker images"
    echo "  start              Start development services"
    echo "  start-prod         Start production services"
    echo "  stop               Stop development services"
    echo "  stop-prod          Stop production services"
    echo "  restart            Restart development services"
    echo "  logs [SERVICE]     View logs (all services or specific service)"
    echo "  logs-prod [SERVICE] View production logs"
    echo "  status             Check development service status"
    echo "  status-prod        Check production service status"
    echo "  scale SERVICE N    Scale service to N replicas"
    echo "  cleanup            Clean up development environment"
    echo "  cleanup-prod       Clean up production environment"
    echo "  info               Show system information"
    echo "  help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start           # Start development environment"
    echo "  $0 start-prod      # Start production environment"
    echo "  $0 logs api        # View API service logs"
    echo "  $0 scale api 3     # Scale API to 3 replicas"
    echo ""
    echo "Note: Run this script from the docker_config directory"
}

# Main script logic
main() {
    # Check prerequisites
    check_docker
    check_compose

    case "${1:-help}" in
        build)
            build_images
            ;;
        start)
            start_services
            ;;
        start-prod)
            start_production
            ;;
        stop)
            stop_services
            ;;
        stop-prod)
            stop_production
            ;;
        restart)
            restart_services
            ;;
        logs)
            view_logs "$2"
            ;;
        logs-prod)
            view_production_logs "$2"
            ;;
        status)
            check_status
            ;;
        status-prod)
            check_production_status
            ;;
        scale)
            scale_services "$2" "$3"
            ;;
        cleanup)
            cleanup
            ;;
        cleanup-prod)
            cleanup_production
            ;;
        info)
            system_info
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
