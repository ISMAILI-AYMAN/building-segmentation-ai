@echo off
setlocal enabledelayedexpansion

REM Building Segmentation Pipeline Deployment Script for Windows
REM This script handles deployment for both development and production environments

set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Function to print colored output
:print_status
echo %BLUE%[INFO]%NC% %~1
goto :eof

:print_success
echo %GREEN%[SUCCESS]%NC% %~1
goto :eof

:print_warning
echo %YELLOW%[WARNING]%NC% %~1
goto :eof

:print_error
echo %RED%[ERROR]%NC% %~1
goto :eof

REM Function to check prerequisites
:check_prerequisites
call :print_status "Checking prerequisites..."

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker is not installed. Please install Docker Desktop first."
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit /b 1
)

REM Check if Docker daemon is running
docker info >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker daemon is not running. Please start Docker Desktop first."
    exit /b 1
)

call :print_success "Prerequisites check passed!"
goto :eof

REM Function to check if model exists
:check_model
call :print_status "Checking if trained model exists..."

if not exist "trained_model_final\final_model.pth" (
    call :print_warning "Trained model not found at trained_model_final\final_model.pth"
    call :print_warning "You may need to train a model first or download a pre-trained model."
    set /p "continue=Do you want to continue anyway? (y/N): "
    if /i not "!continue!"=="y" (
        call :print_error "Deployment cancelled."
        exit /b 1
    )
) else (
    call :print_success "Trained model found!"
)
goto :eof

REM Function to create necessary directories
:create_directories
call :print_status "Creating necessary directories..."

if not exist "logs" mkdir logs
if not exist "data\raw" mkdir data\raw
if not exist "api\uploads" mkdir api\uploads
if not exist "api\results" mkdir api\results
if not exist "api\static" mkdir api\static

call :print_success "Directories created!"
goto :eof

REM Function to deploy development environment
:deploy_dev
call :print_status "Deploying development environment..."

cd docker_config

REM Stop existing containers
call :print_status "Stopping existing containers..."
docker-compose down --remove-orphans

REM Build and start containers
call :print_status "Building and starting containers..."
docker-compose up --build -d

REM Wait for services to be ready
call :print_status "Waiting for services to be ready..."
timeout /t 30 /nobreak >nul

REM Check service health
call :print_status "Checking service health..."

REM Check API health
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    call :print_warning "API service health check failed. Check logs with: docker-compose logs api"
) else (
    call :print_success "API service is healthy!"
)

REM Check frontend health
curl -f http://localhost:5000/health >nul 2>&1
if errorlevel 1 (
    call :print_warning "Frontend service health check failed. Check logs with: docker-compose logs frontend"
) else (
    call :print_success "Frontend service is healthy!"
)

cd ..

call :print_success "Development environment deployed successfully!"
call :print_status "Access points:"
echo   - Frontend: http://localhost:5000
echo   - API: http://localhost:8000
echo   - API Docs: http://localhost:8000/docs
goto :eof

REM Function to deploy production environment
:deploy_prod
call :print_status "Deploying production environment..."

cd docker_config

REM Stop existing containers
call :print_status "Stopping existing containers..."
docker-compose -f docker-compose.prod.yml down --remove-orphans

REM Build and start containers
call :print_status "Building and starting production containers..."
docker-compose -f docker-compose.prod.yml up --build -d

REM Wait for services to be ready
call :print_status "Waiting for services to be ready..."
timeout /t 60 /nobreak >nul

REM Check service health
call :print_status "Checking service health..."

REM Check API health
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    call :print_warning "API service health check failed. Check logs with: docker-compose -f docker-compose.prod.yml logs api"
) else (
    call :print_success "API service is healthy!"
)

REM Check frontend health
curl -f http://localhost:5000/health >nul 2>&1
if errorlevel 1 (
    call :print_warning "Frontend service health check failed. Check logs with: docker-compose -f docker-compose.prod.yml logs frontend"
) else (
    call :print_success "Frontend service is healthy!"
)

REM Check monitoring services
curl -f http://localhost:9090/-/healthy >nul 2>&1
if errorlevel 1 (
    call :print_warning "Prometheus health check failed."
) else (
    call :print_success "Prometheus monitoring is healthy!"
)

curl -f http://localhost:3000/api/health >nul 2>&1
if errorlevel 1 (
    call :print_warning "Grafana health check failed."
) else (
    call :print_success "Grafana dashboard is healthy!"
)

cd ..

call :print_success "Production environment deployed successfully!"
call :print_status "Access points:"
echo   - Frontend: http://localhost:5000
echo   - API: http://localhost:8000
echo   - API Docs: http://localhost:8000/docs
echo   - Prometheus: http://localhost:9090
echo   - Grafana: http://localhost:3000 (admin/admin123)
goto :eof

REM Function to stop services
:stop_services
call :print_status "Stopping services..."

cd docker_config

if "%~1"=="prod" (
    docker-compose -f docker-compose.prod.yml down
    call :print_success "Production services stopped!"
) else (
    docker-compose down
    call :print_success "Development services stopped!"
)

cd ..
goto :eof

REM Function to show logs
:show_logs
call :print_status "Showing logs..."

cd docker_config

if "%~1"=="prod" (
    docker-compose -f docker-compose.prod.yml logs -f
) else (
    docker-compose logs -f
)

cd ..
goto :eof

REM Function to show status
:show_status
call :print_status "Service status..."

cd docker_config

if "%~1"=="prod" (
    docker-compose -f docker-compose.prod.yml ps
) else (
    docker-compose ps
)

cd ..
goto :eof

REM Function to clean up
:cleanup
call :print_status "Cleaning up..."

cd docker_config

if "%~1"=="prod" (
    docker-compose -f docker-compose.prod.yml down -v --remove-orphans
) else (
    docker-compose down -v --remove-orphans
)

cd ..

call :print_success "Cleanup completed!"
goto :eof

REM Function to show help
:show_help
echo Building Segmentation Pipeline Deployment Script for Windows
echo.
echo Usage: %~nx0 [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   dev           Deploy development environment
echo   prod          Deploy production environment
echo   stop [env]    Stop services (env: dev/prod, default: dev)
echo   logs [env]    Show logs (env: dev/prod, default: dev)
echo   status [env]  Show service status (env: dev/prod, default: dev)
echo   cleanup [env] Clean up containers and volumes (env: dev/prod, default: dev)
echo   help          Show this help message
echo.
echo Examples:
echo   %~nx0 dev                    # Deploy development environment
echo   %~nx0 prod                   # Deploy production environment
echo   %~nx0 stop prod              # Stop production services
echo   %~nx0 logs dev               # Show development logs
echo   %~nx0 status                 # Show development service status
echo   %~nx0 cleanup prod           # Clean up production environment
goto :eof

REM Main script logic
if "%~1"=="dev" (
    call :check_prerequisites
    if errorlevel 1 exit /b 1
    call :check_model
    if errorlevel 1 exit /b 1
    call :create_directories
    call :deploy_dev
) else if "%~1"=="prod" (
    call :check_prerequisites
    if errorlevel 1 exit /b 1
    call :check_model
    if errorlevel 1 exit /b 1
    call :create_directories
    call :deploy_prod
) else if "%~1"=="stop" (
    call :stop_services "%~2"
) else if "%~1"=="logs" (
    call :show_logs "%~2"
) else if "%~1"=="status" (
    call :show_status "%~2"
) else if "%~1"=="cleanup" (
    call :cleanup "%~2"
) else if "%~1"=="help" (
    call :show_help
) else (
    call :print_error "Unknown command: %~1"
    echo.
    call :show_help
    exit /b 1
)

endlocal
