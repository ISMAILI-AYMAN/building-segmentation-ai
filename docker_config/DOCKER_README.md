# 🐳 **Docker Setup for Building Segmentation Pipeline**

This document provides comprehensive instructions for deploying and managing the Building Segmentation Pipeline using Docker.

---

## 🎯 **Quick Start**

### **1. Prerequisites**
- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 8GB RAM
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)

### **2. Start Everything**
```bash
# Make the management script executable
chmod +x docker-manage.sh

# Start development environment
./docker-manage.sh start

# Or start production environment
./docker-manage.sh start-prod
```

---

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Server    │    │   Inference     │
│   (Flask)       │◄──►│   (FastAPI)     │◄──►│   Engine        │
│   Port: 5000    │    │   Port: 8000    │    │   (PyTorch)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Redis Cache   │
                    │   Port: 6379    │
                    └─────────────────┘
```

---

## 📁 **File Structure**

```
├── Dockerfile                    # Main application container
├── docker-compose.yml            # Development environment
├── docker-compose.prod.yml       # Production environment
├── docker-manage.sh              # Management script
├── nginx.conf                    # Nginx reverse proxy config
├── redis.conf                    # Redis configuration
├── prometheus.yml                # Monitoring configuration
└── DOCKER_README.md             # This file
```

---

## 🚀 **Deployment Options**

### **Development Environment**
```bash
# Start development services
./docker-manage.sh start

# View logs
./docker-manage.sh logs

# Check status
./docker-manage.sh status
```

### **Production Environment**
```bash
# Start production services
./docker-manage.sh start-prod

# View production logs
./docker-manage.sh logs-prod

# Check production status
./docker-manage.sh status-prod
```

---

## 🔧 **Service Configuration**

### **API Server**
- **Port**: 8000
- **Model Path**: `/app/models/trained_model_final/final_model.pth`
- **Device**: CUDA (GPU) or CPU
- **Memory Limit**: 4GB (dev) / 8GB (prod)
- **Health Check**: `/health` endpoint

### **Frontend**
- **Port**: 5000
- **API URL**: `http://api:8000`
- **Static Files**: Cached with Nginx
- **Health Check**: `/health` endpoint

### **Redis Cache**
- **Port**: 6379
- **Memory Limit**: 256MB
- **Persistence**: AOF + RDB snapshots
- **Policy**: LRU eviction

### **Nginx Reverse Proxy**
- **Port**: 80 (HTTP), 443 (HTTPS)
- **Load Balancing**: Round-robin
- **Rate Limiting**: API (10 req/s), Frontend (30 req/s)
- **Gzip Compression**: Enabled
- **Security Headers**: XSS, CSRF protection

---

## 📊 **Monitoring & Observability**

### **Prometheus**
- **Port**: 9090
- **Metrics**: API, Frontend, Redis, Nginx
- **Scrape Interval**: 15s
- **Retention**: 200 hours

### **Grafana**
- **Port**: 3000
- **Default Credentials**: admin/admin
- **Dashboards**: Pre-configured for all services
- **Alerts**: CPU, memory, response time

---

## 🛠️ **Management Commands**

### **Basic Operations**
```bash
# Build images
./docker-manage.sh build

# Start services
./docker-manage.sh start

# Stop services
./docker-manage.sh stop

# Restart services
./docker-manage.sh restart
```

### **Monitoring & Debugging**
```bash
# View logs
./docker-manage.sh logs              # All services
./docker-manage.sh logs api          # API only
./docker-manage.sh logs frontend     # Frontend only

# Check status
./docker-manage.sh status

# System information
./docker-manage.sh info
```

### **Scaling & Production**
```bash
# Scale API service
./docker-manage.sh scale api 3

# Production deployment
./docker-manage.sh start-prod

# Production cleanup
./docker-manage.sh cleanup-prod
```

---

## 🔒 **Security Features**

### **Container Security**
- Non-root user execution
- Read-only model and data volumes
- Resource limits and reservations
- Health checks and restart policies

### **Network Security**
- Isolated Docker networks
- Rate limiting per service
- Security headers (XSS, CSRF protection)
- Input validation and sanitization

### **Data Security**
- Encrypted volumes (production)
- Secure Redis configuration
- Environment variable management
- Secrets management (production)

---

## 📈 **Performance Optimization**

### **Resource Management**
- **Memory Limits**: Prevent OOM issues
- **CPU Limits**: Fair resource allocation
- **GPU Support**: CUDA acceleration
- **Volume Mounts**: Optimized I/O

### **Caching Strategy**
- **Redis**: In-memory caching
- **Nginx**: Static file caching
- **Browser**: Long-term caching
- **CDN**: Global distribution (production)

### **Load Balancing**
- **Nginx**: Reverse proxy with upstream
- **Docker Swarm**: Service replication
- **Health Checks**: Automatic failover
- **Circuit Breakers**: Fault tolerance

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. Port Conflicts**
```bash
# Check what's using the port
netstat -tulpn | grep :8000

# Stop conflicting services
sudo systemctl stop conflicting-service
```

#### **2. Memory Issues**
```bash
# Check container memory usage
docker stats

# Increase memory limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
```

#### **3. GPU Issues**
```bash
# Check NVIDIA Docker
nvidia-docker run --rm nvidia/cuda:11.0-base nvidia-smi

# Install NVIDIA Docker if needed
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
```

### **Debug Commands**
```bash
# Enter container
docker-compose exec api bash

# View container logs
docker-compose logs -f api

# Check container health
docker-compose ps

# Monitor resources
docker stats
```

---

## 🔄 **Updates & Maintenance**

### **Updating Services**
```bash
# Pull latest images
docker-compose pull

# Rebuild and restart
docker-compose up -d --build

# Zero-downtime update
docker-compose up -d --no-deps --build api
```

### **Backup & Recovery**
```bash
# Backup volumes
docker run --rm -v building-segmentation_redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis_backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v building-segmentation_redis_data:/data -v $(pwd):/backup alpine tar xzf /backup/redis_backup.tar.gz -C /data
```

---

## 🌐 **Production Deployment**

### **Environment Variables**
```bash
# Create .env file
cat > .env << EOF
MODEL_PATH=/app/models/trained_model_final/final_model.pth
HOST=0.0.0.0
PORT=8000
DEVICE=cuda
LOG_LEVEL=INFO
REDIS_URL=redis://redis:6379
EOF
```

### **SSL Configuration**
```bash
# Create SSL directory
mkdir -p ssl

# Generate self-signed certificate (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/nginx.key -out ssl/nginx.crt

# Use Let's Encrypt for production
certbot certonly --webroot -w /var/www/html -d yourdomain.com
```

### **Load Balancer Setup**
```bash
# Scale services
./docker-manage.sh scale api 3
./docker-manage.sh scale frontend 2

# Check status
./docker-manage.sh status-prod
```

---

## 📚 **Additional Resources**

### **Docker Documentation**
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Docker Swarm Mode](https://docs.docker.com/engine/swarm/)
- [Docker Security](https://docs.docker.com/engine/security/)

### **Monitoring & Logging**
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Redis Documentation](https://redis.io/documentation)

### **Performance Tuning**
- [Nginx Performance Tuning](https://nginx.org/en/docs/http/ngx_http_core_module.html)
- [Docker Performance Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

## 🆘 **Support & Troubleshooting**

### **Getting Help**
1. Check the logs: `./docker-manage.sh logs`
2. Verify service status: `./docker-manage.sh status`
3. Check system resources: `./docker-manage.sh info`
4. Review this documentation

### **Emergency Commands**
```bash
# Stop everything
./docker-manage.sh stop

# Clean up completely
./docker-manage.sh cleanup

# Restart from scratch
./docker-manage.sh start
```

---

## 🎉 **Success!**

Your Building Segmentation Pipeline is now running in Docker with:
- ✅ **Scalable Architecture** - Easy to scale up/down
- ✅ **Production Ready** - Monitoring, logging, health checks
- ✅ **Security Hardened** - Non-root users, rate limiting, security headers
- ✅ **Performance Optimized** - Caching, load balancing, resource management
- ✅ **Easy Management** - Simple commands for all operations

**Ready to process building segmentation requests at scale!** 🚀
