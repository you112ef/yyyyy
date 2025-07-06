# 🚀 Deployment Guide - Sperm Analysis System

**Developer:** Youssef Shitiwi (يوسف شتيوي)  
**System:** AI-Powered Computer-Assisted Sperm Analysis

## Overview

This guide covers various deployment options for the Sperm Analysis System, from local development to production-ready deployments with Docker, Kubernetes, and cloud platforms.

## 🖥️ Local Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+ (for Android development)
- Docker & Docker Compose
- Git
- CUDA-compatible GPU (optional, for faster processing)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/youssef-shitiwi/sperm-analysis-system.git
cd sperm-analysis-system

# Install Python dependencies
pip install -r requirements.txt

# Start the development server
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# API will be available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

### Environment Variables
Create a `.env` file in the project root:

```bash
# Application Settings
ENVIRONMENT=development
DEBUG=true
APP_NAME=Sperm Analysis API
APP_VERSION=1.0.0

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=1

# AI Model Configuration
MODEL_PATH=data/models/best/sperm_detection_best.pt
DEFAULT_CONFIDENCE_THRESHOLD=0.3
DEFAULT_IOU_THRESHOLD=0.5

# Directory Configuration
UPLOAD_DIR=data/uploads
OUTPUT_DIR=data/results
TEMP_DIR=data/temp

# Processing Configuration
MAX_CONCURRENT_ANALYSES=2
MAX_UPLOAD_SIZE_MB=500

# Database (Optional)
DATABASE_URL=postgresql://postgres:password@localhost:5432/sperm_analysis
USE_DATABASE=false

# Redis (Optional)
REDIS_URL=redis://localhost:6379/0
USE_REDIS=false

# Security
API_KEY=your-secret-api-key
ENABLE_AUTH=false
CORS_ORIGINS=*

# Logging
LOG_LEVEL=INFO
ENABLE_ACCESS_LOG=true
```

## 🐳 Docker Deployment

### Simple Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Production Deployment
```bash
# Use production profile
docker-compose --profile production up -d

# This includes:
# - API service with production settings
# - PostgreSQL database
# - Redis cache
# - Nginx reverse proxy
# - SSL termination
```

### Development with Hot Reload
```bash
# Use development profile
docker-compose --profile dev up -d api-dev

# API with hot reload at http://localhost:8001
```

### Training Environment
```bash
# Start training environment with Jupyter
docker-compose --profile training up -d training

# Access Jupyter at http://localhost:8888
# Password: Check container logs for token
```

### Service Profiles

#### Core Services (Default)
- `api`: Main FastAPI application
- `postgres`: PostgreSQL database
- `redis`: Redis cache

#### Production Profile
```bash
docker-compose --profile production up -d
```
- All core services
- `nginx`: Reverse proxy with SSL
- Optimized for production

#### Development Profile
```bash
docker-compose --profile dev up -d
```
- `api-dev`: Development API with hot reload
- Simplified setup for development

#### Training Profile
```bash
docker-compose --profile training up -d
```
- `training`: Jupyter environment for model training
- GPU support enabled
- ML development tools

#### Monitoring Profile
```bash
docker-compose --profile monitoring up -d
```
- `prometheus`: Metrics collection
- `grafana`: Monitoring dashboards

## ☸️ Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (local or cloud)
- kubectl configured
- Helm 3+ (optional but recommended)

### Using Kubernetes Manifests

1. **Create namespace:**
```bash
kubectl create namespace sperm-analysis
```

2. **Deploy PostgreSQL:**
```yaml
# k8s/postgres.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: sperm-analysis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: sperm_analysis
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          value: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: sperm-analysis
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

3. **Deploy API:**
```yaml
# k8s/api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sperm-analysis-api
  namespace: sperm-analysis
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sperm-analysis-api
  template:
    metadata:
      labels:
        app: sperm-analysis-api
    spec:
      containers:
      - name: api
        image: sperm-analysis:latest
        env:
        - name: DATABASE_URL
          value: postgresql://postgres:password@postgres:5432/sperm_analysis
        - name: USE_DATABASE
          value: "true"
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /api/v1/health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: sperm-analysis-api
  namespace: sperm-analysis
spec:
  selector:
    app: sperm-analysis-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

4. **Deploy Ingress:**
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sperm-analysis-ingress
  namespace: sperm-analysis
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: sperm-analysis-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sperm-analysis-api
            port:
              number: 80
```

5. **Apply manifests:**
```bash
kubectl apply -f k8s/
```

### Using Helm Chart

1. **Create Helm chart:**
```bash
helm create sperm-analysis-chart
```

2. **Install with Helm:**
```bash
helm install sperm-analysis ./sperm-analysis-chart \
  --namespace sperm-analysis \
  --create-namespace \
  --set image.tag=latest \
  --set database.enabled=true \
  --set ingress.enabled=true \
  --set ingress.host=api.yourdomain.com
```

## ☁️ Cloud Deployment

### AWS Deployment

#### Using AWS ECS with Fargate
```bash
# Build and push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com

docker build -t sperm-analysis .
docker tag sperm-analysis:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/sperm-analysis:latest
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/sperm-analysis:latest

# Deploy with ECS CLI or CloudFormation
```

#### ECS Task Definition
```json
{
  "family": "sperm-analysis",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "sperm-analysis-api",
      "image": "123456789012.dkr.ecr.us-west-2.amazonaws.com/sperm-analysis:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://username:password@rds-endpoint:5432/sperm_analysis"
        },
        {
          "name": "USE_DATABASE",
          "value": "true"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/sperm-analysis",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Platform

#### Using Cloud Run
```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/sperm-analysis

# Deploy to Cloud Run
gcloud run deploy sperm-analysis \
  --image gcr.io/PROJECT_ID/sperm-analysis \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DATABASE_URL=postgresql://user:pass@/db?host=/cloudsql/PROJECT_ID:REGION:INSTANCE \
  --add-cloudsql-instances PROJECT_ID:REGION:INSTANCE
```

### Azure Container Instances

```bash
# Create resource group
az group create --name sperm-analysis-rg --location eastus

# Deploy container
az container create \
  --resource-group sperm-analysis-rg \
  --name sperm-analysis \
  --image sperm-analysis:latest \
  --dns-name-label sperm-analysis \
  --ports 8000 \
  --environment-variables \
    DATABASE_URL=postgresql://user:pass@azure-postgres:5432/sperm_analysis \
    USE_DATABASE=true
```

## 🔧 Production Configuration

### Environment Variables for Production
```bash
# Production settings
ENVIRONMENT=production
DEBUG=false
WORKERS=4

# Security
ENABLE_AUTH=true
API_KEY=your-secure-api-key-here
CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com

# Database
DATABASE_URL=postgresql://user:password@prod-db:5432/sperm_analysis
USE_DATABASE=true

# Redis
REDIS_URL=redis://prod-redis:6379/0
USE_REDIS=true

# Performance
MAX_CONCURRENT_ANALYSES=4
MAX_UPLOAD_SIZE_MB=500

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
```

### Nginx Configuration
```nginx
# /etc/nginx/sites-available/sperm-analysis
upstream sperm_analysis_backend {
    server 127.0.0.1:8000;
    # Add more servers for load balancing
    # server 127.0.0.1:8001;
    # server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/ssl/certs/yourdomain.com.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.com.key;

    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";

    # Upload size limit
    client_max_body_size 500M;

    location / {
        proxy_pass http://sperm_analysis_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://sperm_analysis_backend/api/v1/health/simple;
    }
}
```

### SSL Certificate Setup
```bash
# Using Let's Encrypt with Certbot
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 📊 Monitoring & Logging

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'sperm-analysis-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/api/v1/health/metrics'
    scrape_interval: 30s
```

### Grafana Dashboard
Import the provided dashboard JSON file or create custom dashboards monitoring:
- API response times
- Processing queue status
- System resource usage
- Error rates
- Analysis completion times

### Logging Configuration
```yaml
# docker-compose.override.yml for production logging
version: '3.8'
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    # Or use centralized logging
    # logging:
    #   driver: "fluentd"
    #   options:
    #     fluentd-address: localhost:24224
    #     tag: sperm-analysis-api
```

## 🔐 Security Considerations

### API Security
- Enable authentication in production
- Use HTTPS/TLS encryption
- Implement rate limiting
- Validate all inputs
- Sanitize file uploads

### Database Security
- Use strong passwords
- Enable SSL connections
- Regular backups
- Access control

### Network Security
- Use VPCs/private networks
- Firewall configuration
- VPN access for management

## 📱 Android App Deployment

### Building the APK
```bash
cd android

# Debug build
./gradlew assembleDebug

# Release build (requires keystore)
./gradlew assembleRelease

# APK location: app/build/outputs/apk/
```

### Play Store Deployment
1. Create signed APK with release keystore
2. Upload to Google Play Console
3. Configure app listing and metadata
4. Submit for review

### Alternative Distribution
- Direct APK download from your website
- Amazon Appstore
- F-Droid (if open source)

## 🧪 Testing Deployment

### Health Checks
```bash
# API health
curl http://your-domain.com/api/v1/health

# Database connectivity
curl http://your-domain.com/api/v1/health | jq '.database_connected'

# Processing queue
curl http://your-domain.com/api/v1/queue
```

### Load Testing
```bash
# Using Apache Bench
ab -n 1000 -c 10 http://your-domain.com/api/v1/health/simple

# Using hey
hey -n 1000 -c 10 http://your-domain.com/api/v1/health/simple
```

### End-to-End Testing
```python
# Test video upload and analysis
import requests
import time

# Upload test video
with open('test_video.mp4', 'rb') as f:
    upload_response = requests.post(
        'http://your-domain.com/api/v1/upload',
        files={'file': f}
    )

upload_id = upload_response.json()['upload_id']

# Start analysis
analysis_response = requests.post(
    'http://your-domain.com/api/v1/analyze',
    data={
        'upload_id': upload_id,
        'config': '{"fps": 30.0, "confidence_threshold": 0.3}'
    }
)

analysis_id = analysis_response.json()['analysis_id']

# Poll for completion
while True:
    status_response = requests.get(
        f'http://your-domain.com/api/v1/status/{analysis_id}'
    )
    status = status_response.json()['status']
    
    if status in ['completed', 'failed']:
        break
    
    time.sleep(10)

print(f"Analysis {status}")
```

## 🚨 Troubleshooting

### Common Issues

#### "Model not found" Error
```bash
# Download default model
docker-compose exec api python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

#### High Memory Usage
- Reduce `MAX_CONCURRENT_ANALYSES`
- Increase worker memory limits
- Enable video preprocessing to reduce resolution

#### Slow Processing
- Enable GPU support
- Optimize video preprocessing
- Use faster storage (SSD)
- Scale horizontally with multiple workers

#### Database Connection Issues
- Check database credentials
- Verify network connectivity
- Review firewall rules
- Check connection limits

### Log Analysis
```bash
# View API logs
docker-compose logs -f api

# Database logs
docker-compose logs -f postgres

# System metrics
docker stats
```

## 📋 Maintenance

### Regular Tasks
- Monitor disk space in `data/` directories
- Clean up old analysis results
- Update dependencies
- Backup database
- Monitor SSL certificate expiration
- Review security logs

### Backup Strategy
```bash
# Database backup
pg_dump -h localhost -U postgres sperm_analysis > backup.sql

# Data backup
tar -czf data_backup.tar.gz data/

# Automated backup script
./scripts/backup.sh
```

### Updates and Migrations
```bash
# Update application
git pull origin main
docker-compose build
docker-compose up -d

# Database migrations
docker-compose exec api python -c "
import asyncio
from backend.utils.database import run_migrations
asyncio.run(run_migrations())
"
```

---

**Support:** For deployment assistance, contact Youssef Shitiwi (يوسف شتيوي)  
**Documentation:** Complete API documentation available at `/docs` endpoint  
**Monitoring:** Set up alerts for critical metrics and health checks