# AI Trading Platform Deployment Guide

This guide covers the deployment of the AI Trading Platform using Docker containers and Kubernetes orchestration.

## Prerequisites

### Required Tools

- **Docker** (20.10+): Container runtime
- **kubectl** (1.28+): Kubernetes CLI
- **Helm** (3.12+): Kubernetes package manager
- **AWS CLI** (2.0+): For cloud deployment
- **eksctl** (0.150+): For EKS cluster management

### Required Permissions

- Docker registry push/pull access
- Kubernetes cluster admin access
- AWS EKS and related services access (for cloud deployment)

## Quick Start

### Local Development Deployment

1. **Start with Docker Compose**:
   ```bash
   # Copy environment file
   cp .env.example .env
   
   # Edit configuration
   nano .env
   
   # Start services
   docker-compose up -d
   ```

2. **Verify deployment**:
   ```bash
   # Check service health
   curl http://localhost:8000/health
   
   # Run smoke tests
   python tests/smoke_tests.py --environment local
   ```

### Production Deployment

1. **Setup Kubernetes cluster**:
   ```bash
   # Create EKS cluster (AWS)
   ./scripts/setup-cluster.sh --cluster-name ai-trading-prod --region us-west-2
   
   # Or use existing cluster
   kubectl config use-context your-cluster-context
   ```

2. **Deploy application**:
   ```bash
   # Using deployment script
   ./scripts/deploy.sh --environment production --tag v1.0.0
   
   # Or using Helm
   ./scripts/deploy.sh --environment production --use-helm --tag v1.0.0
   ```

3. **Verify deployment**:
   ```bash
   # Check deployment status
   ./scripts/health-check.sh --environment production
   
   # Run comprehensive tests
   python tests/smoke_tests.py --environment production --verbose
   ```

## Deployment Methods

### Method 1: Docker Compose (Development)

Best for: Local development, testing, small-scale deployments

```bash
# Basic deployment
docker-compose up -d

# With custom configuration
ENVIRONMENT=staging docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d

# Scale services
docker-compose up -d --scale ml-worker=3 --scale data-processor=2
```

**Pros:**
- Simple setup
- Good for development
- Easy to customize

**Cons:**
- Limited scalability
- No high availability
- Manual management

### Method 2: Kubernetes Manifests

Best for: Production deployments, full control over configuration

```bash
# Deploy with kubectl
./scripts/deploy.sh --environment production

# Manual deployment
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/ml-worker-deployment.yaml
kubectl apply -f k8s/data-processor-deployment.yaml
kubectl apply -f k8s/ingress.yaml
```

**Pros:**
- Full control
- Production-ready
- Auto-scaling support

**Cons:**
- More complex
- Requires Kubernetes knowledge
- Manual updates

### Method 3: Helm Charts

Best for: Standardized deployments, multiple environments, easy updates

```bash
# Deploy with Helm
./scripts/deploy.sh --environment production --use-helm

# Manual Helm deployment
helm upgrade --install ai-trading-platform ./helm/ai-trading-platform \
  --namespace ai-trading-platform \
  --create-namespace \
  --values helm/ai-trading-platform/values-production.yaml
```

**Pros:**
- Templated configuration
- Easy updates
- Environment management
- Rollback support

**Cons:**
- Learning curve
- Template complexity

## Configuration

### Environment Variables

Key environment variables for deployment:

```bash
# Application
ENVIRONMENT=production
CONFIG_FILE=/app/config/production.yaml

# Database
DATABASE_URL=postgresql://user:pass@host:5432/db
POSTGRES_PASSWORD=secure_password

# Redis
REDIS_URL=redis://:password@host:6379/0
REDIS_PASSWORD=secure_password

# Exchange APIs
ROBINHOOD_USERNAME=your_username
ROBINHOOD_PASSWORD=your_password
OANDA_API_KEY=your_api_key
COINBASE_API_KEY=your_api_key
COINBASE_API_SECRET=your_api_secret

# Security
JWT_SECRET=your_jwt_secret

# Infrastructure
REGISTRY=ghcr.io/your-org/ai-trading-platform
IMAGE_TAG=v1.0.0
```

### Kubernetes Secrets

Create secrets for sensitive data:

```bash
# Create secrets
kubectl create secret generic trading-platform-secrets \
  --from-literal=postgres-password=secure_password \
  --from-literal=redis-password=secure_password \
  --from-literal=jwt-secret=your_jwt_secret \
  --from-literal=robinhood-username=your_username \
  --from-literal=robinhood-password=your_password \
  --from-literal=oanda-api-key=your_api_key \
  --from-literal=coinbase-api-key=your_api_key \
  --from-literal=coinbase-api-secret=your_api_secret \
  --namespace ai-trading-platform
```

### Resource Requirements

Minimum resource requirements:

| Service | CPU | Memory | Storage |
|---------|-----|--------|---------|
| API | 250m | 512Mi | - |
| ML Worker | 500m | 2Gi | 10Gi |
| Data Processor | 250m | 512Mi | - |
| PostgreSQL | 250m | 512Mi | 20Gi |
| Redis | 100m | 256Mi | 5Gi |

Production recommendations:

| Service | CPU | Memory | Storage |
|---------|-----|--------|---------|
| API | 500m | 1Gi | - |
| ML Worker | 2000m | 4Gi | 50Gi |
| Data Processor | 500m | 1Gi | - |
| PostgreSQL | 500m | 1Gi | 100Gi |
| Redis | 200m | 512Mi | 10Gi |

## Scaling

### Horizontal Pod Autoscaling

HPA is configured for all services:

```yaml
# API Service
minReplicas: 3
maxReplicas: 10
targetCPUUtilization: 70%
targetMemoryUtilization: 80%

# ML Worker
minReplicas: 2
maxReplicas: 8
targetCPUUtilization: 80%
targetMemoryUtilization: 85%

# Data Processor
minReplicas: 2
maxReplicas: 6
targetCPUUtilization: 75%
targetMemoryUtilization: 80%
```

### Manual Scaling

```bash
# Scale deployments manually
kubectl scale deployment api-deployment --replicas=5 -n ai-trading-platform
kubectl scale deployment ml-worker-deployment --replicas=4 -n ai-trading-platform

# Using Helm
helm upgrade ai-trading-platform ./helm/ai-trading-platform \
  --set replicaCount.api=5 \
  --set replicaCount.mlWorker=4 \
  --reuse-values
```

## Monitoring and Observability

### Health Checks

Built-in health check endpoints:

- **Liveness Probe**: `/health` - Basic application health
- **Readiness Probe**: `/ready` - Service readiness for traffic
- **Startup Probe**: `/startup` - Initial startup completion

### Metrics

Prometheus metrics are exposed on `/metrics`:

- HTTP request metrics
- Database connection metrics
- ML inference metrics
- Exchange API metrics
- Resource usage metrics

### Logging

Structured logging with JSON format:

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "service": "api",
  "message": "Request processed",
  "request_id": "req-123",
  "duration": 45.2
}
```

### Dashboards

Grafana dashboards are available for:

- System overview
- API performance
- ML model performance
- Database metrics
- Exchange connectivity

## Security

### Container Security

- Non-root user execution
- Read-only root filesystem
- Security contexts configured
- Resource limits enforced

### Network Security

- Network policies for pod-to-pod communication
- TLS encryption for all external traffic
- Service mesh (optional) for internal encryption

### Secrets Management

- Kubernetes secrets for sensitive data
- External secret management (AWS Secrets Manager, HashiCorp Vault)
- Secret rotation procedures

## Backup and Recovery

### Database Backup

```bash
# Create backup
kubectl exec postgres-pod -n ai-trading-platform -- \
  pg_dump -U trading_user trading_platform > backup.sql

# Restore backup
kubectl exec -i postgres-pod -n ai-trading-platform -- \
  psql -U trading_user trading_platform < backup.sql
```

### Model Backup

```bash
# Backup ML models
kubectl cp ai-trading-platform/ml-worker-pod:/app/checkpoints ./model-backup/

# Restore models
kubectl cp ./model-backup/ ai-trading-platform/ml-worker-pod:/app/checkpoints/
```

## Troubleshooting

### Common Issues

1. **Pod Startup Failures**:
   ```bash
   # Check pod logs
   kubectl logs -f pod-name -n ai-trading-platform
   
   # Check events
   kubectl describe pod pod-name -n ai-trading-platform
   ```

2. **Service Connectivity Issues**:
   ```bash
   # Test service connectivity
   kubectl exec -it api-pod -n ai-trading-platform -- curl postgres-service:5432
   
   # Check service endpoints
   kubectl get endpoints -n ai-trading-platform
   ```

3. **Resource Issues**:
   ```bash
   # Check resource usage
   kubectl top pods -n ai-trading-platform
   kubectl top nodes
   
   # Check resource limits
   kubectl describe pod pod-name -n ai-trading-platform
   ```

### Debugging Commands

```bash
# Get comprehensive status
./scripts/health-check.sh --environment production

# Check deployment rollout
kubectl rollout status deployment/api-deployment -n ai-trading-platform

# Get recent events
kubectl get events -n ai-trading-platform --sort-by='.lastTimestamp'

# Port forward for local access
kubectl port-forward service/api-service 8080:80 -n ai-trading-platform
```

## Rollback Procedures

### Automatic Rollback

```bash
# Rollback to previous version
./scripts/rollback.sh all

# Rollback specific deployment
./scripts/rollback.sh deployment api-deployment

# Rollback to specific revision
./scripts/rollback.sh deployment api-deployment 3
```

### Manual Rollback

```bash
# Using kubectl
kubectl rollout undo deployment/api-deployment -n ai-trading-platform

# Using Helm
helm rollback ai-trading-platform 1 -n ai-trading-platform
```

## Performance Optimization

### Resource Optimization

1. **CPU Optimization**:
   - Use CPU limits and requests
   - Enable CPU throttling
   - Optimize worker processes

2. **Memory Optimization**:
   - Set memory limits
   - Use memory-efficient algorithms
   - Enable garbage collection tuning

3. **Storage Optimization**:
   - Use SSD storage classes
   - Enable compression
   - Implement data lifecycle policies

### Network Optimization

1. **Service Mesh**: Consider Istio for advanced traffic management
2. **CDN**: Use CloudFront for static assets
3. **Caching**: Implement Redis caching strategies

## Maintenance

### Regular Tasks

1. **Update Dependencies**:
   ```bash
   # Update Helm charts
   helm repo update
   
   # Update container images
   docker pull python:3.11-slim
   ```

2. **Certificate Renewal**:
   ```bash
   # Check certificate expiry
   kubectl get certificates -n ai-trading-platform
   
   # Renew certificates (cert-manager handles this automatically)
   ```

3. **Database Maintenance**:
   ```bash
   # Vacuum database
   kubectl exec postgres-pod -n ai-trading-platform -- \
     psql -U trading_user -d trading_platform -c "VACUUM ANALYZE;"
   ```

### Upgrade Procedures

1. **Application Updates**:
   ```bash
   # Deploy new version
   ./scripts/deploy.sh --environment production --tag v1.1.0
   
   # Verify deployment
   ./scripts/health-check.sh --environment production
   ```

2. **Kubernetes Updates**:
   ```bash
   # Update cluster
   eksctl update cluster --name ai-trading-cluster --region us-west-2
   
   # Update node groups
   eksctl update nodegroup --cluster ai-trading-cluster --name general-workers
   ```

## Support and Documentation

### Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [Docker Documentation](https://docs.docker.com/)
- [AWS EKS Documentation](https://docs.aws.amazon.com/eks/)

### Getting Help

1. Check application logs
2. Review health check results
3. Consult troubleshooting section
4. Contact platform team

For production issues, use the emergency rollback procedure:

```bash
./scripts/rollback.sh emergency
```