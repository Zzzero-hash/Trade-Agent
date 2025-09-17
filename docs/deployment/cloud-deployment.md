# Cloud Deployment Guide

This guide covers deploying the AI Trading Platform to production cloud environments including AWS, Google Cloud, and Azure.

## Overview

The platform supports multiple deployment strategies:

- **Kubernetes**: Recommended for production scalability
- **Docker Compose**: Simple multi-container deployment
- **Serverless**: Function-based deployment for specific components
- **Hybrid**: Combination of managed services and containers

## Prerequisites

### Required Tools
- Docker and Docker Compose
- kubectl (Kubernetes CLI)
- Helm 3.x
- Cloud provider CLI (aws, gcloud, or az)
- Terraform (optional, for infrastructure as code)

### Cloud Resources
- Container registry (ECR, GCR, ACR)
- Kubernetes cluster or container service
- Managed database (RDS, Cloud SQL, Azure Database)
- Redis cache service
- Object storage (S3, GCS, Blob Storage)
- Load balancer and CDN

## Kubernetes Deployment

### 1. Prepare Container Images

Build and push images to your container registry:

```bash
# Build all images
docker build -f docker/Dockerfile.api -t your-registry/ai-trading-api:latest .
docker build -f docker/Dockerfile.ml-worker -t your-registry/ai-trading-ml:latest .
docker build -f docker/Dockerfile.data-processor -t your-registry/ai-trading-data:latest .

# Push to registry
docker push your-registry/ai-trading-api:latest
docker push your-registry/ai-trading-ml:latest
docker push your-registry/ai-trading-data:latest
```

### 2. Configure Kubernetes Manifests

Update the Kubernetes manifests in `k8s/` directory:

```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-trading-api
  namespace: ai-trading
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-trading-api
  template:
    metadata:
      labels:
        app: ai-trading-api
    spec:
      containers:
      - name: api
        image: your-registry/ai-trading-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ai-trading-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: ai-trading-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 3. Deploy with Helm

Use the provided Helm chart for easier deployment:

```bash
# Add custom values
cat > values-production.yaml << EOF
image:
  repository: your-registry/ai-trading-api
  tag: latest
  pullPolicy: Always

replicaCount: 3

service:
  type: LoadBalancer
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.your-domain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: api-tls
      hosts:
        - api.your-domain.com

database:
  external: true
  host: your-rds-endpoint.amazonaws.com
  port: 5432
  name: ai_trading_platform

redis:
  external: true
  host: your-redis-cluster.cache.amazonaws.com
  port: 6379

secrets:
  databaseUrl: "postgresql://user:pass@host:5432/db"
  redisUrl: "redis://host:6379/0"
  jwtSecret: "your-jwt-secret"
  exchangeApiKeys:
    robinhood:
      username: "your-username"
      password: "your-password"
    oanda:
      apiKey: "your-api-key"
      accountId: "your-account-id"
    coinbase:
      apiKey: "your-api-key"
      apiSecret: "your-api-secret"
      passphrase: "your-passphrase"

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
EOF

# Deploy with Helm
helm upgrade --install ai-trading-platform ./helm/ai-trading-platform \
  -f values-production.yaml \
  --namespace ai-trading \
  --create-namespace
```

### 4. Configure Ingress and SSL

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-trading-ingress
  namespace: ai-trading
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.your-domain.com
    - app.your-domain.com
    secretName: ai-trading-tls
  rules:
  - host: api.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-trading-api
            port:
              number: 80
  - host: app.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-trading-frontend
            port:
              number: 80
```

## AWS Deployment

### 1. EKS Cluster Setup

```bash
# Create EKS cluster with eksctl
eksctl create cluster \
  --name ai-trading-platform \
  --version 1.28 \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name ai-trading-platform
```

### 2. RDS Database Setup

```bash
# Create RDS PostgreSQL instance
aws rds create-db-instance \
  --db-instance-identifier ai-trading-db \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --engine-version 14.9 \
  --master-username aitrading \
  --master-user-password YourSecurePassword \
  --allocated-storage 100 \
  --storage-type gp2 \
  --vpc-security-group-ids sg-xxxxxxxxx \
  --db-subnet-group-name ai-trading-subnet-group \
  --backup-retention-period 7 \
  --multi-az \
  --storage-encrypted
```

### 3. ElastiCache Redis Setup

```bash
# Create Redis cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id ai-trading-redis \
  --cache-node-type cache.t3.micro \
  --engine redis \
  --num-cache-nodes 1 \
  --cache-subnet-group-name ai-trading-cache-subnet-group \
  --security-group-ids sg-xxxxxxxxx
```

### 4. ECR Repository Setup

```bash
# Create ECR repositories
aws ecr create-repository --repository-name ai-trading-platform/api
aws ecr create-repository --repository-name ai-trading-platform/ml-worker
aws ecr create-repository --repository-name ai-trading-platform/data-processor

# Get login token and login to ECR
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-west-2.amazonaws.com
```

### 5. Application Load Balancer

```yaml
# AWS Load Balancer Controller configuration
apiVersion: v1
kind: Service
metadata:
  name: ai-trading-api-nlb
  namespace: ai-trading
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: http
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: ai-trading-api
```

## Google Cloud Deployment

### 1. GKE Cluster Setup

```bash
# Create GKE cluster
gcloud container clusters create ai-trading-platform \
  --zone us-central1-a \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --machine-type n1-standard-2 \
  --enable-autorepair \
  --enable-autoupgrade

# Get credentials
gcloud container clusters get-credentials ai-trading-platform --zone us-central1-a
```

### 2. Cloud SQL Setup

```bash
# Create Cloud SQL PostgreSQL instance
gcloud sql instances create ai-trading-db \
  --database-version POSTGRES_14 \
  --tier db-n1-standard-2 \
  --region us-central1 \
  --storage-size 100GB \
  --storage-type SSD \
  --backup-start-time 03:00 \
  --enable-bin-log \
  --maintenance-window-day SUN \
  --maintenance-window-hour 04

# Create database
gcloud sql databases create ai_trading_platform --instance ai-trading-db

# Create user
gcloud sql users create aitrading --instance ai-trading-db --password YourSecurePassword
```

### 3. Memorystore Redis

```bash
# Create Redis instance
gcloud redis instances create ai-trading-redis \
  --size 1 \
  --region us-central1 \
  --redis-version redis_6_x \
  --tier standard
```

## Azure Deployment

### 1. AKS Cluster Setup

```bash
# Create resource group
az group create --name ai-trading-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group ai-trading-rg \
  --name ai-trading-aks \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10

# Get credentials
az aks get-credentials --resource-group ai-trading-rg --name ai-trading-aks
```

### 2. Azure Database for PostgreSQL

```bash
# Create PostgreSQL server
az postgres server create \
  --resource-group ai-trading-rg \
  --name ai-trading-db \
  --location eastus \
  --admin-user aitrading \
  --admin-password YourSecurePassword \
  --sku-name GP_Gen5_2 \
  --version 14

# Create database
az postgres db create \
  --resource-group ai-trading-rg \
  --server-name ai-trading-db \
  --name ai_trading_platform
```

### 3. Azure Cache for Redis

```bash
# Create Redis cache
az redis create \
  --resource-group ai-trading-rg \
  --name ai-trading-redis \
  --location eastus \
  --sku Standard \
  --vm-size c1
```

## Environment Configuration

### Production Environment Variables

Create a comprehensive environment configuration:

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/ai_trading_platform
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis
REDIS_URL=redis://host:6379/0
REDIS_MAX_CONNECTIONS=50

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
DEBUG=false
LOG_LEVEL=INFO

# Security
JWT_SECRET_KEY=your-super-secret-jwt-key
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Exchange API Keys (use secrets management)
ROBINHOOD_USERNAME=your-username
ROBINHOOD_PASSWORD=your-password
OANDA_API_KEY=your-oanda-key
OANDA_ACCOUNT_ID=your-account-id
COINBASE_API_KEY=your-coinbase-key
COINBASE_API_SECRET=your-coinbase-secret
COINBASE_PASSPHRASE=your-passphrase

# ML Configuration
MODEL_CACHE_DIR=/app/models
FEATURE_CACHE_SIZE=10000
BATCH_SIZE=64
DEVICE=cuda

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
LOG_AGGREGATION_ENABLED=true

# Ray Configuration
RAY_HEAD_NODE_HOST=ray-head-service
RAY_DASHBOARD_PORT=8265
```

## Secrets Management

### Kubernetes Secrets

```bash
# Create secrets from environment file
kubectl create secret generic ai-trading-secrets \
  --from-env-file=.env.production \
  --namespace ai-trading

# Or create individual secrets
kubectl create secret generic database-secret \
  --from-literal=url="postgresql://user:pass@host:5432/db" \
  --namespace ai-trading

kubectl create secret generic exchange-secrets \
  --from-literal=robinhood-username="username" \
  --from-literal=robinhood-password="password" \
  --from-literal=oanda-api-key="key" \
  --namespace ai-trading
```

### AWS Secrets Manager Integration

```yaml
# Use External Secrets Operator
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: ai-trading
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-west-2
      auth:
        secretRef:
          accessKeyID:
            name: aws-credentials
            key: access-key-id
          secretAccessKey:
            name: aws-credentials
            key: secret-access-key
---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: ai-trading-secrets
  namespace: ai-trading
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: ai-trading-secrets
    creationPolicy: Owner
  data:
  - secretKey: database-url
    remoteRef:
      key: ai-trading/database
      property: url
  - secretKey: jwt-secret
    remoteRef:
      key: ai-trading/auth
      property: jwt-secret
```

## Monitoring and Observability

### Prometheus and Grafana Setup

```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=admin123 \
  --set prometheus.prometheusSpec.retention=30d
```

### Application Metrics

Configure application metrics collection:

```python
# Add to your FastAPI app
from prometheus_client import Counter, Histogram, generate_latest
import time

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    
    REQUEST_DURATION.observe(time.time() - start_time)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Logging Configuration

```yaml
# Fluent Bit configuration for log aggregation
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
  namespace: ai-trading
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         1
        Log_Level     info
        Daemon        off
        Parsers_File  parsers.conf

    [INPUT]
        Name              tail
        Path              /var/log/containers/*ai-trading*.log
        Parser            docker
        Tag               kube.*
        Refresh_Interval  5

    [OUTPUT]
        Name  es
        Match *
        Host  elasticsearch.logging.svc.cluster.local
        Port  9200
        Index ai-trading-logs
```

## Scaling and Performance

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-trading-api-hpa
  namespace: ai-trading
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-trading-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### Vertical Pod Autoscaler

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: ai-trading-api-vpa
  namespace: ai-trading
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-trading-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: api
      maxAllowed:
        cpu: 2
        memory: 4Gi
      minAllowed:
        cpu: 100m
        memory: 128Mi
```

## Backup and Disaster Recovery

### Database Backups

```bash
# Automated PostgreSQL backups
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: ai-trading
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: postgres-backup
            image: postgres:14
            command:
            - /bin/bash
            - -c
            - |
              pg_dump $DATABASE_URL | gzip > /backup/backup-$(date +%Y%m%d-%H%M%S).sql.gz
              aws s3 cp /backup/backup-$(date +%Y%m%d-%H%M%S).sql.gz s3://ai-trading-backups/
            env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: ai-trading-secrets
                  key: database-url
            volumeMounts:
            - name: backup-storage
              mountPath: /backup
          volumes:
          - name: backup-storage
            emptyDir: {}
          restartPolicy: OnFailure
```

### Model Artifacts Backup

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: model-backup
  namespace: ai-trading
spec:
  schedule: "0 4 * * 0"  # Weekly on Sunday at 4 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: model-backup
            image: amazon/aws-cli
            command:
            - /bin/bash
            - -c
            - |
              aws s3 sync /app/models s3://ai-trading-model-backups/$(date +%Y%m%d)/
            volumeMounts:
            - name: model-storage
              mountPath: /app/models
          volumes:
          - name: model-storage
            persistentVolumeClaim:
              claimName: model-storage-pvc
          restartPolicy: OnFailure
```

## Security Best Practices

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ai-trading-network-policy
  namespace: ai-trading
spec:
  podSelector:
    matchLabels:
      app: ai-trading-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
    - protocol: TCP
      port: 443   # HTTPS
```

### Pod Security Standards

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ai-trading-api
  namespace: ai-trading
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: api
    image: ai-trading-api:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: cache
      mountPath: /app/cache
  volumes:
  - name: tmp
    emptyDir: {}
  - name: cache
    emptyDir: {}
```

## CI/CD Pipeline

### GitHub Actions Deployment

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2
    
    - name: Login to ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build and push images
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -f docker/Dockerfile.api -t $ECR_REGISTRY/ai-trading-api:$IMAGE_TAG .
        docker push $ECR_REGISTRY/ai-trading-api:$IMAGE_TAG
    
    - name: Deploy to EKS
      run: |
        aws eks update-kubeconfig --region us-west-2 --name ai-trading-platform
        helm upgrade --install ai-trading-platform ./helm/ai-trading-platform \
          --set image.tag=${{ github.sha }} \
          --namespace ai-trading
```

## Troubleshooting

### Common Issues

#### Pod Startup Failures
```bash
# Check pod status
kubectl get pods -n ai-trading

# View pod logs
kubectl logs -f deployment/ai-trading-api -n ai-trading

# Describe pod for events
kubectl describe pod <pod-name> -n ai-trading
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl run -it --rm debug --image=postgres:14 --restart=Never -- \
  psql postgresql://user:pass@host:5432/db

# Check database service
kubectl get svc -n ai-trading
kubectl describe svc ai-trading-db -n ai-trading
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n ai-trading
kubectl top nodes

# View HPA status
kubectl get hpa -n ai-trading
kubectl describe hpa ai-trading-api-hpa -n ai-trading
```

### Health Checks

Implement comprehensive health checks:

```python
@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # Check database
    try:
        await database.execute("SELECT 1")
        health_status["services"]["database"] = "healthy"
    except Exception as e:
        health_status["services"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check Redis
    try:
        await redis.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check ML models
    try:
        model_status = await model_service.health_check()
        health_status["services"]["ml_models"] = model_status
    except Exception as e:
        health_status["services"]["ml_models"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status
```

This comprehensive cloud deployment guide covers all major cloud providers and deployment scenarios, ensuring your AI Trading Platform can be successfully deployed and scaled in production environments.