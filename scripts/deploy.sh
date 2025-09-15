#!/bin/bash

# AI Trading Platform Deployment Script
# Requirements: 6.1, 6.3, 6.6

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${ENVIRONMENT:-staging}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
NAMESPACE="ai-trading-platform"
TIMEOUT="${TIMEOUT:-600}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required tools
    local required_tools=("docker" "kubectl" "helm")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_error "$tool is not installed or not in PATH"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    local services=("api" "ml-worker" "data-processor")
    local registry="${REGISTRY:-ghcr.io/your-org/ai-trading-platform}"
    
    for service in "${services[@]}"; do
        log_info "Building $service image..."
        
        docker build \
            -f "docker/Dockerfile.$service" \
            -t "$registry/$service:$IMAGE_TAG" \
            --build-arg BUILD_ENV="$ENVIRONMENT" \
            .
        
        if [ $? -eq 0 ]; then
            log_success "Built $service image successfully"
        else
            log_error "Failed to build $service image"
            exit 1
        fi
    done
}

# Push images to registry
push_images() {
    log_info "Pushing images to registry..."
    
    local services=("api" "ml-worker" "data-processor")
    local registry="${REGISTRY:-ghcr.io/your-org/ai-trading-platform}"
    
    for service in "${services[@]}"; do
        log_info "Pushing $service image..."
        
        docker push "$registry/$service:$IMAGE_TAG"
        
        if [ $? -eq 0 ]; then
            log_success "Pushed $service image successfully"
        else
            log_error "Failed to push $service image"
            exit 1
        fi
    done
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace if it doesn't exist..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        log_success "Created namespace $NAMESPACE"
    fi
}

# Deploy with Kubernetes manifests
deploy_kubernetes() {
    log_info "Deploying with Kubernetes manifests..."
    
    # Update image tags in deployment files
    local temp_dir=$(mktemp -d)
    cp -r "$PROJECT_ROOT/k8s" "$temp_dir/"
    
    local registry="${REGISTRY:-ghcr.io/your-org/ai-trading-platform}"
    
    # Replace image tags
    sed -i "s|ai-trading-platform/api:latest|$registry/api:$IMAGE_TAG|g" "$temp_dir/k8s/api-deployment.yaml"
    sed -i "s|ai-trading-platform/ml-worker:latest|$registry/ml-worker:$IMAGE_TAG|g" "$temp_dir/k8s/ml-worker-deployment.yaml"
    sed -i "s|ai-trading-platform/data-processor:latest|$registry/data-processor:$IMAGE_TAG|g" "$temp_dir/k8s/data-processor-deployment.yaml"
    
    # Apply manifests in order
    local manifests=(
        "namespace.yaml"
        "configmap.yaml"
        "secrets.yaml"
        "postgres-deployment.yaml"
        "redis-deployment.yaml"
        "api-deployment.yaml"
        "ml-worker-deployment.yaml"
        "data-processor-deployment.yaml"
        "ingress.yaml"
    )
    
    for manifest in "${manifests[@]}"; do
        log_info "Applying $manifest..."
        kubectl apply -f "$temp_dir/k8s/$manifest"
    done
    
    # Clean up temp directory
    rm -rf "$temp_dir"
    
    log_success "Applied all Kubernetes manifests"
}

# Deploy with Helm
deploy_helm() {
    log_info "Deploying with Helm..."
    
    local registry="${REGISTRY:-ghcr.io/your-org/ai-trading-platform}"
    
    # Add required Helm repositories
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
    helm repo add jetstack https://charts.jetstack.io
    helm repo update
    
    # Deploy infrastructure components
    log_info "Deploying cert-manager..."
    helm upgrade --install cert-manager jetstack/cert-manager \
        --namespace cert-manager \
        --create-namespace \
        --set installCRDs=true \
        --wait \
        --timeout="${TIMEOUT}s"
    
    log_info "Deploying ingress controller..."
    helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
        --namespace ingress-nginx \
        --create-namespace \
        --set controller.metrics.enabled=true \
        --wait \
        --timeout="${TIMEOUT}s"
    
    # Deploy application
    log_info "Deploying AI Trading Platform..."
    helm upgrade --install ai-trading-platform "$PROJECT_ROOT/helm/ai-trading-platform" \
        --namespace "$NAMESPACE" \
        --create-namespace \
        --set image.registry="$(dirname "$registry")" \
        --set image.repository="$(basename "$registry")" \
        --set image.tag="$IMAGE_TAG" \
        --set environment="$ENVIRONMENT" \
        --values "$PROJECT_ROOT/helm/ai-trading-platform/values-$ENVIRONMENT.yaml" \
        --wait \
        --timeout="${TIMEOUT}s"
    
    log_success "Deployed with Helm successfully"
}

# Wait for deployments to be ready
wait_for_deployments() {
    log_info "Waiting for deployments to be ready..."
    
    local deployments=("api-deployment" "ml-worker-deployment" "data-processor-deployment")
    
    for deployment in "${deployments[@]}"; do
        log_info "Waiting for $deployment to be ready..."
        
        kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="${TIMEOUT}s"
        
        if [ $? -eq 0 ]; then
            log_success "$deployment is ready"
        else
            log_error "$deployment failed to become ready"
            exit 1
        fi
    done
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Wait a bit for services to fully start
    sleep 30
    
    # Get service endpoints
    local api_url
    if kubectl get ingress trading-platform-ingress -n "$NAMESPACE" &> /dev/null; then
        api_url="https://$(kubectl get ingress trading-platform-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}')"
    else
        # Use port-forward for testing
        kubectl port-forward service/api-service 8080:80 -n "$NAMESPACE" &
        local port_forward_pid=$!
        sleep 5
        api_url="http://localhost:8080"
    fi
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    local health_response
    health_response=$(curl -s -f "$api_url/health" || echo "FAILED")
    
    if [ "$health_response" != "FAILED" ]; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        
        # Kill port-forward if it was started
        if [ -n "${port_forward_pid:-}" ]; then
            kill $port_forward_pid 2>/dev/null || true
        fi
        
        exit 1
    fi
    
    # Kill port-forward if it was started
    if [ -n "${port_forward_pid:-}" ]; then
        kill $port_forward_pid 2>/dev/null || true
    fi
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Check pod status
    log_info "Checking pod status..."
    kubectl get pods -n "$NAMESPACE"
    
    # Check service status
    log_info "Checking service status..."
    kubectl get services -n "$NAMESPACE"
    
    # Check ingress status
    log_info "Checking ingress status..."
    kubectl get ingress -n "$NAMESPACE"
    
    # Run basic API tests
    log_info "Running basic API tests..."
    python3 "$PROJECT_ROOT/tests/smoke_tests.py" --environment "$ENVIRONMENT"
    
    if [ $? -eq 0 ]; then
        log_success "Smoke tests passed"
    else
        log_error "Smoke tests failed"
        exit 1
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
}

# Main deployment function
main() {
    log_info "Starting deployment to $ENVIRONMENT environment with image tag $IMAGE_TAG"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    check_prerequisites
    
    if [ "${SKIP_BUILD:-false}" != "true" ]; then
        build_images
    fi
    
    if [ "${SKIP_PUSH:-false}" != "true" ]; then
        push_images
    fi
    
    create_namespace
    
    if [ "${USE_HELM:-false}" == "true" ]; then
        deploy_helm
    else
        deploy_kubernetes
    fi
    
    wait_for_deployments
    run_health_checks
    
    if [ "${SKIP_SMOKE_TESTS:-false}" != "true" ]; then
        run_smoke_tests
    fi
    
    log_success "Deployment completed successfully!"
    
    # Display access information
    echo
    log_info "Access Information:"
    echo "  Namespace: $NAMESPACE"
    echo "  Environment: $ENVIRONMENT"
    echo "  Image Tag: $IMAGE_TAG"
    
    if kubectl get ingress trading-platform-ingress -n "$NAMESPACE" &> /dev/null; then
        local ingress_host
        ingress_host=$(kubectl get ingress trading-platform-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}')
        echo "  API URL: https://$ingress_host"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD="true"
            shift
            ;;
        --skip-push)
            SKIP_PUSH="true"
            shift
            ;;
        --use-helm)
            USE_HELM="true"
            shift
            ;;
        --skip-smoke-tests)
            SKIP_SMOKE_TESTS="true"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -e, --environment ENV    Deployment environment (default: staging)"
            echo "  -t, --tag TAG           Image tag (default: latest)"
            echo "  --skip-build            Skip building Docker images"
            echo "  --skip-push             Skip pushing images to registry"
            echo "  --use-helm              Use Helm for deployment instead of kubectl"
            echo "  --skip-smoke-tests      Skip running smoke tests"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main